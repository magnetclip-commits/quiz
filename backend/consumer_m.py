# consumer.py
import redis
import json
from config import REDIS_CONFIG
from task import process_downloadM
from datetime import datetime

redis_client = redis.Redis(**REDIS_CONFIG)
stream_name = "downloadM_queue"
group_name = "download_group"
consumer_name = "consumer_1"

try:
    redis_client.xgroup_create(stream_name, group_name, id='0-0', mkstream=True)
except redis.exceptions.ResponseError as e:
    if "BUSYGROUP" in str(e):
        pass
    else:
        raise e

while True:
    messages = redis_client.xreadgroup(group_name, consumer_name, {stream_name: '>'}, count=1, block=5000)
    if messages:
        for stream, message_list in messages:
            for msg_id, message_data in message_list:
                # message_data의 모든 키와 값을 문자열로 디코딩
                decoded_message_data = {
                    k.decode('utf-8') if isinstance(k, bytes) else k:
                    v.decode('utf-8') if isinstance(v, bytes) else v
                    for k, v in message_data.items()
                }
                # file_ids 필드가 있다면 JSON 역직렬화를 시도하여 리스트로 복원
                if "file_ids" in decoded_message_data:
                    try:
                        decoded_message_data["file_ids"] = json.loads(decoded_message_data["file_ids"])
                    except json.JSONDecodeError:
                        # 만약 JSON 파싱에 실패하면, 기존 문자열을 리스트로 감싸서 처리
                        decoded_message_data["file_ids"] = [decoded_message_data["file_ids"]]
                
                result = process_downloadM.delay(decoded_message_data)
                print(f"{datetime.now().isoformat()} - 메시지 {msg_id}를 Celery 작업 {result.id}에 할당")
                redis_client.xack(stream_name, group_name, msg_id)

