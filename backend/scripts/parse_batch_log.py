import re
import sys
import csv
import argparse

def parse_log_to_csv(log_file_path, output_csv_path=None):
    # Regex patterns
    # "User: 46758, Class: 2025-20-958779-0-00, Year: 2025, Semester: 20"
    # 기존 Regex가 너무 엄격해서 특수문자가 포함된 ID를 놓칠 수 있으므로 수정 ([^,]+ : 쉼표 전까지 모두 잡음)
    user_class_pattern = re.compile(r"User: ([^,]+), Class: ([^,]+),")
    
    # "    - 영상(VOD): 0개"
    vod_pattern = re.compile(r"영상\(VOD\): (\d+)개")
    
    # "    - 자료(PDF/etc): 0개"
    pdf_pattern = re.compile(r"자료\(PDF/etc\): (\d+)개")
    
    # " -> 성공: ..."
    success_pattern = re.compile(r"-> 성공:")
    
    # " -> 실패: ..."
    fail_pattern = re.compile(r"-> 실패:")

    # Data structure
    rows = []
    current_row = {}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # New Block Start
                uc_match = user_class_pattern.search(line)
                if uc_match:
                    # Save previous row if exists
                    if current_row:
                        rows.append(current_row)
                    
                    # Init new row
                    current_row = {
                        "user_id": uc_match.group(1),
                        "cls_id": uc_match.group(2),
                        "vod_cnt": 0,
                        "pdf_cnt": 0,
                        "status": "Unknown"
                    }
                    continue
                
                # VOD Count
                v_match = vod_pattern.search(line)
                if v_match and current_row:
                    current_row["vod_cnt"] = int(v_match.group(1))
                    
                # PDF Count
                p_match = pdf_pattern.search(line)
                if p_match and current_row:
                    current_row["pdf_cnt"] = int(p_match.group(1))
                    
                # Status
                if current_row:
                    if success_pattern.search(line):
                        # 성공 메시지가 여러 번 나올 수 있음 (1단계, 3단계 등). 
                        # 하나라도 성공이면 일단 Processing 중인 것으로 간주하거나, 
                        # 마지막 상태를 따라감. 
                        # 여기서는 "작업 요청 성공"이 뜨면 Success로 봄.
                        current_row["status"] = "Success"
                    elif fail_pattern.search(line):
                        current_row["status"] = "Fail"
                        
            # Append last row
            if current_row:
                rows.append(current_row)
                
    except FileNotFoundError:
        print(f"Error: File not found {log_file_path}")
        return

    # Output to CSV
    header = ["User ID", "Class ID", "VOD Count", "PDF Count", "Status"]
    
    if output_csv_path:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["user_id", "cls_id", "vod_cnt", "pdf_cnt", "status"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to {output_csv_path}")
    else:
        # Print to stdout
        writer = csv.DictWriter(sys.stdout, fieldnames=["user_id", "cls_id", "vod_cnt", "pdf_cnt", "status"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse daily batch log to CSV")
    parser.add_argument("logfile", help="Path to the log file")
    parser.add_argument("--output", help="Path to output CSV file (optional)")
    
    args = parser.parse_args()
    
    parse_log_to_csv(args.logfile, args.output)
