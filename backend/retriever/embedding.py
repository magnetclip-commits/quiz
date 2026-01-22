import requests
from collections import defaultdict
import logging
import os
from google.genai import Client, types
client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

def bge_m3_api_encode(sentences:list[str], return_sparse:bool=False) -> tuple[int, dict]:
    try:
        url = 'https://api.deepinfra.com/v1/inference/BAAI/bge-m3-multi'
        headers = {
            'Authorization': f'bearer {os.getenv("DEEPINFRA_API_KEY")}',
            'Content-Type': 'application/json'
        }
        data = {
            'inputs': sentences,
            'sparse': str(return_sparse).lower(),
        }
        response = requests.post(url, headers=headers, json=data)
        # logger.debug(f"response: {response}")
        response_json:dict = response.json()
        status_code:int = response.status_code
        
        if status_code == 200:
            query_embeddings:dict = {
                'dense_vecs': response_json['embeddings']
            }
            
            if return_sparse:
                sparse_vectors = response_json['sparse']
                lexical_weights = list()
                for sparse_vector in sparse_vectors:
                    lexical_weight = defaultdict(int, {idx:prob for idx, prob in enumerate(sparse_vector) if prob != 0.0})
                    lexical_weights.append(lexical_weight)
                
                query_embeddings['lexical_weights'] = lexical_weights
        else:
            raise Exception(f"{status_code}: {response_json}")        
        return query_embeddings
    except Exception as e:
        logging.error(e)
        
def gemini_encode(texts:list[str]):
    content_embeddings = client.models.embed_content(
                                            model="gemini-embedding-001",
                                            contents=texts,
                                            config= types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                                        ).embeddings
    embeddings = [content_embedding.values for content_embedding in content_embeddings]
    return embeddings