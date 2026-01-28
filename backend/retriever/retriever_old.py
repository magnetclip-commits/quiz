from pymilvus import (
    connections, Collection, AnnSearchRequest, utility
)
from pymilvus.client.types import HybridExtraList
from langchain_core.documents.base import Document
import logging
import os
import sys
import numpy as np
from pathlib import Path
import logging

# .env 설정
from dotenv import load_dotenv
current_dir = str(Path(__file__).parent)
load_dotenv(str(Path(current_dir)/ ".env"))

# local 개발을 위한 python path 설정
sys.path += [current_dir]

from embedding import bge_m3_api_encode, gemini_encode
from vectorstore_config import VectorDBConfig
    
class VectorDB:
    def __init__(self):
        """
        Milvus 연결, Collection schema 및 index들을 불러온 후 적용, Collection 불러오기(load)
        """
        connections.connect("default",
                            host=os.getenv("MILVUS_HOST"),
                            port=os.getenv("MILVUS_PORT")
        )
        self.vdb_cfg = VectorDBConfig()

    def load_collection(self, collection_name:str):
        logging.info(f"collection_name: {collection_name}")
        
        if collection_name in utility.list_collections():
            self.collection = Collection(name=collection_name)
        else:
            self.collection = Collection(name=collection_name,
                                        schema=self.vdb_cfg.get_vdb_schema())
            self.collection = self.vdb_cfg.create_vdb_indices(self.collection)
        
        self.collection.load()
        return self.collection
            
    def _rerank_docs(self, dense_query_embeddings, docs, top_k, similarity_threshold):
        doc_vectors = bge_m3_api_encode([doc.page_content for doc in docs], return_sparse=False)['dense_vecs']
        similarities = [np.dot(dense_query_embeddings, doc_vector) for doc_vector in doc_vectors]
        logging.info(f"rerank_similarities of docs:{similarities}\n\n")
        filtered_pairs = [(sim, doc) for sim, doc in zip(similarities, docs)] # if sim >= similarity_threshold]
        sorted_pairs = sorted(filtered_pairs, key=lambda x: x[0], reverse=True)
        sorted_docs = [pair[1] for pair in sorted_pairs]
        return sorted_docs[:top_k]

    def retriever(self, search_query:str,
                        collection_name:str,
                        expr_str:str=None,
                        top_k:int=15,
                        dense_retriever_limit:int=25,
                        ) -> list[Document]:
        """_summary_
        VectorDB를 사용한 문서 반환기.
        - Document는 langchain의 Document임

        Args:
            search_query (str): 사용자로부터 들어온 질의(쿼리)

        Returns:
            list[Document]: 벡터DB에서 반환된 documents들
        """
        
        query_embeddings = bge_m3_api_encode([search_query], return_sparse=True)
        rerank_query_embeddings = query_embeddings["dense_vecs"]
        dense_query_embeddings = gemini_encode([search_query])[0]
        dense_query_embeddings = [dense_query_embeddings]
        distance_threshold = 0.2
        similarity_threshold = 0.54 # rerank
        
        self.load_collection(collection_name=collection_name)
        if not expr_str:
            expr_str = ""

        dense_req = AnnSearchRequest(dense_query_embeddings,
                                    "embeddings", 
                                    self.vdb_cfg.vector_index, 
                                    limit=dense_retriever_limit,
                                    expr = expr_str
                                    )
        dense_res = self.collection.hybrid_search(
                                                    [dense_req],
                                                    rerank=None,
                                                    limit=dense_retriever_limit,
                                                    output_fields=['*']
                                                )[0]
        logging.info(f"""dense search result{[{"distance": a_res["distance"], "pk": a_res["pk"]} 
                                                for a_res in dense_res]
                                                }\n\n""")
        res = dense_res
        
        docs = []
        for i in res:
            if i.distance < distance_threshold: # distance라 작을수록 가까움.
                text = i.fields['text']
                doc = Document(page_content=text, metadata={**i.fields['metadata'], "distance": i.distance, "pk": i.pk})
                docs.append(doc)
            docs = sorted(docs, key=lambda x: x.metadata["distance"])
        
        return_docs = list()
        if len(docs) == 0:
            logging.info("cause of empty context: no documents")
            pass
        else:
            rerank_docs = self._rerank_docs(rerank_query_embeddings,
                                            docs,
                                            top_k=top_k,
                                            similarity_threshold=similarity_threshold)
            return_docs += rerank_docs

        return return_docs
    
    def query_by_expr(
        self,
        expr_str: str,
        output_fields: list[str] | None = ["user_id", "cls_id", "file_id", "week", "title", "text", "metadata"],
        limit: int = 1000,
        return_in_documents=False
    ) -> list[Document]:
        """
        Milvus vector search 없이 expr 조건만으로 문서 조회

        Args:
            collection_name (str): Milvus collection name
            expr_str (str): Milvus boolean expression
            output_fields (list[str] | None): 조회할 필드 목록 (None이면 전체)
            limit (int): 최대 조회 개수

        Returns:
            list[Document]
        """

        self.load_collection(os.getenv("COLLECTION_NAME"))

        if output_fields is None:
            output_fields = ["*"]

        results:HybridExtraList = self.collection.query(
            expr=expr_str,
            output_fields=output_fields,
            limit=limit,
        )
        if return_in_documents and ("text" in output_fields and "metadata" in output_fields):
            docs: list[Document] = []

            for row in results:
                text = row.get("text", "")
                metadata = row.get("metadata", {}).copy()

                docs.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )
        else:
            docs:list[dict] = [result for result in results]

        return docs
        
if __name__ == "__main__":
    vdb = VectorDB()
    
    # # retreiver 사용 시
    docs = vdb.retriever(
                            search_query = '사후송금방식을 선호하는 사람은 누구야?',
                            collection_name="hallym_dev9", # collection명
                            expr_str='file_id == "0dab821e-74a0-4045-b573-8a4652fb5a34"', # milvus 필터링 https://milvus.io/docs/boolean.md
                            top_k=15,
                            dense_retriever_limit=25,
                        )
    
    # query 사용 시
    # docs = vdb.query_by_expr(
    #             expr_str='file_id == "0dab821e-74a0-4045-b573-8a4652fb5a34"',
    #             # output_fields=["text", "metadata"],
    #             # limit=1000,
    #             # return_in_documents=False
    #         )
    
    # # 그 외 기능이 더 필요할 경우: milvus collection 직접 사용
    # collection = vdb.load_collection(collection_name = os.getenv("COLLECTION_NAME"))
    # docs = collection.query(
    #     expr='file_id == "0dab821e-74a0-4045-b573-8a4652fb5a34"',
    #     output_fields=["text", "metadata"],
    # )
    # # 또는 
    # vdb.load_collection(collection_name = os.getenv("COLLECTION_NAME"))
    # docs = vdb.collection.query(
    #     expr='file_id == "0dab821e-74a0-4045-b573-8a4652fb5a34"',
    #     output_fields=["text", "metadata"],
    # )
    
    print(docs)
