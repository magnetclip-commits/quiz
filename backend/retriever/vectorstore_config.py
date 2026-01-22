from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

class VectorDBConfig:
    vector_index = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",  # 유클리디안 거리 (COSINE(코사인 유사도)도 가능)
        "params": {"nlist": 256,  # 클러스터의 갯수
                "nprobe": 16}  # 최근접 클러스터 16개 방문해서 검색 후 top-k 결과 반환 / top - k는 as_retriever()에서 설정
    }
    # sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    varchar_index = {"index_type": "Trie"}
    varchar_eq_index = {"index_type": "INVERTED"}
    int_eq_index =  {"index_type": "INVERTED"}

    def get_vdb_schema(self):
        vdb_fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=3000, nullable=False),
            FieldSchema(name="cls_id", dtype=DataType.VARCHAR, max_length=3000, nullable=False),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=3000, nullable=False),
            FieldSchema(name="storage_url", dtype=DataType.VARCHAR, max_length=3000, nullable=False),
            FieldSchema(name="week", dtype=DataType.INT64, nullable=True), # AIANT
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=3000, nullable=True), # AIANT
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=9000, nullable=False),
            # FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=3072, nullable=False),
            FieldSchema(name="metadata", dtype=DataType.JSON, nullable=True)
        ]
        vdb_schema = CollectionSchema(vdb_fields, description="data with embeddings")
        return vdb_schema

    def create_vdb_indices(self, collection:Collection):
        collection.create_index("storage_url", self.varchar_eq_index) # retriever filter에 사용
        collection.create_index("embeddings", self.vector_index)
        collection.create_index("user_id", self.varchar_eq_index) # retriever filter에 사용
        collection.create_index("cls_id", self.varchar_eq_index) # retriever filter에 사용
        collection.create_index("file_id", self.varchar_eq_index) # retriever filter에 사용
        collection.create_index("week", self.int_eq_index) # retriever filter에 사용
        collection.create_index("title", self.varchar_eq_index) # retriever filter에 사용
        # collection.create_index("sparse_vector", self.sparse_index)
        return collection