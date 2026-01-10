# # ChromaDB 저장 및 Hybrid Search(RRF 포함)
# # src/vector_db.py

# """
# ChromaDB 저장 및 Hybrid Search(RRF 포함)를 구현할 파일.

# - 이후 단계에서:
#   - processed.parquet를 읽어와서
#   - Chroma 컬렉션에 (id, vector, metadata)로 적재
#   - dense + BM25 하이브리드 검색 & RRF 구현
# """

# def init_vector_db():
#     """TODO: ChromaDB 초기화 로직 작성 예정."""
#     pass


# src/vector_db.py

import os
from typing import List, Dict, Any, Tuple

import pandas as pd
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from .config import PROCESSED_PARQUET_PATH
from .embedder import BGEEmbedder



def load_passage_df(parquet_path: str = PROCESSED_PARQUET_PATH) -> pd.DataFrame:
    """
    processed.parquet를 읽어오는 헬퍼 함수.
    컬럼: venue_id, hall_name, review_idx, aspect, text_chunk, vector
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found at {parquet_path}")
    df = pd.read_parquet(parquet_path)
    # doc_id를 위해 index를 고정시켜 둠
    df = df.reset_index(drop=True)
    df["doc_id"] = df.index.astype(str)  # Chroma용
    return df


class DenseSparseIndex:
    """
    - Dense: ChromaDB (+ BGE 임베딩)
    - Sparse: BM25 (text_chunk)
    - Hybrid: Dense + Sparse 결과를 RRF로 결합
    """

    def __init__(
        self,
        passage_df: pd.DataFrame,
        chroma_path: str = "./chroma_db",
        collection_name: str = "wedding_passages",
    ):
        self.df = passage_df # 이 시점부터 doc_id가 시스템의 절대 기준 키

        # Dense: ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # 컬렉션 가져오기 또는 생성
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Sparse: BM25
        docs = self.df["text_chunk"].fillna("").astype(str).tolist()
        self.tokenized_docs: List[List[str]] = [d.split() for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Dense Query용 임베더
        self.embedder = BGEEmbedder()

    # 인덱스 구축 (Dense / Chroma)
    def build_dense_index(self, batch_size: int = 512):
        """
        Parquet에 저장된 vector를 그대로 Chroma에 올려도 되지만,
        여기서는 df에 vector 컬럼이 있다고 가정하고 그대로 push.
        (만약 vector가 없으면, 다시 embed해서 사용 가능)
        """
        print("📦 Populating ChromaDB collection with existing vectors...")

        # 이미 데이터가 있다면 초기화할지 말지 결정 (지금은 일단 비움)
        if self.collection.count() > 0:
                    print(f"⚠️ 기존 데이터({self.collection.count()}건) 발견. 초기화 후 재구축합니다.")
                    
                    # 1. 현재 컬렉션의 이름을 가져옵니다.
                    target_name = self.collection.name 
                    
                    # 2. 컬렉션을 삭제하고 새로 만듭니다.
                    self.client.delete_collection(name=target_name)
                    self.collection = self.client.create_collection(name=target_name) 
        
        ids = self.df["doc_id"].tolist()
        documents = self.df["text_chunk"].tolist()
        metadatas = self.df[
            ["venue_id", "hall_name", "aspect"]
        ].to_dict(orient="records")

        vectors = self.df["vector"].tolist()  # list[list[float]]

        # batch로 Chroma에 적재
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_embs = vectors[i : i + batch_size]

            batch_embs = [
                emb.tolist() if hasattr(emb, "tolist") else list(emb)
                for emb in batch_embs
            ]

            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embs,
            )

        print(f"✅ Chroma collection populated ({self.collection.count()} docs).")


    # Dense Retrieval
    def dense_search(
        self,
        query_text: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Dense retriever.
        - 쿼리 텍스트를 BGE로 임베딩
        - ChromaDB에서 유사한 벡터 Top-K 검색
        return: [(doc_id, score), ...]
        """
        # 1) 쿼리 임베딩
        q_vec = self.embedder.embed_texts([query_text])[0]

        # 2) Chroma에 질의
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            include=["distances"],
        )

        ids = results["ids"][0]
        distances = results["distances"][0]

        dense_results: List[Tuple[str, float]] = []
        for doc_id, dist in zip(ids, distances):
            # 거리가 작을수록 유사 → 간단히 score = -distance 로 변환
            score = -float(dist)
            dense_results.append((doc_id, score))

        return dense_results


    # Sparse Retrieval (BM25)
    def sparse_search(
        self,
        query_text: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Sparse retriever (BM25).
        - text_chunk 전체를 BM25 인덱스
        - 쿼리 텍스트를 공백 기준 토큰화 후 BM25 점수 계산
        return: [(doc_id, score), ...]
        """
        query_tokens = query_text.split()
        # 각 문서에 대한 BM25 점수
        scores = self.bm25.get_scores(query_tokens)  # len = num_docs

        scores_series = pd.Series(scores)
        top_idx = scores_series.nlargest(top_k).index.tolist()

        sparse_results: List[Tuple[str, float]] = []
        for idx in top_idx:
            doc_id = self.df.iloc[idx]["doc_id"]
            score = float(scores[idx])
            sparse_results.append((doc_id, score))

        return sparse_results



    # RRF Fusion (dense + sparse)
    @staticmethod
    def rrf_fusion(
        ranked_lists: List[List[str]],  # 각 리스트는 doc_id 순서대로
        k: int = 60,
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion:
        score(d) = sum_{lists} 1 / (k + rank(d, list))
        여기서는 rank는 1-based index.
        """
        scores: Dict[str, float] = {}

        for results in ranked_lists:
            for rank, doc_id in enumerate(results, start=1):
                scores.setdefault(doc_id, 0.0)
                scores[doc_id] += 1.0 / (k + rank)

        return scores

    def hybrid_search(
        self,
        query_text: str,
        top_k_dense: int = 50,
        top_k_sparse: int = 50,
        k_final: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Retrieval:
        1) Dense Top-K
        2) Sparse Top-K
        3) RRF로 두 랭킹을 결합
        4) 최종 상위 k_final passage 반환

        return: [
          {
            "doc_id": ...,
            "venue_id": ...,
            "hall_name": ...,
            "review_idx": ...,
            "aspect": ...,
            "text_chunk": ...,
            "rrf_score": ...,
            "dense_score": ... (optional),
            "sparse_score": ... (optional)
          },
          ...
        ]
        """

        print(f"🔍 Hybrid search for query: {query_text}")

        # 1) dense / sparse 각각 검색
        dense = self.dense_search(query_text, top_k=top_k_dense)
        sparse = self.sparse_search(query_text, top_k=top_k_sparse)

        # doc_id 순서 리스트만 추출 (RRF는 순위만 필요)
        dense_ids = [doc_id for doc_id, _ in dense]
        sparse_ids = [doc_id for doc_id, _ in sparse]

        # 2) RRF 점수 계산
        rrf_scores = self.rrf_fusion([dense_ids, sparse_ids])

        # 3) doc_id → dense/sparse raw score 매핑 (디버깅/설명용)
        dense_dict = {doc_id: score for doc_id, score in dense}
        sparse_dict = {doc_id: score for doc_id, score in sparse}

        # 4) RRF 점수 기준으로 상위 k_final 선택
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:k_final]

        results: List[Dict[str, Any]] = []
        for doc_id, rrf_score in top_docs:
            row = self.df[self.df["doc_id"] == doc_id].iloc[0]
            results.append(
                {
                    "doc_id": doc_id,
                    "venue_id": row["venue_id"],
                    "hall_name": row["hall_name"],
                    "aspect": row["aspect"],
                    "text_chunk": row["text_chunk"],
                    "rrf_score": float(rrf_score),
                    "dense_score": float(dense_dict.get(doc_id, 0.0)),
                    "sparse_score": float(sparse_dict.get(doc_id, 0.0)),
                }
            )

        return results
