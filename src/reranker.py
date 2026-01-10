"""
Cross-Encoder 기반 재순위화 로직을 구현할 파일.

- retrieve된 Top-K passage들에 대해
  (query, text_chunk)를 Cross-Encoder에 넣어 점수 재계산
  
  [역할]
- recall 단계에서 끌고 온 passage 후보들에 대해
- query + passage를 함께 읽고 의미적 정합성 점수 계산

  [중요]
  - 여기서 계산된 rerank_score만이
    "이 문장이 쿼리에 실제로 맞는가?"를 판단하는 점수임
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        query + passage를 함께 읽는 cross-encoder
        """
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        passages: List[Dict],
        top_k: int = 50,
    ) -> List[Dict]:
        """
        passages: hybrid_search 결과 (rrf 기반 recall 결과)
        passages: [
          {
            "text_chunk": str,
            "aspect": str,
            "venue_id": int,
            ...
          }
        ]
        """
        pairs = [(query, p["text_chunk"]) for p in passages]
        scores = self.model.predict(pairs)

        for p, s in zip(passages, scores):
            p["rerank_score"] = float(s)

        passages = sorted(
            passages,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )
        return passages[:top_k]
