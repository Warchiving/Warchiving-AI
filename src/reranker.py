from sentence_transformers import CrossEncoder
from .config import CROSS_ENCODER_NAME

class CrossEncoderReranker:
    def __init__(self):
        print(f"\n[Step 3] Cross-Encoder 로드 중: {CROSS_ENCODER_NAME}")
        self.model = CrossEncoder(CROSS_ENCODER_NAME)

    def rerank(self, query, candidates):
        """
        Input: 유저 쿼리, RRF 후보 리스트
        Process: 쿼리와 후보 문장을 Pair로 묶어 정밀 채점
        Output: cross_score가 추가된 후보 리스트
        """
        if not candidates: return []
        print(f"  - 총 {len(candidates)}개 후보 문장 정밀 재순위화 시작...")
        
        # Cross-Encoder 전용 입력 포맷: [[질문, 문장1], [질문, 문장2], ...]
        pairs = [[query, c['text']] for c in candidates]
        scores = self.model.predict(pairs)
        
        for i, score in enumerate(scores):
            candidates[i]['cross_score'] = float(score)
            
        return sorted(candidates, key=lambda x: x['cross_score'], reverse=True)