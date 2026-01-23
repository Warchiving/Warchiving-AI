import chromadb
from rank_bm25 import BM25Okapi
import numpy as np
from .config import VENUE_ID_COL

class DenseSparseIndex:
    def __init__(self, df_processed, client_path="./chroma_db"):
        self.df = df_processed
        self.client = chromadb.PersistentClient(path=client_path)
        # 코사인 유사도 기반 컬렉션 생성/로드
        self.collection = self.client.get_or_create_collection(
            name="wedding_collection", 
            metadata={"hnsw:space": "cosine"}
        )

    def build_chroma_db(self):
        """Parquet 데이터를 ChromaDB에 적재"""
        print("\n[Step 2-1] ChromaDB에 데이터 적재를 시작합니다...")
        ids, embeddings, metadatas = [], [], []
        print(f"📊 현재 DF 컬럼 목록: {self.df.columns.tolist()}")
        
        for idx, row in self.df.iterrows():
            doc_id = f"{row[VENUE_ID_COL]}_{row['aspect']}_{idx}"
            ids.append(doc_id)
            embeddings.append(row['vector'].tolist())
            metadatas.append({
                "hall_id": str(row[VENUE_ID_COL]),
                "hall_name": row['hall_name'],
                "aspect": row['aspect'],
                "text": row['text_chunk']
            })
            
            
        BATCH_SIZE = 512 
        total_len = len(ids)
        
        print(f"📊 총 {total_len}개의 데이터를 {BATCH_SIZE}개씩 나누어 적재합니다.")
        
        for i in range(0, total_len, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, total_len)
            
            # 슬라이싱 (i부터 end_idx까지)
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            # upsert를 사용하여 안전하게 적재
            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            
            # 512개 단위로 진행 상황 출력
            if end_idx % (BATCH_SIZE * 2) == 0 or end_idx == total_len:
                print(f"   - Progress: {end_idx}/{total_len} 적재 완료")

        print(f"✅ ChromaDB 구축 완료: 총 {total_len}개의 리뷰 구절이 저장되었습니다.")
        
        
    def get_dense_hits(self, query, aspect, embedder_model, n_results=50):
        """BGE-M3 벡터를 이용한 의미 기반 검색 (Dense)"""
        print(f"  - [{aspect}] 측면 Dense 검색 중...")
        query_vector = embedder_model.embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_vector],
            where={"aspect": aspect}, # 특정 컬럼만 필터링
            n_results=n_results
        )
        ranks = {res_id: i + 1 for i, res_id in enumerate(results['ids'][0])}
        meta_map = {res_id: meta for res_id, meta in zip(results['ids'][0], results['metadatas'][0])}
        return ranks, meta_map

    def get_sparse_hits(self, query, aspect, n_results=50):
            """BM25를 이용한 키워드 기반 검색 (Sparse)"""
            print(f"  - [{aspect}] 측면 Sparse 검색 중...")
            
            # [수정] reset_index를 하지 않아야 원본 데이터프레임의 idx를 보존할 수 있습니다.
            target_df = self.df[self.df['aspect'] == aspect] 
            if target_df.empty: return {}
            
            tokenized_corpus = [doc.split() for doc in target_df['text_chunk'].tolist()]
            bm25 = BM25Okapi(tokenized_corpus)
            
            scores = bm25.get_scores(query.split())
            top_indices_in_target = np.argsort(scores)[::-1][:n_results]
            
            ranks = {}
            for rank, i in enumerate(top_indices_in_target):
                # target_df.index[i]를 통해 원본의 고유 index(idx)를 가져옵니다.
                actual_idx = target_df.index[i]
                venue_id = target_df.iloc[i][VENUE_ID_COL]
                
                # [수정] build_chroma_db와 동일한 포맷의 ID 생성
                doc_id = f"{venue_id}_{aspect}_{actual_idx}"
                ranks[doc_id] = rank + 1
                
            return ranks
        
    def calculate_rrf(self, d_ranks, s_ranks, metadata_map, k=60, top_n=50):
        """Dense와 Sparse의 순위를 결합 (RRF)"""
        all_ids = set(d_ranks.keys()) | set(s_ranks.keys())
        rrf_list = []
        
        for doc_id in all_ids:
            # 기본 순위를 크게 잡아서(100) 결과에 없는 경우 점수를 낮게 줌
            d_rank = d_ranks.get(doc_id, 100)
            s_rank = s_ranks.get(doc_id, 100)
            
            score = (1 / (k + d_rank)) + (1 / (k + s_rank))
            
            # [체크] metadata_map은 Dense 검색 결과에서 채워지므로, 
            # Sparse에서만 나온 결과는 원본 df에서 메타데이터를 직접 가져와야 할 수도 있습니다.
            meta = metadata_map.get(doc_id)
            
            if not meta:
                # Sparse에서만 발견된 경우 원본 df에서 정보 추출 (안전장치)
                try:
                    # doc_id 형태: "1001_catering_58" -> 마지막 숫자가 idx
                    original_idx = int(doc_id.split('_')[-1])
                    row = self.df.loc[original_idx]
                    meta = {
                        "hall_id": str(row[VENUE_ID_COL]),
                        "hall_name": row['hall_name'],
                        "aspect": row['aspect'],
                        "text": row['text_chunk']
                    }
                except:
                    continue

            rrf_list.append({**meta, "rrf_score": score})
        
        if not rrf_list: return []
        
        sorted_res = sorted(rrf_list, key=lambda x: x['rrf_score'], reverse=True)[:top_n]
        print(f"  - [{sorted_res[0]['aspect']}] RRF 완료: 상위 {len(sorted_res)}개 추출")
        return sorted_res