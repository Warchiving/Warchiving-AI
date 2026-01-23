# src/main.py
import time
import pandas as pd
import numpy as np
from transformers import pipeline
import csv
from datetime import datetime
from .config import *
from .embedder import BGEEmbedder
from .vector_db import DenseSparseIndex
from .reranker import CrossEncoderReranker
from .aggregator import aggregate_to_providers

# 의도 분류기 (Zero-shot)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 평가를 위한 골드셋
EVAL_GROUND_TRUTH = {
    # "주차공간이 넓고, 신부대기실이 넓고 채광이 좋은 곳": [1199, 1269, 1135, 1030, 1031],
    # "호텔에서 식을 올리고 역과 가까운 곳" : [1003, 1025, 1109, 1112, 1107],
    # "채플홀이고 식사가 훌륭한 곳": [1008, 1111, 1209, 1210],
    "야외 웨딩홀에 식대가 맛있는 곳": [1066, 1133, 1216, 1191, 1163, 1113],
    # "단독홀에 하객동선이 좋은 곳" : [1009, 1012, 1019, 1029, 1031],
}

# 초기 raw data를 벡터화 할 때, 한번만 주석을 풀고 실행합니다.
# def run_build_embeddings():
#     """
#     (이미 완성된 함수)
#     chromaDB에 적재할 벡터 파일 만들기
#     1) raw csv → 2) parquet with vectors
#     """
#     embedder = BGEEmbedder()
#     embedder.build_vector_parquet(
#         input_csv_path=RAW_CSV_PATH,
#         output_parquet_path=PROCESSED_PARQUET_PATH,
#     )

def classifier_user_intent(query):
    """
    [Task 1] 유저 쿼리 분석 (Target Aspect 추출)
    Zero-shot Classification을 사용하여 쿼리에서 관련 속성을 추출합니다.
    """
    print("\n" + "="*30 + "\n[Task 1] Intent Analysis\n" + "="*30)
    
    target_labels = ASPECT_COLUMNS[:7]  # 설정된 주요 Aspect들
    
    # 1) 모델을 통한 의도 분류 (실제 가동 시 주석 해제)
    # clf_res = classifier(query, target_labels, multi_label=True)
    # relevant_aspects = [l for l, s in zip(clf_res['labels'], clf_res['scores']) if s > 0.5]
    
    # 테스트용 하드코딩 (분류기 속도가 느릴 경우를 대비)
    relevant_aspects = ['hall_vibe', 'catering']  # 예: 단독홀에 하객동선이 좋은 곳
    
    num_query_aspects = len(relevant_aspects)
    
    # 방어 로직: 추출된 의도가 없을 경우
    if num_query_aspects == 0:
        num_query_aspects = 1
        print("⚠️ 유저 의도에서 추출된 Aspect가 없습니다. 기본값 1로 설정합니다.")
    
    print(f"🎯 유저 의도 분석 결과: {relevant_aspects} (총 {num_query_aspects}개)")
    return relevant_aspects, num_query_aspects


def hybrid_retrieval(query, relevant_aspects, index_service, embedder, df):
    """
    [Task 2] Hybrid Search Stage
    각 Aspect별로 Dense + Sparse 검색 후 RRF로 통합 후보군 인출
    """
    
    print("\n" + "="*30 + "\n[Task 2] Hybrid Search (Recall)\n" + "="*30)
    

    all_candidates = []
    
    # 2) Aspect별 독립 검색 루프
    for aspect in relevant_aspects:
        d_ranks, meta_map = index_service.get_dense_hits(query, aspect, embedder, n_results=50)
        s_ranks = index_service.get_sparse_hits(query, aspect, n_results=50)
        rrf_hits = index_service.calculate_rrf(d_ranks, s_ranks, meta_map)
        all_candidates.extend(rrf_hits)
        
    # --- [수정] 검색된 실제 원문들 출력 ---
    print(f"\n🔍 총 {len(all_candidates)}개의 후보 구절을 찾았습니다.")
    print("-" * 80)
    print(f"{'Aspect':<15} | {'Venue Name':<15} | {'Review Snippet'}")
    print("-" * 80)
    
    for c in all_candidates:
        name = c.get('hall_name', 'Unknown')
        snippet = c['text'].replace('\n', ' ')[:50] # 추천 근거 보고싶으면 50보다 더 크게하세요.
        print(f"{c['aspect']:<15} | {name:<15} | {snippet}...")
    print("-" * 80)
        
    return all_candidates

def evaluate_retrieval(candidates, ground_truth_ids):
    #Recall: 후보군 안에 정답 id가 하나라도 포함되어 있는지 확인
    retrieved_ids = set([int(c['hall_id']) for c in candidates])
    hits = [gt_id for gt_id in ground_truth_ids if gt_id in retrieved_ids]
    
    recall = len(hits) / len(ground_truth_ids) if ground_truth_ids else 0
    print(f"\n📊 [Retrieval Evaluation] Recall@{len(candidates)}: {recall:.2%}")
    print(f"   (찾은 정답: {hits} / 전체 정답: {ground_truth_ids})")
    return recall


def reranking(query, candidates, reranker):
    """
    [Task 3] Cross-Encoder Reranking Stage
    앞서 선택된 후보군을 유저 쿼리와 비교하여 유사도 기반으로 정밀 재순위화
    """
    print("\n" + "="*30 + "\n[Task 3] Cross-Encoder Reranking\n" + "="*30)
    
    # 1) 정밀 재순위화 수행
    reranked_res = reranker.rerank(query, candidates)
    
    # 2) 파일 저장을 위한 경로 설정
    timestamp = datetime.now().strftime("%H%M%S")
    log_filename = f"./data/logs/rerank_result_{timestamp}.csv"
    
    print(f"💾 전체 결과({len(reranked_res)}개)를 파일에 저장합니다: {log_filename}")
    
    # 3) 파일 쓰기 및 터미널 전체 출력
    with open(log_filename, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['hall_id', 'hall_name', 'aspect', 'cross_score', 'text'])
        writer.writeheader()
        
        print("\n" + "-"*100)
        print(f"{'Rank':<5} | {'Score':<8} | {'Aspect':<15} | {'Venue':<15} | {'Text'}")
        print("-" * 100)
        
        for i, res in enumerate(reranked_res):
            # 파일 저장
            writer.writerow({
                'hall_id': res['hall_id'],
                'hall_name': res.get('hall_name', ''),
                'aspect': res['aspect'],
                'cross_score': f"{res['cross_score']:.4f}",
                'text': res['text']
            })
            
            # 터미널 전체 출력 (텍스트는 60자만)
            clean_text = res['text'].replace('\n', ' ')
            print(f"{i+1:<5} | {res['cross_score']:<8.4f} | {res['aspect']:<15} | {res.get('hall_name', 'N/A'):<15} | {clean_text[:500]}...")
            
    return reranked_res

def aggregate_results(reranked_res, num_query_aspects, df_processed):
    """최종 추천 업체 집계"""
    
    print("\n" + "="*30 + "\n[Task 4] Final Aggregation\n" + "="*30)
    
    # 1) 집계 수행 (aggregator.py의 로직 호출)
    # 반환값은 {hall_id: score} 형태의 Series라고 가정
    final_ranking_series = aggregate_to_providers(reranked_res, num_query_aspects)
    
    # 2) 로그 저장
    timestamp = datetime.now().strftime("%H%M%S")
    log_filename = f"./data/logs/final_aggregation_{timestamp}.csv"
    
    final_log_data = []
    
    print(f"💾 최종 집계 결과를 파일에 저장합니다: {log_filename}")
    print(f"\n🏆 최종 추천 업체 TOP 20 (상위 업체부터 정렬)")
    print("-" * 60)
    
    # 3) 상위 20개 추출 및 루프
    # final_ranking_series가 점수 내림차순으로 정렬되어 있으므로 상위 20개만 슬라이싱
    top_20 = final_ranking_series.head(20)
    
    for rank, (h_id, score) in enumerate(top_20.items()):
        name_row = df_processed[df_processed[VENUE_ID_COL] == int(h_id)]
        name = name_row[HALL_NAME_COL].iloc[0] if not name_row.empty else "Unknown"
        
        print(f"  {rank+1:>2}위: {name:<20} (ID: {h_id:<5}) | 통합 점수: {score:.4f}")
        
        # 로그 데이터 축적
        final_log_data.append({
            'rank': rank + 1,
            'hall_id': h_id,
            'hall_name': name,
            'total_score': f"{score:.4f}"
        })
        
    # 4) CSV 파일 저장
    log_df = pd.DataFrame(final_log_data)
    log_df.to_csv(log_filename, index=False, encoding='utf-8-sig')
    
    print("-" * 60)
    print(f"✅ 로그 저장 완료: {len(final_log_data)}개 업체")
    
    return final_ranking_series
    
    
def calculate_metrics(top_ids, ground_truth, k=10):
    """
    다양한 검색 성능 지표 계산 (MRR, Hit Rate, nDCG)
    """
    if not ground_truth:
        return {"mrr": 0.0, "hit_rate": 0.0, "ndcg": 0.0}

    # 1. MRR (Mean Reciprocal Rank)
    # 정답 업체들 중 가장 높은 순위에 있는 업체의 역수 순위 합의 평균
    rr_sum = 0
    for gt in ground_truth:
        if gt in top_ids:
            rank = top_ids.index(gt) + 1
            rr_sum += (1 / rank)
    mrr = rr_sum / len(ground_truth)

    # 2. Hit Rate @ K
    # 상위 K개 결과 중에 정답이 하나라도 포함되어 있는지 여부
    hits = [gt for gt in ground_truth if gt in top_ids[:k]]
    hit_rate = 1.0 if len(hits) > 0 else 0.0

    # 3. nDCG @ K (Normalized Discounted Cumulative Gain)
    # 정답이 상단에 있을수록 높은 가중치를 부여
    dcg = 0.0
    for i, _id in enumerate(top_ids[:k]):
        if _id in ground_truth:
            dcg += 1 / np.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(len(ground_truth), k)):
        idcg += 1 / np.log2(i + 2)
        
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"mrr": mrr, "hit_rate": hit_rate, "ndcg": ndcg}
    

if __name__ == "__main__":
    
    # 1) 초기화 및 데이터 로딩
    df_processed = pd.read_parquet(PROCESSED_PARQUET_PATH)
    
    user_query = "야외 웨딩홀에 식대가 맛있는 곳"
    ground_truth = EVAL_GROUND_TRUTH.get(user_query, [])
    
    print(f"\n🔍 테스트 쿼리: {user_query}")
    print(f"✅ 정답 업체 리스트: {ground_truth}")
    
    # 2) 서비스 클래스 초기화
    embedder = BGEEmbedder()
    index_service = DenseSparseIndex(df_processed)
    reranker = CrossEncoderReranker()

    
    # --- 파이프라인 단계별 실행 ---
    # index_service.build_chroma_db() # 2) ChromaDB 적재 (최초 1회만 주석을 풀고 실행하세요.)
    relevant_aspects, num_query_aspects = classifier_user_intent(user_query)
    
    # 3) Retrieval (Aspecst별 Dense+Sparse+RRF)
    start_ret = time.time()
    candidates = hybrid_retrieval(user_query, relevant_aspects, index_service, embedder, df_processed)
    ret_latency = time.time() - start_ret
    print(f"⏱️ 검색 시간: {ret_latency:.2f}초")
    print(f"✅ 검색된 총 후보 수: {len(candidates)}개")
    
    # 중간 테스트 평가: 리트리버의 Recall
    print("\n" + "="*40)
    print("📊 [Step 1] Retrieval(예선) 성능 평가")
    print("="*40)
    evaluate_retrieval(candidates, ground_truth)
    print(f"⏱️ Retrieval 소요 시간: {ret_latency:.4f}s")
    
    # 4) Reranking
    start_rerank = time.time()
    reranked_res = reranking(user_query, candidates, reranker)
    rerank_latency = time.time() - start_rerank
    print(f"⏱️ 재정렬 시간: {rerank_latency:.2f}초")

    # 5) Aggregation
    start_agg = time.time()
    aggregate_results(reranked_res, num_query_aspects, df_processed)
    
    # 최종 결과 리스트 추출 및 성능 지표 계산
    final_ranking_series = aggregate_to_providers(reranked_res, num_query_aspects)
    top_20_ids = [int(h_id) for h_id in final_ranking_series.head(20).index]
    agg_latency = time.time() - start_agg
    
    # 5) 전체 성능 지표 계산
    metrics = calculate_metrics(top_20_ids, ground_truth, k=10)
    recall_ret = evaluate_retrieval(candidates, ground_truth)

    # 6) 최종 통합 리포트 출력
    print("\n" + "="*50)
    print("🏆 시스템 최종 성능 검증 리포트")
    print("="*50)
    
    print(f"📊 [품질 지표 - 정밀도 및 순위]")
    print(f"   - Recall@Ret      : {recall_ret:.4f} (후보군 내 정답 비율)")
    print(f"   - MRR             : {metrics['mrr']:.4f}")
    print(f"   - Hit Rate@10     : {metrics['hit_rate']:.0f}")
    print(f"   - nDCG@10         : {metrics['ndcg']:.4f}")
    
    print(f"\n⏱️ [효율 지표 - 지연 시간]")
    print(f"   - Retrieval       : {ret_latency:.4f}s")
    print(f"   - Reranking       : {rerank_latency:.4f}s")
    print(f"   - Aggregation     : {agg_latency:.4f}s")
    print(f"   - Total Latency   : {ret_latency + rerank_latency + agg_latency:.4f}s")
    print("="*50)
    