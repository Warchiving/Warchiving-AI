# # src/aggregator.py
import pandas as pd

# def aggregate_to_providers(reranked_res):
#     if not reranked_res: return pd.Series()
#     df = pd.DataFrame(reranked_res)
    
#     final_ranking = {}
    
#     for h_id, group in df.groupby('hall_id'):
#         # 1. 이 업체가 가진 Aspect 종류 확인 (예: {'parking_facility', 'bridal_room'})
#         found_aspects = set(group['aspect'].unique())
        
#         # 2. 각 Aspect의 최고 점수 합산
#         # (한 업체에 주차 리뷰가 여러개면 그 중 제일 점수 높은 거 하나만 씀)
#         total_score = group.groupby('aspect')['cross_score'].max().sum()
        
#         # 3. [핵심] 교집합 보너스 및 필터링
#         # 유저가 2개를 물어봤는데, 이 업체는 1개 측면 리뷰만 검색되었다면? 과감히 점수 깎기
#         if len(found_aspects) < 2:
#             total_score = total_score * 0.5  # 페널티: 한쪽만 잘하는 놈은 뒤로 밀어버림
#         else:
#             total_score = total_score * 1.5  # 보너스: 둘 다 잘하면 2배 뻥튀기
            
#         final_ranking[h_id] = total_score
        
#     return pd.Series(final_ranking).sort_values(ascending=False)

# src/aggregator.py

def aggregate_to_providers(reranked_res, num_query_aspects):
    # 1. 입력 검증: 후보가 없거나 의도 개수가 정상적이지 않으면 빈 결과 반환
    if not reranked_res or num_query_aspects < 1:
        return pd.Series(dtype=float)
    
    df = pd.DataFrame(reranked_res)
    final_ranking = {}
    
    for h_id, group in df.groupby('hall_id'):
        # 해당 업체가 충족한 Aspect 종류
        found_aspects = set(group['aspect'].unique())
        num_found = len(found_aspects)
        
        # 각 Aspect별 최고점 합산
        base_score = group.groupby('aspect')['cross_score'].max().sum()
        
        # [Soft Weight] 
        # 유저 의도 대비 충족 비율을 가중치로 사용 (1.0 ~ 2.0배)
        coverage_ratio = num_found / num_query_aspects
        soft_weight = 1.0 + coverage_ratio
        
        final_ranking[h_id] = base_score * soft_weight
        
    return pd.Series(final_ranking).sort_values(ascending=False)