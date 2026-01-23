# 모델명 (BGE-M3, Cross-Encoder), 하이퍼파라미터# src/config.py

# ====== 모델 설정 ======
# Dense Encoder (Late Chunking용)
DENSE_MODEL_NAME = "BAAI/bge-m3"

# 나중에 Reranker에서 쓸 Cross-Encoder도 여기에 적어두면 좋아
CROSS_ENCODER_NAME = "BAAI/bge-reranker-base"  # 예시, 나중에 바꿔도 됨

# ====== 데이터 설정 ======
ASPECT_COLUMNS = [
    "hall_vibe",
    "catering",
    "parking_facility",
    "transport_access",
    "bridal_room",
    "guest_flow",
    "pricing_detail",
    "service_etc",
    "disadvantage"
]

# Hard flitering col: "in_guarantee" ("catering", "pricing_detail")

VENUE_ID_COL = "venue_id" #hall_id
HALL_NAME_COL = "hall_name" #name

RAW_CSV_PATH = "./data/raw/20260107_final_wedding_augmented.csv"
PROCESSED_PARQUET_PATH = "./data/processed/processed.parquet"

# ====== 임베딩 관련 하이퍼파라미터 ======
MAX_LENGTH = 512      # 컬럼 텍스트가 길어질 경우 대비
BATCH_SIZE = 32       # 한 번에 임베딩할 텍스트 개수
USE_FP16 = True       # GPU 쓸 때 half precision 사용 (가능하면 True)

# ====== 기타 ======
RANDOM_SEED = 42
