from huggingface_hub import snapshot_download
import os

# --- 설정 ---
repo_id = "Seono/Instruction_Fine_Tune"
# 다운로드할 어댑터 및 토크나이저 관련 파일 목록
# 이미지에 보이는 주요 파일들을 포함합니다.
files_to_include = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    # 필요에 따라 다른 파일 추가 가능 (예: README, training_args 등)
    "README.md",
    # ".gitattributes" # 보통 로컬 사용에는 불필요
    # "training_args.bin" # 필요하다면 포함
]

# 파일을 저장할 로컬 디렉토리 경로
local_save_directory = "./Instruction_Adapter"

# --- 디렉토리 생성 ---
os.makedirs(local_save_directory, exist_ok=True)
print(f"파일을 저장할 디렉토리: {local_save_directory}")

# --- 파일 다운로드 ---
try:
    print(f"'{repo_id}' 리포지토리에서 파일 다운로드를 시작합니다...")
    # snapshot_download를 사용하여 특정 파일만 다운로드
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=files_to_include, # 지정된 파일 패턴만 허용
        local_dir=local_save_directory,  # 저장할 로컬 경로 지정
        local_dir_use_symlinks=False,    # 캐시에서 심볼릭 링크 대신 파일 직접 복사 (선택 사항)
        repo_type="model",               # 리포지토리 타입 명시 (보통 'model')
        # token="YOUR_HF_READ_TOKEN"    # 리포지토리가 private이거나 rate limit 문제가 있을 경우 토큰 필요
    )
    print(f"파일 다운로드 완료! 저장 위치: {local_save_directory}")
    print("다운로드된 파일 목록:")
    for filename in os.listdir(local_save_directory):
        print(f"- {filename}")

except Exception as e:
    print(f"다운로드 중 오류가 발생했습니다: {e}")