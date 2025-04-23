import os
import json
import requests # HTTP 요청을 위한 라이브러리
import logging
import time
from tqdm import tqdm # 진행률 표시
from typing import Optional # <--- 이 라인을 추가하세요


# --- Configuration ---
INPUT_DIR = "/scratch/jsong132/Technical_Llama3.2/DB/Test_From_Course"
OUTPUT_DIR = "/scratch/jsong132/Technical_Llama3.2/DB/Test_From_Course/Generated_Answers" # 결과를 저장할 디렉토리
SERVER_URL = "http://127.0.0.1:8000/generate" # 로컬에서 실행 중인 서버 주소 (필요시 IP 변경)
REQUEST_TIMEOUT = 120 # 서버 응답 대기 시간 (초) - 필요시 늘리기

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Function to Send Request ---
def get_answer_from_server(prompt: str, session: requests.Session) -> Optional[dict]:
    """FastAPI 서버에 질문을 보내고 응답(답변과 사용된 어댑터)을 받아옵니다."""
    payload = {
        "prompt": prompt,
        # 필요한 경우 다른 생성 파라미터 추가 가능
        # "max_new_tokens": 512,
        # "temperature": 0.7
    }
    try:
        response = session.post(SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # 200 OK가 아니면 에러 발생
        return response.json() # {"response": "...", "adapter_used": "..."} 형태의 응답 기대
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out for prompt: '{prompt[:50]}...'")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for prompt '{prompt[:50]}...': {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from server for prompt: '{prompt[:50]}...'")
        return None

# --- Main Processing Function ---
def process_json_files(input_dir: str, output_dir: str):
    """
    Input 디렉토리의 모든 .json 파일을 읽어 서버에 질문을 보내고,
    결과를 포함하여 Output 디렉토리에 _answer.json 파일을 생성합니다.
    """
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True) # Output 디렉토리 생성
    logger.info(f"Output will be saved to: {output_dir}")

    # Use a session object for potential connection reuse
    with requests.Session() as session:
        # 입력 디렉토리 내의 모든 파일 순회
        for filename in os.listdir(input_dir):
            if filename.endswith(".json") and not filename.endswith("_answer.json"): # _answer.json 파일은 제외
                input_filepath = os.path.join(input_dir, filename)
                output_filename = os.path.splitext(filename)[0] + "_answer.json"
                output_filepath = os.path.join(output_dir, output_filename)

                logger.info(f"Processing file: {input_filepath}")

                try:
                    with open(input_filepath, 'r', encoding='utf-8') as f:
                        original_data = json.load(f) # JSON 파일 로드 (리스트 형태 기대)
                except FileNotFoundError:
                    logger.error(f"File not found: {input_filepath}. Skipping.")
                    continue
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from {input_filepath}. Skipping.")
                    continue
                except Exception as e:
                    logger.error(f"Error reading file {input_filepath}: {e}. Skipping.")
                    continue

                # 데이터가 리스트 형태인지 확인
                if not isinstance(original_data, list):
                    logger.warning(f"Expected a list in {input_filepath}, but got {type(original_data)}. Skipping.")
                    continue

                results_data = [] # 결과를 저장할 새 리스트
                total_questions = len(original_data)
                processed_count = 0
                failed_count = 0

                # 파일 내 각 질문에 대해 처리 (tqdm으로 진행률 표시)
                for item in tqdm(original_data, desc=f"Querying {filename}", total=total_questions):
                    if isinstance(item, dict) and "question" in item:
                        question = item["question"]
                        if not question:
                            logger.warning(f"Skipping item with empty question in {filename}")
                            failed_count += 1
                            item['generated_answer'] = None # 답변 필드 추가 (None)
                            item['adapter_used'] = None
                            results_data.append(item) # 원본 아이템에 None 추가하여 저장
                            continue

                        # 서버로부터 답변 받아오기
                        answer_data = get_answer_from_server(question, session)

                        # 원본 item 딕셔너리에 결과 추가
                        if answer_data and "response" in answer_data:
                            item['generated_answer'] = answer_data.get("response")
                            item['adapter_used'] = answer_data.get("adapter_used", "unknown")
                            processed_count += 1
                        else:
                            logger.warning(f"Failed to get answer for question: '{question[:50]}...'")
                            item['generated_answer'] = None # 실패 시 None으로 표시
                            item['adapter_used'] = None
                            failed_count += 1

                        results_data.append(item) # 수정된 item을 결과 리스트에 추가
                        # time.sleep(0.1) # 서버 부하를 줄이기 위해 약간의 딜레이 추가 (선택 사항)
                    else:
                        logger.warning(f"Skipping invalid item format in {filename}: {item}")
                        failed_count += 1
                        # 유효하지 않은 아이템도 결과에 포함시킬지 결정 (여기서는 포함 안 함)
                        # results_data.append(item) # 원본 그대로 추가하려면 주석 해제

                logger.info(f"Finished processing {filename}. Success: {processed_count}, Failed/Skipped: {failed_count}")

                # 결과 데이터를 새 JSON 파일에 저장
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(results_data, f, indent=2, ensure_ascii=False) # 들여쓰기 적용하여 저장
                    logger.info(f"Successfully saved results to: {output_filepath}")
                except Exception as e:
                    logger.error(f"Failed to save results to {output_filepath}: {e}")

    logger.info("Finished processing all files.")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    process_json_files(INPUT_DIR, OUTPUT_DIR)