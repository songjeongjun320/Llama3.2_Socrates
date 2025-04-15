import os
import logging
from pathlib import Path
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 설정 ---
BASE_DIR = Path("/scratch/jsong132/Technical_Llama3.2/DB/PoT/train") # 작업 대상 디렉토리
OUTPUT_SUFFIX = "_math_io_format.jsonl" # 최종 결과 파일 접미사 변경 (.jsonl 로 명시)
INPUT_KEY = "question" # 원본 객체에서 찾을 키 이름
OUTPUT_KEY_INPUT = "input" # 결과 객체의 입력 키 이름
OUTPUT_KEY_OUTPUT = "output" # 결과 객체의 출력 키 이름
FIXED_OUTPUT_VALUE = "math" # 'output' 키에 들어갈 고정 값
# --- ---

def transform_pot_object(original_object: dict) -> dict | None:
    """
    원본 JSON 객체에서 'question' 키를 찾아
    {"input": "...", "output": "math"} 형식의 새 객체를 반환합니다.

    Args:
        original_object (dict): 원본 JSON 객체.

    Returns:
        dict | None: 변환된 새로운 JSON 객체 또는 실패 시 None.
    """
    if INPUT_KEY not in original_object:
        # item_id 같은 식별자가 있다면 로깅에 포함하면 좋음
        item_id = original_object.get("item_id", "Unknown item")
        logging.warning(f"Object (ID: {item_id}) does not contain '{INPUT_KEY}' key.")
        return None

    input_text = original_object[INPUT_KEY] # 'question' 키의 값을 가져옴

    # 새로운 JSON 객체 생성
    output_data = {
        OUTPUT_KEY_INPUT: input_text,      # 'input' 키에 'question' 값 할당
        OUTPUT_KEY_OUTPUT: FIXED_OUTPUT_VALUE # 'output' 키에 'math' 값 할당
    }
    return output_data

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    logging.info(f"Starting JSON transformation process for PoT data (Array -> JSON Lines) in: {BASE_DIR}")
    processed_file_count = 0
    processed_objects_count = 0
    transformed_objects_count = 0

    # BASE_DIR 자체에서 .json 파일 검색
    if not BASE_DIR.is_dir():
        logging.error(f"Base directory not found: {BASE_DIR}. Exiting.")
        exit()

    logging.info(f"Looking for original .json files (excluding *{OUTPUT_SUFFIX}) in: {BASE_DIR}")
    # .json 파일만 찾고, 이미 생성된 결과 파일은 제외
    input_files = [f for f in BASE_DIR.glob('*.json') if OUTPUT_SUFFIX not in f.name]

    if not input_files:
        logging.info(f"No original .json files found in {BASE_DIR}.")
    else:
        logging.info(f"Found {len(input_files)} original .json files to process.")

        for input_file_path in input_files:
            # 출력 파일 경로 생성 (e.g., original.json -> original_math_io_format.jsonl)
            output_file_name = f"{input_file_path.stem}{OUTPUT_SUFFIX}"
            output_file_path = input_file_path.with_name(output_file_name)

            # 이미 최종 결과 파일이 존재하면 건너뛰기
            if output_file_path.exists():
                 logging.info(f"Output file already exists, skipping: {output_file_path}")
                 continue

            logging.info(f"Processing file: {input_file_path} -> {output_file_path}")
            current_file_transformed_count = 0
            try:
                # 원본 JSON 파일(배열) 읽기
                with open(input_file_path, 'r', encoding='utf-8') as infile:
                    try:
                        original_data_array = json.load(infile) # 파일 전체를 배열로 로드
                        if not isinstance(original_data_array, list):
                             logging.error(f"File {input_file_path} does not contain a JSON array. Skipping.")
                             continue
                    except json.JSONDecodeError:
                        logging.error(f"Failed to decode JSON array from file: {input_file_path}. Skipping.")
                        continue

                # 변환된 객체를 JSON Lines 형식으로 출력 파일에 쓰기
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    # 배열 내의 각 객체 처리
                    for original_object in original_data_array:
                        processed_objects_count += 1
                        # 각 객체 변환
                        transformed_object = transform_pot_object(original_object)
                        if transformed_object is not None:
                            # 변환된 객체를 JSON 문자열로 변환하고 개행 문자와 함께 쓰기
                            outfile.write(json.dumps(transformed_object, ensure_ascii=False) + '\n')
                            transformed_objects_count += 1
                            current_file_transformed_count += 1

                logging.info(f"Finished processing {input_file_path}. Transformed {current_file_transformed_count} objects.")
                processed_file_count += 1

            except FileNotFoundError:
                logging.error(f"Error: Input file not found during processing: {input_file_path}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing {input_file_path}: {e}")

    logging.info(f"PoT JSON transformation finished. Processed {processed_file_count} original files, "
                 f"{processed_objects_count} total original objects. Transformed and saved {transformed_objects_count} objects to JSON Lines format.")