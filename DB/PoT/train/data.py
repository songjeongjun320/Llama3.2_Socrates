import os
import json
import glob
import logging

# 로깅 설정 (진행 상황 및 오류 확인용)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_max_string_length_in_json_files(directory_path: str) -> int:
    """
    지정된 디렉토리 내 모든 .json 파일에서 'input' 또는 'output' 키 값 중
    가장 긴 문자열의 길이를 찾습니다.

    Args:
        directory_path: JSON 파일이 있는 디렉토리 경로.

    Returns:
        찾은 최대 문자열 길이. 해당 키나 유효한 문자열이 없으면 0 반환.
    """
    if not os.path.isdir(directory_path):
        logger.error(f"오류: 디렉토리를 찾을 수 없습니다 - {directory_path}")
        return 0

    max_overall_length = 0
    max_overall_sentence = ""
    # 디렉토리 내의 모든 .json 파일 찾기
    file_pattern = os.path.join(directory_path, "*.json")
    json_files = glob.glob(file_pattern)

    if not json_files:
        logger.warning(f"경고: '{directory_path}' 디렉토리에서 .json 파일을 찾지 못했습니다.")
        return 0

    logger.info(f"총 {len(json_files)}개의 .json 파일을 처리합니다.")

    for file_path in json_files:
        logger.info(f"파일 처리 중: {file_path}")
        current_file_max = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 파일 전체를 하나의 JSON 객체(보통 리스트)로 로드
                data = json.load(f)

                # 데이터가 리스트 형태라고 가정 (일반적인 JSONL이나 JSON 배열 형식)
                if not isinstance(data, list):
                    logger.warning(f"경고: {file_path} 파일의 최상위 구조가 리스트가 아닙니다. 건너<0xEB><0x9B><0x84>니다.")
                    continue

                # 리스트 내의 각 항목(딕셔너리) 순회
                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        # logger.debug(f"항목 {i+1} 건너<0xEB><0x9B><0x84>: 딕셔너리 아님")
                        continue

                    # 'input' 키와 'output' 키 확인
                    for key in ["question", "answer_cot"]:
                        if key in item:
                            value = item[key]
                            # 값이 문자열인 경우에만 길이 계산
                            if isinstance(value, str):
                                current_length = len(value)
                                current_file_max = max(current_file_max, current_length)
                                # 전체 최대 길이 업데이트
                                if current_length > max_overall_length:
                                     logger.info(f"새로운 최대 길이 발견: {current_length} (파일: {os.path.basename(file_path)}, 항목: {i+1}, 키: '{key}')")
                                     max_overall_length = current_length
                                     max_overall_sentence = value
                            # else:
                                # logger.debug(f"항목 {i+1}의 키 '{key}' 값이 문자열 아님: {type(value)}")

            logger.info(f"'{os.path.basename(file_path)}' 내 최대 길이: {current_file_max}")

        except json.JSONDecodeError:
            logger.error(f"오류: {file_path} 파일 파싱 실패 (유효한 JSON 아님).")
        except FileNotFoundError:
            logger.error(f"오류: {file_path} 파일을 찾을 수 없음 (처리 중 삭제되었을 수 있음).")
        except Exception as e:
            logger.error(f"오류: {file_path} 처리 중 예외 발생 - {e}")

    return max_overall_length, max_overall_sentence

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    target_directory = "/scratch/jsong132/Technical_Llama3.2/DB/PoT/train" # 대상 디렉토리 경로 지정
    max_len, max_sentence = find_max_string_length_in_json_files(target_directory)

    if max_len > 0:
        print("-" * 50)
        logger.info(f"최종 결과: '{target_directory}' 내 모든 .json 파일에서")
        logger.info(f"'input' 또는 'output' 키의 가장 긴 문자열 길이는 【 {max_len} 】입니다.")
        print("-" * 50)
        logger.info(f"'input' 또는 'output' 키의 가장 긴 문자열은 【 {max_sentence} 】입니다.")
    else:
        logger.warning(f"'{target_directory}' 내 .json 파일에서 유효한 'input' 또는 'output' 문자열을 찾지 못했습니다.")