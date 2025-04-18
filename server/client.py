import requests
import sys
import json
import logging # 로깅 사용을 위해 추가 (선택 사항)

# 로깅 설정 (선택 사항, 오류 디버깅에 도움)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 서버 URL
server_url = "http://localhost:8000/generate"

# 구분선 스타일
SEPARATOR_LONG = "=" * 60
SEPARATOR_SHORT = "-" * 40

def run_client():
    """대화형 클라이언트를 실행합니다 (가독성 개선)."""
    print(SEPARATOR_LONG)
    print(" Hello! Welcome to Llama3.2 Socrates.")
    print(" Sample Question: there are 10 girls and 20 boys in a classroom . what is the ratio of girls to boys ?")
    print(" Type your prompt or enter '/bye' to exit.")
    print(SEPARATOR_LONG)

    while True:
        try:
            # 사용자 입력 받기
            user_input = input("You: ")

            # 종료 명령어 확인
            if user_input.strip().lower() == "/bye":
                print(SEPARATOR_LONG)
                print("Socrates: Goodbye!")
                print(SEPARATOR_LONG)
                break

            # 빈 입력 건너뛰기
            if not user_input.strip():
                continue

            # 서버에 보낼 데이터 준비
            data = {"prompt": user_input}

            print(SEPARATOR_SHORT) # 요청 보내기 전 구분선

            try:
                # 서버에 POST 요청 보내기
                response = requests.post(server_url, json=data, timeout=60)

                # 응답 상태 코드 확인
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        socrates_response = response_data.get("response", "[Error: No 'response' key in server reply]")
                        adapter_used = response_data.get("adapter_used", "N/A")
                        print(f"Socrates (via {adapter_used}): {socrates_response}")
                    except json.JSONDecodeError:
                        print(f"Socrates: [Error: Could not parse server response]")
                        # logger.error(...) # 필요 시 로깅
                else:
                    # HTTP 오류 처리
                    print(f"Socrates: [Error: Received status code {response.status_code}]")
                    try:
                        error_detail = response.json().get("detail", response.text[:200])
                        print(f"         Detail: {error_detail}")
                    except json.JSONDecodeError:
                        print(f"         Response: {response.text[:200]}")

            except requests.exceptions.ConnectionError:
                print("Socrates: [Error: Could not connect to the server. Is it running?]")
            except requests.exceptions.Timeout:
                print("Socrates: [Error: Request timed out. The server might be busy.]")
            except requests.exceptions.RequestException as e:
                print(f"Socrates: [Error: An unexpected request error occurred: {e}]")

            print(SEPARATOR_SHORT) # 응답 또는 오류 출력 후 구분선

        except KeyboardInterrupt:
            # Ctrl+C 입력 시 종료
            print("\n" + SEPARATOR_LONG) # 구분선 추가
            print("Socrates: Interrupted by user. Goodbye!")
            print(SEPARATOR_LONG) # 구분선 추가
            break
        except EOFError:
            # 입력 스트림이 닫혔을 때
             print("\n" + SEPARATOR_LONG) # 구분선 추가
             print("Socrates: Input stream closed. Goodbye!")
             print(SEPARATOR_LONG) # 구분선 추가
             break

if __name__ == "__main__":
    run_client()
    sys.exit(0)