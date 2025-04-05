import requests
import json

# --- 설정 ---
# 서버 URL 수정: localhost(127.0.0.1)와 엔드포인트 경로(/generate/) 명시
SERVER_URL = "http://127.0.0.1:8080/generate/"

# 테스트 입력 텍스트
input_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

# 요청 데이터 (QueryRequest 모델과 일치)
payload = {
    "query": input_text
}

# --- API 호출 및 결과 처리 ---
try:
    print(f"FastAPI 서버({SERVER_URL})에 요청 전송 중...")
    # 수정된 SERVER_URL 사용
    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status() # HTTP 오류 확인

    print("응답 수신 완료.")
    result = response.json()

    # 단순화된 응답 처리
    generated_text = result.get("response", "N/A")

    print("\n--- 결과 ---")
    print("입력:", input_text)
    print("-" * 20)
    print("출력:", generated_text)

except requests.exceptions.ConnectionError as e:
    print(f"서버 연결 실패: {e}")
    # 에러 메시지에서 경로 제거
    print(f"{SERVER_URL.replace('/generate/', '')} 주소에서 서버가 실행 중인지 확인하세요.")
except requests.exceptions.HTTPError as e:
    print(f"HTTP 오류 발생: {e.response.status_code}")
    try:
        error_detail = e.response.json().get('detail', e.response.text)
        print(f"서버 오류 메시지: {error_detail}")
    except json.JSONDecodeError:
        print(f"서버 응답 내용: {e.response.text}")
except Exception as e:
    print(f"알 수 없는 오류 발생: {e}")