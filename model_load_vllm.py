from openai import OpenAI

# vLLM 서버에 연결
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",  # vLLM은 API 키를 요구하지 않지만, 형식상 필요
)

# 테스트 입력 텍스트
input_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

# 텍스트 생성 요청
response = client.chat.completions.create(
    model="jeongjunsong/llama3.2-Socrates",
    messages=[
        {"role": "user", "content": input_text}
    ],
    max_tokens=50,
    temperature=0.7,
    top_p=0.95,
)

# 결과 출력
generated_text = response.choices[0].message.content
print("입력:", input_text)
print("출력:", generated_text)