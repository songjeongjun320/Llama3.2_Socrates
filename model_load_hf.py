from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 모델과 토크나이저 불러오기
model_name = "jeongjunsong/llama3.2-Socrates"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. GPU가 있다면 GPU로 모델 이동 (선택 사항)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. 테스트 입력 텍스트 준비
input_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 4. 텍스트 생성
outputs = model.generate(
    inputs["input_ids"],
    max_length=50,  # 생성할 텍스트의 최대 길이
    num_return_sequences=1,  # 생성할 시퀀스 수
    no_repeat_ngram_size=2,  # 반복 방지
    do_sample=True,  # 샘플링 방식으로 생성
    top_k=50,  # 상위 k개 토큰 중 선택
    top_p=0.95,  # 누적 확률 기반 샘플링
    temperature=0.7  # 생성 다양성 조절
)

# 5. 결과 디코딩 및 출력
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("입력:", input_text)
print("출력:", generated_text)