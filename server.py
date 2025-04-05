import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import torch # 예시로 추가

print("스크립트 시작") # 추가

# FastAPI 앱 초기화 (vLLM 로딩 전에 정의)
app = FastAPI(title="vLLM LoRA Inference Server", version="1.0.0")
print("FastAPI 앱 객체 생성 완료") # 추가

model_path = "/scratch/jsong132/Technical_Llama3.2/llama3.2_3b"

print(f"vLLM 모델 로딩 시작: {model_path}") # 추가
print(f"사용 가능한 GPU 수: {torch.cuda.device_count()}") # 추가
print(f"현재 GPU: {torch.cuda.current_device()}") # 추가

try:
    llm = LLM(model=model_path, gpu_memory_utilization=0.9, tensor_parallel_size=2)
    print("vLLM 모델 로딩 성공") # 추가
except Exception as e:
    print(f"vLLM 모델 로딩 중 오류 발생: {e}") # 추가
    raise # 오류 발생 시 프로세스 종료

sampling_params = SamplingParams(temperature=0.5, top_p=0.7, repetition_penalty=1.1, max_tokens=1024)
print("SamplingParams 설정 완료") # 추가

class QueryRequest(BaseModel):
    query: str

@app.post("/generate/")
async def generate_response(request: QueryRequest):
    print("'/generate/' 요청 수신") # 추가
    try:
        response = llm.generate(request.query, sampling_params)
        result_text = response[0].outputs[0].text
        print("응답 생성 완료") # 추가
        return {"response": result_text}
    except Exception as e:
        print(f"'/generate/' 처리 중 오류: {e}") # 추가
        raise HTTPException(status_code = 500, detail = str(e)) # detail 오타 수정

print("FastAPI 라우트 정의 완료") # 추가
print("Uvicorn 서버 시작 직전") # 추가

# Uvicorn 실행은 커맨드라인에서 하므로 이 아래 코드는 필요 없음
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)

# uvicorn server:app --host 0.0.0.0 --port 8080