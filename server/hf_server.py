# Standard library imports
import json
import logging
import signal
import os
import re
import contextlib
from io import StringIO
import time
import traceback
from typing import Optional, Any

# Third-party imports
import torch
from fastapi import FastAPI, Request, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # PeftModel import 확인
import uvicorn

# --- Configuration ---
# Model paths (제공된 경로로 수정)
BASE_MODEL_PATH = '/scratch/jsong132/Technical_Llama3.2/llama3.2_3b'
JUDGE_ADAPTER_PATH = '/scratch/jsong132/Technical_Llama3.2/Adapters/Judge_Adapter/final'
MATH_ADAPTER_PATH = '/scratch/jsong132/Technical_Llama3.2/Adapters/Math_Adapter/final'
INSTRUCTION_ADAPTER_PATH = '/scratch/jsong132/Technical_Llama3.2/Adapters/Instruction_Adapter/final' # Instruction 경로 확인

# Code Execution Configuration
TIMEOUT_SECONDS = 5 # Timeout for executing math code

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables for model and tokenizer ---
model = None # PeftModel 객체가 저장될 변수
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --- Code Extraction and Execution Functions (이전과 동일) ---
class TimeoutError(Exception):
    """Custom Timeout Error"""
    pass

def _timeout_handler(signum, frame):
    """Signal handler function to raise TimeoutError"""
    logger.warning("Execution timed out via SIGALRM.")
    raise TimeoutError(f"Code execution exceeded timeout of {TIMEOUT_SECONDS} seconds")

def extract_python_code(generated_text: str) -> Optional[str]:
    """Extracts Python code block from text (simplified version)."""
    if not generated_text:
        return None
    match_py = re.search(r"```python\s*(.*?)\s*```", generated_text, re.DOTALL | re.IGNORECASE)
    if match_py: return match_py.group(1).strip()
    match_any = re.search(r"```\s*(.*?)\s*```", generated_text, re.DOTALL)
    if match_any:
        code = match_any.group(1).strip()
        if any(kw in code for kw in ['def ', 'return ', '=', 'import ', 'print(']): return code
    match_def = re.search(r"^(def\s+solution\s*\(.*?\):.*)", generated_text, re.DOTALL | re.MULTILINE)
    if match_def: return match_def.group(1).strip()
    code_part_cleaned = generated_text.split("</s>")[0].split("<|endoftext|>")[0].strip()
    if not code_part_cleaned.startswith('```') and not code_part_cleaned.endswith('```'):
        if any(kw in code_part_cleaned for kw in ['def ', 'return ', '=', 'import ', 'print(']):
            logger.warning("Using full cleaned text as code (last resort).")
            return code_part_cleaned
    logger.warning("Failed to extract any plausible code block.")
    return None

def safe_execute_code(code: str, timeout: int = TIMEOUT_SECONDS) -> tuple[Optional[Any], Optional[str]]:
    """Executes the extracted Python code safely with a timeout."""
    if not code:
        logger.warning("No code provided to execute.")
        return None, "No code provided to execute."
    logger.info(f"Attempting to execute code (timeout={timeout}s)...")
    output_buffer = StringIO()
    execution_error = None; result = None; original_handler = None
    has_sigalrm = hasattr(signal, "SIGALRM")
    if not has_sigalrm: logger.warning("signal.SIGALRM not available. Timeout disabled.")
    try:
        if has_sigalrm:
            original_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
        local_scope = {}; safe_builtins = {
             'print': print, 'range': range, 'len': len, 'abs': abs, 'min': min,
             'max': max, 'sum': sum, 'int': int, 'float': float, 'str': str,
             'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
             'pow': pow, 'round': round, 'divmod': divmod, 'enumerate':enumerate,
             'zip':zip, 'sorted':sorted, 'map':map, 'filter':filter, 'all':all, 'any':any }
        global_scope = {'__builtins__': safe_builtins, 'math': __import__('math')}
        with contextlib.redirect_stdout(output_buffer):
            exec(code, global_scope, local_scope)
            if 'solution' in local_scope and callable(local_scope['solution']): result = local_scope['solution']()
            elif 'result' in local_scope: result = local_scope['result']
            else:
                 printed_output = output_buffer.getvalue().strip()
                 if printed_output:
                      try:
                           last_line = printed_output.splitlines()[-1].strip()
                           if '.' in last_line or 'e' in last_line.lower(): result = float(last_line)
                           else: result = int(last_line)
                      except: pass
        if has_sigalrm: signal.alarm(0)
        if result is None:
             execution_error = "Code executed without error, but yielded no discernible result (None)."
             logger.warning(execution_error)
        logger.info(f"Code execution finished. Result: {result}, Error: {execution_error}")
        return result, execution_error
    except TimeoutError as e: logger.error(f"Code execution failed: {e}"); return None, f"Timeout Error: Execution exceeded {timeout} seconds."
    except Exception as e:
        execution_error = f"Execution Error: {type(e).__name__}: {e}"
        logger.error(f"{execution_error}\n{traceback.format_exc()}")
        return None, execution_error
    finally:
        if has_sigalrm and original_handler is not None:
            signal.alarm(0); signal.signal(signal.SIGALRM, original_handler)

# --- Model Loading (어댑터 로딩 방식 수정) ---
def load_models_and_tokenizer():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        logger.info("Model and tokenizer already loaded.")
        return

    logger.info(f"Loading tokenizer from base model path: {BASE_MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True) # trust_remote_code 추가 고려
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set Tokenizer PAD token to EOS token.")
            else:
                # Add a pad token if none exists
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.warning("Added new PAD token '[PAD]'. Resizing embeddings needed.")
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal error loading tokenizer: {e}", exc_info=True)
        raise RuntimeError(f"Could not load tokenizer from {BASE_MODEL_PATH}") from e

    logger.info(f"Loading base model from: {BASE_MODEL_PATH}")
    try:
        # device_map='auto'는 multi-GPU 환경에 유용
        # 단일 GPU 또는 CPU에서는 device=device 로 명시적 지정 가능
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            # device_map="auto" # Accelerate 사용 시 자동
            device_map={"": device} # 단일 장치에 명시적 할당 시도
        )
        # 토크나이저에 새 토큰 추가 시 임베딩 크기 조정
        if len(tokenizer) > base_model.config.vocab_size:
             logger.warning(f"Resizing token embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
             base_model.resize_token_embeddings(len(tokenizer))
             # 모델 config 업데이트 (중요)
             base_model.config.vocab_size = len(tokenizer)

        logger.info("Base model loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal error loading base model: {e}", exc_info=True)
        raise RuntimeError(f"Could not load base model from {BASE_MODEL_PATH}") from e

    logger.info("Loading and attaching adapters...")
    try:
        # --- 첫 번째 어댑터 로드 (PeftModel 사용) ---
        logger.info(f"Loading JUDGE adapter from: {JUDGE_ADAPTER_PATH}")
        if not os.path.isdir(JUDGE_ADAPTER_PATH): # 디렉토리 존재 확인
             raise FileNotFoundError(f"Judge adapter path not found or not a directory: {JUDGE_ADAPTER_PATH}")
        # PeftModel.from_pretrained 는 베이스 모델과 첫 어댑터를 결합
        model = PeftModel.from_pretrained(
            base_model,
            JUDGE_ADAPTER_PATH,
            adapter_name="judge" # 어댑터 이름 지정
        )
        logger.info("JUDGE adapter attached.")

        # --- 이후 어댑터 로드 (model.load_adapter 사용) ---
        logger.info(f"Loading MATH adapter from: {MATH_ADAPTER_PATH}")
        if not os.path.isdir(MATH_ADAPTER_PATH):
             raise FileNotFoundError(f"Math adapter path not found or not a directory: {MATH_ADAPTER_PATH}")
        model.load_adapter(MATH_ADAPTER_PATH, adapter_name="math")
        logger.info("MATH adapter attached.")

        logger.info(f"Loading INSTRUCTION adapter from: {INSTRUCTION_ADAPTER_PATH}")
        if not os.path.isdir(INSTRUCTION_ADAPTER_PATH):
            # Instruction 어댑터는 필수 아닐 수 있으므로 경고만 로깅
            logger.warning(f"Instruction adapter path not found or not a directory: {INSTRUCTION_ADAPTER_PATH}. Instruction fallback might not work.")
            # raise FileNotFoundError(f"Instruction adapter path not found: {INSTRUCTION_ADAPTER_PATH}") # 필수로 만들려면 주석 해제
        else:
            model.load_adapter(INSTRUCTION_ADAPTER_PATH, adapter_name="instruction")
            logger.info("INSTRUCTION adapter attached.")

        model.eval() # 전체 모델을 평가 모드로 설정
        logger.info("All available adapters loaded. Model set to evaluation mode.")
        logger.info(f"Loaded adapters: {list(model.peft_config.keys())}") # 로드된 어댑터 이름 확인
        logger.info(f"Model is on device: {model.device}") # 모델이 할당된 주 장치 확인

    except Exception as e:
        logger.error(f"Fatal error loading adapters: {e}", exc_info=True)
        raise RuntimeError("Could not load one or more adapters.") from e


# --- FastAPI App ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델과 토크나이저를 로드합니다."""
    logger.info("Server startup: Initiating model loading...")
    try:
        load_models_and_tokenizer()
        logger.info("Model loading complete. Server ready.")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR DURING STARTUP: {e}. Server might be unusable.", exc_info=True)
        # 서버 시작 실패 시 프로세스 종료 고려
        # import sys; sys.exit(1)

async def generate_with_adapter(adapter_name: str, prompt: str, **gen_kwargs):
    """특정 어댑터를 사용하여 텍스트를 생성하는 헬퍼 함수."""
    global model, tokenizer # 전역 변수 사용 명시
    if model is None or tokenizer is None:
        logger.error("Model or tokenizer not loaded properly.")
        raise HTTPException(status_code=503, detail="Model not ready.")

    if adapter_name not in model.peft_config:
        logger.error(f"Adapter '{adapter_name}' not found in loaded adapters: {list(model.peft_config.keys())}")
        # Instruction 어댑터가 로드되어 있다면 거기로 fallback 시도
        if "instruction" in model.peft_config:
             logger.warning(f"Falling back to 'instruction' adapter because '{adapter_name}' is not available.")
             adapter_name = "instruction"
        else: # Instruction 어댑터도 없으면 에러
             raise HTTPException(status_code=500, detail=f"Requested adapter '{adapter_name}' is not available, and no fallback.")

    logger.info(f"Activating adapter: {adapter_name}")
    try:
        model.set_adapter(adapter_name) # 어댑터 활성화
        logger.debug(f"Active adapter set to: {model.active_adapter}")
    except Exception as e:
         logger.error(f"Failed to set adapter '{adapter_name}': {e}. Trying fallback if possible.", exc_info=True)
         if adapter_name != "instruction" and "instruction" in model.peft_config:
              logger.warning("Falling back to 'instruction' adapter due to error setting previous adapter.")
              adapter_name = "instruction"
              try:
                   model.set_adapter("instruction")
              except Exception as fallback_e:
                   logger.error(f"Failed to set fallback 'instruction' adapter: {fallback_e}", exc_info=True)
                   raise HTTPException(status_code=500, detail="Failed to set active adapter.")
         else:
              raise HTTPException(status_code=500, detail=f"Adapter '{adapter_name}' could not be activated.")

    logger.info(f"Generating with {adapter_name} adapter. Prompt: '{prompt[:100]}...'")
    logger.debug(f"Generation args: {gen_kwargs}")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048) # 입력 길이 조절 필요 시
    # 입력 텐서를 모델과 동일한 장치로 이동
    target_device = model.device # 모델의 주 장치 사용
    input_ids = inputs.input_ids.to(target_device)
    attention_mask = inputs.attention_mask.to(target_device)
    logger.debug(f"Input tensors moved to: {target_device}")

    try:
        with torch.inference_mode(): # torch.no_grad() 대신 사용 권장 (더 효율적)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, # 패딩 토큰 ID 명시
                **gen_kwargs
            )
        # 생성된 부분만 디코딩 (입력 프롬프트 제외)
        generated_ids = outputs[0, input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info(f"Raw response from {adapter_name}: '{response[:200]}...'")
        return response.strip()
    except Exception as e:
        logger.error(f"Error during generation with adapter {adapter_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating response with {adapter_name}")


@app.post("/generate")
async def generate_route(request: Request):
    """사용자 요청을 받아 적절한 어댑터로 라우팅하고 결과를 반환합니다."""
    start_time = time.time()
    try:
        data = await request.json()
        user_prompt = data.get("prompt")
        # Get generation parameters or use defaults
        max_new_tokens = data.get("max_new_tokens", 512)
        temperature = data.get("temperature", 0.6)
        top_p = data.get("top_p", 0.9)
        repetition_penalty = data.get("repetition_penalty", 1.1)

        if not user_prompt:
            raise HTTPException(status_code=400, detail="Prompt is missing.")

        # --- Step 1: Judge the prompt type ---
        logger.info("Step 1: Judging prompt type using 'judge' adapter...")
        try:
            # 분류 작업이므로 샘플링 없이 결정적으로 생성
            judge_response = await generate_with_adapter(
                adapter_name="judge",
                prompt=user_prompt,
                max_new_tokens=10, # 분류 결과는 짧음
                temperature=0.0,   # 샘플링 끄기 (가장 확률 높은 토큰 선택)
                do_sample=False
            )
            # Judge 응답 정규화 (소문자 변환, 앞뒤 공백/마침표 제거 등)
            judged_type = judge_response.lower().strip().replace(".","").split()[0] # 첫 단어만 사용 고려
            logger.info(f"Judge adapter classified prompt as: '{judged_type}'")
        except Exception as judge_e:
            logger.error(f"Error using Judge adapter: {judge_e}. Defaulting to instruction.", exc_info=True)
            judged_type = "instruction" # Judge 어댑터 오류 시 기본값

        # --- Step 2: Route based on judgment ---
        final_response = None
        adapter_used = "instruction" # 기본값

        if "math" in judged_type:
            logger.info("Step 2: Routing to MATH adapter.")
            adapter_used = "math"
            try:
                math_code_response = await generate_with_adapter(
                    adapter_name="math",
                    prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True
                )

                # --- Step 2a: Execute Math Code ---
                logger.info("Step 2a: Extracting and executing code from MATH response.")
                extracted_code = extract_python_code(math_code_response)
                if extracted_code:
                    logger.info("Python code extracted successfully.")
                    exec_result, exec_error = safe_execute_code(extracted_code)
                    if exec_error:
                        logger.warning(f"Math code execution failed: {exec_error}. Falling back to instruction adapter.")
                    elif exec_result is not None:
                        logger.info(f"Math code execution successful. Result: {exec_result}")
                        final_response = f"Python Code Execution Result:\n```\n{exec_result}\n```"
                        # final_response += f"\n\nGenerated Code:\n```python\n{extracted_code}\n```" # 코드 포함 옵션
                    else:
                        logger.warning("Math code executed but returned None. Falling back to instruction adapter.")
                else:
                    logger.warning("Failed to extract Python code from Math adapter response. Falling back to instruction adapter.")

            except Exception as math_e:
                logger.error(f"Error during MATH adapter processing or execution: {math_e}. Falling back to instruction.", exc_info=True)

        # --- Step 3: Instruction Adapter (if not math or math failed) ---
        if final_response is None: # Math 처리가 성공적으로 끝나지 않았을 경우
            if "instruction" not in model.peft_config:
                 logger.error("Instruction adapter was not loaded during startup. Cannot generate fallback response.")
                 raise HTTPException(status_code=501, detail="Processing failed and Instruction adapter is not available.")

            logger.info("Step 3: Routing to INSTRUCTION adapter (or fallback).")
            adapter_used = "instruction"
            try:
                final_response = await generate_with_adapter(
                    adapter_name="instruction",
                    prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True
                )
                logger.info("Generated response using INSTRUCTION adapter.")
            except Exception as instr_e:
                 logger.error(f"CRITICAL: Error using INSTRUCTION adapter: {instr_e}", exc_info=True)
                 raise HTTPException(status_code=500, detail="Failed to generate response using instruction adapter.")

        end_time = time.time()
        logger.info(f"Request processing completed in {end_time - start_time:.2f} seconds using '{adapter_used}' adapter flow.")
        return {"response": final_response, "adapter_used": adapter_used}

    except HTTPException as http_exc:
        logger.warning(f"HTTP Exception: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in /generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {type(e).__name__}")

# --- Run Server ---
if __name__ == "__main__":
    logger.info("Starting FastAPI server with Uvicorn...")
    # 서버 시작 전에 모델 로딩 시도
    try:
        load_models_and_tokenizer()
        logger.info("Pre-loading models and adapters finished successfully.")
        # reload=False: 프로덕션 환경에서 권장, 코드 변경 시 서버 재시작 필요 없음 (이미 로드됨)
        # reload=True: 개발 중 편리, 코드 변경 시 서버 자동 재시작 (매번 모델 로드)
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except RuntimeError as load_error:
         logger.critical(f"Failed to load models/adapters during pre-start: {load_error}. Server cannot start.", exc_info=True)
    except Exception as startup_e:
         logger.critical(f"An unexpected error occurred before starting Uvicorn: {startup_e}", exc_info=True)