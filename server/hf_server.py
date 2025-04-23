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
MATH_ADAPTER_PATH = '/scratch/jsong132/Technical_Llama3.2/Adapters/Math_Adapter/llama3.2_Socrates_Math_v1/final_checkpoint'
INSTRUCTION_ADAPTER_PATH = '/scratch/jsong132/Technical_Llama3.2/Adapters/Instruction_Adapter/final' # Instruction 경로 확인

# Code Execution Configuration
TIMEOUT_SECONDS = 5 # Timeout for executing math code

# Keywords to identify math questions (case-insensitive)
MATH_KEYWORDS = ["what is", "calculate", "how much", "how many", "how old", "how"]
# Keywords to identify code generation requests (case-insensitive)
CODE_GEN_KEYWORDS = ["write a python", "write a function", "generate python", "python code for"]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables for model and tokenizer ---
model = None # PeftModel 객체가 저장될 변수
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --- Code Extraction and Execution Functions (변경 없음) ---
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
    # 가장 일반적인 Markdown 코드 블록 먼저 시도
    match_py = re.search(r"```python\s*(.*?)\s*```", generated_text, re.DOTALL | re.IGNORECASE)
    if match_py: return match_py.group(1).strip()
    # 언어 명시 없는 코드 블록 시도
    match_any = re.search(r"```\s*(.*?)\s*```", generated_text, re.DOTALL)
    if match_any:
        code = match_any.group(1).strip()
        # 간단한 휴리스틱으로 파이썬 코드인지 확인
        if any(kw in code for kw in ['def ', 'return ', '=', 'import ', 'print(', 'class ', 'for ', 'while ']):
            logger.debug("Extracted code block without language specifier, assuming Python.")
            return code
    # 'def solution(...):' 패턴 시도 (주로 문제 해결 컨텍스트)
    match_def = re.search(r"^(def\s+solution\s*\(.*?\):.*?)(?:\n\n|\Z)", generated_text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
    if match_def:
        logger.debug("Extracted code using 'def solution(...):' pattern.")
        return match_def.group(1).strip()
    # 코드 블록 없이 코드처럼 보이는 경우 (마지막 수단)
    # 모델 출력에서 종료 토큰 제거 시도
    code_part_cleaned = generated_text.split("</s>")[0].split("<|endoftext|>")[0].split("<|eot_id|>")[0].strip()
    # 코드 블록 마커가 없는지 재확인
    if not code_part_cleaned.startswith('```') and not code_part_cleaned.endswith('```'):
        # 더 많은 키워드로 파이썬 코드 가능성 확인
        if any(kw in code_part_cleaned for kw in ['def ', 'return ', '=', 'import ', 'print(', 'class ', 'for ', 'while ', 'if ', 'else', '[', '{']):
            logger.warning("Using cleaned full text as code (last resort, no code blocks found).")
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
        # 필요한 모듈만 안전하게 import 허용 (예: math)
        global_scope = {'__builtins__': safe_builtins, 'math': __import__('math')}
        with contextlib.redirect_stdout(output_buffer):
            # 코드를 실행
            exec(code, global_scope, local_scope)
            # 'solution' 함수가 정의되어 있고 호출 가능하면 실행
            if 'solution' in local_scope and callable(local_scope['solution']):
                 result = local_scope['solution']()
            # 'result' 변수가 있으면 그 값을 사용
            elif 'result' in local_scope:
                 result = local_scope['result']
            # 위 두 경우가 아니면, print된 마지막 줄을 결과로 간주 시도
            else:
                 printed_output = output_buffer.getvalue().strip()
                 if printed_output:
                      try:
                           last_line = printed_output.splitlines()[-1].strip()
                           # 간단한 타입 추론 (정수/실수)
                           if '.' in last_line or 'e' in last_line.lower():
                               result = float(last_line)
                           else:
                               result = int(last_line)
                      except Exception as parse_err:
                            logger.warning(f"Could not parse last printed line as result: {parse_err}. Output was: {printed_output}")
                            result = printed_output # 파싱 실패 시 전체 출력 반환 고려
        if has_sigalrm: signal.alarm(0) # 타임아웃 해제
        # 실행은 됐으나 결과가 없는 경우
        if result is None and execution_error is None:
             # exec 자체가 에러 없이 끝났고, result/solution도 없고, print도 안 찍혔거나 파싱 불가
             printed_output = output_buffer.getvalue().strip()
             if printed_output: # print는 찍혔으나 파싱 불가
                 result = printed_output # 파싱 안되면 그냥 출력값 자체를 결과로
                 logger.warning(f"Code executed, couldn't parse final result, returning full print output.")
             else: # print 조차 안찍힘
                 execution_error = "Code executed without error, but yielded no discernible result (None)."
                 logger.warning(execution_error)

        logger.info(f"Code execution finished. Result: {result}, Error: {execution_error}")
        return result, execution_error
    except TimeoutError as e:
        logger.error(f"Code execution failed: {e}")
        return None, f"Timeout Error: Execution exceeded {timeout} seconds."
    except Exception as e:
        execution_error = f"Execution Error: {type(e).__name__}: {e}"
        logger.error(f"{execution_error}\n{traceback.format_exc()}")
        return None, execution_error
    finally:
        # SIGALRM 핸들러 복원
        if has_sigalrm and original_handler is not None:
            signal.alarm(0); signal.signal(signal.SIGALRM, original_handler)


# --- Model Loading (변경 없음) ---
def load_models_and_tokenizer():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        logger.info("Model and tokenizer already loaded.")
        return

    logger.info(f"Loading tokenizer from base model path: {BASE_MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set Tokenizer PAD token to EOS token.")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.warning("Added new PAD token '[PAD]'. Resizing embeddings needed.")
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal error loading tokenizer: {e}", exc_info=True)
        raise RuntimeError(f"Could not load tokenizer from {BASE_MODEL_PATH}") from e

    logger.info(f"Loading base model from: {BASE_MODEL_PATH}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": device}
        )
        if len(tokenizer) > base_model.config.vocab_size:
             logger.warning(f"Resizing token embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
             base_model.resize_token_embeddings(len(tokenizer))
             base_model.config.vocab_size = len(tokenizer)
        logger.info("Base model loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal error loading base model: {e}", exc_info=True)
        raise RuntimeError(f"Could not load base model from {BASE_MODEL_PATH}") from e

    logger.info("Loading and attaching adapters...")
    try:
        logger.info(f"Loading MATH adapter from: {MATH_ADAPTER_PATH} as initial adapter")
        if not os.path.isdir(MATH_ADAPTER_PATH):
             raise FileNotFoundError(f"Math adapter path not found or not a directory: {MATH_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(
            base_model,
            MATH_ADAPTER_PATH,
            adapter_name="math"
        )
        logger.info("MATH adapter attached as initial.")

        logger.info(f"Loading INSTRUCTION adapter from: {INSTRUCTION_ADAPTER_PATH}")
        if not os.path.isdir(INSTRUCTION_ADAPTER_PATH):
            logger.warning(f"Instruction adapter path not found or not a directory: {INSTRUCTION_ADAPTER_PATH}. Instruction routing might not work.")
        else:
            model.load_adapter(INSTRUCTION_ADAPTER_PATH, adapter_name="instruction")
            logger.info("INSTRUCTION adapter attached.")

        model.eval()
        logger.info("All available adapters loaded. Model set to evaluation mode.")
        logger.info(f"Loaded adapters: {list(model.peft_config.keys())}")
        logger.info(f"Model is on device: {model.device}")

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

# --- Helper function to generate text with a specific adapter (변경 없음) ---
async def generate_with_adapter(adapter_name: str, prompt: str, **gen_kwargs):
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.error("Model or tokenizer not loaded properly.")
        raise HTTPException(status_code=503, detail="Model not ready.")

    # Check if adapter exists, fallback to instruction if needed and available
    if adapter_name not in model.peft_config:
        logger.error(f"Adapter '{adapter_name}' not found. Loaded: {list(model.peft_config.keys())}")
        if "instruction" in model.peft_config:
             logger.warning(f"Falling back to 'instruction' adapter.")
             adapter_name = "instruction"
        else:
             raise HTTPException(status_code=500, detail=f"Adapter '{adapter_name}' not available, no fallback.")

    # Activate the adapter
    logger.info(f"Activating adapter: {adapter_name}")
    try:
        model.set_adapter(adapter_name)
        logger.debug(f"Active adapter set to: {model.active_adapter}")
    except Exception as e:
         logger.error(f"Failed to set adapter '{adapter_name}': {e}. Trying fallback.", exc_info=True)
         # Attempt fallback again if activation fails and it wasn't instruction already
         if adapter_name != "instruction" and "instruction" in model.peft_config:
              logger.warning("Falling back to 'instruction' due to activation error.")
              adapter_name = "instruction"
              try: model.set_adapter("instruction")
              except Exception as fallback_e:
                   logger.error(f"Failed to set fallback 'instruction' adapter: {fallback_e}", exc_info=True)
                   raise HTTPException(status_code=500, detail="Failed to set active adapter.")
         else: # If fallback also fails or wasn't possible
              raise HTTPException(status_code=500, detail=f"Adapter '{adapter_name}' could not be activated.")

    # Prepare inputs
    logger.info(f"Generating with {adapter_name}. Prompt: '{prompt[:100]}...'")
    logger.debug(f"Generation args: {gen_kwargs}")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048) # Adjust max_length if needed
    target_device = model.device
    input_ids = inputs.input_ids.to(target_device)
    attention_mask = inputs.attention_mask.to(target_device)

    # Generate
    try:
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                **gen_kwargs
            )
        # Decode response (excluding prompt)
        generated_ids = outputs[0, input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info(f"Raw response ({adapter_name}): '{response[:200]}...'")
        return response.strip()
    except Exception as e:
        logger.error(f"Generation error ({adapter_name}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating response with {adapter_name}")


# --- Main generation route (/generate) ---
@app.post("/generate")
async def generate_route(request: Request):
    """
    Handles user requests, classifies them based on keywords,
    routes to the appropriate adapter (Math or Instruction),
    executes code for math problems (unless code generation is requested),
    and returns the final response.
    """
    start_time = time.time()
    try:
        data = await request.json()
        user_prompt = data.get("prompt")
        # Generation parameters
        max_new_tokens = data.get("max_new_tokens", 512)
        temperature = data.get("temperature", 0.6)
        top_p = data.get("top_p", 0.9)
        repetition_penalty = data.get("repetition_penalty", 1.1)

        if not user_prompt:
            raise HTTPException(status_code=400, detail="Prompt is missing.")

        # --- Step 1: Classify prompt type ---
        logger.info("Step 1: Judging prompt type using keywords...")
        prompt_lower = user_prompt.lower()
        is_math_question = False
        is_code_gen_request = False
        adapter_to_use = "instruction" # Default adapter

        # Check for code generation keywords first (higher priority than math calc)
        for keyword in CODE_GEN_KEYWORDS:
             # Use startswith for simple check, might need more robust matching
             if prompt_lower.startswith(keyword):
                 is_code_gen_request = True
                 adapter_to_use = "math" # Use math adapter for code generation
                 logger.info(f"Prompt identified as CODE GENERATION based on keyword: '{keyword}'")
                 break

        # If not code gen, check for math calculation keywords
        if not is_code_gen_request:
             for keyword in MATH_KEYWORDS:
                 if prompt_lower.startswith(keyword + " ") or prompt_lower.startswith(keyword + "?"):
                     is_math_question = True
                     adapter_to_use = "math" # Use math adapter for calculation
                     logger.info(f"Prompt identified as MATH CALCULATION based on keyword: '{keyword}'")
                     break

        if adapter_to_use == "instruction":
            logger.info("Prompt identified as INSTRUCTION (no specific keywords found).")

        # Check if the designated adapter is actually loaded
        if adapter_to_use not in model.peft_config:
             logger.error(f"{adapter_to_use.upper()} adapter requested but not loaded.")
             if "instruction" in model.peft_config:
                  logger.warning("Falling back to INSTRUCTION adapter.")
                  adapter_to_use = "instruction"
             else:
                  logger.critical("Target adapter and fallback INSTRUCTION adapter are unavailable.")
                  raise HTTPException(status_code=501, detail=f"{adapter_to_use.upper()} adapter not available.")

        # --- Step 2: Generate Response using chosen adapter ---
        final_response = None
        adapter_used_for_generation = adapter_to_use # Track which adapter generated the text

        logger.info(f"Step 2: Routing to {adapter_to_use.upper()} adapter.")
        try:
            generated_text = await generate_with_adapter(
                adapter_name=adapter_to_use,
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True
            )

            # --- Step 3: Post-processing (Code Execution or Direct Output) ---
            logger.info("Step 3: Post-processing generated text...")

            # Case 1: It was a math calculation request AND we used the math adapter
            if is_math_question and adapter_to_use == "math":
                logger.info("Attempting code extraction and execution for math question...")
                extracted_code = extract_python_code(generated_text)
                if extracted_code:
                    logger.info("Python code extracted successfully.")
                    exec_result, exec_error = safe_execute_code(extracted_code)
                    if exec_error:
                        logger.warning(f"Math code execution failed: {exec_error}. Returning generated text instead.")
                        final_response = generated_text # Fallback: return the generated code/text
                        adapter_used_for_generation = "math (execution failed)" # Indicate execution failure
                    elif exec_result is not None:
                        logger.info(f"Math code execution successful. Result: {exec_result}")
                        # Format result clearly
                        final_response = f"Execution Result:\n```\n{exec_result}\n```"
                        # Optionally include the code that was run
                        # final_response += f"\n\nExecuted Code:\n```python\n{extracted_code}\n```"
                    else: # Code ran but returned None
                        logger.warning("Math code executed but returned None. Returning generated text.")
                        final_response = generated_text # Fallback: return the generated code/text
                        adapter_used_for_generation = "math (execution returned None)"
                else: # Code extraction failed
                    logger.warning("Failed to extract Python code from Math adapter response for calculation. Returning raw text.")
                    final_response = generated_text # Return the raw text generated by math adapter

            # Case 2: It was a code generation request OR Any other case (Instruction or Math fallback)
            else:
                 logger.info("Returning generated text directly (Instruction, Code Gen Request, or Math Fallback).")
                 final_response = generated_text
                 # adapter_used_for_generation already set correctly

        except Exception as gen_e:
            logger.error(f"Error during generation or post-processing with {adapter_to_use.upper()} adapter: {gen_e}", exc_info=True)
            # Attempt fallback to instruction if error occurred with Math adapter
            if adapter_to_use == "math" and "instruction" in model.peft_config:
                 logger.warning("Error with MATH adapter, attempting fallback to INSTRUCTION.")
                 try:
                      adapter_used_for_generation = "instruction (fallback)"
                      final_response = await generate_with_adapter(
                          adapter_name="instruction", prompt=user_prompt, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=True )
                      logger.info("Fallback generation using INSTRUCTION adapter successful.")
                 except Exception as fallback_e:
                      logger.error(f"CRITICAL: Fallback to INSTRUCTION adapter also failed: {fallback_e}", exc_info=True)
                      raise HTTPException(status_code=500, detail="Failed to generate response after fallback.")
            else: # Error occurred with Instruction adapter or Math adapter with no Instruction fallback
                 raise HTTPException(status_code=500, detail=f"Failed to generate response using {adapter_used_for_generation} adapter.")

        # Ensure final_response is not None before returning
        if final_response is None:
             logger.error("Processing completed, but final_response is unexpectedly None.")
             raise HTTPException(status_code=500, detail="Internal error: Failed to produce a final response.")

        end_time = time.time()
        logger.info(f"Request processing completed in {end_time - start_time:.2f} seconds. Final adapter logic: '{adapter_used_for_generation}'.")
        return {"response": final_response, "adapter_used": adapter_used_for_generation}

    except HTTPException as http_exc:
        logger.warning(f"HTTP Exception: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in /generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {type(e).__name__}")

# --- Run Server ---
if __name__ == "__main__":
    logger.info("Starting FastAPI server with Uvicorn...")
    try:
        load_models_and_tokenizer()
        logger.info("Pre-loading models and adapters finished successfully.")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except RuntimeError as load_error:
         logger.critical(f"Failed to load models/adapters during pre-start: {load_error}. Server cannot start.", exc_info=True)
    except Exception as startup_e:
         logger.critical(f"An unexpected error occurred before starting Uvicorn: {startup_e}", exc_info=True)