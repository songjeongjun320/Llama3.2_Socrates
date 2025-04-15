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
from peft import PeftModel
import uvicorn

# --- Configuration ---
# Model paths
BASE_MODEL_PATH = '/scratch/jsong132/Technical_Llama3.2/llama3.2_3b'
JUDGE_ADAPTER_PATH = '/scratch/jsong132/Technical_Llama3.2/FineTuning/Judge/Tune_Results/llama3.2_Judge/final_checkpoint'
MATH_ADAPTER_PATH = '/scratch/jsong132/Technical_Llama3.2/FineTuning/Math/Tune_Results/llama3.2_Socrates_Math_v3/final_checkpoint'
# *** PLACEHOLDER: Update this when the instruction adapter is ready ***
INSTRUCTION_ADAPTER_PATH = '/scratch/jsong132/Technical_Llama3.2/FineTuning/Instruction/final_checkpoint' # UPDATE THIS PATH

# Code Execution Configuration
TIMEOUT_SECONDS = 5 # Timeout for executing math code

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables for model and tokenizer ---
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}") # Note: device_map='auto' might override this

# --- Code Extraction and Execution Functions (Adapted from reeval script) ---
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
    # Look for ```python ... ```
    match_py = re.search(r"```python\s*(.*?)\s*```", generated_text, re.DOTALL | re.IGNORECASE)
    if match_py:
        return match_py.group(1).strip()
    # Look for ``` ... ```
    match_any = re.search(r"```\s*(.*?)\s*```", generated_text, re.DOTALL)
    if match_any:
        # Basic check if it looks like code
        code = match_any.group(1).strip()
        if any(kw in code for kw in ['def ', 'return ', '=', 'import ', 'print(']):
             return code
    # Look for code starting with "def solution" without fences
    match_def = re.search(r"^(def\s+solution\s*\(.*?\):.*)", generated_text, re.DOTALL | re.MULTILINE)
    if match_def:
        return match_def.group(1).strip()

    # Last resort: Check if the whole text (cleaned) looks like code
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
    execution_error = None
    result = None
    original_handler = None

    # Check for signal availability (Windows compatibility)
    has_sigalrm = hasattr(signal, "SIGALRM")
    if not has_sigalrm:
         logger.warning("signal.SIGALRM not available. Timeout disabled.")

    try:
        if has_sigalrm:
            original_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        local_scope = {}
        safe_builtins = {
             'print': print, 'range': range, 'len': len, 'abs': abs, 'min': min,
             'max': max, 'sum': sum, 'int': int, 'float': float, 'str': str,
             'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
             'pow': pow, 'round': round, 'divmod': divmod, 'enumerate':enumerate,
             'zip':zip, 'sorted':sorted, 'map':map, 'filter':filter, 'all':all, 'any':any
        }
        # Allow common math functions
        global_scope = {'__builtins__': safe_builtins, 'math': __import__('math')}

        with contextlib.redirect_stdout(output_buffer):
            exec(code, global_scope, local_scope) # Define functions/classes

            if 'solution' in local_scope and callable(local_scope['solution']):
                result = local_scope['solution']() # Execute the main function
            elif 'result' in local_scope:
                result = local_scope['result'] # Check for a 'result' variable
            else: # Check printed output
                 printed_output = output_buffer.getvalue().strip()
                 if printed_output:
                      try: # Attempt to parse last line as number
                           last_line = printed_output.splitlines()[-1].strip()
                           if '.' in last_line or 'e' in last_line.lower(): result = float(last_line)
                           else: result = int(last_line)
                      except: pass # Ignore errors

        if has_sigalrm:
            signal.alarm(0) # Disable alarm if finished

        if result is None:
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
        if has_sigalrm and original_handler is not None:
            signal.alarm(0) # Ensure alarm is off
            signal.signal(signal.SIGALRM, original_handler) # Restore

# --- Model Loading ---
def load_models_and_tokenizer():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        logger.info("Model and tokenizer already loaded.")
        return

    logger.info(f"Loading tokenizer from base model path: {BASE_MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set Tokenizer PAD token to EOS token.")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.warning("Added new PAD token '[PAD]'.")
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal error loading tokenizer: {e}", exc_info=True)
        raise RuntimeError(f"Could not load tokenizer from {BASE_MODEL_PATH}") from e

    logger.info(f"Loading base model from: {BASE_MODEL_PATH}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto" # Use accelerate for multi-GPU/CPU+GPU
        )
        # Resize embeddings if a new pad token was added to tokenizer *after* base model load config was read
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= base_model.config.vocab_size:
            logger.warning("Resizing base model token embeddings for new pad token.")
            base_model.resize_token_embeddings(len(tokenizer))

        logger.info("Base model loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal error loading base model: {e}", exc_info=True)
        raise RuntimeError(f"Could not load base model from {BASE_MODEL_PATH}") from e

    logger.info("Loading and attaching adapters...")
    try:
        # Load the first adapter (Judge) using PeftModel.from_pretrained
        logger.info(f"Loading JUDGE adapter from: {JUDGE_ADAPTER_PATH}")
        if not os.path.exists(JUDGE_ADAPTER_PATH):
             raise FileNotFoundError(f"Judge adapter path not found: {JUDGE_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, JUDGE_ADAPTER_PATH, adapter_name="judge")
        logger.info("JUDGE adapter attached.")

        # Load subsequent adapters using model.load_adapter
        logger.info(f"Loading MATH adapter from: {MATH_ADAPTER_PATH}")
        if not os.path.exists(MATH_ADAPTER_PATH):
             raise FileNotFoundError(f"Math adapter path not found: {MATH_ADAPTER_PATH}")
        model.load_adapter(MATH_ADAPTER_PATH, adapter_name="math")
        logger.info("MATH adapter attached.")

        logger.info(f"Loading INSTRUCTION adapter from: {INSTRUCTION_ADAPTER_PATH}")
        if not os.path.exists(INSTRUCTION_ADAPTER_PATH):
            # Log a warning but continue if the instruction adapter is optional/not ready
            logger.warning(f"Instruction adapter path not found: {INSTRUCTION_ADAPTER_PATH}. Instruction fallback might not work.")
            # Optionally raise an error if it's mandatory:
            # raise FileNotFoundError(f"Instruction adapter path not found: {INSTRUCTION_ADAPTER_PATH}")
        else:
            model.load_adapter(INSTRUCTION_ADAPTER_PATH, adapter_name="instruction")
            logger.info("INSTRUCTION adapter attached.")

        model.eval() # Set the entire model (including all adapters) to evaluation mode
        logger.info("All adapters loaded. Model set to evaluation mode.")
        logger.info(f"Model device: {model.device}") # Get the primary device

    except Exception as e:
        logger.error(f"Fatal error loading adapters: {e}", exc_info=True)
        # If adapters fail, the server might be unusable. Raise or handle gracefully.
        raise RuntimeError("Could not load one or more adapters.") from e


# --- FastAPI App ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("Server startup: Initiating model loading...")
    try:
        load_models_and_tokenizer()
        logger.info("Model loading complete. Server ready.")
    except Exception as e:
        logger.error(f"CRITICAL ERROR DURING STARTUP: {e}", exc_info=True)
        # Optional: exit the application if models are essential and failed to load
        # import sys
        # sys.exit(1)

async def generate_with_adapter(adapter_name: str, prompt: str, **gen_kwargs):
    """Helper function to generate text using a specific adapter."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    logger.info(f"Activating adapter: {adapter_name}")
    try:
        model.set_adapter(adapter_name)
    except ValueError as e:
         # Handle case where adapter might not have loaded successfully during startup
         logger.error(f"Failed to set adapter '{adapter_name}': {e}. Falling back to instruction if possible.")
         if adapter_name != "instruction" and "instruction" in model.peft_config:
              logger.warning("Falling back to 'instruction' adapter due to error setting previous adapter.")
              adapter_name = "instruction"
              model.set_adapter("instruction")
         else:
              raise HTTPException(status_code=500, detail=f"Adapter '{adapter_name}' not available.")


    logger.info(f"Generating with {adapter_name} adapter. Prompt: '{prompt[:100]}...'")
    logger.debug(f"Generation args: {gen_kwargs}")

    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to the model's primary device
    # model.device gives the device of the base model when using device_map='auto'
    target_device = model.device
    input_ids = inputs.input_ids.to(target_device)
    attention_mask = inputs.attention_mask.to(target_device)
    logger.debug(f"Input tensors moved to: {target_device}")

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id, # Use EOS for padding during generation
                **gen_kwargs
            )
        # Decode only the newly generated tokens
        generated_ids = outputs[0, input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info(f"Raw response from {adapter_name}: '{response[:200]}...'")
        return response.strip()
    except Exception as e:
        logger.error(f"Error during generation with adapter {adapter_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating response with {adapter_name}")


@app.post("/generate")
async def generate_route(request: Request):
    start_time = time.time()
    try:
        data = await request.json()
        user_prompt = data.get("prompt")
        # Get generation parameters or use defaults
        max_new_tokens = data.get("max_new_tokens", 512)
        temperature = data.get("temperature", 0.6) # Adjusted default slightly
        top_p = data.get("top_p", 0.9)
        repetition_penalty = data.get("repetition_penalty", 1.1)

        if not user_prompt:
            raise HTTPException(status_code=400, detail="Prompt is missing.")

        # --- Step 1: Judge the prompt type ---
        logger.info("Step 1: Judging prompt type...")
        try:
            # Use simple generation params for classification
            judge_response = await generate_with_adapter(
                "judge",
                user_prompt,
                max_new_tokens=10,
                do_sample=False # Deterministic output for classification
            )
            # Normalize judge response (e.g., lowercase, strip)
            judged_type = judge_response.lower().strip().replace(".","")
            logger.info(f"Judge adapter classified prompt as: '{judged_type}'")
        except Exception as judge_e:
            logger.error(f"Error using Judge adapter: {judge_e}. Defaulting to instruction.", exc_info=True)
            judged_type = "instruction" # Fallback on judge error


        # --- Step 2: Route based on judgment ---
        final_response = None
        adapter_used = "instruction" # Default if math fails or not chosen

        if "math" in judged_type:
            logger.info("Step 2: Routing to MATH adapter.")
            adapter_used = "math"
            try:
                math_code_response = await generate_with_adapter(
                    "math",
                    user_prompt,
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
                        # Fall through to instruction adapter block
                    elif exec_result is not None:
                        logger.info(f"Math code execution successful. Result: {exec_result}")
                        # Format the successful result for the user
                        final_response = f"Python Code Execution Result:\n```\n{exec_result}\n```"
                        # Optional: Include the code itself in the response
                        # final_response += f"\n\nGenerated Code:\n```python\n{extracted_code}\n```"
                    else:
                        logger.warning("Math code executed but returned None. Falling back to instruction adapter.")
                        # Fall through to instruction adapter block
                else:
                    logger.warning("Failed to extract Python code from Math adapter response. Falling back to instruction adapter.")
                    # Fall through to instruction adapter block

            except Exception as math_e:
                logger.error(f"Error during MATH adapter processing or execution: {math_e}. Falling back to instruction.", exc_info=True)
                # Fall through to instruction adapter block

        # --- Step 3: Instruction Adapter (if not math or math failed) ---
        if final_response is None:
            # Check if instruction adapter exists before trying to use it
            if "instruction" not in model.peft_config:
                 logger.error("Instruction adapter was not loaded. Cannot generate fallback response.")
                 raise HTTPException(status_code=501, detail="Math processing failed and Instruction adapter is not available.")

            logger.info("Step 3: Routing to INSTRUCTION adapter (or fallback).")
            adapter_used = "instruction"
            try:
                final_response = await generate_with_adapter(
                    "instruction",
                    user_prompt, # Send the original user prompt
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True
                )
                logger.info("Generated response using INSTRUCTION adapter.")
            except Exception as instr_e:
                 logger.error(f"CRITICAL: Error using INSTRUCTION adapter as fallback: {instr_e}", exc_info=True)
                 raise HTTPException(status_code=500, detail="Failed to generate response.")

        end_time = time.time()
        logger.info(f"Request processing completed in {end_time - start_time:.2f} seconds using '{adapter_used}' adapter flow.")
        return {"response": final_response, "adapter_used": adapter_used}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error in /generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: An unexpected error occurred.")

# --- Run Server ---
if __name__ == "__main__":
    logger.info("Starting FastAPI server with Uvicorn...")
    # Ensure models are loaded *before* starting the server workers if not using reload=True
    try:
        load_models_and_tokenizer()
        logger.info("Pre-loading models finished.")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except RuntimeError as load_error:
         logger.critical(f"Failed to load models during pre-start: {load_error}. Server cannot start.", exc_info=True)
    except Exception as startup_e:
         logger.critical(f"An unexpected error occurred before starting Uvicorn: {startup_e}", exc_info=True)