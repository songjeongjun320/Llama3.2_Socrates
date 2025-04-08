# Standard library imports
import json
import logging
import os
import re
import gc
import time
import traceback
import signal # Added for timeout in safe_execute
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal, Tuple # Added Tuple
from io import StringIO
import contextlib

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Configure logging (Set to DEBUG for detailed eval logs like Script 2)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str
    model_path: str
    is_local: bool = True
    model_type: Literal["causal", "encoder", "encoder-decoder"] = "causal"
    is_adapter_model: bool = False
    base_model_path_for_adapter: Optional[str] = None

    def __post_init__(self):
        if self.is_adapter_model and not self.base_model_path_for_adapter:
            raise ValueError(f"Model '{self.name}' is marked as adapter model, but 'base_model_path_for_adapter' is not provided.")
        if self.is_adapter_model and self.base_model_path_for_adapter == self.model_path:
             raise ValueError(f"For adapter model '{self.name}', 'base_model_path_for_adapter' cannot be the same as 'model_path'.")

# --- Define Models to Evaluate ---
BASE_LLAMA_PATH = "/scratch/jsong132/Technical_Llama3.2/llama3.2_3b"

MODEL_CONFIGS = [
    ModelConfig(
        name="Llama-3.2-3b-Socrates-Math-Tuned-Adapter-Final", # Example specific name
        model_path="/scratch/jsong132/Technical_Llama3.2/FineTuning/Math/Tune_Results/llama3.2_Socrates_Math/final_checkpoint",
        is_local=True,
        is_adapter_model=True,
        base_model_path_for_adapter=BASE_LLAMA_PATH
    ),
    ModelConfig(
        name="Llama-3.2-3b-Socrates-Math-AllTarget-Adapter-Final", # Example specific name
        model_path="/scratch/jsong132/Technical_Llama3.2/FineTuning/Math/Tune_Results/llama3.2_Socrates_Math_all_target/final_checkpoint",
        is_local=True,
        is_adapter_model=True,
        base_model_path_for_adapter=BASE_LLAMA_PATH
    ),
    # Add other models if needed
]

# --- Evaluation Configuration ---
TEST_DATA_DIR = "/scratch/jsong132/Technical_Llama3.2/DB/PoT/test"
RESULTS_FILE = "detailed_evaluation_results.json" # New filename for results
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATION_MAX_NEW_TOKENS = 512
GENERATION_TEMPERATURE = 0.1 # Set to 0 for deterministic if needed, >0 for sampling
GENERATION_TOP_P = 0.9
TIMEOUT_SECONDS = 10 # Timeout for code execution
EVALUATION_BATCH_SIZE = 8 # Adjust based on GPU VRAM

# --- Helper Functions (Adapted from Script 1 & 2) ---

def load_pot_test_data(directory_path: str) -> List[Dict[str, str]]:
    """Load all JSON test files from a directory."""
    logger.info(f"Loading PoT test data from directory: {directory_path}")
    all_test_items = []
    if not os.path.isdir(directory_path):
        logger.error(f"Test data directory not found: {directory_path}")
        return all_test_items

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = 0
                        for item in data:
                             if isinstance(item, dict) and "question" in item and "answer_value" in item and item["answer_value"] is not None:
                                all_test_items.append({
                                    "question": str(item["question"]),
                                    "answer_value": str(item["answer_value"])
                                })
                                count += 1
                             else:
                                logger.warning(f"Skipping item in {filename} due to missing keys or null answer_value: {item}")
                        logger.debug(f"Loaded {count} valid items from {filename}") # Changed to debug
                    else:
                        logger.warning(f"File {filename} does not contain a list.")
            except Exception as e:
                logger.error(f"Error loading test file {file_path}: {str(e)}")

    logger.info(f"Total valid test items loaded: {len(all_test_items)}")
    return all_test_items

def format_prompt(question: str) -> str:
    """Formats the question into the prompt structure used during training."""
    # Ensure this matches the EXACT prompt format used for fine-tuning
    return f"Question: {question}\nAnswer:\n```python\n"

# --- DETAILED EVALUATION FUNCTIONS (ADAPTED FROM SCRIPT 2) ---

class TimeoutError(Exception):
    """Custom Timeout Error"""
    pass

def _timeout_handler(signum, frame):
    """Signal handler function to raise TimeoutError"""
    logger.warning("Execution timed out via SIGALRM.")
    raise TimeoutError(f"Code execution exceeded timeout of {TIMEOUT_SECONDS} seconds")

def extract_python_code(generated_text: str) -> Optional[str]:
    """
    Extracts Python code block (Detailed version from Script 2).
    """
    func_name = "extract_python_code" # For logger
    if not generated_text:
         logger.debug(f"Input text is empty or None.")
         return None

    log_text_snippet = generated_text[:100].replace('\n', '\\n')
    logger.debug(f"Input text (first 100 chars): {log_text_snippet}")

    code_part = generated_text.strip()
    extracted_code: Optional[str] = None

    # 1. Try matching ```python ... ```
    logger.debug(f"Attempting ```python match...")
    match_py = re.search(r"```python\s*(.*?)\s*```", code_part, re.DOTALL | re.IGNORECASE)
    if match_py:
        extracted_code = match_py.group(1).strip()
        logger.debug(f"Extracted using ```python fence.")
    else:
        logger.debug(f"No ```python match found.")
        # 2. Try matching ``` ... ```
        logger.debug(f"Attempting ``` (any) match...")
        match_any = re.search(r"```\s*(.*?)\s*```", code_part, re.DOTALL)
        if match_any:
            potential_code = match_any.group(1).strip()
            logger.debug(f"Found ``` block. Checking for keywords...")
            if any(kw in potential_code for kw in ['def ', 'return ', '=', 'import ', 'print(']):
                extracted_code = potential_code
                logger.debug(f"Keywords found, extracted using ``` fence.")
            else:
                logger.warning(f"Code block ```...``` found but lacks Python keywords: {potential_code[:100]}...")
        else:
            logger.debug(f"No ``` (any) match found.")
            # 3. Try matching based on 'def solution():'
            logger.debug(f"Attempting 'def solution():' match...")
            match_def = re.search(r"^(def\s+solution\s*\(.*?\):.*)(?:```)?$", code_part, re.DOTALL | re.MULTILINE)
            if match_def:
                extracted_code = match_def.group(1).strip()
                logger.debug(f"Extracted based on 'def solution():'.")
            else:
                 logger.debug(f"No 'def solution():' match found.")

    # Cleanup Step
    if extracted_code:
        original_extracted = extracted_code
        log_extracted_snippet_before = extracted_code[:100].replace('\n','\\n')
        logger.debug(f"Running cleanup on extracted code (first 100): {log_extracted_snippet_before}")

        if extracted_code.endswith('```'):
            extracted_code = extracted_code[:-3].strip()
            logger.debug(f"Removed trailing ```.")
        if extracted_code.startswith('```'):
             extracted_code = extracted_code[3:].strip()
             logger.debug(f"Removed leading ```.")
        if extracted_code.lower().startswith('python'):
             extracted_code = extracted_code[6:].lstrip()
             logger.debug(f"Removed leading 'python'.")

        if extracted_code != original_extracted:
             log_extracted_snippet_after = extracted_code[:100].replace('\n','\\n')
             logger.debug(f"Code after cleanup (first 100): {log_extracted_snippet_after}")
        else:
             logger.debug(f"No cleanup needed.")

    # Validation
    if extracted_code:
         logger.debug(f"Validating extracted code...")
         if any(kw in extracted_code for kw in ['def ', 'return ', '=', 'import ', 'print(']):
              if not re.match(r"^\s*(SyntaxError|Error|Traceback)", extracted_code, re.IGNORECASE):
                   logger.debug(f"Validation passed.")
                   return extracted_code
              else:
                   logger.warning(f"Validation failed: Looks like an error message.")
                   return None
         else:
              logger.warning(f"Validation failed: Lacks common Python keywords.")
              return None
    else:
        # Last Resort Check
        logger.debug(f"No code extracted yet. Trying last resort check on full cleaned text.")
        code_part_cleaned = code_part.split("</s>")[0].split("<|endoftext|>")[0].strip()
        # Avoid if it clearly still has fences or looks like an instruction/refusal
        fence_pattern = r"^\s*```|```\s*$"
        refusal_pattern = r"^\s*(I cannot|I am unable|As a large language model)"
        if not re.search(fence_pattern, code_part_cleaned) and not re.search(refusal_pattern, code_part_cleaned, re.IGNORECASE):
            if any(kw in code_part_cleaned for kw in ['def ', 'return ', '=', 'import ', 'print(']):
                 log_cleaned_snippet = code_part_cleaned[:150].replace('\n', '\\n')
                 logger.warning(f"Using full cleaned text as code (last resort): {log_cleaned_snippet}")
                 return code_part_cleaned
            else:
                 logger.debug(f"Last resort check failed: No keywords in cleaned text.")
        else:
             logger.debug(f"Last resort check failed: Cleaned text still has fences or looks like refusal.")

    logger.debug(f"Failed to extract any plausible code block.")
    return None

def safe_execute_code(code: str, timeout: int = TIMEOUT_SECONDS) -> Tuple[Optional[Any], Optional[str]]:
    """
    Executes extracted Python code safely with timeout (Detailed version from Script 2).
    Returns: (result, error_message_or_none)
    """
    func_name = "safe_execute_code" # For logger
    if not code:
        logger.debug(f"No code provided.")
        return None, "No code provided to execute."

    # --- Security Check ---
    # Be cautious with allowed keywords. Add more restrictions if needed.
    unsafe_keywords = ['import os', 'import sys', 'subprocess', 'eval(', 'exec(', 'open(', '__import__'] # Added __import__
    if any(kw in code for kw in unsafe_keywords):
         logger.error(f"Potential unsafe code detected. Execution blocked.\nCode snippet:\n{code[:500]}...")
         return None, "Execution blocked due to potentially unsafe keywords."

    logger.debug(f"Attempting to execute code (first 200 chars):\n--- CODE START ---\n{code[:200]}\n--- CODE END ---")

    # --- Use signal for timeout only if available (Unix-like) ---
    can_use_timeout = hasattr(signal, "SIGALRM")
    original_handler = None
    output_buffer = StringIO()
    execution_error = None
    result = None

    try:
        # --- Setup Timeout ---
        if can_use_timeout:
            logger.debug(f"Setting timeout alarm for {timeout} seconds.")
            original_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
        else:
            logger.warning("signal.SIGALRM not available. Timeout disabled.")

        # --- Execute Code ---
        local_scope = {}
        # Define allowed built-ins explicitly
        safe_builtins = {
             'print': print, 'range': range, 'len': len, 'abs': abs, 'min': min,
             'max': max, 'sum': sum, 'int': int, 'float': float, 'str': str,
             'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
             'pow': pow, 'round': round, 'divmod': divmod, 'enumerate':enumerate,
             'zip':zip, 'sorted':sorted, 'map':map, 'filter':filter, 'all':all, 'any':any,
             'isinstance': isinstance, 'type': type, # Add more safe built-ins if needed
             # Be very careful adding more built-ins
        }
        # Allow specific safe modules
        safe_modules = {'math': __import__('math')} # Only allow math
        global_scope = {'__builtins__': safe_builtins, **safe_modules}

        logger.debug(f"Prepared execution scope. Calling exec()...")
        start_time = time.time()
        with contextlib.redirect_stdout(output_buffer):
             exec(code, global_scope, local_scope) # Define functions/variables

             # --- Try to get result ---
             if 'solution' in local_scope and callable(local_scope['solution']):
                 logger.debug(f"Found callable 'solution' function. Calling it...")
                 result = local_scope['solution']()
                 logger.debug(f"'solution()' returned: {result} (type: {type(result)})")
             elif 'result' in local_scope:
                  result = local_scope['result']
                  logger.debug(f"Found 'result' variable: {result} (type: {type(result)})")
             else:
                 # Fallback: Check printed output (last numerical line)
                 printed_output = output_buffer.getvalue().strip()
                 logger.debug(f"No solution()/result variable. Captured print output: '{printed_output}'")
                 if printed_output:
                     try:
                         # Find last line that looks like a number
                         numerical_lines = re.findall(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", printed_output, re.MULTILINE)
                         if numerical_lines:
                             last_num_str = numerical_lines[-1][0] # Get the full number match
                             logger.debug(f"Last numerical line found: '{last_num_str}'. Attempting conversion...")
                             if '.' in last_num_str or 'e' in last_num_str.lower():
                                  result = float(last_num_str)
                                  logger.debug(f"Parsed as float: {result}")
                             else:
                                  result = int(last_num_str)
                                  logger.debug(f"Parsed as int: {result}")
                         else:
                             logger.warning("No lines in print output looked like numbers.")
                             execution_error = "Execution finished, but no numerical print output found."

                     except ValueError as e:
                         logger.warning(f"Conversion of last numerical printed line failed: {e}")
                         execution_error = "Execution finished, but final numerical print output failed conversion."
                     except IndexError:
                          logger.warning("Printed output parsing failed (IndexError).")
                          execution_error = "Execution finished, but parsing print output failed."
                 else:
                      logger.warning("No solution()/result variable and no print output captured.")
                      execution_error = "Execution finished, but no result could be determined."

        execution_time = time.time() - start_time
        logger.debug(f"Code execution block finished in {execution_time:.4f} seconds.")

        # Disable alarm if execution finished on time
        if can_use_timeout:
            signal.alarm(0)
            logger.debug("Timeout alarm disabled (execution finished).")

        # Check for None result if no error occurred yet
        if result is None and execution_error is None:
             logger.warning(f"Code executed without error, but result is None.")
             execution_error = "Code executed without error, but yielded no discernible result (None)."

        logger.debug(f"Returning result: {result}, error: {execution_error}")
        return result, execution_error

    except TimeoutError as e:
        logger.warning(f"Caught TimeoutError explicitly.")
        return None, f"Timeout Error: Execution exceeded {timeout} seconds."
    except SyntaxError as e: execution_error = f"Syntax Error: {e}"; logger.error(f"{execution_error}", exc_info=False) # Less verbose log
    except NameError as e: execution_error = f"Name Error: {e}"; logger.error(f"{execution_error}", exc_info=False)
    except TypeError as e: execution_error = f"Type Error: {e}"; logger.error(f"{execution_error}", exc_info=False)
    except ZeroDivisionError as e: execution_error = f"ZeroDivisionError: {e}"; logger.error(f"{execution_error}", exc_info=False)
    except MemoryError as e: execution_error = f"Memory Error: {e}"; logger.error(f"{execution_error}", exc_info=False)
    except Exception as e:
        execution_error = f"Unexpected Execution Error: {type(e).__name__}: {e}"
        logger.error(f"{execution_error}\n{traceback.format_exc()}") # Log full traceback for unexpected

    # Ensure alarm is disabled even if error occurred before signal.alarm(0)
    finally:
        if can_use_timeout:
            signal.alarm(0)
            if original_handler is not None:
                signal.signal(signal.SIGALRM, original_handler)
            logger.debug("Timeout alarm disabled and original handler restored in finally block.")

    logger.debug(f"Returning result: None, error: {execution_error}")
    return None, execution_error # Return None result if any exception occurred


def compare_results(generated_result: Any, ground_truth_str: str) -> bool:
    """
    Compares the generated result with ground truth (Detailed version from Script 2).
    """
    func_name = "compare_results" # For logger
    logger.debug(f"Comparing Generated: '{generated_result}' (type: {type(generated_result)}) with Ground Truth: '{ground_truth_str}' (type: {type(ground_truth_str)})")
    if generated_result is None:
        logger.debug("Comparison returning False because generated_result is None.")
        return False

    try:
        # Convert generated result to a comparable string format
        if isinstance(generated_result, float):
             # Handle potential floating point inaccuracies for integers
             if abs(generated_result - round(generated_result)) < 1e-9: # Tolerance for int conversion
                  generated_str = str(int(round(generated_result)))
             else:
                  generated_str = f"{generated_result:.10f}" # Format to avoid sci notation sometimes
                  generated_str = generated_str.rstrip('0').rstrip('.') # Clean trailing zeros/dots
        elif isinstance(generated_result, (int, bool)): # Handle bools as 0/1? Or compare directly? Let's compare directly for now.
             generated_str = str(generated_result)
        else: # Attempt to convert other types to string
             generated_str = str(generated_result)
        generated_str = generated_str.strip()
        logger.debug(f"Converted generated result to string: '{generated_str}'")
    except Exception as e:
        logger.warning(f"Conversion of generated result '{generated_result}' to string failed: {e}")
        return False

    ground_truth_str_cleaned = str(ground_truth_str).strip()
    logger.debug(f"Cleaned ground truth string: '{ground_truth_str_cleaned}'")

    logger.debug(f"Attempting direct string comparison...")
    if generated_str == ground_truth_str_cleaned:
        logger.debug("Direct string comparison PASSED.")
        return True
    # Case-insensitive comparison for potential textual answers (like True/False)
    if generated_str.lower() == ground_truth_str_cleaned.lower():
         logger.debug("Case-insensitive string comparison PASSED.")
         return True
    logger.debug("String comparisons failed.")

    try:
        logger.debug(f"Attempting numerical comparison...")
        # Remove common formatting characters for robust numerical comparison
        gen_cleaned_num_str = re.sub(r'[,\s$]', '', generated_str)
        gt_cleaned_num_str = re.sub(r'[,\s$]', '', ground_truth_str_cleaned)
        logger.debug(f"Cleaned numerical strings - Gen: '{gen_cleaned_num_str}', GT: '{gt_cleaned_num_str}'")

        # More robust check if they *look* like numbers before converting
        num_pattern = r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$"
        is_gen_num = re.match(num_pattern, gen_cleaned_num_str)
        is_gt_num = re.match(num_pattern, gt_cleaned_num_str)
        logger.debug(f"Look like numbers? Gen: {bool(is_gen_num)}, GT: {bool(is_gt_num)}")

        if is_gen_num and is_gt_num:
            gen_num = float(gen_cleaned_num_str)
            gt_num = float(gt_cleaned_num_str)
            logger.debug(f"Converted to floats - Gen: {gen_num}, GT: {gt_num}")

            # 1. Check for exact float equality (rarely reliable, but worth a quick check)
            # if gen_num == gt_num:
            #      logger.debug(f"Exact float comparison PASSED.") # Usually handled by tolerance
            #      return True

            # 2. Check with tolerance
            tolerance = 1e-6
            logger.debug(f"Comparing floats with tolerance {tolerance}...")
            if abs(gen_num - gt_num) < tolerance:
                logger.debug(f"Numerical comparison PASSED (tolerance check).")
                return True
            logger.debug(f"Tolerance check failed (|{gen_num} - {gt_num}| = {abs(gen_num - gt_num)}).")

            # 3. Check if both are effectively integers and compare ints
            # Use a tolerance check for is_integer as well
            gen_is_int = abs(gen_num - round(gen_num)) < tolerance
            gt_is_int = abs(gt_num - round(gt_num)) < tolerance
            logger.debug(f"Are they integers (within tolerance)? Gen: {gen_is_int}, GT: {gt_is_int}")
            if gen_is_int and gt_is_int:
                 logger.debug(f"Comparing as integers (rounded)...")
                 if int(round(gen_num)) == int(round(gt_num)):
                      logger.debug(f"Numerical comparison PASSED (integer check).")
                      return True
                 logger.debug(f"Integer check failed.")

    except (ValueError, TypeError) as e:
        logger.debug(f"Numerical comparison failed during conversion/check: {e}")
        pass # Fall through if conversion fails

    logger.debug(f"All comparison methods failed.")
    return False

# --- Main Evaluation Loop (Modified for Detailed PoT Evaluation) ---
evaluation_summary = {}
test_data = load_pot_test_data(TEST_DATA_DIR)

if not test_data:
    logger.error("No valid test data loaded. Exiting.")
    exit(1)

for config in MODEL_CONFIGS:
    logger.info(f"\n--- Evaluating Model: {config.name} ({config.model_path}) ---")
    model = None
    tokenizer = None
    # Initialize detailed results structure
    results = {
        "correct_pot_eval": 0,
        "pot_extraction_failures_eval": 0,
        "pot_execution_failures_eval": 0, # Includes exec errors, timeouts, logic errors, None results
        "pipeline_errors": 0, # Errors in the overall processing (batching, decoding, etc.)
        "total": 0,
        "details": []
    }
    tokenizer_path = config.base_model_path_for_adapter if config.is_adapter_model else config.model_path

    try:
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, local_files_only=config.is_local
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set EOS token as PAD token.")
        tokenizer.padding_side = 'left' # Crucial for batch generation

        logger.info("Loading model...")
        load_start_time = time.time()
        if config.is_adapter_model:
            logger.info(f"Loading base model from: {config.base_model_path_for_adapter}")
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_path_for_adapter,
                torch_dtype=torch.bfloat16,
                device_map=DEVICE,
                trust_remote_code=True,
                local_files_only=config.is_local
            )
            logger.info(f"Loading adapter weights from: {config.model_path}")
            model = PeftModel.from_pretrained(
                base_model, config.model_path, is_trainable=False
            )
            logger.info("Successfully loaded base model and applied adapters.")
        else:
            logger.info(f"Loading standard model from: {config.model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.bfloat16,
                device_map=DEVICE,
                trust_remote_code=True,
                local_files_only=config.is_local
            )
            logger.info("Successfully loaded standard model.")
        load_end_time = time.time()
        logger.info(f"Model loading took {load_end_time - load_start_time:.2f} seconds.")

        model.eval()
        logger.info(f"Model ready for evaluation on {DEVICE}.")

        # --- Evaluation Loop using Batches ---
        with tqdm(total=len(test_data), desc=f"Evaluating {config.name}") as pbar:
            for i in range(0, len(test_data), EVALUATION_BATCH_SIZE):
                # 1. Prepare Batch
                batch_indices = range(i, min(i + EVALUATION_BATCH_SIZE, len(test_data)))
                batch_items = [test_data[j] for j in batch_indices]
                batch_questions = [item["question"] for item in batch_items]
                batch_prompts = [format_prompt(q) for q in batch_questions]
                batch_ground_truths = [item["answer_value"] for item in batch_items]

                try:
                    # 2. Batch Tokenization
                    inputs = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024 # Adjust if needed, depends on expected prompt+answer length
                    ).to(DEVICE)
                    input_length = inputs['input_ids'].shape[1]

                    # 3. Batch Generation
                    gen_start_time = time.time()
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                            temperature=GENERATION_TEMPERATURE if GENERATION_TEMPERATURE > 0 else 1.0, # Temp > 0 requires do_sample=True
                            top_p=GENERATION_TOP_P if GENERATION_TEMPERATURE > 0 else 1.0,
                            do_sample=True if GENERATION_TEMPERATURE > 0 else False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            # num_return_sequences=1 # Default
                        )
                    gen_end_time = time.time()
                    logger.debug(f"Batch {i // EVALUATION_BATCH_SIZE + 1}: Generation took {gen_end_time - gen_start_time:.2f}s")


                    # 4. Batch Decoding
                    decode_start_time = time.time()
                    # Only decode the newly generated tokens
                    if outputs.shape[1] > input_length:
                        generated_tokens = outputs[:, input_length:]
                        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    else:
                         generated_texts = [""] * len(batch_items) # Handle cases where no new tokens were generated
                         logger.warning(f"Batch starting index {i}: No new tokens generated for {len(batch_items)} items (input length {input_length}, output length {outputs.shape[1]}).")
                    decode_end_time = time.time()
                    logger.debug(f"Batch {i // EVALUATION_BATCH_SIZE + 1}: Decoding took {decode_end_time - decode_start_time:.2f}s")


                    # 5. Process Each Item in the Batch with Detailed PoT Evaluation
                    eval_start_time = time.time()
                    for j, original_item_index in enumerate(batch_indices):
                        item_log_prefix = f"Item {original_item_index}:" # For easier log reading
                        logger.info(f"--- {item_log_prefix} Processing START ---")

                        generated_text = generated_texts[j]
                        question = batch_questions[j]
                        ground_truth = batch_ground_truths[j]

                        # Reset detailed eval results for this item
                        extracted_code_eval = None
                        executed_result_eval = None
                        exec_error_eval = None
                        is_correct_eval = False
                        evaluation_method_eval = "N/A"
                        error_message_eval = None # More specific error tracking

                        log_gen_text_snippet = generated_text[:150].replace('\n', '\\n')
                        logger.debug(f"{item_log_prefix} Generated Text (len: {len(generated_text)}): {log_gen_text_snippet}...")


                        try:
                            # --- Attempt Detailed Program-of-Thought (PoT) Evaluation ---
                            logger.info(f"{item_log_prefix} Calling extract_python_code...")
                            extracted_code_eval = extract_python_code(generated_text)

                            if extracted_code_eval:
                                logger.info(f"{item_log_prefix} Code extracted successfully.")
                                log_extracted_code_snippet = extracted_code_eval[:150].replace('\n', '\\n')
                                logger.debug(f"{item_log_prefix} Extracted Code (first 150):\n{log_extracted_code_snippet}\n---")

                                logger.info(f"{item_log_prefix} Calling safe_execute_code...")
                                executed_result_eval, exec_error_eval = safe_execute_code(extracted_code_eval, timeout=TIMEOUT_SECONDS)

                                if exec_error_eval:
                                    logger.warning(f"{item_log_prefix} Code execution failed or yielded error: {exec_error_eval}")
                                    error_message_eval = f"PoT Eval: Execution Failed - {exec_error_eval}"
                                    evaluation_method_eval = "PoT Execution Failed"
                                    results["pot_execution_failures_eval"] += 1
                                    is_correct_eval = False
                                elif executed_result_eval is not None:
                                    logger.info(f"{item_log_prefix} Code execution successful. Result: {executed_result_eval}")
                                    logger.info(f"{item_log_prefix} Calling compare_results...")
                                    is_correct_eval = compare_results(executed_result_eval, ground_truth)
                                    logger.info(f"{item_log_prefix} Comparison result: {is_correct_eval}")
                                    if is_correct_eval:
                                        evaluation_method_eval = "PoT Correct"
                                        results["correct_pot_eval"] += 1
                                        error_message_eval = None # Clear error if correct
                                    else:
                                        evaluation_method_eval = "PoT Incorrect Logic/Result"
                                        error_message_eval = f"PoT Eval: Result mismatch (Got: '{executed_result_eval}', Expected: '{ground_truth}')"
                                        results["pot_execution_failures_eval"] += 1 # Count mismatch as a type of execution failure
                                else: # Execution finished without error, but result was None
                                    logger.warning(f"{item_log_prefix} Code execution returned None result without explicit error.")
                                    error_message_eval = "PoT Eval: Execution returned None"
                                    evaluation_method_eval = "PoT Execution Returned None"
                                    results["pot_execution_failures_eval"] += 1
                                    is_correct_eval = False
                            else:
                                logger.warning(f"{item_log_prefix} Failed to extract code.")
                                error_message_eval = "PoT Eval: Failed to extract code"
                                evaluation_method_eval = "PoT Extraction Failed"
                                results["pot_extraction_failures_eval"] += 1
                                is_correct_eval = False

                        except Exception as item_eval_e:
                            logger.error(f"{item_log_prefix} UNEXPECTED error during detailed PoT evaluation logic: {item_eval_e}")
                            logger.error(traceback.format_exc())
                            error_message_eval = f"PoT Eval Logic Error: {str(item_eval_e)}"
                            evaluation_method_eval = "Evaluation Logic Error"
                            results["pipeline_errors"] += 1 # Count this as a pipeline error
                            is_correct_eval = False

                        # Store detailed results for this item
                        logger.debug(f"{item_log_prefix} Storing detailed results...")
                        results["details"].append({
                            "item_index": original_item_index,
                            "question": question,
                            "ground_truth": ground_truth,
                            "generated_full_text": generated_text,
                            "extracted_code_eval": extracted_code_eval,
                            "executed_result_pot_eval": str(executed_result_eval) if executed_result_eval is not None else None,
                            "execution_error_eval": exec_error_eval, # Specific error from execution
                            "is_correct_eval": is_correct_eval,
                            "evaluation_method_eval": evaluation_method_eval,
                            "error_message_eval": error_message_eval, # Overall item evaluation error summary
                        })
                        pbar.update(1) # Update progress for this single item
                        logger.info(f"--- {item_log_prefix} Processing END ---")
                    eval_end_time = time.time()
                    logger.debug(f"Batch {i // EVALUATION_BATCH_SIZE + 1}: Evaluation took {eval_end_time - eval_start_time:.2f}s")


                except Exception as batch_e:
                    # Critical error during batch tokenization, generation, or decoding
                    logger.error(f"CRITICAL Error during processing BATCH starting at index {i} for {config.name}: {batch_e}")
                    logger.error(traceback.format_exc())
                    # Mark all items in this batch as errors
                    for k, failed_item_index in enumerate(batch_indices):
                         # Try to get question/gt if available, otherwise use placeholders
                        failed_question = batch_questions[k] if k < len(batch_questions) else "N/A - Batch Error"
                        failed_gt = batch_ground_truths[k] if k < len(batch_ground_truths) else "N/A - Batch Error"
                        results["details"].append({
                            "item_index": failed_item_index,
                            "question": failed_question,
                            "ground_truth": failed_gt,
                            "generated_full_text": "N/A - Batch Error",
                            "extracted_code_eval": None,
                            "executed_result_pot_eval": None,
                            "execution_error_eval": None,
                            "is_correct_eval": False,
                            "evaluation_method_eval": "Batch Error",
                            "error_message_eval": f"Critical Batch Error: {str(batch_e)}"
                        })
                        results["pipeline_errors"] += 1
                    pbar.update(len(batch_indices)) # Update progress bar for the failed batch


        # --- Calculate Final Results (using detailed eval fields) ---
        results["total"] = len(test_data)
        accuracy_pot = (results["correct_pot_eval"] / results["total"]) * 100 if results["total"] > 0 else 0
        extraction_fail_rate = (results["pot_extraction_failures_eval"] / results["total"]) * 100 if results["total"] > 0 else 0
        execution_fail_rate = (results["pot_execution_failures_eval"] / results["total"]) * 100 if results["total"] > 0 else 0
        pipeline_error_rate = (results["pipeline_errors"] / results["total"]) * 100 if results["total"] > 0 else 0

        logger.info(f"--- Detailed Results for {config.name} ---")
        logger.info(f"Total Items: {results['total']}")
        logger.info(f"Correct via PoT (Eval): {results['correct_pot_eval']}")
        logger.info(f"PoT Extraction Failures (Eval): {results['pot_extraction_failures_eval']}")
        logger.info(f"PoT Execution/Logic Failures (Eval): {results['pot_execution_failures_eval']}")
        logger.info(f"Pipeline Processing Errors (Batch/Item): {results['pipeline_errors']}")
        logger.info(f"Accuracy (PoT Correct / Total): {accuracy_pot:.2f}%")
        logger.info(f"PoT Extraction Failure Rate: {extraction_fail_rate:.2f}%")
        logger.info(f"PoT Execution/Logic Failure Rate: {execution_fail_rate:.2f}%")
        logger.info(f"Pipeline Error Rate: {pipeline_error_rate:.2f}%")

        evaluation_summary[config.name] = {
            "model_path": config.model_path,
            "base_model_path": config.base_model_path_for_adapter if config.is_adapter_model else config.model_path,
            "is_adapter": config.is_adapter_model,
            "batch_size": EVALUATION_BATCH_SIZE,
            "generation_temperature": GENERATION_TEMPERATURE,
            "generation_top_p": GENERATION_TOP_P,
            "total_items": results["total"],
            "correct_pot_eval": results["correct_pot_eval"],
            "pot_extraction_failures_eval": results["pot_extraction_failures_eval"],
            "pot_execution_failures_eval": results["pot_execution_failures_eval"],
            "pipeline_errors": results["pipeline_errors"],
            "accuracy_pot_eval_percent": round(accuracy_pot, 2),
            "pot_extraction_failure_eval_percent": round(extraction_fail_rate, 2),
            "pot_execution_failure_eval_percent": round(execution_fail_rate, 2),
            "pipeline_error_rate_percent": round(pipeline_error_rate, 2),
            "details": results["details"] # Keep detailed logs
        }

    except Exception as e:
        logger.error(f"Critical error setting up or running evaluation for model {config.name}: {e}")
        logger.error(traceback.format_exc())
        evaluation_summary[config.name] = {"error": f"Setup/Run Error: {str(e)}", "model_path": config.model_path}

    finally:
        # Clean up memory
        logger.info(f"Cleaning up resources for {config.name}...")
        del model
        if 'base_model' in locals(): del base_model
        del tokenizer
        if 'inputs' in locals(): del inputs
        if 'outputs' in locals(): del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Cleaned up resources for {config.name}.")
        time.sleep(5) # Add a small delay before loading the next model

# --- Save and Print Summary ---
try:
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        # Use default=str to handle potential non-serializable types like numpy ints/floats if any crept in
        json.dump(evaluation_summary, f, ensure_ascii=False, indent=4, default=str)
    logger.info(f"\nSaved detailed evaluation results to {RESULTS_FILE}")
except Exception as e:
    logger.error(f"Error saving results to {RESULTS_FILE}: {e}")

logger.info("\n--- Final Detailed Evaluation Summary ---")
for name, summary in evaluation_summary.items():
    if "error" in summary:
        logger.info(f"{name}: OVERALL ERROR ({summary['error']})")
    else:
        logger.info(f"{name} (Batch: {summary.get('batch_size', 'N/A')}, Temp: {summary.get('generation_temperature', 'N/A')}, TopP: {summary.get('generation_top_p', 'N/A')}):")
        logger.info(f"  Accuracy (PoT Eval): {summary['accuracy_pot_eval_percent']:.2f}% ({summary['correct_pot_eval']}/{summary['total_items']})")
        logger.info(f"  PoT Failures (Extract: {summary['pot_extraction_failures_eval']} [{summary['pot_extraction_failure_eval_percent']:.2f}%], Exec/Logic: {summary['pot_execution_failures_eval']} [{summary['pot_execution_failure_eval_percent']:.2f}%])")
        logger.info(f"  Pipeline Errors: {summary['pipeline_errors']} ({summary['pipeline_error_rate_percent']:.2f}%)")


logger.info("\n--- Evaluation Complete ---")