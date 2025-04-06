# Standard library imports
import json
import logging
import os
import re # Regular expression import added
import gc
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
from io import StringIO
import contextlib

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Configuration (Keep as before) ---
@dataclass
class ModelConfig:
    name: str
    model_path: str # Path to the base model OR the adapter checkpoint directory
    is_local: bool = True
    model_type: Literal["causal", "encoder", "encoder-decoder"] = "causal"
    is_adapter_model: bool = False # Flag to indicate if model_path points to LoRA adapters
    base_model_path_for_adapter: Optional[str] = None # Path to the base model if is_adapter_model is True

    def __post_init__(self):
        if self.is_adapter_model and not self.base_model_path_for_adapter:
            raise ValueError(f"Model '{self.name}' is marked as adapter model, but 'base_model_path_for_adapter' is not provided.")
        if self.is_adapter_model and self.base_model_path_for_adapter == self.model_path:
             raise ValueError(f"For adapter model '{self.name}', 'base_model_path_for_adapter' cannot be the same as 'model_path'.")

# --- Define Models to Evaluate (Keep as before) ---
BASE_LLAMA_PATH = "/scratch/jsong132/Technical_Llama3.2/llama3.2_3b"

MODEL_CONFIGS = [
    ModelConfig(
        name="Llama-3.2-3b-Base",
        model_path="/scratch/jsong132/Technical_Llama3.2/llama3.2_3b",
        is_local=True,
        is_adapter_model=False
    ),
    ModelConfig(
        name="Llama-3.2-3b-Socrates-Math-Tuned-Adapter",
        model_path="/scratch/jsong132/Technical_Llama3.2/FineTuning/Math/Tune_Results/llama3.2_Socrates_Math",
        is_local=True,
        is_adapter_model=True,
        base_model_path_for_adapter=BASE_LLAMA_PATH
    ),
]

# --- Evaluation Configuration (Keep as before) ---
TEST_DATA_DIR = "/scratch/jsong132/Technical_Llama3.2/DB/PoT/test"
RESULTS_FILE = "math_pot_evaluation_results_adapters_v2.json" # Changed filename slightly
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATION_MAX_NEW_TOKENS = 512
GENERATION_TEMPERATURE = 0.1
GENERATION_TOP_P = 0.9
TIMEOUT_SECONDS = 10

# --- Helper Functions (load_pot_test_data, format_prompt, extract_python_code, safe_execute_code, compare_results - keep as before) ---
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
                             # Ensure essential keys exist, especially answer_value
                            if isinstance(item, dict) and "question" in item and "answer_value" in item and item["answer_value"] is not None:
                                all_test_items.append({
                                    "question": str(item["question"]),
                                    "answer_value": str(item["answer_value"]) # Keep as string
                                })
                                count += 1
                            else:
                                logger.warning(f"Skipping item in {filename} due to missing keys or null answer_value: {item}")
                        logger.info(f"Loaded {count} valid items from {filename}")
                    else:
                        logger.warning(f"File {filename} does not contain a list.")
            except Exception as e:
                logger.error(f"Error loading test file {file_path}: {str(e)}")

    logger.info(f"Total valid test items loaded: {len(all_test_items)}")
    return all_test_items

def format_prompt(question: str) -> str:
    """Formats the question into the prompt structure used during training."""
    # Using a simpler prompt might be better for base models not trained on PoT
    # return f"Question: {question}\nAnswer:"
    # Keep original PoT prompt for consistency with potential fine-tuning
    return f"Question: {question}\nAnswer:\n```python\n"

def extract_python_code(generated_text: str, prompt: str) -> Optional[str]:
    """Extracts Python code block from the model's generation."""
    if generated_text.startswith(prompt):
        code_part = generated_text[len(prompt):]
    else:
        code_part = generated_text

    match_py = re.search(r"```python\s*(.*?)\s*```", code_part, re.DOTALL | re.IGNORECASE)
    if match_py:
        return match_py.group(1).strip()

    match_any = re.search(r"```\s*(.*?)\s*```", code_part, re.DOTALL)
    if match_any:
        potential_code = match_any.group(1).strip()
        # Basic check if it looks like code
        if any(kw in potential_code for kw in ['def ', 'return ', '=', 'import ', 'print(']):
             return potential_code
        else:
             logger.warning(f"Code block ```...``` found but might not be Python: {potential_code[:100]}...")
             # Return it anyway, let execution handle errors
             return potential_code

    # Fallback: try to find 'def solution' even without fences
    match_def = re.search(r"(def\s+solution\s*\(.*?\):.*?)(?=\n\w|#|$)", code_part, re.DOTALL)
    if match_def:
         logger.warning("No markdown fences found, attempting to extract based on 'def solution():'")
         return match_def.group(1).strip()

    # Final fallback: maybe the code starts immediately? Often problematic.
    code_part_cleaned = code_part.split("</s>")[0].split("<|endoftext|>")[0].strip()
    if any(kw in code_part_cleaned for kw in ['def ', 'return ', '=', 'import ', 'print(']):
        logger.warning("Could not reliably extract code block, using potentially incomplete remaining text.")
        return code_part_cleaned

    logger.debug("Failed to extract any plausible code block.") # Changed level to debug
    return None # Explicitly return None if no code found

@contextlib.contextmanager
def time_limit(seconds):
    """Basic context manager for time limits (might not be thread-safe or perfect)."""
    start = time.time()
    def check():
        if time.time() - start > seconds:
            raise TimeoutError(f"Code execution exceeded {seconds} seconds.")
    yield check # Not actually used in safe_execute_code, but kept for structure

def safe_execute_code(code: str, timeout: int = TIMEOUT_SECONDS) -> Optional[Any]:
    """Executes the extracted Python code safely and returns the result."""
    if not code: return None

    # Simple check for potentially unsafe operations (basic protection)
    if any(kw in code for kw in ['import os', 'import sys', 'subprocess', 'eval(', 'exec(', 'open(']):
        logger.error(f"Potential unsafe code detected. Execution blocked.\nCode:\n{code[:500]}...")
        return None

    output_buffer = StringIO() # Capture print statements if any
    try:
        local_scope = {}
        # More restricted builtins
        safe_builtins = {
             'print': print, 'range': range, 'len': len, 'abs': abs, 'min': min,
             'max': max, 'sum': sum, 'int': int, 'float': float, 'str': str,
             'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
             'pow': pow, 'round': round, 'divmod': divmod, 'enumerate':enumerate,
             'zip':zip, 'sorted':sorted, 'map':map, 'filter':filter, 'all':all, 'any':any
        }
        # Allow math module safely
        global_scope = {'__builtins__': safe_builtins, 'math': __import__('math')}

        start_time = time.time()
        # Redirect stdout to capture prints
        with contextlib.redirect_stdout(output_buffer):
             exec(code, global_scope, local_scope)
        execution_time = time.time() - start_time

        if execution_time > timeout:
             logger.warning(f"Code execution finished but took longer than timeout: {execution_time:.2f}s")
             # Treat as timeout for safety? Or allow result? Let's allow for now.

        # Priority: result from solution() function
        if 'solution' in local_scope and callable(local_scope['solution']):
            result = local_scope['solution']()
            logger.debug(f"Execution successful via solution(), result: {result}")
            return result

        # Fallback 1: 'result' variable in scope
        if 'result' in local_scope:
             result = local_scope['result']
             logger.warning("Executed code, 'solution' function not found, using 'result' variable.")
             return result

        # Fallback 2: Last printed value (if numerical)
        printed_output = output_buffer.getvalue().strip()
        if printed_output:
            try:
                # Try to interpret the last line of print as the result
                last_line = printed_output.splitlines()[-1].strip()
                # Attempt conversion to float/int
                if '.' in last_line:
                    result = float(last_line)
                else:
                    result = int(last_line)
                logger.warning(f"Executed code, no solution()/result, using last printed value: {result}")
                return result
            except ValueError:
                logger.warning(f"Last printed value '{last_line}' is not numerical.")


        logger.warning("Executed code, but no 'solution' function, 'result' variable, or numerical print found.")
        return None # Indicate execution happened but no usable result found

    except SyntaxError as e: logger.error(f"Syntax Error: {e}\nCode:\n{code[:500]}..."); return None
    except NameError as e: logger.error(f"Name Error: {e}\nCode:\n{code[:500]}..."); return None
    except TypeError as e: logger.error(f"Type Error: {e}\nCode:\n{code[:500]}..."); return None
    except ZeroDivisionError as e: logger.error(f"ZeroDivisionError: {e}\nCode:\n{code[:500]}..."); return None
    except TimeoutError as e: logger.error(f"Timeout Error: {e}\nCode:\n{code[:500]}..."); return None
    except Exception as e:
        logger.error(f"Unexpected Execution Error: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc()); return None


def compare_results(generated_result: Any, ground_truth_str: str) -> bool:
    """Compares the generated result (from code exec OR direct extract) with the ground truth."""
    if generated_result is None: return False

    # Convert both to string first for direct comparison
    generated_str = str(generated_result).strip()
    ground_truth_str_cleaned = ground_truth_str.strip()

    if generated_str == ground_truth_str_cleaned: return True

    # Try numerical comparison (handle potential formatting like commas, units)
    try:
        # Basic cleaning: remove commas, maybe units if necessary (keep simple for now)
        gen_cleaned = re.sub(r'[,\s]', '', generated_str)
        gt_cleaned = re.sub(r'[,\s]', '', ground_truth_str_cleaned)

        # Check if they look like numbers before converting
        if re.fullmatch(r"[-+]?\d*\.?\d+", gen_cleaned) and re.fullmatch(r"[-+]?\d*\.?\d+", gt_cleaned):
            gen_num = float(gen_cleaned)
            gt_num = float(gt_cleaned)

            # Integer comparison if both are integers
            if gen_num.is_integer() and gt_num.is_integer():
                if int(gen_num) == int(gt_num): return True
            # Floating point comparison with tolerance
            if abs(gen_num - gt_num) < 1e-6: return True

    except (ValueError, TypeError) as e:
        # logger.debug(f"Numerical comparison failed: Gen='{generated_str}', GT='{ground_truth_str}', Error: {e}")
        pass # Expected if values aren't purely numeric

    logger.debug(f"Comparison failed: Generated='{generated_str}', GroundTruth='{ground_truth_str_cleaned}'")
    return False

# --- *** NEW HELPER FUNCTION *** ---
def extract_direct_answer(generated_text: str) -> Optional[str]:
    """
    Fallback function to extract the likely numerical answer directly from text.
    Finds the *last* number in the string as a heuristic.
    """
    # Regex to find numbers (integers or floats, potentially with signs)
    # Handles numbers like 123, 45.6, -7, +8.0
    numbers_found = re.findall(r"[-+]?\d*\.?\d+", generated_text)

    if numbers_found:
        last_number = numbers_found[-1]
        logger.info(f"Direct extraction found last number: '{last_number}'")
        # Basic validation: Avoid ridiculously long numbers or just signs/dots
        if len(last_number) > 1 or last_number.isdigit():
             # Avoid returning just "." or "+" or "-"
             if last_number not in ['.', '+', '-']:
                 return last_number
        logger.warning(f"Direct extraction ignored invalid number candidate: '{last_number}'")

    logger.info("Direct extraction found no suitable number.")
    return None
# --- *** END NEW HELPER FUNCTION *** ---


# --- Main Evaluation Loop ---
evaluation_summary = {}
test_data = load_pot_test_data(TEST_DATA_DIR)

if not test_data:
    logger.error("No valid test data loaded. Exiting.")
    exit(1)

for config in MODEL_CONFIGS:
    logger.info(f"\n--- Evaluating Model: {config.name} ({config.model_path}) ---")
    model = None
    tokenizer = None
    # Reset results for each model
    results = {
        "correct_pot": 0,           # Count corrects via PoT
        "correct_fallback": 0,    # Count corrects via fallback
        "total_correct": 0,       # Total correct (PoT + fallback)
        "total": 0,
        "pipeline_errors": 0,       # Count critical exceptions during processing
        "pot_failures": 0,        # Count items where PoT failed (extraction or exec)
        "fallback_attempts": 0,   # Count items where fallback was attempted
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
        tokenizer.padding_side = 'left' # Use left padding for generation

        logger.info("Loading model...")
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
            # Ensure the merged model is on the correct device if device_map wasn't enough
            # model.to(DEVICE) # Usually handled by device_map in base, but good practice
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

        model.eval()
        logger.info(f"Model ready for evaluation on {DEVICE}.")

        # --- *** MODIFIED Evaluation Loop for items *** ---
        for item_index, item in enumerate(tqdm(test_data, desc=f"Evaluating {config.name}")):
            question = item["question"]
            ground_truth = item["answer_value"]
            prompt = format_prompt(question)

            generated_code = None
            executed_result = None
            direct_answer_str = None
            is_correct = False
            error_msg = None
            generated_text = "N/A"
            evaluation_method = "N/A" # To track how it was evaluated

            try:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE) # Added truncation

                # Generate text
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                        temperature=GENERATION_TEMPERATURE,
                        top_p=GENERATION_TOP_P,
                        do_sample=True if GENERATION_TEMPERATURE > 0 else False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                input_length = inputs['input_ids'].shape[1]
                # Handle potential case where output is shorter than input (shouldn't happen with max_new_tokens > 0)
                if outputs.shape[1] > input_length:
                     output_tokens = outputs[0, input_length:]
                     generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                else:
                     generated_text = "" # No new tokens generated
                     logger.warning(f"Item {item_index}: No new tokens generated.")


                # --- Attempt 1: Program-of-Thought (PoT) ---
                pot_succeeded = False
                generated_code = extract_python_code(generated_text, "") # Pass empty prompt as we only decode new tokens

                if generated_code:
                    executed_result = safe_execute_code(generated_code)
                    if executed_result is not None:
                        is_correct = compare_results(executed_result, ground_truth)
                        evaluation_method = "PoT"
                        pot_succeeded = True
                        if is_correct:
                            results["correct_pot"] += 1
                        else:
                             error_msg = "PoT: Execution result mismatch"
                    else: # Code execution failed
                         error_msg = "PoT: Code execution failed or returned None"
                         results["pot_failures"] += 1
                else: # Code extraction failed
                    error_msg = "PoT: Failed to extract code"
                    results["pot_failures"] += 1 # Count as PoT failure

                # --- Attempt 2: Fallback Direct Answer Extraction (if PoT failed) ---
                if not pot_succeeded:
                    results["fallback_attempts"] += 1
                    logger.debug(f"Item {item_index}: PoT failed ({error_msg}). Trying direct answer extraction.")
                    direct_answer_str = extract_direct_answer(generated_text)
                    if direct_answer_str is not None:
                        is_correct = compare_results(direct_answer_str, ground_truth)
                        evaluation_method = "Fallback Direct Extraction"
                        if is_correct:
                            results["correct_fallback"] += 1
                            error_msg = None # Clear previous PoT error if fallback succeeded
                        else:
                             # Keep the PoT error message, but add fallback info
                             error_msg += " | Fallback: Direct extraction result mismatch"
                    else:
                        # Keep the PoT error message, add fallback info
                        error_msg += " | Fallback: Direct extraction found no number"
                        # is_correct remains False


            except Exception as e:
                logger.error(f"CRITICAL Error during processing item {item_index} for {config.name}: {e}")
                logger.error(traceback.format_exc())
                error_msg = f"Critical Pipeline Error: {str(e)}"
                results["pipeline_errors"] += 1
                is_correct = False # Ensure incorrect on critical error
                evaluation_method = "Error"


            # Store results for this item
            results["details"].append({
                "item_index": item_index,
                "question": question,
                "ground_truth": ground_truth,
                # "prompt": prompt, # Optional: can make JSON large
                "generated_full_text": generated_text,
                "extracted_code": generated_code,
                "executed_result (PoT)": str(executed_result) if executed_result is not None else None,
                "direct_answer_extraction (Fallback)": direct_answer_str,
                "evaluation_method": evaluation_method,
                "is_correct": is_correct,
                "error_message": error_msg
            })
        # --- End Evaluation Loop for items ---

        results["total"] = len(test_data)
        results["total_correct"] = results["correct_pot"] + results["correct_fallback"]
        accuracy = (results["total_correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        pipeline_error_rate = (results["pipeline_errors"] / results["total"]) * 100 if results["total"] > 0 else 0

        logger.info(f"--- Results for {config.name} ---")
        logger.info(f"Total Items: {results['total']}")
        logger.info(f"Correct via PoT: {results['correct_pot']}")
        logger.info(f"Correct via Fallback Extraction: {results['correct_fallback']}")
        logger.info(f"Total Correct: {results['total_correct']}")
        logger.info(f"Pipeline Processing Errors: {results['pipeline_errors']}")
        logger.info(f"Accuracy (Total Correct / Total): {accuracy:.2f}%")
        logger.info(f"Pipeline Error Rate: {pipeline_error_rate:.2f}%")
        logger.info(f"PoT Failures (Extraction/Execution): {results['pot_failures']}")
        logger.info(f"Fallback Extraction Attempts: {results['fallback_attempts']}")


        evaluation_summary[config.name] = {
            "model_path": config.model_path,
            "base_model_path": config.base_model_path_for_adapter if config.is_adapter_model else config.model_path,
            "is_adapter": config.is_adapter_model,
            "total_items": results["total"],
            "correct_pot": results["correct_pot"],
            "correct_fallback": results["correct_fallback"],
            "total_correct": results["total_correct"],
            "pipeline_errors": results["pipeline_errors"],
            "accuracy_percent": round(accuracy, 2),
            "pipeline_error_rate_percent": round(pipeline_error_rate, 2),
            "pot_failures": results["pot_failures"],
            "fallback_attempts": results["fallback_attempts"],
            "details": results["details"] # Keep details for debugging
        }

    except Exception as e:
        logger.error(f"Critical error setting up or running evaluation for model {config.name}: {e}")
        logger.error(traceback.format_exc())
        evaluation_summary[config.name] = {"error": f"Setup/Run Error: {str(e)}", "model_path": config.model_path}

    finally:
        # Clean up memory
        del model
        if 'base_model' in locals(): del base_model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Cleaned up resources for {config.name}.")

# --- Save and Print Summary ---
try:
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        # Use default=str to handle potential non-serializable types in results (though should be fine now)
        json.dump(evaluation_summary, f, ensure_ascii=False, indent=4, default=str)
    logger.info(f"\nSaved detailed evaluation results to {RESULTS_FILE}")
except Exception as e:
    logger.error(f"Error saving results to {RESULTS_FILE}: {e}")

logger.info("\n--- Final Evaluation Summary ---")
for name, summary in evaluation_summary.items():
    if "error" in summary:
        logger.info(f"{name}: OVERALL ERROR ({summary['error']})")
    else:
        logger.info(f"{name}: Accuracy = {summary['accuracy_percent']:.2f}% ({summary['total_correct']}/{summary['total_items']}) | Correct (PoT: {summary['correct_pot']}, Fallback: {summary['correct_fallback']}) | Pipeline Errors: {summary['pipeline_errors']}")

logger.info("\n--- Evaluation Complete ---")