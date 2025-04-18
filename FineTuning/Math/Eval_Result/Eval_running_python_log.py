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

VERSION = "V4"
# --- Configuration ---
INPUT_JSON_PATH = "/scratch/jsong132/Technical_Llama3.2/FineTuning/Math/Eval_Result/V3_generation_math_logs.json"
# --- MODIFIED: Define separate output files ---
OUTPUT_SUMMARY_PATH = f"{VERSION}_reeval_summary.json"
OUTPUT_SUCCESS_DETAILS_PATH = f"{VERSION}_reeval_success_details.json"
OUTPUT_FAILURE_DETAILS_PATH = f"{VERSION}_reeval_failure_details.json"
# --- END MODIFIED ---
TARGET_MODEL_KEY = f"Llama-3.2-3b-Socrates-Math-{VERSION}"
TIMEOUT_SECONDS = 4

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Functions (extract_python_code, safe_execute_code, compare_results) ---
# Assume these functions (extract_python_code, safe_execute_code, compare_results)
# are defined exactly as in the previous version you provided.
# Keeping them here for completeness would make the code very long.
# ... (Paste the definitions of extract_python_code, safe_execute_code, compare_results here) ...
class TimeoutError(Exception):
    """Custom Timeout Error"""
    pass

def _timeout_handler(signum, frame):
    """Signal handler function to raise TimeoutError"""
    logger.warning("Execution timed out via SIGALRM.")
    raise TimeoutError(f"Code execution exceeded timeout of {TIMEOUT_SECONDS} seconds")

def extract_python_code(generated_text: str) -> Optional[str]:
    """
    Extracts Python code block from the model's generation,
    with improved handling for missing fences and trailing fences.
    (Version 2 with extra logging - FIX for f-string backslash)
    """
    func_name = "extract_python_code"
    if not generated_text:
         logger.debug(f"Input text is empty or None.")
         return None

    # --- FIX: Perform replace outside f-string ---
    log_text_snippet = generated_text[:100].replace('\n', '\\n')
    logger.debug(f"Input text (first 100 chars): {log_text_snippet}")
    # --- End FIX ---

    code_part = generated_text.strip()
    extracted_code: Optional[str] = None

    # 1. Try matching ```python ... ``` (Highest priority)
    logger.debug(f"Attempting ```python match...")
    match_py = re.search(r"```python\s*(.*?)\s*```", code_part, re.DOTALL | re.IGNORECASE)
    if match_py:
        extracted_code = match_py.group(1).strip()
        logger.debug(f"Extracted using ```python fence.")
    else:
        logger.debug(f"No ```python match found.")
        # 2. Try matching ``` ... ``` (any language or none specified)
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
            # Use greedy match (.*)
            match_def = re.search(r"^(def\s+solution\s*\(.*?\):.*)(?:```)?$", code_part, re.DOTALL | re.MULTILINE)
            if match_def:
                extracted_code = match_def.group(1).strip()
                logger.debug(f"Extracted based on 'def solution():'.")
            else:
                 logger.debug(f"No 'def solution():' match found.")


    # --- CRITICAL CLEANUP STEP ---
    if extracted_code:
        original_extracted = extracted_code # For logging comparison
        # --- FIX: Perform replace outside f-string ---
        log_extracted_snippet_before = extracted_code[:100].replace('\n','\\n')
        logger.debug(f"Running cleanup on extracted code (first 100): {log_extracted_snippet_before}")
        # --- End FIX ---

        # Remove potential trailing ```
        if extracted_code.endswith('```'):
            extracted_code = extracted_code[:-3].strip()
            logger.debug(f"Removed trailing ```.")
        # Remove potential leading ```
        if extracted_code.startswith('```'):
             extracted_code = extracted_code[3:].strip()
             logger.debug(f"Removed leading ```.")
        # Remove potential leading python identifier
        if extracted_code.lower().startswith('python'):
             extracted_code = extracted_code[6:].lstrip()
             logger.debug(f"Removed leading 'python'.")

        if extracted_code != original_extracted:
             # --- FIX: Perform replace outside f-string ---
             log_extracted_snippet_after = extracted_code[:100].replace('\n','\\n')
             logger.debug(f"Code after cleanup (first 100): {log_extracted_snippet_after}")
             # --- End FIX ---
        else:
             logger.debug(f"No cleanup needed.")


    # 4. Final validation
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
        # 5. Last Resort Check (less reliable)
        logger.debug(f"No code extracted yet. Trying last resort check on full cleaned text.")
        code_part_cleaned = code_part.split("</s>")[0].split("<|endoftext|>")[0].strip()
        if not code_part_cleaned.startswith('```') and not code_part_cleaned.endswith('```'):
            if any(kw in code_part_cleaned for kw in ['def ', 'return ', '=', 'import ', 'print(']):
                 # --- FIX: Perform replace outside f-string for logging ---
                 log_cleaned_snippet = code_part_cleaned[:150].replace('\n', '\\n')
                 logger.warning(f"Using full cleaned text as code (last resort): {log_cleaned_snippet}")
                 # --- End FIX ---
                 return code_part_cleaned
            else:
                 logger.debug(f"Last resort check failed: No keywords in cleaned text.")
        else:
             logger.debug(f"Last resort check failed: Cleaned text still has fences.")


    logger.debug(f"Failed to extract any plausible code block.")
    return None

def safe_execute_code(code: str, timeout: int = TIMEOUT_SECONDS) -> tuple[Optional[Any], Optional[str]]:
    """
    Executes the extracted Python code safely with a timeout mechanism (using signal)
    and returns the result and any error message.
    """
    func_name = "safe_execute_code"
    if not code:
        logger.debug(f"No code provided.")
        return None, "No code provided to execute."
    logger.debug(f"Attempting to execute code (first 200 chars):\n--- CODE START ---\n{code[:200]}\n--- CODE END ---")

    # --- Check if signal module is available (for non-Unix safety) ---
    if not hasattr(signal, "SIGALRM"):
         logger.warning("signal.SIGALRM not available on this platform (likely Windows). Timeout disabled.")
         # Fallback to original execution without robust timeout
         try:
             output_buffer = StringIO()
             local_scope = {}
             safe_builtins = {
                 'print': print, 'range': range, 'len': len, 'abs': abs, 'min': min,
                 'max': max, 'sum': sum, 'int': int, 'float': float, 'str': str,
                 'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
                 'pow': pow, 'round': round, 'divmod': divmod, 'enumerate':enumerate,
                 'zip':zip, 'sorted':sorted, 'map':map, 'filter':filter, 'all':all, 'any':any
             }
             global_scope = {'__builtins__': safe_builtins, 'math': __import__('math')}
             with contextlib.redirect_stdout(output_buffer):
                  exec(code, global_scope, local_scope)

             result = None # Initialize result
             if 'solution' in local_scope and callable(local_scope['solution']):
                 result = local_scope['solution']()
             elif 'result' in local_scope:
                 result = local_scope['result']
             else:
                 # Try getting result from print output
                 printed_output = output_buffer.getvalue().strip()
                 if printed_output:
                     try:
                         last_line = printed_output.splitlines()[-1].strip()
                         if '.' in last_line or 'e' in last_line.lower():
                              result = float(last_line)
                         else:
                              result = int(last_line)
                     except (ValueError, IndexError):
                          pass # Ignore if conversion fails or no lines

             execution_error = None
             if result is None:
                 execution_error = "Code executed without error (no timeout), but yielded no discernible result (None)."

             return result, execution_error

         except Exception as e:
             execution_error = f"Execution Error (no timeout): {type(e).__name__}: {e}"
             logger.error(f"{execution_error}\n{traceback.format_exc()}")
             return None, execution_error

    # --- Execution with signal-based timeout (Unix-like systems) ---
    output_buffer = StringIO()
    execution_error = None
    result = None
    original_handler = None # To restore previous handler

    try:
        # --- Setup Timeout ---
        logger.debug(f"Setting timeout alarm for {timeout} seconds.")
        original_handler = signal.signal(signal.SIGALRM, _timeout_handler) # Register our handler
        signal.alarm(timeout) # Set the alarm

        # --- Execute Code ---
        local_scope = {}
        safe_builtins = {
             'print': print, 'range': range, 'len': len, 'abs': abs, 'min': min,
             'max': max, 'sum': sum, 'int': int, 'float': float, 'str': str,
             'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
             'pow': pow, 'round': round, 'divmod': divmod, 'enumerate':enumerate,
             'zip':zip, 'sorted':sorted, 'map':map, 'filter':filter, 'all':all, 'any':any
        }
        global_scope = {'__builtins__': safe_builtins, 'math': __import__('math')}
        logger.debug(f"Prepared execution scope. Calling exec() and potentially solution()...")
        start_time = time.time()
        with contextlib.redirect_stdout(output_buffer):
             exec(code, global_scope, local_scope) # Define the function

             # Now try to call the function (this is where the infinite loop might happen)
             if 'solution' in local_scope and callable(local_scope['solution']):
                 logger.debug(f"Found callable 'solution' function. Calling it...")
                 result = local_scope['solution']() # <--- Potential hang point
                 logger.debug(f"'solution()' returned: {result} (type: {type(result)})")
             elif 'result' in local_scope:
                  result = local_scope['result']
                  logger.debug(f"Found 'result' variable: {result} (type: {type(result)})")
             else:
                 # Try getting result from print output
                 printed_output = output_buffer.getvalue().strip()
                 logger.debug(f"No solution()/result variable. Captured print output: '{printed_output}'")
                 if printed_output:
                     try:
                         last_line = printed_output.splitlines()[-1].strip()
                         logger.debug(f"Last printed line: '{last_line}'. Attempting conversion...")
                         if '.' in last_line or 'e' in last_line.lower():
                              parsed_result = float(last_line)
                              logger.debug(f"Parsed as float: {parsed_result}")
                         else:
                              parsed_result = int(last_line)
                              logger.debug(f"Parsed as int: {parsed_result}")
                         result = parsed_result
                     except ValueError as e:
                         logger.warning(f"Conversion of last printed line failed: {e}")
                         execution_error = "Execution finished, but final printed output was not numerical."
                     except IndexError:
                          logger.warning("Printed output was empty after stripping.")
                          execution_error = "Execution finished, but no result variable or numerical print found."
                 else:
                      logger.warning("No solution()/result variable and no print output captured.")
                      execution_error = "Execution finished, but no result could be determined."

        execution_time = time.time() - start_time
        logger.debug(f"Code execution block finished in {execution_time:.4f} seconds (before timeout check).")

        # Disable the alarm if execution finished on time
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
        # The error message is already set by the handler's raise
        return None, f"Timeout Error: Execution exceeded {timeout} seconds." # Return specific timeout message
    except SyntaxError as e: execution_error = f"Syntax Error: {e}"; logger.debug(f"{execution_error}")
    except NameError as e: execution_error = f"Name Error: {e}"; logger.debug(f"{execution_error}")
    except TypeError as e: execution_error = f"Type Error: {e}"; logger.debug(f"{execution_error}")
    except ZeroDivisionError as e: execution_error = f"ZeroDivisionError: {e}"; logger.debug(f"{execution_error}")
    except Exception as e:
        execution_error = f"Unexpected Execution Error: {type(e).__name__}: {e}"
        logger.error(f"{execution_error}\n{traceback.format_exc()}")
        return None, execution_error


    finally:
        # --- CRUCIAL: Always restore original signal handler and disable alarm ---
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0) # Ensure alarm is off
            if original_handler is not None:
                signal.signal(signal.SIGALRM, original_handler) # Restore previous handler
            logger.debug("Timeout alarm disabled and original handler restored in finally block.")

    # This part is generally only reached if an exception other than TimeoutError occurred
    logger.debug(f"Returning None due to caught exception: {execution_error}")
    return None, execution_error


def compare_results(generated_result: Any, ground_truth_str: str) -> bool:
    """Compares the generated result (from code exec) with the ground truth string. (With extra logging)"""
    func_name = "compare_results"
    logger.debug(f"Comparing Generated: '{generated_result}' (type: {type(generated_result)}) with Ground Truth: '{ground_truth_str}' (type: {type(ground_truth_str)})")
    if generated_result is None:
        logger.debug("Comparison returning False because generated_result is None.")
        return False

    try:
        # Attempt more robust float to int conversion before string comparison
        if isinstance(generated_result, float):
             # Use a small tolerance for checking if it's basically an integer
             if abs(generated_result - round(generated_result)) < 1e-9:
                  generated_str = str(int(round(generated_result)))
                  logger.debug(f"Converted float to int string: '{generated_str}'")
             else:
                  # Standard float conversion, maybe limit precision if needed later
                  generated_str = str(generated_result)
                  logger.debug(f"Converted float to string: '{generated_str}'")
        else:
             generated_str = str(generated_result) # Convert other types directly
        generated_str = generated_str.strip()
        logger.debug(f"Final generated string for comparison: '{generated_str}'")
    except Exception as e:
        logger.warning(f"Conversion of generated result '{generated_result}' to string failed: {e}")
        return False

    ground_truth_str_cleaned = str(ground_truth_str).strip()
    logger.debug(f"Cleaned ground truth string: '{ground_truth_str_cleaned}'")

    logger.debug(f"Attempting direct string comparison...")
    if generated_str == ground_truth_str_cleaned:
        logger.debug("Direct string comparison PASSED.")
        return True
    # Case-insensitive comparison (e.g., for True/False)
    if generated_str.lower() == ground_truth_str_cleaned.lower():
         logger.debug("Case-insensitive string comparison PASSED.")
         return True
    logger.debug("String comparisons failed.")

    # Numerical comparison as fallback
    try:
        logger.debug(f"Attempting numerical comparison...")
        # Clean strings for numerical conversion (remove commas, spaces, etc.)
        gen_cleaned_num_str = re.sub(r'[,\s$]', '', generated_str) # Allow $? Maybe not for math.
        gt_cleaned_num_str = re.sub(r'[,\s$]', '', ground_truth_str_cleaned)
        logger.debug(f"Cleaned numerical strings - Gen: '{gen_cleaned_num_str}', GT: '{gt_cleaned_num_str}'")

        # Pattern to check if they look like numbers
        num_pattern = r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$"
        is_gen_num = re.match(num_pattern, gen_cleaned_num_str)
        is_gt_num = re.match(num_pattern, gt_cleaned_num_str)
        logger.debug(f"Look like numbers? Gen: {bool(is_gen_num)}, GT: {bool(is_gt_num)}")

        if is_gen_num and is_gt_num:
            gen_num = float(gen_cleaned_num_str)
            gt_num = float(gt_cleaned_num_str)
            logger.debug(f"Converted to floats - Gen: {gen_num}, GT: {gt_num}")

            # Compare with tolerance
            tolerance = 1e-6
            logger.debug(f"Comparing floats with tolerance {tolerance}...")
            if abs(gen_num - gt_num) < tolerance:
                logger.debug(f"Numerical comparison PASSED (tolerance check).")
                return True
            logger.debug(f"Tolerance check failed (|{gen_num} - {gt_num}| = {abs(gen_num - gt_num)}).")

            # Check if both are effectively integers and compare ints
            gen_is_int_check = abs(gen_num - round(gen_num)) < tolerance
            gt_is_int_check = abs(gt_num - round(gt_num)) < tolerance
            logger.debug(f"Are they integers (within tolerance)? Gen: {gen_is_int_check}, GT: {gt_is_int_check}")
            if gen_is_int_check and gt_is_int_check:
                 logger.debug(f"Comparing as integers (rounded)...")
                 if int(round(gen_num)) == int(round(gt_num)):
                      logger.debug(f"Numerical comparison PASSED (integer check).")
                      return True
                 logger.debug(f"Integer check failed.")

    except (ValueError, TypeError) as e:
        # Log if conversion to float fails
        logger.debug(f"Numerical comparison failed during conversion/check: {e}")
        pass # Fall through if conversion fails

    logger.debug(f"All comparison methods failed.")
    return False


# --- Main Re-evaluation Logic ---
if __name__ == "__main__":
    logger.info(f"Starting re-evaluation using log file: {INPUT_JSON_PATH}")
    logger.info(f"Logging level set to DEBUG.")

    if not os.path.exists(INPUT_JSON_PATH):
        logger.error(f"Input JSON file not found: {INPUT_JSON_PATH}")
        exit(1)
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            original_results = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {INPUT_JSON_PATH}: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Error reading file {INPUT_JSON_PATH}: {e}")
        exit(1)

    if TARGET_MODEL_KEY not in original_results:
        logger.error(f"Target model key '{TARGET_MODEL_KEY}' not found in the JSON file.")
        logger.info(f"Available keys: {list(original_results.keys())}")
        exit(1)

    model_data = original_results[TARGET_MODEL_KEY]
    if "details" not in model_data or not isinstance(model_data["details"], list):
        logger.error(f"Key 'details' not found or is not a list in the data for '{TARGET_MODEL_KEY}'.")
        exit(1)

    original_details = model_data["details"]
    total_items = len(original_details)
    logger.info(f"Found {total_items} items in the 'details' log for model '{TARGET_MODEL_KEY}'.")

    # --- Re-evaluation counters ---
    new_correct_pot = 0
    new_pot_extraction_failures = 0
    new_pot_execution_failures = 0 # Includes logic errors/None results/timeouts
    new_processing_errors = 0
    # --- MODIFIED: Separate lists for details ---
    success_details = []
    failure_details = []
    # --- END MODIFIED ---

    logger.info("Starting loop through items...")
    processed_count = 0

    for item_log in original_details:
        item_index = item_log.get("item_index", "N/A")
        logger.info(f"--- Processing item {item_index}/{total_items} START ---")

        question = item_log.get("question")
        ground_truth = item_log.get("ground_truth")
        generated_text = item_log.get("generated_full_text")

        # Initialize re-evaluation status vars
        extracted_code_new = None
        executed_result_new = None
        is_correct_new = False
        evaluation_method_new = "N/A"
        error_message_new = None
        current_item_failed = True # Assume failure unless proven otherwise

        if question is None or ground_truth is None or generated_text is None:
            logger.warning(f"Item {item_index}: Skipping due to missing essential data.")
            new_processing_errors += 1
            evaluation_method_new = "Skipped - Missing Data"
            error_message_new = "Essential data (question, ground_truth, or generated_full_text) missing in log."
            # Keep current_item_failed = True
        else:
            # Proceed with re-evaluation only if essential data is present
            logger.debug(f"Item {item_index}: Question: {question[:100]}...")
            logger.debug(f"Item {item_index}: Ground Truth: {ground_truth}")
            log_gen_text_snippet = generated_text[:150].replace('\n', '\\n')
            logger.debug(f"Item {item_index}: Generated Text (len: {len(generated_text)}): {log_gen_text_snippet}...")

            try:
                logger.info(f"Item {item_index}: Calling extract_python_code...")
                extracted_code_new = extract_python_code(generated_text)

                if extracted_code_new:
                    logger.info(f"Item {item_index}: Code extracted successfully.")
                    log_extracted_code_snippet = extracted_code_new[:150].replace('\n', '\\n')
                    logger.debug(f"Item {item_index}: Extracted Code (first 150):\n{log_extracted_code_snippet}\n---")

                    logger.info(f"Item {item_index}: Calling safe_execute_code...")
                    executed_result_new, exec_error = safe_execute_code(extracted_code_new)

                    if exec_error:
                        logger.warning(f"Item {item_index}: Code execution failed or yielded error: {exec_error}")
                        error_message_new = f"PoT (Re-eval): Execution Failed - {exec_error}"
                        evaluation_method_new = "PoT Execution Failed (Re-eval)"
                        new_pot_execution_failures += 1
                        # Keep current_item_failed = True
                    elif executed_result_new is not None:
                        logger.info(f"Item {item_index}: Code execution successful. Result: {executed_result_new}")
                        logger.info(f"Item {item_index}: Calling compare_results...")
                        is_correct_new = compare_results(executed_result_new, ground_truth)
                        logger.info(f"Item {item_index}: Comparison result: {is_correct_new}")
                        if is_correct_new:
                            evaluation_method_new = "PoT Correct (Re-eval)"
                            new_correct_pot += 1
                            error_message_new = None
                            current_item_failed = False # <<< Mark as success
                        else:
                            evaluation_method_new = "PoT Incorrect (Re-eval)"
                            error_message_new = f"PoT (Re-eval): Result mismatch (Got: {executed_result_new}, Expected: {ground_truth})"
                            new_pot_execution_failures += 1
                            # Keep current_item_failed = True
                    else: # Execution finished ok, but result was None
                        logger.warning(f"Item {item_index}: Code execution returned None result.")
                        error_message_new = "PoT (Re-eval): Execution returned None"
                        evaluation_method_new = "PoT Execution Returned None (Re-eval)"
                        new_pot_execution_failures += 1
                        # Keep current_item_failed = True
                else: # Failed to extract code
                    logger.warning(f"Item {item_index}: Failed to extract code.")
                    error_message_new = "PoT (Re-eval): Failed to extract code"
                    evaluation_method_new = "PoT Extraction Failed (Re-eval)"
                    new_pot_extraction_failures += 1
                    # Keep current_item_failed = True
            except Exception as reeval_e:
                logger.error(f"Item {item_index}: UNEXPECTED error during re-evaluation: {reeval_e}")
                logger.error(traceback.format_exc())
                error_message_new = f"Re-evaluation Error: {str(reeval_e)}"
                evaluation_method_new = "Re-evaluation Error"
                new_processing_errors += 1
                # Keep current_item_failed = True

        # --- Store results in appropriate list ---
        logger.debug(f"Item {item_index}: Final status - Failed: {current_item_failed}")
        updated_item = {
            **item_log, # Include original logged data
            "extracted_code_reeval": extracted_code_new,
            "executed_result_pot_reeval": str(executed_result_new) if executed_result_new is not None else None,
            "is_correct_reeval": is_correct_new,
            "evaluation_method_reeval": evaluation_method_new,
            "error_message_reeval": error_message_new,
        }

        if current_item_failed:
            failure_details.append(updated_item)
        else:
            success_details.append(updated_item)
        # --- END Store results ---

        processed_count += 1
        logger.info(f"--- Processing item {item_index}/{total_items} END ---")

        # Optional intermediate logging
        if processed_count % 100 == 0:
            logger.info(f"--- Progress: Processed {processed_count}/{total_items} items ---")
            current_acc = (new_correct_pot / processed_count) * 100 if processed_count > 0 else 0
            logger.info(f"--- Intermediate Accuracy (PoT): {current_acc:.2f}% ---")

    if processed_count == total_items:
        logger.info(f"--- Loop finished successfully after processing {processed_count} items ---")
    else:
        logger.warning(f"--- Loop ended prematurely! Processed only {processed_count}/{total_items} items ---")

    logger.info("--- Re-evaluation Complete ---")

    # --- Calculate final summary statistics ---
    # Use total_items for percentages to reflect the full dataset
    new_pot_accuracy = (new_correct_pot / total_items) * 100 if total_items > 0 else 0
    new_pot_extraction_failure_rate = (new_pot_extraction_failures / total_items) * 100 if total_items > 0 else 0
    new_pot_execution_failure_rate = (new_pot_execution_failures / total_items) * 100 if total_items > 0 else 0
    new_processing_error_rate = (new_processing_errors / total_items) * 100 if total_items > 0 else 0

    # --- Prepare final summary output structure ---
    final_summary = {
        "re_evaluation_summary": {
            "source_file": INPUT_JSON_PATH,
            "target_model_key": TARGET_MODEL_KEY,
            "total_items": total_items,
            "correct_pot_reeval": new_correct_pot,
            "accuracy_pot_reeval_percent": round(new_pot_accuracy, 2),
            "pot_extraction_failures_reeval": new_pot_extraction_failures,
            "pot_extraction_failure_reeval_percent": round(new_pot_extraction_failure_rate, 2),
            "pot_execution_failures_reeval": new_pot_execution_failures, # Combined exec/logic/timeout/none errors
            "pot_execution_failure_reeval_percent": round(new_pot_execution_failure_rate, 2),
            "processing_errors_reeval": new_processing_errors, # Errors in the script/data itself
            "processing_error_reeval_percent": round(new_processing_error_rate, 2),
            "total_success_details_count": len(success_details),
            "total_failure_details_count": len(failure_details),
        }
    }

    # --- Log final summary ---
    logger.info(f"--- Final Re-evaluation Summary ---")
    logger.info(f"Total Items: {total_items}")
    logger.info(f"Correct (Re-eval): {new_correct_pot} ({new_pot_accuracy:.2f}%)")
    logger.info(f"Extraction Failures (Re-eval): {new_pot_extraction_failures} ({new_pot_extraction_failure_rate:.2f}%)")
    logger.info(f"Execution/Logic/Timeout/None Failures (Re-eval): {new_pot_execution_failures} ({new_pot_execution_failure_rate:.2f}%)")
    logger.info(f"Processing Errors (Re-eval): {new_processing_errors} ({new_processing_error_rate:.2f}%)")
    logger.info(f"Success details count: {len(success_details)}")
    logger.info(f"Failure details count: {len(failure_details)}")


    # --- Save the new results into THREE files ---
    try:
        # Save Summary
        with open(OUTPUT_SUMMARY_PATH, 'w', encoding='utf-8') as f_summary:
            json.dump(final_summary, f_summary, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved re-evaluated summary to: {OUTPUT_SUMMARY_PATH}")

        # Save Success Details
        with open(OUTPUT_SUCCESS_DETAILS_PATH, 'w', encoding='utf-8') as f_success:
            json.dump(success_details, f_success, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved success details to: {OUTPUT_SUCCESS_DETAILS_PATH}")

        # Save Failure Details
        with open(OUTPUT_FAILURE_DETAILS_PATH, 'w', encoding='utf-8') as f_failure:
            json.dump(failure_details, f_failure, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved failure details to: {OUTPUT_FAILURE_DETAILS_PATH}")

    except Exception as e:
        logger.error(f"Error saving re-evaluated results: {e}")