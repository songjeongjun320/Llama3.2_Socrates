# Standard library imports
import json
import logging
import os
import re # Keep re in case needed elsewhere, though less likely now
import gc
import time
import traceback
# Removed signal as safe_execute is removed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal, Tuple # Added Tuple
# Removed io and contextlib as safe_execute is removed

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Configure logging (Set to DEBUG for detailed eval logs)
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
        name="Llama-3.2-3b-Socrates-Math-v2", # Example specific name
        model_path="/scratch/jsong132/Technical_Llama3.2/FineTuning/Math/Tune_Results/llama3.2_Socrates_Math_v2/final_checkpoint",
        is_local=True,
        is_adapter_model=True,
        base_model_path_for_adapter=BASE_LLAMA_PATH
    ),
    # ModelConfig(
    #     name="Llama-3.2-3b-Socrates-Math-AllTarget-Adapter-Final", # Example specific name
    #     model_path="/scratch/jsong132/Technical_Llama3.2/FineTuning/Math/Tune_Results/llama3.2_Socrates_Math_all_target/final_checkpoint",
    #     is_local=True,
    #     is_adapter_model=True,
    #     base_model_path_for_adapter=BASE_LLAMA_PATH
    # ),
    # Add other models if needed
]

# --- Evaluation Configuration ---
TEST_DATA_DIR = "/scratch/jsong132/Technical_Llama3.2/DB/PoT/test"
RESULTS_FILE = "generation_math_logs.json" # Renamed file for clarity
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATION_MAX_NEW_TOKENS = 1024
GENERATION_TEMPERATURE = 0.1 # Set to 0 for deterministic if needed, >0 for sampling
GENERATION_TOP_P = 0.9
# TIMEOUT_SECONDS removed
EVALUATION_BATCH_SIZE = 8 # Adjust based on GPU VRAM

# --- Helper Functions ---

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
                             # Keep loading structure, ground truth is still useful context in results file
                             if isinstance(item, dict) and "question" in item and "answer_value" in item and item["answer_value"] is not None:
                                all_test_items.append({
                                    "question": str(item["question"]),
                                    "answer_value": str(item["answer_value"]) # Keep GT for context
                                })
                                count += 1
                             else:
                                logger.warning(f"Skipping item in {filename} due to missing keys or null answer_value: {item}")
                        logger.debug(f"Loaded {count} valid items from {filename}")
                    else:
                        logger.warning(f"File {filename} does not contain a list.")
            except Exception as e:
                logger.error(f"Error loading test file {file_path}: {str(e)}")

    logger.info(f"Total valid test items loaded: {len(all_test_items)}")
    return all_test_items

def format_prompt(question: str) -> str:
    """Formats the question into the prompt structure used during training."""
    # Ensure this matches the EXACT prompt format used for fine-tuning
    return f"Question: {question}\nAnswer:\n```python\n" # Keep format consistent with training

# --- REMOVED DETAILED EVALUATION FUNCTIONS ---
# extract_python_code, safe_execute_code, compare_results are removed.

# --- Main Evaluation Loop (Simplified for Generation Logging) ---
evaluation_summary = {}
test_data = load_pot_test_data(TEST_DATA_DIR)

if not test_data:
    logger.error("No valid test data loaded. Exiting.")
    exit(1)

for config in MODEL_CONFIGS:
    logger.info(f"\n--- Evaluating Model: {config.name} ({config.model_path}) ---")
    model = None
    tokenizer = None
    # Initialize simplified results structure
    results = {
        "successful_generations": 0,
        "pipeline_errors": 0, # Errors in batching, generation, decoding, etc.
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
        with tqdm(total=len(test_data), desc=f"Generating for {config.name}") as pbar:
            for i in range(0, len(test_data), EVALUATION_BATCH_SIZE):
                # 1. Prepare Batch
                batch_indices = range(i, min(i + EVALUATION_BATCH_SIZE, len(test_data)))
                batch_items = [test_data[j] for j in batch_indices]
                batch_questions = [item["question"] for item in batch_items]
                batch_prompts = [format_prompt(q) for q in batch_questions]
                batch_ground_truths = [item["answer_value"] for item in batch_items] # Keep for context

                try:
                    # 2. Batch Tokenization
                    inputs = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024 # Adjust if needed
                    ).to(DEVICE)
                    input_length = inputs['input_ids'].shape[1]

                    # 3. Batch Generation
                    gen_start_time = time.time()
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                            temperature=GENERATION_TEMPERATURE if GENERATION_TEMPERATURE > 0 else 1.0,
                            top_p=GENERATION_TOP_P if GENERATION_TEMPERATURE > 0 else 1.0,
                            do_sample=True if GENERATION_TEMPERATURE > 0 else False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    gen_end_time = time.time()
                    logger.debug(f"Batch {i // EVALUATION_BATCH_SIZE + 1}: Generation took {gen_end_time - gen_start_time:.2f}s")

                    # 4. Batch Decoding
                    decode_start_time = time.time()
                    if outputs.shape[1] > input_length:
                        generated_tokens = outputs[:, input_length:]
                        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    else:
                         generated_texts = [""] * len(batch_items)
                         logger.warning(f"Batch starting index {i}: No new tokens generated for {len(batch_items)} items (input length {input_length}, output length {outputs.shape[1]}).")
                    decode_end_time = time.time()
                    logger.debug(f"Batch {i // EVALUATION_BATCH_SIZE + 1}: Decoding took {decode_end_time - decode_start_time:.2f}s")

                    # 5. Process and Log Each Item in the Batch
                    log_start_time = time.time()
                    for j, original_item_index in enumerate(batch_indices):
                        item_log_prefix = f"Item {original_item_index}:"
                        logger.debug(f"--- {item_log_prefix} Logging START ---") # Changed log message

                        generated_text = generated_texts[j]
                        question = batch_questions[j]
                        ground_truth = batch_ground_truths[j] # Keep for context

                        log_gen_text_snippet = generated_text[:150].replace('\n', '\\n')
                        logger.info(f"{item_log_prefix} Generated Text (len: {len(generated_text)}): {log_gen_text_snippet}...")

                        # Simply store the generated text and context
                        logger.debug(f"{item_log_prefix} Storing generated text...")
                        results["details"].append({
                            "item_index": original_item_index,
                            "question": question,
                            "ground_truth": ground_truth, # Keep ground truth for context
                            "generated_full_text": generated_text,
                            "status": "Generated" # Mark as successfully generated
                        })
                        results["successful_generations"] += 1
                        pbar.update(1)
                        logger.debug(f"--- {item_log_prefix} Logging END ---") # Changed log message
                    log_end_time = time.time()
                    logger.debug(f"Batch {i // EVALUATION_BATCH_SIZE + 1}: Logging took {log_end_time - log_start_time:.2f}s")

                except Exception as batch_e:
                    # Critical error during batch tokenization, generation, or decoding
                    logger.error(f"CRITICAL Error during processing BATCH starting at index {i} for {config.name}: {batch_e}")
                    logger.error(traceback.format_exc())
                    # Mark all items in this batch as errors
                    for k, failed_item_index in enumerate(batch_indices):
                        failed_question = batch_questions[k] if k < len(batch_questions) else "N/A - Batch Error"
                        failed_gt = batch_ground_truths[k] if k < len(batch_ground_truths) else "N/A - Batch Error"
                        results["details"].append({
                            "item_index": failed_item_index,
                            "question": failed_question,
                            "ground_truth": failed_gt,
                            "generated_full_text": "N/A - Batch Error",
                            "status": f"Critical Batch Error: {str(batch_e)}" # Update status
                        })
                        results["pipeline_errors"] += 1
                    pbar.update(len(batch_indices)) # Update progress bar for the failed batch

        # --- Calculate Final Results (Simplified) ---
        results["total"] = len(test_data)
        generation_success_rate = (results["successful_generations"] / results["total"]) * 100 if results["total"] > 0 else 0
        pipeline_error_rate = (results["pipeline_errors"] / results["total"]) * 100 if results["total"] > 0 else 0

        logger.info(f"--- Results for {config.name} ---")
        logger.info(f"Total Items: {results['total']}")
        logger.info(f"Successful Generations: {results['successful_generations']}")
        logger.info(f"Pipeline Processing Errors (Batch/Item): {results['pipeline_errors']}")
        logger.info(f"Generation Success Rate: {generation_success_rate:.2f}%")
        logger.info(f"Pipeline Error Rate: {pipeline_error_rate:.2f}%")

        evaluation_summary[config.name] = {
            "model_path": config.model_path,
            "base_model_path": config.base_model_path_for_adapter if config.is_adapter_model else config.model_path,
            "is_adapter": config.is_adapter_model,
            "batch_size": EVALUATION_BATCH_SIZE,
            "generation_temperature": GENERATION_TEMPERATURE,
            "generation_top_p": GENERATION_TOP_P,
            "total_items": results["total"],
            "successful_generations": results["successful_generations"],
            "pipeline_errors": results["pipeline_errors"],
            "generation_success_rate_percent": round(generation_success_rate, 2),
            "pipeline_error_rate_percent": round(pipeline_error_rate, 2),
            "details": results["details"] # Keep detailed generated texts
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
        time.sleep(5)

# --- Save and Print Summary ---
try:
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, ensure_ascii=False, indent=4, default=str)
    logger.info(f"\nSaved generation log results to {RESULTS_FILE}")
except Exception as e:
    logger.error(f"Error saving results to {RESULTS_FILE}: {e}")

logger.info("\n--- Final Generation Summary ---")
for name, summary in evaluation_summary.items():
    if "error" in summary:
        logger.info(f"{name}: OVERALL ERROR ({summary['error']})")
    else:
        logger.info(f"{name} (Batch: {summary.get('batch_size', 'N/A')}, Temp: {summary.get('generation_temperature', 'N/A')}, TopP: {summary.get('generation_top_p', 'N/A')}):")
        logger.info(f"  Generation Success Rate: {summary['generation_success_rate_percent']:.2f}% ({summary['successful_generations']}/{summary['total_items']})")
        logger.info(f"  Pipeline Errors: {summary['pipeline_errors']} ({summary['pipeline_error_rate_percent']:.2f}%)")


logger.info("\n--- Evaluation Complete ---")