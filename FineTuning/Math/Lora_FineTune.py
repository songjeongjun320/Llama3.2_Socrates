# Standard library imports
import json
import logging
import os
import gc
from datasets import Dataset, concatenate_datasets # Added concatenate_datasets
from typing import List, Dict

# Third-party imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments # Use TrainingArguments directly with SFTTrainer now
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer # SFTTrainer handles formatting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pot_data_from_directory(directory_path: str) -> Dict[str, List[str]]:
    """Load all JSON files from a directory containing PoT data."""
    logger.info(f"Loading PoT data from directory: {directory_path}")
    all_data = {"question": [], "answer_cot": []} # Use 'answer_cot' key explicitly

    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return all_data

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            logger.info(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = 0
                        for item in data:
                            # Check for the correct keys for PoT data
                            if isinstance(item, dict) and "question" in item and "answer_cot" in item:
                                question = str(item["question"])
                                # Make sure 'answer_cot' holds the code string
                                answer_code = str(item["answer_cot"])
                                all_data["question"].append(question)
                                all_data["answer_cot"].append(answer_code)
                                count += 1
                            else:
                                logger.warning(f"Skipping item in {filename} due to missing 'question' or 'answer_cot' keys: {item}")
                        logger.info(f"Loaded {count} items from {filename}")
                    else:
                        logger.warning(f"File {filename} does not contain a list.")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {file_path}: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")

    logger.info(f"Total processed data: questions={len(all_data['question'])}, answers(code)={len(all_data['answer_cot'])}")
    if all_data["question"]:
        logger.info(f"First question: {all_data['question'][0][:150]}")
        logger.info(f"First answer (code): {all_data['answer_cot'][0][:150]}")
    return all_data

# --- SFTTrainer Formatting ---
# SFTTrainer can handle the prompt formatting directly.
# We define a function that returns the formatted string.
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        # Define the prompt structure for PoT
        # Ensure the model knows to generate code after "Answer:"
        text = f"Question: {example['question'][i]}\nAnswer:\n```python\n{example['answer_cot'][i]}\n```"
        output_texts.append(text)
    return output_texts

if __name__ == "__main__":
    # --- Updated Paths and Constants ---
    DATA_DIR = "/scratch/jsong132/Technical_Llama3.2/DB/PoT/train" # Directory containing JSON files
    BASE_MODEL = "/scratch/jsong132/Technical_Llama3.2/llama3.2_3b"
    OUTPUT_DIR = "Tune_Results/llama3.2_Socrates_Math_v4" # Updated output directory

    # --- Load and Prepare Data ---
    all_data = load_pot_data_from_directory(DATA_DIR)
    if not all_data["question"] or not all_data["answer_cot"]:
        logger.error("No data loaded. Check JSON files and structure in the directory.")
        exit(1)
    assert len(all_data["question"]) == len(all_data["answer_cot"]), "Question/answer_cot counts mismatch."

    # --- Convert to Dataset and Split ---
    dataset = Dataset.from_dict(all_data)
    # Add EOS token if tokenizer does not add it automatically during formatting
    # dataset = dataset.map(lambda example: {'text': example['text'] + tokenizer.eos_token})
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset['test'])} samples")

    # --- Load Tokenizer and Model ---
    logger.info(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set EOS token as PAD token.")
    # Set chat template if needed, or rely on manual formatting
    # tokenizer.chat_template = ... # Optional: Define if using specific chat format

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency on compatible GPUs
        device_map="auto", # Automatically distribute model across available GPUs
        trust_remote_code=True
    )

    # --- Apply LoRA ---
    # Ensure target modules are correct for Llama 3.2 (common ones listed)
    peft_params = LoraConfig(
        lora_alpha=16, # Often set to 2*r
        lora_dropout=0.1,
        r=64,          # Rank, higher might capture more complex patterns but uses more memory
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Common Llama target modules
    )
    # No need for prepare_model_for_kbit_training if not using k-bit quantization (like 8-bit or 4-bit)
    # model = prepare_model_for_kbit_training(model) # Remove or comment out if using bfloat16/float16
    model = get_peft_model(model, peft_params)
    model.config.use_cache = False # Important for training stability with LoRA/gradient checkpointing
    model.print_trainable_parameters()

    # Verify trainable parameters
    trainable_params = [(name, param.shape) for name, param in model.named_parameters() if param.requires_grad]
    logger.info(f"Trainable parameters ({len(trainable_params)}):")
    for name, shape in trainable_params:
        logger.info(f"  - {name}: {shape}")
    if not trainable_params:
        logger.error("No trainable parameters found. Check LoRA config and target_modules.")
        exit(1)

    # --- Training Arguments ---
    # Use TrainingArguments, SFTTrainer wraps it
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5, # Adjust epochs based on dataset size and convergence
        per_device_train_batch_size=8, # Adjust based on GPU memory
        per_device_eval_batch_size=8,  # Adjust based on GPU memory
        gradient_accumulation_steps=2, # Effective batch size = 4 * 4 = 16
        optim="adamw_torch",          # Recommended optimizer
        save_strategy="steps",
        save_steps=500,               # Save checkpoints periodically
        save_total_limit=2,           # Keep only the last 2 checkpoints
        logging_steps=50,            # Log training progress more frequently
        learning_rate=2e-5,           # Common starting point for LoRA fine-tuning
        weight_decay=0.01,
        fp16=False,                   # Set to False if using bf16
        bf16=True,                    # Use bfloat16 if available (Ampere GPUs or newer)
        max_grad_norm=0.3,            # Gradient clipping
        warmup_ratio=0.03,            # Warmup steps
        lr_scheduler_type="cosine",   # Learning rate scheduler
        evaluation_strategy="steps",  # Evaluate during training
        eval_steps=250,               # Evaluate every 200 steps
        # group_by_length=True,       # Speeds up training by batching similar length sequences (Optional)
        report_to="none",             # Disable external reporting (like wandb) if not needed
        gradient_checkpointing=False,  # Enable gradient checkpointing to save memory (set model.config.use_cache=False)
        load_best_model_at_end=True,  # Load the best model based on eval loss at the end
        metric_for_best_model="eval_loss",
    )

    # --- Initialize SFTTrainer ---
    # SFTTrainer handles tokenization and formatting internally if formatting_func is provided
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_params, # Pass PEFT config here
        formatting_func=formatting_prompts_func,
    )

    # --- Pre-training Optimizations ---
    torch.cuda.empty_cache()
    gc.collect()
    # Enable TF32 for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model.train() # Ensure model is in training mode

    # --- Train ---
    logger.info("Starting fine-tuning...")
    train_result = trainer.train(resume_from_checkpoint=False) # Set to True or path to resume

    # --- Save Final Model ---
    logger.info("Saving final model...")
    # SFTTrainer automatically saves the best model if load_best_model_at_end=True
    # If you want to save the final state regardless:
    final_save_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(final_save_path) # Saves adapter model and tokenizer
    logger.info(f"Final model adapters saved to: {final_save_path}")

    # Optional: If you need to merge LoRA weights with the base model
    # logger.info("Merging LoRA adapters...")
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     BASE_MODEL,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     trust_remote_code=True
    # )
    # merged_model = PeftModel.from_pretrained(base_model, final_save_path) # Load PEFT model
    # merged_model = merged_model.merge_and_unload() # Merge adapters
    # merged_model.save_pretrained(os.path.join(OUTPUT_DIR, "final_merged_model"))
    # tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_merged_model"))
    # logger.info(f"Merged model saved to: {os.path.join(OUTPUT_DIR, 'final_merged_model')}")


    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Fine-tuning completed successfully!")