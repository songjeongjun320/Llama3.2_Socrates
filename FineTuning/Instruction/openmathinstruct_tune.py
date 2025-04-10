import json
import logging
import os
import gc
import math
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BASEM_LOC = "/scratch/syeom3/Technical_Llama3.2/llama3.2_3b" 
USED_DATA = "nvidia/OpenMathInstruct-1"
OUT_LOC = "Tune_Results/llama3.2_FFN_Instruction"
ETC_LOC = "./hf_cache"

# LoRA Configuration - Targeting FFN layers only
LORA_AL = 16
LORA_DROP = 0.1
LORA_R = 64
LORA_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]

# Training Arguments Configuration
EPOCHS = 1 
TRAIN_SIZE = 8 
EVAL_SIZE = 8 
GRADIENTSTEP = 8 # Effective batch size = batch_size * num_gpus * grad_accum
LR = 0.00002
OPTIMIZER = "adamw_torch"
SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3
BF16_ENABLED = True 
FP16_ENABLED = False 
MAX_SEQ_LENGTH = 1024 
LOGGING_STEPS = 25
SAVE_STEPS = 250 
EVAL_STEPS = 250 
SAVE_TOTAL_LIMIT = 2

def load_with_convert_dat(dataset_name: str, cache_dir: str = None):
    """Load and prepare the OpenMathInstruct-1 dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        logger.info(f"Successfully loaded dataset '{dataset_name}'")

        # OpenMathInstruct-1 has only a 'train' split.
        if "train" not in dataset:
            logger.error(f"Dataset {dataset_name} does not contain a 'train' split.")
            raise ValueError("Dataset requires a 'train' split.")

        logger.info("Splitting 'train' dataset into train and validation sets (98% train, 2% validation)")
        split_dataset = dataset["train"].train_test_split(test_size=0.02, shuffle=True, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(eval_dataset)}")

        # Log a sample to verify structure
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info("Sample Data:")
            logger.info(f"  Instruction: {sample['question'][:150]}...")
            logger.info(f"  Output: {sample['generated_solution'][:150]}...")

        return train_dataset, eval_dataset

    except Exception as e:
        logger.error(f"Error loading or preparing dataset {dataset_name}: {str(e)}")
        raise

# Formatting function for SFTTrainer
def prompt_SFT(example):
    """Formats the OpenMathInstruct examples for instruction tuning."""
    instruction_col_name = 'question'
    response_col_name = 'generated_solution'
    output_texts = []

    if instruction_col_name not in example or response_col_name not in example:
        logger.error(f"Formatting function missing expected columns: {instruction_col_name} or {response_col_name}")
        return []

    instructions = example[instruction_col_name]
    responses = example[response_col_name]

    for i in range(len(instructions)):
        instruction = instructions[i]
        response = responses[i] if responses[i] is not None else ""
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{tokenizer.eos_token}"
        output_texts.append(text)
    return output_texts

# --- Main Execution ---
if __name__ == "__main__":
    # --- Create output directory ---
    os.makedirs(OUT_LOC, exist_ok=True)

    # --- Load Dataset ---
    xtrain, xeval = load_with_convert_dat(USED_DATA, ETC_LOC)

    # --- Load Tokenizer ---
    logger.info(f"Loading tokenizer: {BASEM_LOC}")
    tokenizer = AutoTokenizer.from_pretrained(BASEM_LOC, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load Model ---

    model = AutoModelForCausalLM.from_pretrained(
        BASEM_LOC,
        torch_dtype=torch.bfloat16 if BF16_ENABLED else torch.float32,
        device_map="auto",  
        trust_remote_code=True,
    )

    # Optional: Prepare model for k-bit training if using quantization
    # if bnb_config:
    #     logger.info("Preparing model for k-bit training.")
    #     model = prepare_model_for_kbit_training(model)

    # --- Configure PEFT (LoRA) ---
    logger.info("Configuring LoRA for FFN layers...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_AL,
        lora_dropout=LORA_DROP,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    logger.info("Applied LoRA to the Instruction tuned model.")
    model.print_trainable_parameters()

    trainable_params = []
    all_param = 0
    trainable_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_param += param.numel()
            trainable_params.append((name, param.shape))
            is_ffn = any(target_module in name for target_module in LORA_TARGET_MODULES)
            if not is_ffn:
                 logger.warning(f"WARNING: Trainable parameter '{name}' might not be an intended FFN layer.")

    logger.info(f"Trainable parameters ({len(trainable_params)} tensors):")
    logger.info(
        f"Total params: {all_param:,} || Trainable params: {trainable_param:,} || Trainable %: {100 * trainable_param / all_param:.4f}"
    )
    if not trainable_params:
        logger.error("FATAL: No trainable parameters found. Check LoRA config and target_modules.")
        exit(1)

    model.config.use_cache = False

    print(f"OUTPUT_DIR: {OUT_LOC}")
    print(f"NUM_TRAIN_EPOCHS: {EPOCHS}")
    print(f"SAVE_STEPS : {SAVE_STEPS}")
    print(f"EVAL_STEPS : {EVAL_STEPS}")
    # --- Configure Training Arguments ---
    logger.info("Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUT_LOC,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_SIZE,
        per_device_eval_batch_size=EVAL_SIZE,
        gradient_accumulation_steps=GRADIENTSTEP,
        gradient_checkpointing=True, # to reduce GPU usage
        optim=OPTIMIZER,
        learning_rate=LR,
        lr_scheduler_type=SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_steps=EVAL_STEPS,
        bf16=BF16_ENABLED,
        fp16=FP16_ENABLED,

        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=True # Important for SFTTrainer
    )

    # --- Initialize SFTTrainer ---
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=xtrain,
        eval_dataset=xeval,
        peft_config=peft_config,
        # tokenizer=tokenizer,
        formatting_func=prompt_SFT,
        # max_seq_length=MAX_SEQ_LENGTH, # Set maximum sequence length
    )

    # --- Pre-training Optimizations ---
    torch.cuda.empty_cache()
    gc.collect()
    if BF16_ENABLED and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
         logger.info("Enabling TF32 for matmul and cuDNN.")
         torch.backends.cuda.matmul.allow_tf32 = True
         torch.backends.cudnn.allow_tf32 = True

    # --- Train ---
    logger.info("Starting fine-tuning...")
    train_result = trainer.train()
    logger.info("Fine-tuning finished.")

    # --- Save Final Model (LoRA adapters) ---
    logger.info("Saving final LoRA adapters...")
    final_save_path = os.path.join(OUT_LOC, "final_checkpoint")
    trainer.save_model(final_save_path)
    logger.info(f"Final LoRA adapters saved to: {final_save_path}")
    # Optionally save the tokenizer as well
    tokenizer.save_pretrained(final_save_path)
    logger.info(f"Tokenizer saved to: {final_save_path}")

    # --- Log and Save Metrics ---
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # Save trainer state

    # --- Evaluate on Validation Set (using SFTTrainer's evaluate) ---
    logger.info("Running evaluation on the validation set...")
    eval_metrics = trainer.evaluate(eval_dataset=xeval)

    # Calculate perplexity
    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
        eval_metrics["perplexity"] = perplexity
        logger.info(f"Validation Perplexity: {perplexity}")
    except KeyError:
        logger.warning("Could not calculate perplexity: 'eval_loss' not found in metrics.")
    except OverflowError:
         perplexity = float("inf")
         eval_metrics["perplexity"] = perplexity
         logger.warning(f"Could not calculate perplexity due to OverflowError (eval_loss likely too high): {eval_metrics.get('eval_loss', 'N/A')}")


    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # --- Final Notes ---
    logger.info("Training complete!")
    logger.info(f"LoRA adapters saved in: {final_save_path}")
    logger.info("To use the fine-tuned model for inference:")
    logger.info("1. Load the base model (Llama 3.2 3B).")
    logger.info(f"2. Load the LoRA adapters from '{final_save_path}'.")
    logger.info("3. Merge the adapters with the base model OR use PeftModel directly for inference.")
    logger.info("Example loading for inference:")
    logger.info("  from peft import PeftModel")
    logger.info(f"  base_model = AutoModelForCausalLM.from_pretrained('{BASEM_LOC}', ...)")
    logger.info(f"  model = PeftModel.from_pretrained(base_model, '{final_save_path}')")
    logger.info("  model = model.merge_and_unload() # Optional: Merge for faster inference")
    logger.info("\nNEXT STEP: Evaluate the saved model using the MT-Bench framework.")
    logger.info("Refer to the MT-Bench repository for instructions: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge")