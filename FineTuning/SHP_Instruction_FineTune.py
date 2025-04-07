# Standard library imports
import json
import logging
import os
import gc
from datasets import load_dataset, Dataset
from typing import List, Dict

# Third-party imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_shp_data():
    """Load Stanford Human Preferences dataset from Hugging Face."""
    logger.info("Loading Stanford Human Preferences (SHP) dataset")
    try:
        # Load directly from Hugging Face
        dataset = load_dataset("stanfordnlp/SHP")
        logger.info(f"Successfully loaded SHP dataset")
        logger.info(f"Train: {len(dataset['train'])} samples")
        logger.info(f"Validation: {len(dataset['validation'])} samples")
        logger.info(f"Test: {len(dataset['test'])} samples")
        
        # Log a sample to verify structure
        if len(dataset['train']) > 0:
            sample = dataset['train'][0]
            logger.info(f"Sample question: {sample['question'][:150]}...")
            logger.info(f"Sample preferred answer (human_ref_A): {sample['human_ref_A'][:150]}...")
        
        return dataset
    except Exception as e:
        logger.error(f"Error loading SHP dataset: {str(e)}")
        raise

# SFTTrainer formatting function for SHP dataset
def formatting_prompts_func(example):
    """Format SHP examples for instruction tuning."""
    output_texts = []
    for i in range(len(example['question'])):
        # Use the preferred human reference (human_ref_A) for instruction tuning
        # Format as a simple instruction-following example
        text = f"Question: {example['question'][i]}\nAnswer: {example['human_ref_A'][i]}"
        output_texts.append(text)
    return output_texts

if __name__ == "__main__":
    # --- Paths and Constants ---
    BASE_MODEL = "/scratch/jsong132/Technical_Llama3.2/llama3.2_3b"
    OUTPUT_DIR = "Tune_Results/llama3.2_SHP_Instruction"
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Load and Prepare Data ---
    dataset = load_shp_data()
    
    # --- Load Tokenizer and Model ---
    logger.info(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set EOS token as PAD token.")
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map="auto",           # Automatically distribute model across GPUs
        trust_remote_code=True
    )
    
    # --- Apply LoRA ---
    # Target only FFNN layers (gate_proj, up_proj, down_proj) as requested
    # These are the feed-forward neural network components in Llama architecture
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        # Target only FFNN layers, not attention layers (q_proj, k_proj, v_proj, o_proj)
        target_modules=["gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_params)
    model.config.use_cache = False  # Important for training stability
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
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=250,
        report_to="none",
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # --- Initialize SFTTrainer ---
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_params,
        formatting_func=formatting_prompts_func,
    )
    
    # --- Pre-training Optimizations ---
    torch.cuda.empty_cache()
    gc.collect()
    # Enable TF32 for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model.train()
    
    # --- Train ---
    logger.info("Starting fine-tuning...")
    train_result = trainer.train(resume_from_checkpoint=False)
    
    # --- Save Final Model ---
    logger.info("Saving final model...")
    final_save_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(final_save_path)
    logger.info(f"Final model adapters saved to: {final_save_path}")
    
    # --- Evaluation Summary ---
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Run evaluation on test set
    logger.info("Running evaluation on test set...")
    eval_metrics = trainer.evaluate(dataset["test"])
    trainer.log_metrics("test", eval_metrics)
    trainer.save_metrics("test", eval_metrics)
    
    logger.info("Fine-tuning complete!")
