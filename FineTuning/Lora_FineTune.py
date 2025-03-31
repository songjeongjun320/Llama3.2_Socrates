from datasets import Dataset
import json
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_single_json_file(file_path):
    """Load a single JSON file containing question/answer pairs."""
    logger.info(f"Loading data from: {file_path}")
    all_data = {"question": [], "answer": []}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Data type: {type(data)}, length: {len(data)}")
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"First item type: {type(data[0])}, keys: {data[0].keys() if isinstance(data[0], dict) else 'N/A'}")
                for item in data:
                    if isinstance(item, dict) and "question" in item and "answer" in item:
                        question = str(item["question"])
                        answer = str(item["answer"])
                        all_data["question"].append(question)
                        all_data["answer"].append(answer)
            logger.info(f"Processed data: questions={len(all_data['question'])}, answers={len(all_data['answer'])}")
            if all_data["question"]:
                logger.info(f"First question: {all_data['question'][0][:100]}")
                logger.info(f"First answer: {all_data['answer'][0][:100]}")
            return all_data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return all_data

def preprocess_function(examples, tokenizer, max_length=1024):
    """Preprocess function for tokenizing question/answer pairs."""
    prompts = [f"Question: {q}\nAnswer: " for q in examples["question"]]
    answers = [a for a in examples["answer"]]
    inputs = [p + a for p, a in zip(prompts, answers)]
    
    tokenized = tokenizer(
        inputs,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].clone()
    }

def preprocess_with_tokenizer(examples):
    return preprocess_function(examples, tokenizer)

if __name__ == "__main__":
    # Paths and constants
    FILE_PATH = "/scratch/jsong132/Technical_Llama3.2/DB/dev_data.json"
    BASE_MODEL = "/scratch/jsong132/Technical_Llama3.2/llama3.2_3b"
    OUTPUT_DIR = "Tune_Results/llama3.2_Socrates"
    
    # Load and prepare data
    all_data = load_single_json_file(FILE_PATH)
    if not all_data["question"] or not all_data["answer"]:
        logger.error("No data loaded. Check JSON file structure.")
        exit(1)
    assert len(all_data["question"]) == len(all_data["answer"]), "Question/answer counts mismatch."

    # Convert to Dataset and split
    dataset = Dataset.from_dict(all_data)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset['test'])} samples")

    # Load tokenizer and model
    logger.info(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Tokenize datasets and format as PyTorch tensors
    logger.info("Tokenizing datasets...")
    train_dataset = dataset["train"].map(
        preprocess_with_tokenizer,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count(),
        desc="Tokenizing training data"
    ).with_format("torch")

    val_dataset = dataset["test"].map(
        preprocess_with_tokenizer,
        batched=True,
        remove_columns=dataset["test"].column_names,
        num_proc=os.cpu_count(),
        desc="Tokenizing validation data"
    ).with_format("torch")

    # Apply LoRA
    peft_params = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_params)
    model.config.use_cache = False  # For gradient checkpointing compatibility
    model.print_trainable_parameters()

    # Verify trainable parameters
    trainable_params = [(name, param.requires_grad) for name, param in model.named_parameters() if param.requires_grad]
    for name, req_grad in trainable_params:
        logger.info(f"Trainable: {name}, requires_grad: {req_grad}")
    if not trainable_params:
        logger.error("No trainable parameters found. Check LoRA config.")
        exit(1)

    # Manual forward pass test
    logger.info("Running manual forward pass test...")
    sample_batch = train_dataset[:2]
    inputs = {
        "input_ids": sample_batch["input_ids"].to(model.device),
        "attention_mask": sample_batch["attention_mask"].to(model.device),
        "labels": sample_batch["labels"].to(model.device)
    }
    model.train()
    outputs = model(**inputs)
    loss = outputs.loss
    logger.info(f"Manual loss: {loss.item() if loss is not None else 'None'}")
    logger.info(f"Loss requires grad: {loss.requires_grad if loss is not None else 'N/A'}")
    if loss is None or not loss.requires_grad:
        logger.error("Loss computation failed or no gradients. Check model/data.")
        exit(1)

    # Training arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        dataloader_num_workers=1,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=400,
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=False,  # Temporarily disable to test
        optim="adamw_torch",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Pre-training optimizations
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model.train()  # Ensure training mode

    # Train
    logger.info("Starting fine-tuning...")
    trainer.train(resume_from_checkpoint=False)

    # Save model
    logger.info(f"Saving model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Fine-tuning completed successfully!")