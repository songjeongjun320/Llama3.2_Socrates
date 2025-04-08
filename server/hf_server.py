import logging
import torch
from fastapi import FastAPI, Request, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn # Import uvicorn directly

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths
base_model_path = '/scratch/jsong132/Technical_Llama3.2/llama3.2_3b'
lora_model_path = '/scratch/jsong132/Technical_Llama3.2/FineTuning/Math/Tune_Results/llama3.2_Socrates_Math/final_checkpoint'

# Global variables for model and tokenizer (loaded on server start)
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Function to load model and tokenizer
def load_model_and_tokenizer():
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.info("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            # If tokenizer lacks a pad_token, set it to eos_token (common practice)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}", exc_info=True)
            raise

        logger.info("Loading base model...")
        try:
            # Consider using bfloat16 for memory efficiency during model loading (if GPU supports)
            # device_map='auto' can be useful if 'accelerate' library is installed
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16, # Or torch.float16 (depending on GPU performance/compatibility)
                device_map="auto" # Automatically distributes across multiple GPUs or CPU+GPU (requires 'pip install accelerate')
                # device_map=device # Can be explicitly specified for single GPU usage
            )
            logger.info("Base model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading base model: {e}", exc_info=True)
            raise

        logger.info("Loading LoRA adapter...")
        try:
            # Load the LoRA adapter on top of the base model using PeftModel.from_pretrained
            model = PeftModel.from_pretrained(base_model, lora_model_path)
            # Set to evaluation mode (disables dropout, etc.)
            model.eval()
            logger.info("LoRA model loaded and combined successfully.")
            # Check the device(s) the model is loaded on (can be multiple devices with device_map='auto')
            logger.info(f"Model loaded on device(s): {model.device if hasattr(model, 'device') else 'check layers'}") 

        except Exception as e:
            logger.error(f"Error loading LoRA adapter: {e}", exc_info=True)
            raise
    else:
        logger.info("Model and tokenizer already loaded.")

# Create FastAPI app instance
app = FastAPI()

# Load model on server startup (called within an async function)
@app.on_event("startup")
async def startup_event():
    logger.info("Starting model loading process...")
    load_model_and_tokenizer()
    logger.info("Model loading completed. Server is ready.")

# Text generation endpoint
@app.post("/generate")
async def generate(request: Request):
    # Check if the model is loaded
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet. Please wait.")

    try:
        data = await request.json()
        prompt = data.get("prompt")
        # Get sampling parameters from the request (use defaults if missing)
        max_new_tokens = data.get("max_new_tokens", 512) # Maximum number of new tokens to generate
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        repetition_penalty = data.get("repetition_penalty", 1.1)

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is missing in the request.")

        logger.info(f"Received prompt: {prompt}")
        logger.info(f"Generation parameters: max_new_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}, repetition_penalty={repetition_penalty}")

        # Tokenize the input prompt
        # When using device_map='auto', no need to specify device for input tensors; the model handles it
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        target_device = model.device 
        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device)
        logger.info(f"Moved input tensors to device: {target_device}") # Log for confirmation

        logger.info("Generating response...")
        # Generate text using the model
        with torch.no_grad(): # Disable gradient calculations during inference
            # Use the generate function - inputs are now on the correct device
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True, # Set to True when using temperature, top_p
                pad_token_id=tokenizer.eos_token_id # Specify pad_token_id (important)
            )

        # Decode the generated tokens (excluding the input prompt part)
        # outputs[0] is the first result in the batch
        # input_ids.shape[1] is the length of the input tokens
        generated_token_ids = outputs[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        logger.info(f"Generated response: {response_text}")
        return {"response": response_text}

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Run the server (using uvicorn)
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    # host="0.0.0.0" allows external connections
    # Set reload=False to prevent automatic restart (considering model loading time)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    # Note: When running directly from the terminal, use 'uvicorn server:app --host 0.0.0.0 --port 8000' (without reload)