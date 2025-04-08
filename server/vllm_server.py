import logging
from vllm import LLM, SamplingParams # Import LoRARequest
from vllm.lora.request import LoRARequest
from fastapi import FastAPI, Request
# Keep only necessary imports at the top if not merging beforehand
# from transformers import AutoTokenizer # Might only need tokenizer path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_model_path = '/scratch/jsong132/Technical_Llama3.2/llama3.2_3b'
lora_model_path = '/scratch/jsong132/Technical_Llama3.2/FineTuning/Math/Tune_Results/llama3.2_Socrates_Math/final_checkpoint'
# Optional: Give your LoRA adapter a name/ID for vLLM
lora_id = "socrates_math"

app = FastAPI()

logger.info("Initializing LLM with LoRA support...")
llm = LLM(
    model=base_model_path,           # Pass the BASE model path
    tokenizer=base_model_path,       # Usually tokenizer path is the same
    enable_lora=True,                # Enable LoRA support in vLLM
    max_loras=1,                     # Max number of LoRAs you might load concurrently
    max_model_len=4096,
    # You might need to configure max_lora_rank depending on your adapter
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1
)
logger.info("LLM initialized successfully.")

@app.post("/generate")
async def generate(request: Request):
    logger.info("Received request to generate response.")
    data = await request.json()
    prompt = data.get("prompt", "")

    logger.info(f"Generating response for prompt: {prompt}")

    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.7,
        repetition_penalty=1.1,
        max_tokens=1024
    )

    # Create a LoRARequest to specify which adapter to use for this generation
    # The lora_int_id should match the order/index vLLM assigns (often starts at 1, but check docs or experiment)
    # Or you might load it dynamically if needed - vLLM API evolves.
    # For simplicity here, assuming it's loaded at startup or identifiable.
    # You might need a mechanism to tell vLLM about lora_model_path during init or first request.
    # A common pattern is to use LoRARequest referencing the path:
    lora_request = LoRARequest(
        lora_name=lora_id, # A unique name you assign
        lora_int_id=1,       # An internal ID vLLM uses (often 1 for the first loaded)
        lora_local_path=lora_model_path # The actual path to the adapter
    )


    # Pass the lora_request to the generate method
    responses = llm.generate(prompt, sampling_params, lora_request=lora_request)

    # Process responses (vLLM returns RequestOutput objects)
    generated_texts = [output.outputs[0].text for output in responses]
    logger.info(f"Generated {len(generated_texts)} response(s).")
    # Return just the text part
    return {"responses": generated_texts}

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    import uvicorn
    # Make sure peft is installed in the environment uvicorn uses
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("FastAPI server started successfully.")