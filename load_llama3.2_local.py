import os
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

download_path = "/scratch/jsong132/Technical_Llama3.2/llama3.2_3b"

os.makedirs(download_path, exist_ok=True)

model_file_1 = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="model-00001-of-00002.safetensors",
    local_dir=download_path
)

model_file_2 = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="model-00002-of-00002.safetensors",
    local_dir=download_path
)

index_file = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="model.safetensors.index.json",
    local_dir=download_path
)

config_file = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="config.json",
    local_dir=download_path
)

generation_config_file = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="generation_config.json",
    local_dir=download_path
)

special_tokens_map = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="special_tokens_map.json",
    local_dir=download_path
)

tokenizer = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="tokenizer.json",
    local_dir=download_path
)

tokenizer_config = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="tokenizer_config.json",
    local_dir=download_path
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", 
    cache_dir=download_path
)