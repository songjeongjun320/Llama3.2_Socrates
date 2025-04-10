import subprocess
import os
import shlex

# define setting
model_path = "/scratch/syeom3/Technical_Llama3.2/llama3.2_3b"
results_dir = "/scratch/syeom3/eval_results"
results_filename = "llama-3.2-3b-base-mmlu-gsm8k.json"
results_path = os.path.join(results_dir, results_filename)

os.makedirs(results_dir, exist_ok=True)

command = [
    "lm_eval",  
    "--model", "hf",
    "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
    "--tasks", "mmlu,gsm8k",
    "--num_fewshot", "5",
    "--device", "cuda:0",
    "--batch_size", "auto",
    "--output_path", results_path
]

print("--- running command ---")
try:
    print(shlex.join(command))
except ImportError:
    print(" ".join(command))
print("--------------------------")

try:
    process = subprocess.run(command, check=True, text=True)

    print("--------------------------")
    print(f"successfully evaluated.")
    print(f"result file path: {results_path}")

except FileNotFoundError:
    print("--------------------------")
    print("cannot find lm_eval file")

except subprocess.CalledProcessError as e:
    print("--------------------------")
    print(f"ERROR WITH : {e.returncode})")

except Exception as e:
    print("--------------------------")
    print(f" Exceptional Error: {e}")