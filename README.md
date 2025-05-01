# Llama3.2 Socrates: Enhanced Mathematical and Instructional Assistant

![Llama Socrates Overview](Socrates_Llama.png)

Llama3.2 Socrates is an AI assistant based on Meta's Llama-3.2-3b model, designed to provide enhanced performance in solving mathematical problems and handling general inquiries. This project utilizes **Mixture of Experts (MoE)** and **LoRA** (Low-Rank Adaptation) techniques, providing a modular architecture for different types of queries.

## Key Features

- **Base Model**: `meta-llama/llama-3.2-3b`
- **Fine-Tuning Techniques**: MoE (Mixture of Experts), LoRA (Low-Rank Adaptation)
- **Adapter Architecture**:
  - `judge_adapter`: A classifier that determines whether a user’s query is related to mathematics or general inquiries.
  - `math_adapter`: An adapter specialized in mathematical problem-solving, using the **Prompt of Thought (PoT)** technique to improve problem-solving accuracy.
  - `instruction_adapter`: Handles general queries and instructions. It is also integrated with a subset of mathematical data to handle potential math-related requests.

## Architecture

Llama3.2 Socrates operates by adapting the base model using **LoRA** for various tasks, with different adapters loaded depending on the type of query. The **judge_adapter** classifies the query type, and for math-related questions, the request is passed to the **math_adapter**. For general inquiries, it is directed to the **instruction_adapter**. 

The **math_adapter** leverages the **Prompt of Thought (PoT)** method, where the model generates Python code to solve mathematical problems, significantly improving its performance on such tasks.

### Performance Improvement

- **Accuracy Improvement**: The base Llama3.2-3b model achieved 8% accuracy in solving mathematical problems, but with the use of our adapter system, accuracy was improved to **32%**.
- **Datasets Used**:
  - Mathematical Datasets: `gsm8k`, `mathcodeinstruct`, `mathqa`, `mbpp`, `svamp`, `codeparrot`
  - Instruction Datasets: `openmath`, `shp`, `alpaca`, `dolly`, `openbookQA`, `mmlu`

### Handling Errors in Math Adapter

Given that the **math_adapter** might occasionally generate incorrect code, the **instruction_adapter** has been integrated with some mathematical data to handle math-related queries in case of errors or limitations in the math-specific model.
