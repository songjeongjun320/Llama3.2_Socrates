# llama3.2 socrates

a fine-tuned llama-3.2-3b assistant with enhanced mathematical and general reasoning.

## overview

this project implements a general-purpose ai assistant based on meta's llama-3.2-3b model, with specialized fine-tuning for mathematical problem-solving and instruction following. the system includes a client-server architecture for easy interaction.

## components

- **base model**: meta-llama/llama-3.2-3b
- **fine-tuning approaches**: 
  - shp (stanford human preferences) instruction tuning
  - mathematics-specific "socrates" method tuning
- **server**: vllm-based inference api
- **client**: simple interface for sending queries

## current status (april 8, 2025)

- ✓ base model loading implemented
- ✓ stanford human preferences (shp) instruction tuning complete
- ✓ mathematical reasoning fine-tuning via "socrates" method complete
- ✓ working server/client api for query processing
- ✓ evaluation framework established with test datasets
- ✓ performance testing on gsm8k, mathqa and svamp datasets

## usage

### start server
```
python server.py
```

### send queries
```
python client.py
```

## evaluation

our enhanced model shows improvements over the base llama-3.2-3b in:
- instruction following
- step-by-step mathematical reasoning
- program-of-thought (pot) generation for solving problems

detailed evaluation metrics available in the `FineTuning/Math` directory.

## next steps

- optimize inference speed
- explore further training data augmentation
- implement safety guardrails
- expand evaluation to more diverse tasks 