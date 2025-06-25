# Fine-tuning Llama-2-7b-chat-hf for Hawaiian Wildfire Q&A

This repository contains the code and resources for fine-tuning the `meta-llama/Llama-2-7b-chat-hf` model on a custom dataset related to Hawaiian wildfires. The goal is to enhance the model's ability to answer questions accurately and concisely about Hawaiian wildfires.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project demonstrates the process of fine-tuning a pre-trained Large Language Model (LLM), specifically Llama-2-7b-chat-hf, on a domain-specific dataset. The chosen domain is Hawaiian wildfires, and the fine-tuning aims to make the model a more reliable source of information for questions related to this topic.

The fine-tuning process leverages:
- **Quantization (4-bit)**: To reduce memory footprint and enable training on GPUs with limited memory (e.g., T4 on Colab)
- **LoRA (Low-Rank Adaptation)**: For efficient fine-tuning by only training a small number of additional parameters
- **Hugging Face Transformers & Datasets**: For streamlined model loading, training, and data handling

## Dataset

The custom dataset used for fine-tuning consists of text data related to Hawaiian wildfires. The data is loaded from two text files:
- `/content/Fine-tuning-LLMs/data/hawaii_wf_4.txt`
- `/content/Fine-tuning-LLMs/data/hawaii_wf_2.txt`

The dataset is loaded and prepared using the `datasets` library. Each line in the text files is treated as a separate training example.

**Example of a data point:**
```
'had taken refuge in the ocean to escape the fire, ensuring they reached the emergency shelter safely.'
```

## Model Architecture

The base model used is `meta-llama/Llama-2-7b-chat-hf`.

### Quantization Configuration
The model is loaded with a `BitsAndBytesConfig` for 4-bit quantization:
```python
load_in_4bit=True
bnb_4bit_use_double_quant=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
```

### LoRA Configuration
LoRA configuration for efficient fine-tuning:
```python
r=8
lora_alpha=64
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
bias="none"
lora_dropout=0.05
task_type="CAUSAL_LM"
```

## Setup and Installation

To run this notebook, you'll need a GPU (e.g., Google Colab T4 GPU).

### 1. Clone the repository (if applicable):
```bash
!git clone https://github.com/poloclub/Fine-tuning-LLMs.git
```

### 2. Install necessary libraries:
```bash
!pip install -qq peft transformers bitsandBytes accelerate datasets
!pip install -qq GPUtil
```

### 3. Hugging Face Login:
You will need to log in to Hugging Face to access the Llama 2 model.

```python
from huggingface_hub import notebook_login
notebook_login()
```

If running in Colab, use:
```bash
!huggingface-cli login
```

## Training

The model is trained using the `transformers.Trainer` class.

### Training Arguments:
```python
output_dir: ./finetunedModel
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
num_train_epochs: 3
learning_rate: 1e-4
max_steps: 20  # for demonstration/quick testing
bf16: False
optim: "paged_adamw_8bit"
logging_dir: ./logs
save_strategy: "epoch"
save_steps: 50
logging_steps: 10
```

The training process can be initiated by running the corresponding cell in the Jupyter Notebook.

## Inference

After training, the fine-tuned model can be loaded and used for inference.

### Loading the Fine-tuned Model:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from peft import PeftModel

base_model_id = "meta-llama/Llama-2-7b-chat-hf"

nf4Config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained(
    base_model_id, 
    use_fast=False, 
    trust_remote_code=True, 
    add_eos_token=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=nf4Config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

modelFinetuned = PeftModel.from_pretrained(base_model, "./finetunedModel/checkpoint-20")
```

### Performing Inference:
```python
user_question = "When did Hawaii wildfires start?"  # Example question
eval_prompt = f"Question: {user_question} Just answer this question accurately and concisely.\n"

promptTokenized = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

modelFinetuned.eval()

with torch.no_grad():
    print(tokenizer.decode(
        modelFinetuned.generate(**promptTokenized, max_new_tokens=1024)[0], 
        skip_special_tokens=True
    ))
    torch.cuda.empty_cache()
```

## Results

The fine-tuned model aims to provide more relevant and accurate answers to questions related to Hawaiian wildfires.

### Example Inferences:

**Question: When did Hawaii wildfires start?**
- **Original Model (Hypothetical)**: Might give a general answer or hallucinate
- **Fine-tuned Model**: Provides detailed information about Hawaiian wildfires

**Question: What caused the Hawaii wildfires to start?**
The fine-tuned model provides comprehensive answers covering:
1. Dry conditions due to unusually dry winter
2. Strong winds contributing to fire spread
3. Potential human activity factors (e.g., downed power lines)

**Question: Can Hawaii wildfires spread to inner part of the country?**
The model addresses the possibility of spread through trade winds and airborne embers.

### Notes on Model Performance
- The first example output occasionally mentions California fires, suggesting that more data or different prompt engineering might be needed to avoid confusion
- Subsequent responses show better adherence to the Hawaiian wildfire context
- Further refinement may be needed for optimal performance

## Contributing

Contributions are welcome! If you have suggestions for improving the dataset, model architecture, or training process, please open an issue or submit a pull request.

## License

This project is open-sourced under the MIT License.

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/) and [Datasets](https://huggingface.co/datasets/) libraries
- [Meta](https://ai.meta.com/) for the Llama-2 model
- The provided Colab Notebook for the initial setup and fine-tuning pipeline
