# LLM Fine-Tuning with Llama 2

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![HuggingFace](https://img.shields.io/badge/huggingface-%F0%9F%A4%97-yellow)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This repository contains a comprehensive implementation for fine-tuning Llama 2 models using Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically focusing on Low-Rank Adaptation (LoRA). The project demonstrates how to efficiently adapt large language models for specific downstream tasks while minimizing computational resources.

By using LoRA, we can fine-tune large models with a fraction of the parameters, making it feasible to run on consumer hardware while maintaining performance comparable to full fine-tuning.

## Features

- **Parameter-Efficient Fine-Tuning**: Implements LoRA technique to fine-tune LLMs with significantly fewer parameters
- **Flexible Model Selection**: Support for various Llama 2 model sizes (7B, 13B, 70B)
- **Custom Dataset Handling**: Tools for preprocessing and loading custom datasets
- **Mixed Precision Training**: Utilizes bitsandbytes for 8-bit quantization and mixed precision training
- **Evaluation Metrics**: Built-in evaluation pipeline with various NLP metrics
- **Inference Optimization**: Techniques for optimizing inference speed and memory usage
- **Checkpoint Management**: Save and load model checkpoints efficiently

## Project Structure

```
├── notebooks/
│   ├── LLMS_finetuning_Models_llama2.ipynb    # Main notebook for fine-tuning
│   └── ... (other related notebooks)
├── data/
│   ├── processed/                             # Processed datasets
│   └── raw/                                   # Raw data files
├── models/
│   └── checkpoints/                           # Saved model checkpoints
├── scripts/
│   ├── preprocess.py                          # Data preprocessing utilities
│   └── evaluate.py                            # Evaluation scripts
├── configs/                                   # Configuration files
├── requirements.txt                           # Project dependencies
└── README.md                                  # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30.0+
- Accelerate 0.20.0+
- PEFT 0.4.0+
- bitsandbytes 0.40.0+
- tqdm
- datasets
- CUDA-capable GPU (recommended 16GB+ VRAM for 7B models)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-finetuning-llama2.git
cd llm-finetuning-llama2
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up Hugging Face credentials to access Llama 2 models:
```bash
huggingface-cli login
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/LLMS_finetuning_Models_llama2.ipynb
```

2. Follow the step-by-step instructions in the notebook to:
   - Load and prepare your dataset
   - Configure the model and LoRA parameters
   - Train the model
   - Evaluate results
   - Save the fine-tuned model

## Model Training

The project uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA to efficiently fine-tune Llama 2 models. Key training parameters include:

- **LoRA Rank**: Controls the rank of the low-rank adaptation matrices (typically 8-128)
- **LoRA Alpha**: Scaling factor for the LoRA parameters (typically 16-64)
- **Learning Rate**: Typically in the range of 1e-5 to 5e-4
- **Batch Size**: Adjusted based on available GPU memory
- **Training Epochs**: Typically 3-5 epochs are sufficient

Example configuration:
```python
peft_config = LoraConfig(
    r=16,                     # LoRA rank
    lora_alpha=32,            # LoRA alpha parameter
    lora_dropout=0.05,        # Dropout probability for LoRA layers
    bias="none",              # Bias type for LoRA
    task_type="CAUSAL_LM",    # Task type
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
```

## Evaluation

The notebook includes evaluation code to assess model performance using metrics such as:
- Perplexity
- ROUGE scores
- BLEU scores
- Custom task-specific metrics

## Results

Typical improvements seen with LoRA fine-tuning on Llama 2:
- 20-30% reduction in perplexity on domain-specific tasks
- Significant improvement in task-specific metrics
- Comparable performance to full fine-tuning with only ~0.1% of the parameters

## Examples

### Sample code for inference:
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the LoRA adapter
adapter_path = "path/to/your/adapter"
model = PeftModel.from_pretrained(model, adapter_path)

# Generate text
input_text = "Your prompt here"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Troubleshooting

### Common Issues:

1. **Out of Memory Errors**
   - Reduce batch size
   - Use a smaller model variant
   - Enable gradient checkpointing
   - Increase LoRA rank for better parameter efficiency

2. **Slow Training**
   - Use mixed precision training (fp16 or bf16)
   - Enable flash attention if supported

3. **Poor Performance**
   - Tune learning rate and LoRA hyperparameters
   - Increase training epochs
   - Review data quality and preprocessing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta AI for releasing the Llama 2 models
- Hugging Face for the Transformers library
- The PEFT library developers for implementing efficient fine-tuning methods
- The open-source AI community for continuous improvements and knowledge sharing
