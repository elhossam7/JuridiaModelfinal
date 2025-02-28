# Juridia Fine-Tuning Project

## Overview
The Juridia Fine-Tuning Project aims to fine-tune a legal language model using a dataset of legal documents and questions. The project is designed to support modular development and scalability, allowing for easy updates and enhancements.

## Project Structure
```
juridia-finetuning-project
├── src
│   ├── models
│   │   ├── base_llm.py          # Base legal language model implementation
│   │   └── juridia_model.py     # Fine-tuned model wrapper
│   ├── data
│   │   ├── legal_dataset.py     # Class for loading and processing the training dataset
│   │   ├── multilingual_data.py  # Handling bilingual data (French and Arabic)
│   │   └── preprocess.py        # Functions for preprocessing legal text data
│   ├── peft
│   │   └── lora.py              # Implementation of PEFT techniques (LoRA)
│   ├── training
│   │   ├── trainer.py           # Main training loop for fine-tuning the model
│   │   └── config.py            # Configuration parameters for training
│   ├── evaluation
│   │   ├── juridia_metrics.py    # Evaluation metrics for model outputs
│   │   └── evaluator.py          # Script to evaluate the model on the test dataset
│   ├── multilingual
│   │   ├── translation.py        # Functionality for translating between French and Arabic
│   │   └── tokenizer.py          # Tokenization for both languages
│   ├── submission
│   │   └── generate_submission.py # Generates submission files for competitions
│   ├── utils
│   │   └── helpers.py            # General utility functions
│   └── tests                     # Unit tests for all modules
├── notebooks                     # For experimentation and analysis
├── requirements.txt              # Python dependencies for the project
├── pyproject.toml                # Project configuration file
└── README.md                     # Documentation on setup, usage, and development
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/elhossam7/Juridia-Fine-Tuning-Project.git 
cd Juridia-Fine-Tuning-Project
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Use the `legal_dataset.py` to load and preprocess the training data from `train.csv`.
   ```bash
   python src/data/legal_dataset.py --input_file "data/train.csv" --output_file "data/processed_data.pkl"
   ```
2. **Model Training**: Run the `trainer.py` script to fine-tune the model using the prepared dataset.
   ```bash
   python -m src.training.trainer
   ```
3. **Evaluation**: After training, evaluate the model using `evaluator.py` to assess its performance on the test dataset.
   ```bash
   python -m src.evaluation.evaluator
   ```
4. **Submission**: Generate submission files using `generate_submission.py` for competitions or assessments.
   ```bash
   python -m src.submission.generate_submission
   ```

## Development Guidelines
- Follow the modular structure to add new features or improve existing ones.
- Write unit tests for new functionalities in the `tests` directory.
- Document any changes made to the codebase for future reference.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.# juridia-finetuning-project
