# FILE: /juridia-finetuning-project/juridia-finetuning-project/src/submission/generate_submission.py

import os
import pandas as pd
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from typing import Dict, List, Optional

# Suppress symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class SubmissionGenerator:
    def __init__(self, model_dir: str):
        """Initialize the submission generator with a trained model directory."""
        self.model_dir = model_dir
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.label_classes = self._load_label_encoder()
        
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer from the saved directory."""
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        return model, tokenizer
    
    def _load_label_encoder(self) -> List[str]:
        """Load the label encoder classes from the saved YAML file."""
        with open(os.path.join(self.model_dir, 'label_encoder.yml'), 'r') as f:
            label_info = yaml.safe_load(f)
        return label_info['classes']
    
    def predict(self, texts: List[str]) -> List[str]:
        """Generate predictions for a list of texts."""
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_indices = predictions.argmax(dim=-1)
        
        # Convert to document types
        return [self.label_classes[idx] for idx in predicted_indices]
    
    def generate_submission(self, test_file: str, output_file: str, output_dir: Optional[str] = None) -> None:
        """
        Generate a submission file from test data.
        
        Args:
            test_file: Path to the test CSV file
            output_file: Name of the output submission file
            output_dir: Optional directory for the output file
        """
        # Load test data
        test_df = pd.read_csv(test_file)
        
        # Generate predictions
        predictions = self.predict(test_df['question'].tolist())
        
        # Create single-row submission DataFrame with all predictions in one column
        submission_df = pd.DataFrame({
            'predictions': [predictions]  # All predictions as a list in a single row
        })
        
        # Prepare output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_file)
        else:
            output_path = output_file
        
        # Save submission
        if output_path.endswith('.parquet'):
            submission_df.to_parquet(output_path, index=False)
        else:
            submission_df.to_csv(output_path, index=False)
        print(f'Submission file saved to: {output_path}')
        
        # Print submission statistics
        print('\nSubmission Statistics:')
        print(f'Total predictions: {len(predictions)}')
        print('\nPrediction counts:')
        for doc_type, count in pd.Series(predictions).value_counts().items():
            print(f'{doc_type}: {count}')

def main():
    """Main function to generate submission."""
    # Load configuration
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize submission generator
    generator = SubmissionGenerator(config['output_dir'])
    
    # Generate submission
    generator.generate_submission(
        test_file=config['test_data_path'],
        output_file='submission.csv',
        output_dir=config['output_dir']
    )

if __name__ == "__main__":
    main()