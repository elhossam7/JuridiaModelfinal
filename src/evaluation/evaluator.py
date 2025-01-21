# FILE: /juridia-finetuning-project/juridia-finetuning-project/src/evaluation/evaluator.py

import os
import pandas as pd
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, List, Tuple

class Evaluator:
    def __init__(self, model_dir: str):
        """Initialize the evaluator with a trained model directory."""
        self.model_dir = model_dir
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.label_classes = self._load_label_encoder()
        
    def _load_model_and_tokenizer(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Load the model and tokenizer from the saved directory."""
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        return model, tokenizer
    
    def _load_label_encoder(self) -> List[str]:
        """Load the label encoder classes from the saved YAML file."""
        with open(os.path.join(self.model_dir, 'label_encoder.yml'), 'r') as f:
            label_info = yaml.safe_load(f)
        return label_info['classes']
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on a list of texts."""
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return predictions.numpy()
    
    def predict_labels(self, texts: List[str]) -> List[str]:
        """Predict document types for a list of texts."""
        predictions = self.predict(texts)
        return [self.label_classes[idx] for idx in predictions.argmax(axis=1)]
    
    def evaluate_and_save(self, test_file: str, output_dir: str) -> Dict:
        """Evaluate the model on test data and save predictions."""
        # Load test data
        test_df = pd.read_csv(test_file)
        
        # Make predictions
        predicted_labels = self.predict_labels(test_df['question'].tolist())
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'Id': test_df['Id'],
            'Predicted_Type': predicted_labels
        })
        
        # Save predictions
        os.makedirs(output_dir, exist_ok=True)
        submission_path = os.path.join(output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        
        # Calculate statistics
        stats = {
            'prediction_counts': pd.Series(predicted_labels).value_counts().to_dict(),
            'num_predictions': len(predicted_labels),
            'unique_types': len(set(predicted_labels))
        }
        
        # Save statistics
        stats_path = os.path.join(output_dir, 'evaluation_stats.yml')
        with open(stats_path, 'w') as f:
            yaml.dump(stats, f)
        
        return stats

def main():
    """Main function to run evaluation."""
    # Load configuration
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize evaluator
    evaluator = Evaluator(config['output_dir'])
    
    # Run evaluation
    stats = evaluator.evaluate_and_save(
        test_file=config['test_data_path'],
        output_dir=config['output_dir']
    )
    
    # Print statistics
    print("\nEvaluation Statistics:")
    print(f"Total predictions: {stats['num_predictions']}")
    print(f"Unique document types: {stats['unique_types']}")
    print("\nPrediction counts:")
    for doc_type, count in stats['prediction_counts'].items():
        print(f"{doc_type}: {count}")

if __name__ == "__main__":
    main()