# FILE: /juridia-finetuning-project/juridia-finetuning-project/src/training/trainer.py

import os
import pandas as pd
import yaml
from datasets import Dataset
from src.models.juridia_model import JuridiaModel
from sklearn.preprocessing import LabelEncoder

# Suppress symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.train_data_path = config['train_data_path']
        self.test_data_path = config['test_data_path']
        self.batch_size = int(config['batch_size'])  # Ensure integer
        self.learning_rate = float(config['learning_rate'])  # Ensure float
        self.num_epochs = int(config['num_epochs'])  
        self.model_name = config['model_name']
        self.max_length = int(config['max_length']) 
        self.output_dir = config['output_dir']

def load_training_data(file_path):
    df = pd.read_csv(file_path)
    # Combine title and document type for better context
    texts = df['long_title'].fillna('') + ' [SEP] ' + df['doc_type'].fillna('')
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['doc_type'])
    return Dataset.from_dict({'text': texts, 'label': labels}), label_encoder

def load_test_data(file_path):
    df = pd.read_csv(file_path)
    # For test data, we only have questions
    texts = df['question'].fillna('')
    return Dataset.from_dict({'text': texts})

def main():
    # Load configuration
    config = Config('config.yml')
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load datasets
    train_dataset, label_encoder = load_training_data(config.train_data_path)
    test_dataset = load_test_data(config.test_data_path)
    
    # Save label encoder classes
    config.num_labels = len(label_encoder.classes_)
    with open(os.path.join(config.output_dir, 'label_encoder.yml'), 'w') as f:
        yaml.dump({'classes': label_encoder.classes_.tolist()}, f)
    
    # Initialize model
    model = JuridiaModel(config)
    
    # Train the model
    model.train(train_dataset)
    
    # Save the model
    model.save_model(config.output_dir)

if __name__ == "__main__":
    main()