# FILE: /juridia-finetuning-project/juridia-finetuning-project/src/utils/helpers.py

# This file contains general utility functions that can be used throughout the project.

import logging
import pandas as pd

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded CSV file: {file_path} with {len(df)} records.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file: {file_path}. Error: {e}")
        return None

def save_csv(df, file_path):
    """Save a pandas DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Saved DataFrame to CSV file: {file_path}.")
    except Exception as e:
        logging.error(f"Error saving CSV file: {file_path}. Error: {e}")

def clean_text(text):
    """Clean and preprocess text data."""
    if isinstance(text, str):
        cleaned_text = text.strip().replace('\n', ' ').replace('\r', '')
        logging.debug(f"Cleaned text: {cleaned_text}")
        return cleaned_text
    return text

def normalize_date(date_str):
    """Normalize date strings to a standard format."""
    try:
        normalized_date = pd.to_datetime(date_str).strftime('%Y-%m-%d')
        logging.debug(f"Normalized date: {normalized_date}")
        return normalized_date
    except Exception as e:
        logging.error(f"Error normalizing date: {date_str}. Error: {e}")
        return date_str