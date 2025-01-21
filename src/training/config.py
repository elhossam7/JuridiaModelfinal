# File: /juridia-finetuning-project/juridia-finetuning-project/src/training/config.py

import torch

# Configuration parameters for the training process

class Config:
    # Model hyperparameters
    MODEL_NAME = "juridia-legal-model"
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    EPOCHS = 3
    MAX_SEQ_LENGTH = 512

    # Paths to datasets
    TRAIN_DATA_PATH = "src/data/train.csv"
    TEST_DATA_PATH = "src/data/test.csv"

    # Checkpoint settings
    CHECKPOINT_DIR = "checkpoints/"
    SAVE_STEPS = 500
    LOGGING_STEPS = 100

    # PEFT settings
    USE_LORA = True
    LORA_RANK = 16
    LORA_ALPHA = 32

    # Evaluation settings
    EVAL_STEPS = 200
    EVAL_METRICS = ["relevance", "completeness", "legal_soundness", "fluency", "latency"]

    # Device settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"