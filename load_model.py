from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
import os

token = os.getenv("HF_TOKEN", None)
if token is None:
	model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
	tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
else:
    print("Warning: HF_TOKEN environment variable not set.")

model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

# Save the model and tokenizer locally
output_dir = "saved_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
