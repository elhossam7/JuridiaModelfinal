from transformers import AutoTokenizer

class MultilingualTokenizer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str, max_length: int = 512):
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_length, return_tensors='pt')
        return tokens

    def decode(self, tokens):
        text = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        return text

    def batch_tokenize(self, texts: list, max_length: int = 512):
        tokens = self.tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors='pt'
        )
        return tokens['input_ids'], tokens['attention_mask']