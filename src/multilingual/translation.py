# FILE: /juridia-finetuning-project/juridia-finetuning-project/src/multilingual/translation.py

from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, model_name='Helsinki-NLP/opus-mt-fr-ar'):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text, src_lang='fr', target_lang='ar'):
        if src_lang == 'fr' and target_lang == 'ar':
            translated = self._translate_fr_to_ar(text)
        elif src_lang == 'ar' and target_lang == 'fr':
            translated = self._translate_ar_to_fr(text)
        else:
            raise ValueError("Unsupported language pair.")
        return translated

    def _translate_fr_to_ar(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = self.model.generate(**inputs)
        return self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    def _translate_ar_to_fr(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = self.model.generate(**inputs)
        return self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)