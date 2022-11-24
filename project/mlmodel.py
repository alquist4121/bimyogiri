from random import choice
from time import sleep
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class MLModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self, filepath=''):
        """
        when server is activated, load weight or use joblib or pickle for performance improvement.
        then, assign pretrained model instance to self.model.
        """
        model_name = 'rinna/japanese-gpt2-medium'

        model =  AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        tokenizer.do_lower_case = False
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        print('done loading')
        pass

    def predict(self, sentences):
        """implement followings
        - Load data
        - Preprocess
        - Prediction using self.model
        - Post-process
        """
        try:
            with torch.no_grad():
                input_sentences = [sentence.text for sentence in sentences]
                encodings = self.tokenizer.batch_encode_plus(input_sentences, padding=True, return_tensors="pt").to(self.device)
                output = self.model.generate(**encodings,
                    do_sample=True,
                    max_length=50,
                    num_return_sequences=1,
                    top_k=500,
                    top_p=0.95,
                    num_beams=10,
                    repetition_penalty=10.0,
                    no_repeat_ngram_size=2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=[[self.tokenizer.unk_token_id]], 
                )
            output_sentences = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            print(output_sentences)
            outputs = [{'original': input_sentence, 'output': output_sentence} for input_sentence, output_sentence in zip(input_sentences, output_sentences)]
        except IndexError as e:
            print(e)
        return outputs