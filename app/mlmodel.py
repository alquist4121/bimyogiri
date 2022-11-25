from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from pprint import pprint

class MLModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self):
        """
        when server is activated, load weight or use joblib or pickle for performance improvement.
        then, assign pretrained model instance to self.model.
        """
        model_name = 'rinna/japanese-gpt2-medium'
        model =  AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.do_lower_case = True

        pprint(model)
        pprint(tokenizer)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
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
                input_sentences = [sentence.text.lower() for sentence in sentences]
                encodings = self.tokenizer.batch_encode_plus(input_sentences, padding=True, return_tensors="pt").to(self.device)
                num_return_sentences = 3

                output = self.model.generate(**encodings,
                    do_sample=True,
                    max_length=100,
                    num_return_sequences=num_return_sentences,
                    top_k=500,
                    top_p=0.95,
                    num_beams=10,
                    repetition_penalty=5.0,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=[[self.tokenizer.unk_token_id]],
                )
            output_sentence_batches = np.array(self.tokenizer.batch_decode(output, skip_special_tokens=True)).reshape(-1, num_return_sentences).tolist()
            pprint(output_sentence_batches)
            outputs = [{'original': input_sentence, 'generated': output_sentences} for input_sentence, output_sentences in zip(input_sentences, output_sentence_batches)]
        except Exception as e:
            print(e)
        return outputs