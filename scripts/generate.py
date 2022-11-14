from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pprint import pprint
from argparse import ArgumentParser

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--model_dir', type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.do_lower_case = True
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)

    print(args.model_dir)

    prompt = "150歳まで生きたおばあちゃんが遂に亡くなった。"

    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(input_ids, 
            do_sample=True,
            max_length=30,
            num_return_sequences=3,
            top_k=500,
            top_p=0.95,
            num_beams=10,
            repetition_penalty=10.0,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.unk_token_id]], 
        )
        decoded_output = [x.replace('</s>', "") for x in tokenizer.batch_decode(output)]
        pprint(decoded_output)

        with open('output.txt', 'w+') as f:
            f.write("\n".join(decoded_output))


if __name__=="__main__":
    main()