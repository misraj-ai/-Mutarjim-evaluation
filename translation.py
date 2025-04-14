# @title translatioin script
import argparse
import json
import os
import time
import logging
import pandas as pd
import torch
import sentencepiece as spm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vllm import LLM, SamplingParams
from sacrebleu import corpus_bleu, CHRF, TER
from rouge import Rouge
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from evaluate import load
from comet import download_model, load_from_checkpoint

# Environment & Logging Setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Pretrained Models
sp = spm.SentencePieceProcessor(model_file='flores200_sacrebleu_tokenizer_spm.model')
bleurt = load("bleurt")
comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

class TranslationDataset(Dataset):
    def __init__(self, data_path, input_column, output_column):
        df = pd.read_csv(data_path, usecols=[input_column, output_column])
        self.inputs = df[input_column].tolist()
        self.outputs = df[output_column].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def translate_batch(sentences, llm, sampling_params, system_prompt):
    prompts = [system_prompt.format(text=sent) if system_prompt else f'{sent}\n' for sent in sentences]
    return [output.outputs[0].text.strip() for output in llm.generate(prompts, sampling_params)]

def compute_metrics(metric, references, hypotheses, sources, lang):
    metrics_map = {
        "bleu": lambda: corpus_bleu(hypotheses, [references]).score,
        "rouge-1": lambda: Rouge().get_scores(hypotheses, references, avg=True)['rouge-1']['f'],
        "rouge-2": lambda: Rouge().get_scores(hypotheses, references, avg=True)['rouge-2']['f'],
        "rouge-l": lambda: Rouge().get_scores(hypotheses, references, avg=True)['rouge-l']['f'],
        "meteor": lambda: sum(meteor_score([ref.split()], hyp.split()) for ref, hyp in zip(references, hypotheses)) / len(hypotheses),
        "bleurt": lambda: sum(bleurt.compute(predictions=hypotheses, references=references)["scores"]) / len(hypotheses),
        "chrf": lambda: CHRF().corpus_score(hypotheses, [references]).score,
        "bert": lambda: bert_score(hypotheses, references, lang=lang, device='cuda' if torch.cuda.is_available() else 'cpu')[2].mean().item(),
        "ter": lambda: TER().corpus_score(hypotheses, [references]).score,
        "comet": lambda: sum(comet_model.predict([{ "src": src, "ref": ref, "mt": mt } for src, ref, mt in zip(sources, references, hypotheses)], batch_size=8, gpus=1 if torch.cuda.is_available() else 0)["scores"]) / len(hypotheses),
        "spbleu": lambda: corpus_bleu([' '.join(sp.encode(hyp, out_type=str)) for hyp in hypotheses], [[' '.join(sp.encode(ref, out_type=str))] for ref in references]).score,
    }
    return metrics_map.get(metric, lambda: None)()

def evaluate_translations(model_name, data_name, references, hypotheses, sources, lang):
    logging.info("Computing evaluation metrics...")
    metrics = ["bleu", "rouge-1", "rouge-2", "rouge-l", "meteor", "bleurt", "chrf", "bert", "ter", "comet", "spbleu"]
    return {"model": model_name, "data": data_name, **{metric: compute_metrics(metric, references, hypotheses, sources, lang) for metric in metrics}}

def main():
    parser = argparse.ArgumentParser(description='Evaluate translation model with various metrics')
    parser.add_argument('--model', required=True, help='Model name or path')
    parser.add_argument('--data_path', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--input_column', required=True, help='Input text column name')
    parser.add_argument('--output_column', required=True, help='Reference translation column')
    parser.add_argument('--output_dir', required=True, help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for translation')
    parser.add_argument('--system_prompt', default=None, help='System prompt for translation')
    parser.add_argument('--lang', default='en', help='Language code for BERTScore')
    parser.add_argument('--stop_token', default='</s>', help='Stop token for translation')
    parser.add_argument('--max_tokens', type=int, default=256, help='Maximum number of tokens')
    parser.add_argument('--min_tokens', type=int, default=5, help='Minimum number of tokens')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.model.split('/')[-1]}.csv")
    checkpoint_file = os.path.join(args.output_dir, "checkpoint.json")
    results_file = os.path.join(args.output_dir, "results.json")

    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = TranslationDataset(args.data_path, args.input_column, args.output_column)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    llm = LLM(model=args.model, tensor_parallel_size=1)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        stop=[args.stop_token]
    )
    
    translations = []
    with torch.no_grad():
        
        for i, (inputs, _) in enumerate(tqdm(dataloader, desc="Processing batches", leave=True, dynamic_ncols=True)):
            translated_batch = translate_batch(inputs, llm, sampling_params, args.system_prompt)

            translations.extend(translated_batch)
            pd.DataFrame({
                args.input_column: dataset.inputs[:len(translations)],
                args.output_column: dataset.outputs[:len(translations)],
                'Translated': translations
            }).to_csv(output_file, index=False)
            with open(checkpoint_file, 'w') as f:
                json.dump({'last_idx': len(translations)}, f)
    
    logging.info("Translation and evaluation completed.")

    results = evaluate_translations(args.model, args.data_path, dataset.outputs[:len(translations)], translations, dataset.inputs[:len(translations)], args.lang)
    results['system_prompt'] = args.system_prompt
    results['stop_token'] =  args.stop_token
    results['data_size'] = len(translations)
    results['time_taken'] = time.time() - start_time

    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info("Translation and evaluation completed.")

if __name__ == "__main__":
    main()