import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm
import os
import json
import argparse
from datetime import datetime

def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling on the last hidden state of the model.
    """
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def save_benchmark_result(model_name, dataset_name, metric, score):
    """
    Saves the benchmark result to a timestamped JSON file.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    results_dir = "benchmarks"
    filename = f"{safe_model_name}_{dataset_name.replace('/', '_')}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    result_data = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "evaluation_timestamp_utc": datetime.utcnow().isoformat(),
        "metric": metric,
        "score": score
    }
    
    os.makedirs(results_dir, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(result_data, f, indent=4)
        
    print(f"\nBenchmark results saved to: {filepath}")

def evaluate_stsb_benchmark(model_name_or_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads a model and evaluates its performance on the STS-B benchmark.
    """
    print(f"--- Evaluating Model: {model_name_or_path} on STS-B ---")
    print(f"Using device: {device}")

    try:
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model '{model_name_or_path}': {e}")
        return

    dataset_name = 'glue'
    dataset_subset = 'stsb'
    print(f"Loading {dataset_name}/{dataset_subset} dataset ('validation' split)...")
    dataset = load_dataset(dataset_name, dataset_subset, split='validation')

    all_model_scores = []
    all_gold_scores = []

    print(f"\n--- Generating embeddings for {len(dataset)} sentence pairs ---")
    
    batch_size = 32
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        sentences1 = batch['sentence1']
        sentences2 = batch['sentence2']
        
        encoded_input = tokenizer(
            sentences1 + sentences2, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings1 = embeddings[:len(sentences1)]
        embeddings2 = embeddings[len(sentences1):]
        
        cosine_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)
        
        all_model_scores.extend(cosine_scores.cpu().numpy())
        all_gold_scores.extend(batch['label'])

    spearman_corr, _ = spearmanr(all_model_scores, all_gold_scores)
    
    print("\n--- Evaluation Complete ---")
    print(f"Spearman Correlation on STS-B (validation set): {spearman_corr:.4f}")
    
    save_benchmark_result(
        model_name=model_name_or_path,
        dataset_name=f"{dataset_name}/{dataset_subset}",
        metric="spearman_correlation",
        score=spearman_corr
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a base encoder model on the STS-B benchmark.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="xlm-roberta-base",
        help="The name or local path of the model to evaluate."
    )
    args = parser.parse_args()
    
    evaluate_stsb_benchmark(args.model_name_or_path)