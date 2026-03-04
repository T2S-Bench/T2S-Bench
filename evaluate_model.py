#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciStruct Benchmark Evaluation Script

评估模型在测试集上的表现，支持 HuggingFace 模型和 API 模型（ChatGPT/Gemini等）

用法:
  # HuggingFace 模型
  python evaluate_model.py --test_dir dataset_used_multichoice_test --model_type hf --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --output results_llama.json
  
  # API 模型（ChatGPT/Gemini）
  export OPENAI_API_KEY="..."
  export OPENAI_BASE_URL="..."
  python evaluate_model.py --test_dir dataset_used_multichoice_test --model_type api --model_name gpt-4o --output results_gpt4o.json
"""

import argparse
import json
import os
import re
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# -------------------------
# Parquet data loading (local HuggingFace data/)
# -------------------------
def _parse_mr_question(q_text: str, answer_str: str, question_class: str) -> Dict[str, Any]:
    """Convert parquet MR question fields into the expected question_data dict."""
    parts = re.split(r'\n\s*Options:\s*\n', q_text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        question_text = parts[0].strip()
        options_text = parts[1].strip()
    else:
        question_text = q_text.strip()
        options_text = ''

    options: List[Dict[str, str]] = []
    for m in re.finditer(r'^([A-D])\.\s*(.+?)(?=\n[A-D]\.|$)', options_text, re.MULTILINE | re.DOTALL):
        options.append({'id': m.group(1), 'text': m.group(2).strip()})

    answer = sorted([a.strip() for a in answer_str.split(',') if a.strip() in ('A', 'B', 'C', 'D')])
    qtype = 'multi' if len(answer) > 1 else 'single'

    return {
        'question': question_text,
        'options': options,
        'answer': answer,
        'type': qtype,
        'question_class': question_class,
    }


def load_samples_from_parquet_mr(data_dir: Path) -> List[Dict[str, Any]]:
    """Load MR samples from local parquet file under data_dir/data/."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("pyarrow is required to load parquet data: pip install pyarrow")

    parquet_files = sorted((data_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir / 'data'}")

    samples = []
    for pf in parquet_files:
        table = pq.read_table(str(pf))
        cols = table.schema.names
        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in cols}
            pid = row.get('pid', str(i))
            question_data = _parse_mr_question(
                row.get('question', ''),
                row.get('answer', ''),
                row.get('question_class', 'Unknown'),
            )
            samples.append({
                'sample_name': f"sample_{pid}",
                'major_class': row.get('question_class', 'Unknown'),
                'minor_class': row.get('question_class', 'Unknown'),
                'text': row.get('text', ''),
                'question_data': question_data,
            })
    return samples

import torch
from tqdm import tqdm

# Set multiprocessing start method for CUDA compatibility
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Try import transformers (for HF models)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available. HF models will not work.")

# Try import openai (for API models)
try:
    import openai
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    print("Warning: openai not available. API models will not work.")


# -------------------------
# System Prompt
# -------------------------
SYSTEM_PROMPT = """You are a helpful assistant. Output only the final answer in the required format.
""".strip()


FORMAT_INSTRUCTION = """
Provide your final answer in the following format:
- For single-select: [Answer] A
- For multi-select: [Answer] A,C,D (comma-separated option letters, no spaces)
Output only the final answer with no extra explanation
""".strip()


# -------------------------
# File utils
# -------------------------
def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def find_extracted_md(sample_dir: Path) -> Optional[Path]:
    """Find extracted.md file"""
    cands = sorted(sample_dir.glob("*_extracted.md"))
    return cands[0] if cands else None


# -------------------------
# Model wrappers
# -------------------------
class BaseModel:
    """Base class for models"""
    
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class HFModel(BaseModel):
    """HuggingFace model wrapper"""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not available")
        
        print(f"Loading HF model: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.device = device
        print("Model loaded successfully")
    
    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """Generate response from HF model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and return only the new tokens
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if full_text.startswith(prompt):
            return full_text[len(prompt):].strip()
        return full_text.strip()


class APIModel(BaseModel):
    """API model wrapper (ChatGPT/Gemini etc)"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str):
        if not API_AVAILABLE:
            raise RuntimeError("openai library not available")
        
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        print(f"API model initialized: {model_name}")
    
    def generate(self, prompt: str, max_retries: int = 5) -> str:
        """Generate response from API model with retry on rate limit"""
        import time
        import re as regex

        # Gemma models don't support system prompts
        if "gemma" in self.model_name.lower():
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"}
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower():
                    # Try to extract retry delay from error message
                    match = regex.search(r'retry in (\d+(?:\.\d+)?)', error_str.lower())
                    if match:
                        wait_time = float(match.group(1)) + 1
                    else:
                        wait_time = (2 ** attempt) * 10  # Exponential backoff: 10, 20, 40, 80, 160s

                    print(f"[Rate limit] Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise  # Re-raise non-rate-limit errors

        raise Exception(f"Max retries ({max_retries}) exceeded due to rate limiting")


# -------------------------
# Answer extraction and evaluation
# -------------------------
def extract_answer(response: str) -> Optional[str]:
    """
    Extract answer from model response.
    Looking for pattern: [Answer] A or [Answer] A,C,D
    """
    # Try to find [Answer] pattern
    match = re.search(r'\[Answer\]\s*([A-D,\s]+)', response, re.IGNORECASE)
    if match:
        answer_str = match.group(1).strip()
        # Remove spaces and convert to uppercase
        answer_str = answer_str.replace(" ", "").upper()
        return answer_str

    ## For Gemini 2-5 Flash MultiQA
    # text = response.strip()

    # patterns = [
    #     r'\[Answer\]\s*([A-D](?:\s*,\s*[A-D])*)',
    #     r'【Answer】\s*([A-D](?:\s*,\s*[A-D])*)',
    #     r'(?i)\bfinal\s*answer\b\s*[:：]?\s*([A-D](?:\s*,\s*[A-D])*)',
    #     r'(?i)\banswer\b\s*[:：]?\s*([A-D](?:\s*,\s*[A-D])*)',
    #     r'(?m)^\s*([A-D](?:\s*,\s*[A-D])*)\s*$',
    # ]

    # for pat in patterns:
    #     m = re.search(pat, text)
    #     if m:
    #         ans = m.group(1).replace(" ", "").upper()
    #         ans = re.sub(r'[^A-D,]', '', ans)
    #         return ans
    
    return None


def parse_answer(answer_str: str) -> List[str]:
    """Parse answer string to list of options"""
    if not answer_str:
        return []
    
    # Split by comma and filter
    options = [opt.strip() for opt in answer_str.split(",")]
    options = [opt for opt in options if opt in ["A", "B", "C", "D"]]
    return sorted(options)


def print_sample_result(result: Dict[str, Any], use_tqdm_write: bool = False) -> None:
    """Print result for a single sample"""
    em = result['em']
    correct_mark = "✅" if em == 1.0 else "❌"
    
    output_lines = [
        f"\n{correct_mark} [{result['sample_name']}]",
        f"  Question Type: {result['question_type']}",
        f"  Gold: {result['gold_answer']}, Pred: {result['pred_answer']}",
        f"  EM: {em:.2f}, F1: {result['f1']:.2f}",
    ]
    
    response = result['response']
    if len(response) > 500:
        output_lines.append(f"  Response: {response[:500]}...")
    else:
        output_lines.append(f"  Response: {response}")
    
    output = "\n".join(output_lines)
    
    if use_tqdm_write:
        tqdm.write(output)
    else:
        print(output)


def compute_em(pred: List[str], gold: List[str]) -> float:
    """Compute Exact Match"""
    return 1.0 if pred == gold else 0.0


def compute_f1(pred: List[str], gold: List[str]) -> float:
    """Compute F1 score"""
    if len(pred) == 0 and len(gold) == 0:
        return 1.0
    if len(pred) == 0 or len(gold) == 0:
        return 0.0
    
    pred_set = set(pred)
    gold_set = set(gold)
    
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# -------------------------
# Extract metadata from folder name
# -------------------------
def extract_major_class(folder_name: str) -> str:
    """Extract major class (e.g., CS, Life-Science)"""
    # Get first component before underscore
    parts = folder_name.split("_")
    if parts:
        major = parts[0]
        # Handle Life-Sciences vs Life-Science
        if major == "Life-Sciences":
            major = "Life-Science"
        return major
    return "Unknown"


def extract_minor_class(folder_name: str) -> str:
    """Extract minor class (first two components)"""
    parts = folder_name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return folder_name


# -------------------------
# Build prompt
# -------------------------
def build_test_prompt(extracted_text: str, question_data: Dict[str, Any]) -> str:
    """Build the full prompt for testing"""
    question_text = question_data.get("question", "")
    options = question_data.get("options", [])
    question_type = question_data.get("type", "single")
    
    # Format options
    options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in options])
    
    # Construct prompt
    prompt = f"""TEXT PARAGRAPH:
{extracted_text}

QUESTION:
{question_text}

OPTIONS:
{options_text}

{FORMAT_INSTRUCTION}
"""
    
    return prompt.strip()


# -------------------------
# Global model for multiprocessing (HF models)
# -------------------------
_global_model = None


def init_worker_hf(model_name_or_path: str, device: str, gpu_id: int):
    """Initialize HF model in worker process with specific GPU"""
    global _global_model
    
    # Set CUDA_VISIBLE_DEVICES for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"[Worker] Initializing on GPU {gpu_id} (PID: {os.getpid()})")
    _global_model = HFModel(model_name_or_path, device)


def process_single_sample_hf(folder: Path) -> Optional[Dict[str, Any]]:
    """Process a single test sample using global HF model. For multiprocessing."""
    global _global_model
    
    if _global_model is None:
        return None
    
    folder_name = folder.name
    info_path = folder / "information.json"
    
    if not info_path.exists():
        return None
    
    try:
        info = load_json(info_path)
        question_data = info.get("question")
        
        if not question_data:
            return None
        
        # Find extracted text
        md_path = find_extracted_md(folder)
        if not md_path:
            return None
        
        extracted_text = read_text(md_path)
        
        # Build prompt
        prompt = build_test_prompt(extracted_text, question_data)
        
        # Generate answer using global model
        response = _global_model.generate(prompt)
        
        # Extract answer from response
        pred_answer_str = extract_answer(response)
        pred_answer = parse_answer(pred_answer_str) if pred_answer_str else []
        
        # Get gold answer
        gold_answer = question_data.get("answer", [])
        if isinstance(gold_answer, str):
            gold_answer = [gold_answer]
        gold_answer = sorted(gold_answer)
        
        # Compute metrics
        em = compute_em(pred_answer, gold_answer)
        f1 = compute_f1(pred_answer, gold_answer)
        
        # Extract metadata
        major_class = extract_major_class(folder_name)
        minor_class = extract_minor_class(folder_name)
        question_type = question_data.get("question_class", "Unknown")
        
        # Get GPU ID from environment
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
        
        # Store result
        result = {
            "sample_name": folder_name,
            "major_class": major_class,
            "minor_class": minor_class,
            "question_type": question_type,
            "question": question_data.get("question", ""),
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "pred_answer_raw": pred_answer_str,
            "em": em,
            "f1": f1,
            "response": response,
            "gpu_id": gpu_id,
        }
        
        # Print immediately for GPU 0
        if gpu_id == "0":
            correct_mark = "✅" if em == 1.0 else "❌"
            print(f"\n[GPU 0] {correct_mark} {folder_name}")
            print(f"  Type: {question_type} | Gold: {gold_answer} | Pred: {pred_answer} | EM: {em:.2f}")
            print(f"  Response: {response[:300]}..." if len(response) > 300 else f"  Response: {response}")
            print(flush=True)
        
        return result
    
    except Exception as e:
        print(f"[ERROR] {folder_name}: {e}", flush=True)
        return None


# -------------------------
# Process single sample (for parallel processing with API models)
# -------------------------
def process_single_sample(args_tuple) -> Optional[Dict[str, Any]]:
    """Process a single test sample. Used for parallel processing with API models."""
    folder, model, verbose = args_tuple
    folder_name = folder.name
    info_path = folder / "information.json"
    
    if not info_path.exists():
        return None
    
    try:
        info = load_json(info_path)
        question_data = info.get("question")
        
        if not question_data:
            return None
        
        # Find extracted text
        md_path = find_extracted_md(folder)
        if not md_path:
            return None
        
        extracted_text = read_text(md_path)
        
        # Build prompt
        prompt = build_test_prompt(extracted_text, question_data)
        
        # Generate answer
        response = model.generate(prompt)
        
        # Extract answer from response
        pred_answer_str = extract_answer(response)
        pred_answer = parse_answer(pred_answer_str) if pred_answer_str else []
        
        # Get gold answer
        gold_answer = question_data.get("answer", [])
        if isinstance(gold_answer, str):
            gold_answer = [gold_answer]
        gold_answer = sorted(gold_answer)
        
        # Compute metrics
        em = compute_em(pred_answer, gold_answer)
        f1 = compute_f1(pred_answer, gold_answer)
        
        # Extract metadata
        major_class = extract_major_class(folder_name)
        minor_class = extract_minor_class(folder_name)
        question_type = question_data.get("question_class", "Unknown")
        
        # Store result
        result = {
            "sample_name": folder_name,
            "major_class": major_class,
            "minor_class": minor_class,
            "question_type": question_type,
            "question": question_data.get("question", ""),
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "pred_answer_raw": pred_answer_str,
            "em": em,
            "f1": f1,
            "response": response,
        }
        
        return result
    
    except Exception as e:
        print(f"[ERROR] {folder_name}: {e}")
        return None


# -------------------------
# Process dict-based sample (for parquet data)
# -------------------------
def process_sample_from_dict(args_tuple) -> Optional[Dict[str, Any]]:
    """Process a single sample from a pre-loaded dict (parquet data). Used for API models."""
    sample_dict, model, verbose = args_tuple
    try:
        question_data = sample_dict['question_data']
        extracted_text = sample_dict['text']
        sample_name = sample_dict['sample_name']

        prompt = build_test_prompt(extracted_text, question_data)
        response = model.generate(prompt)

        pred_answer_str = extract_answer(response)
        pred_answer = parse_answer(pred_answer_str) if pred_answer_str else []

        gold_answer = question_data.get("answer", [])
        if isinstance(gold_answer, str):
            gold_answer = [gold_answer]
        gold_answer = sorted(gold_answer)

        em = compute_em(pred_answer, gold_answer)
        f1 = compute_f1(pred_answer, gold_answer)

        return {
            "sample_name": sample_name,
            "major_class": sample_dict.get('major_class', 'Unknown'),
            "minor_class": sample_dict.get('minor_class', 'Unknown'),
            "question_type": question_data.get("question_class", "Unknown"),
            "question": question_data.get("question", ""),
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "pred_answer_raw": pred_answer_str,
            "em": em,
            "f1": f1,
            "response": response,
        }
    except Exception as e:
        print(f"[ERROR] {sample_dict.get('sample_name', '?')}: {e}")
        return None


def evaluate_model_from_samples(
    model: BaseModel,
    samples: List[Dict[str, Any]],
    num_workers: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate model on pre-loaded samples (from parquet data)."""
    print(f"Found {len(samples)} samples")
    print(f"Using {num_workers} worker(s) for parallel processing")
    print("=" * 80)

    all_results = []

    if num_workers == 1:
        worker_args = [(s, model, verbose) for s in samples]
        for args in tqdm(worker_args, desc="Evaluating"):
            result = process_sample_from_dict(args)
            if result:
                all_results.append(result)
                if verbose:
                    print_sample_result(result, use_tqdm_write=True)
    else:
        worker_args = [(s, model, verbose) for s in samples]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_sample_from_dict, args) for args in worker_args]
            completed = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                        completed += 1
                        if verbose:
                            tqdm.write(f"\n{'='*80}")
                            tqdm.write(f"Sample {completed}/{len(samples)}")
                            print_sample_result(result, use_tqdm_write=True)
                            tqdm.write(f"{'='*80}")
                except Exception as e:
                    tqdm.write(f"[ERROR] Future failed: {e}")

    return _aggregate_results(all_results)


def _aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-sample results into overall metrics."""
    num_samples = len(all_results)
    if num_samples == 0:
        print("No samples evaluated!")
        return {}

    stats_by_major = defaultdict(lambda: {"em": [], "f1": []})
    stats_by_minor = defaultdict(lambda: {"em": [], "f1": []})
    stats_by_question_type = defaultdict(lambda: {"em": [], "f1": []})

    for result in all_results:
        major = result["major_class"]
        minor = result["minor_class"]
        qtype = result["question_type"]
        em = result["em"]
        f1 = result["f1"]
        stats_by_major[major]["em"].append(em)
        stats_by_major[major]["f1"].append(f1)
        stats_by_minor[minor]["em"].append(em)
        stats_by_minor[minor]["f1"].append(f1)
        stats_by_question_type[qtype]["em"].append(em)
        stats_by_question_type[qtype]["f1"].append(f1)

    overall_em = sum(r["em"] for r in all_results) / num_samples
    overall_f1 = sum(r["f1"] for r in all_results) / num_samples

    results = {
        "overall": {"em": overall_em, "f1": overall_f1, "num_samples": num_samples},
        "by_major_class": {},
        "by_minor_class": {},
        "by_question_type": {},
        "all_samples": all_results,
    }
    for major, stats in stats_by_major.items():
        results["by_major_class"][major] = {
            "em": sum(stats["em"]) / len(stats["em"]),
            "f1": sum(stats["f1"]) / len(stats["f1"]),
            "num_samples": len(stats["em"]),
        }
    for minor, stats in stats_by_minor.items():
        results["by_minor_class"][minor] = {
            "em": sum(stats["em"]) / len(stats["em"]),
            "f1": sum(stats["f1"]) / len(stats["f1"]),
            "num_samples": len(stats["em"]),
        }
    for qtype, stats in stats_by_question_type.items():
        results["by_question_type"][qtype] = {
            "em": sum(stats["em"]) / len(stats["em"]),
            "f1": sum(stats["f1"]) / len(stats["f1"]),
            "num_samples": len(stats["em"]),
        }
    return results


# -------------------------
# Evaluation
# -------------------------
def evaluate_model(
    model: BaseModel,
    test_dir: Path,
    num_workers: int = 1,
    gpu_ids: Optional[List[int]] = None,
    model_type: str = "api",
    model_name_or_path: str = "",
    device: str = "cuda",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Args:
        model: Model instance (for API models or single-process HF)
        test_dir: Test directory path
        num_workers: Number of parallel workers
        gpu_ids: List of GPU IDs to use for HF models (e.g., [0, 1, 2, 3])
        model_type: "api" or "hf"
        model_name_or_path: HF model path (for multi-GPU)
        device: Device for HF model
        verbose: Print detailed results
        
    Returns:
        Detailed evaluation results
    """
    
    # Get all test folders
    test_folders = sorted([p for p in test_dir.iterdir() if p.is_dir()])
    
    print(f"Found {len(test_folders)} test samples")
    print(f"Using {num_workers} worker(s) for parallel processing")
    if gpu_ids and model_type == "hf":
        print(f"GPUs: {gpu_ids}")
    print("=" * 80)
    
    # Store all results
    all_results = []
    
    # Process samples
    if num_workers == 1 or (model_type == "hf" and not gpu_ids):
        # Sequential processing (original behavior)
        worker_args = [(folder, model, verbose) for folder in test_folders]
        for args in tqdm(worker_args, desc="Evaluating"):
            result = process_single_sample(args)
            if result:
                all_results.append(result)
                if verbose:
                    print_sample_result(result, use_tqdm_write=True)
    
    elif model_type == "api":
        # Parallel API calls using ThreadPool
        worker_args = [(folder, model, verbose) for folder in test_folders]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_sample, args) for args in worker_args]
            
            completed = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                        completed += 1
                        if verbose:
                            tqdm.write(f"\n{'='*80}")
                            tqdm.write(f"Sample {completed}/{len(test_folders)}")
                            print_sample_result(result, use_tqdm_write=True)
                            tqdm.write(f"{'='*80}")
                except Exception as e:
                    tqdm.write(f"[ERROR] Future failed: {e}")
    
    else:  # model_type == "hf" and gpu_ids provided
        # Multi-GPU HF model processing using separate pools per GPU
        print(f"Multi-GPU mode: distributing work across {len(gpu_ids)} GPUs")
        
        # Distribute folders across GPUs
        gpu_work_queues = [[] for _ in range(len(gpu_ids))]
        for idx, folder in enumerate(test_folders):
            gpu_idx = idx % len(gpu_ids)
            gpu_work_queues[gpu_idx].append(folder)
        
        print("Work distribution:")
        for gpu_idx, work_queue in enumerate(gpu_work_queues):
            print(f"  GPU {gpu_ids[gpu_idx]}: {len(work_queue)} samples")
        print()
        
        # Create separate process pool for each GPU
        from multiprocessing import get_context
        import threading
        import queue
        ctx = get_context('spawn')
        
        pools = []
        result_queue = queue.Queue()
        
        def collect_gpu_results(pool, work_queue, gpu_id):
            """Collect results from a single GPU pool in a separate thread"""
            try:
                # Use imap_unordered to process tasks asynchronously
                for result in pool.imap_unordered(process_single_sample_hf, work_queue):
                    result_queue.put((result, gpu_id))
            except Exception as e:
                tqdm.write(f"[ERROR] GPU {gpu_id} collection thread failed: {e}")
        
        # Start process pools and collection threads for each GPU
        collection_threads = []
        for gpu_idx, work_queue in enumerate(gpu_work_queues):
            if len(work_queue) == 0:
                continue
            
            gpu_id = gpu_ids[gpu_idx]
            
            # Create a pool with 1 worker for this GPU
            pool = ctx.Pool(
                processes=1,
                initializer=init_worker_hf,
                initargs=(model_name_or_path, device, gpu_id)
            )
            pools.append(pool)
            
            # Start a thread to collect results from this GPU
            thread = threading.Thread(
                target=collect_gpu_results,
                args=(pool, work_queue, gpu_id),
                daemon=True
            )
            thread.start()
            collection_threads.append(thread)
        
        # Collect results from all GPUs in parallel
        completed = 0
        with tqdm(total=len(test_folders), desc="Evaluating") as pbar:
            while completed < len(test_folders):
                try:
                    result, gpu_id = result_queue.get(timeout=1.0)
                    if result:
                        all_results.append(result)
                        completed += 1
                        if verbose:
                            # Use tqdm.write to print without breaking progress bar
                            tqdm.write(f"\n{'='*80}")
                            tqdm.write(f"Sample {completed}/{len(test_folders)} [GPU {gpu_id}]")
                            print_sample_result(result, use_tqdm_write=True)
                            tqdm.write(f"{'='*80}")
                    pbar.update(1)
                except queue.Empty:
                    # Check if all threads are done
                    if all(not thread.is_alive() for thread in collection_threads):
                        break
                    continue
        
        # Clean up pools
        for pool in pools:
            pool.close()
            pool.join()
        
        # Wait for all collection threads to finish
        for thread in collection_threads:
            thread.join(timeout=5.0)
    
    # Compute overall metrics
    num_samples = len(all_results)
    if num_samples == 0:
        print("No samples evaluated!")
        return {}
    
    # Aggregate statistics
    stats_by_major = defaultdict(lambda: {"em": [], "f1": []})
    stats_by_minor = defaultdict(lambda: {"em": [], "f1": []})
    stats_by_question_type = defaultdict(lambda: {"em": [], "f1": []})
    
    for result in all_results:
        major = result["major_class"]
        minor = result["minor_class"]
        qtype = result["question_type"]
        em = result["em"]
        f1 = result["f1"]
        
        stats_by_major[major]["em"].append(em)
        stats_by_major[major]["f1"].append(f1)
        
        stats_by_minor[minor]["em"].append(em)
        stats_by_minor[minor]["f1"].append(f1)
        
        stats_by_question_type[qtype]["em"].append(em)
        stats_by_question_type[qtype]["f1"].append(f1)
    
    overall_em = sum(r["em"] for r in all_results) / num_samples
    overall_f1 = sum(r["f1"] for r in all_results) / num_samples
    
    # Aggregate results
    results = {
        "overall": {
            "em": overall_em,
            "f1": overall_f1,
            "num_samples": num_samples,
        },
        "by_major_class": {},
        "by_minor_class": {},
        "by_question_type": {},
        "all_samples": all_results,
    }
    
    # By major class
    for major, stats in stats_by_major.items():
        results["by_major_class"][major] = {
            "em": sum(stats["em"]) / len(stats["em"]),
            "f1": sum(stats["f1"]) / len(stats["f1"]),
            "num_samples": len(stats["em"]),
        }
    
    # By minor class
    for minor, stats in stats_by_minor.items():
        results["by_minor_class"][minor] = {
            "em": sum(stats["em"]) / len(stats["em"]),
            "f1": sum(stats["f1"]) / len(stats["f1"]),
            "num_samples": len(stats["em"]),
        }
    
    # By question type
    for qtype, stats in stats_by_question_type.items():
        results["by_question_type"][qtype] = {
            "em": sum(stats["em"]) / len(stats["em"]),
            "f1": sum(stats["f1"]) / len(stats["f1"]),
            "num_samples": len(stats["em"]),
        }
    
    return results


# -------------------------
# Print results as table
# -------------------------
def print_results_table(results: Dict[str, Any]):
    """Print evaluation results in table format"""
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Overall
    print("\n【Overall Performance】")
    print("-" * 80)
    overall = results["overall"]
    print(f"{'Metric':<20} {'Value':<15} {'Num Samples':<15}")
    print(f"{'EM':<20} {overall['em']:.4f} ({overall['em']*100:.2f}%)   {overall['num_samples']}")
    print(f"{'F1':<20} {overall['f1']:.4f} ({overall['f1']*100:.2f}%)   {overall['num_samples']}")
    
    # By major class
    print("\n【Performance by Major Class】")
    print("-" * 80)
    print(f"{'Major Class':<30} {'EM':<15} {'F1':<15} {'Num':<10}")
    print("-" * 80)
    for major, stats in sorted(results["by_major_class"].items()):
        print(f"{major:<30} {stats['em']:.4f} ({stats['em']*100:5.1f}%)  {stats['f1']:.4f} ({stats['f1']*100:5.1f}%)  {stats['num_samples']:<10}")
    
    # By question type
    print("\n【Performance by Question Type】")
    print("-" * 80)
    print(f"{'Question Type':<40} {'EM':<15} {'F1':<15} {'Num':<10}")
    print("-" * 80)
    for qtype, stats in sorted(results["by_question_type"].items()):
        print(f"{qtype:<40} {stats['em']:.4f} ({stats['em']*100:5.1f}%)  {stats['f1']:.4f} ({stats['f1']*100:5.1f}%)  {stats['num_samples']:<10}")
    
    # By minor class (top 10)
    print("\n【Performance by Minor Class (Top 10 by sample count)】")
    print("-" * 80)
    print(f"{'Minor Class':<60} {'EM':<10} {'F1':<10} {'Num':<10}")
    print("-" * 80)
    
    sorted_minor = sorted(
        results["by_minor_class"].items(),
        key=lambda x: x[1]["num_samples"],
        reverse=True
    )
    
    for minor, stats in sorted_minor[:10]:
        print(f"{minor:<60} {stats['em']:.3f}    {stats['f1']:.3f}    {stats['num_samples']:<10}")
    
    if len(sorted_minor) > 10:
        print(f"  ... and {len(sorted_minor) - 10} more minor classes")
    
    print("\n" + "=" * 80)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate model on SciStruct benchmark")

    # Data source: either --test_dir (folder-based) or --data_dir (local HF parquet)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--test_dir", default="", help="Test dataset directory (folder-based format)")
    data_group.add_argument("--data_dir", default="", help="Local HuggingFace data directory (e.g. data/T2S-Bench-MR)")

    parser.add_argument("--model_type", required=True, choices=["hf", "api"], help="Model type: hf (HuggingFace) or api (ChatGPT/Gemini)")

    # HF model args
    parser.add_argument("--model_name_or_path", default="", help="HF model name or path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for HF model")
    parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated GPU IDs for multi-GPU HF inference (e.g., '0,1,2,3')")

    # API model args
    parser.add_argument("--model_name", default="", help="API model name (e.g., gpt-4o, gemini-2.5-pro)")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""), help="API key")
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", ""), help="API base URL")

    # Output
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results for each sample")

    # Parallel processing
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers (recommended: 4-8 for API models, 1 for HF models)")

    args = parser.parse_args()

    # Parse GPU IDs if provided
    gpu_ids = None
    if args.gpu_ids.strip():
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        print(f"Using GPUs: {gpu_ids}")

    # Load model
    model = None
    if args.model_type == "hf":
        if not args.model_name_or_path:
            raise ValueError("--model_name_or_path required for HF models")

        # For multi-GPU, model will be loaded in each worker
        if not gpu_ids or len(gpu_ids) == 1:
            model = HFModel(args.model_name_or_path, args.device)
        else:
            print(f"Multi-GPU mode: model will be loaded in {len(gpu_ids)} worker processes")
    else:  # api
        if not args.model_name:
            raise ValueError("--model_name required for API models")
        if not args.api_key or not args.base_url:
            raise ValueError("--api_key and --base_url required for API models")
        model = APIModel(args.model_name, args.api_key, args.base_url)

    # Evaluate
    if args.data_dir:
        # Load from local HuggingFace parquet data
        data_dir = Path(args.data_dir)
        print(f"Loading data from parquet: {data_dir}")
        samples = load_samples_from_parquet_mr(data_dir)
        results = evaluate_model_from_samples(
            model,
            samples,
            num_workers=args.num_workers,
            verbose=args.verbose,
        )
    else:
        test_dir = Path(args.test_dir)
        results = evaluate_model(
            model,
            test_dir,
            num_workers=args.num_workers if not gpu_ids else len(gpu_ids),
            gpu_ids=gpu_ids,
            model_type=args.model_type,
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            verbose=args.verbose
        )

    # Print results table
    print_results_table(results)

    # Save results
    output_path = Path(args.output)
    save_json(output_path, results)
    print(f"\n✅ Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
