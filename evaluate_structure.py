#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate structure extraction against key_frame.

Two-stage evaluation:
1. Stage 1 (Node Labeling): Given graph structure (links), model generates label for each node -> compute average SBERT similarity
2. Stage 2 (Link Extraction): Given text + GT nodes, model extracts links -> compute Link F1

Metrics:
- Node Avg Similarity: Average SBERT semantic similarity between predicted and GT node labels (direct correspondence by ID)
- Link F1: Connection-based F1 (ignoring link labels, only checking if connection exists)

Usage:
  # API model
  export OPENAI_API_KEY="..."
  export OPENAI_BASE_URL="..."
  python evaluate_structure.py --dataset_dir dataset_used_score1 --model_type api --model_name gpt-4o --output results_struct.json

  # HF model
  python evaluate_structure.py --dataset_dir dataset_used_score1 --model_type hf --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --output results_struct.json
"""

import argparse
import json
import os
import re
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# -------------------------
# Parquet data loading (local HuggingFace data/)
# -------------------------
def load_samples_from_parquet_e2e(data_dir: Path) -> List[Dict[str, Any]]:
    """Load E2E samples from local parquet file under data_dir/data/."""
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
            key_frame_raw = row.get('key_frame', '{}')
            if isinstance(key_frame_raw, str):
                try:
                    key_frame = json.loads(key_frame_raw)
                except Exception:
                    key_frame = {}
            else:
                key_frame = key_frame_raw or {}
            samples.append({
                'sample_name': f"sample_{pid}",
                'major_class': row.get('topic', 'Unknown'),
                'text': row.get('text', ''),
                'key_frame': key_frame,
            })
    return samples

import numpy as np
from tqdm import tqdm

# Try import transformers (for HF models)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Try import OpenAI SDK (for API models)
try:
    # new style
    from openai import OpenAI
    API_AVAILABLE = True
    OPENAI_SDK_NEW = True
except Exception:
    try:
        import openai  # old style still provides openai.OpenAI in many versions
        API_AVAILABLE = True
        OPENAI_SDK_NEW = False
    except Exception:
        API_AVAILABLE = False
        OPENAI_SDK_NEW = False

# Optional: httpx for fine-grained timeouts (recommended)
try:
    import httpx
    HTTPX_AVAILABLE = True
except Exception:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

# Try import sentence-transformers (for SBERT)
try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# Try import scipy for Hungarian
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


SYSTEM_PROMPT_NODES = (
    "You are a helpful assistant. You will be given a text and a graph structure (links between nodes). "
    "Your task is to read the text and provide a concise label (single word or short phrase) for each node "
    "that best represents its meaning based on the text content. Output only valid JSON with no extra text. "
    "Put your JSON after [Nodes]."
)

FORMAT_INSTRUCTION_NODES = """
Output format: Put the JSON after [Nodes]
[Nodes]
{
  "nodes": [
    { "id": "n1", "label": "Your Label for n1" },
    { "id": "n2", "label": "Your Label for n2" }
  ]
}
""".strip()

SYSTEM_PROMPT_LINKS = (
    "You are a helpful assistant. Given the text and a list of nodes, identify the relationships "
    "(links/edges) between these nodes based on the text. Each link should connect two nodes and "
    "optionally have a label describing the relationship. Output only valid JSON with no extra text. "
    "Put your JSON after [Links]."
)

FORMAT_INSTRUCTION_LINKS = """
Output format: Put the JSON after [Links]
[Links]
{
  "links": [
    { "source": "n1", "target": "n2", "label": "Link Label or empty" }
  ]
}
""".strip()

# -------------------------
# Provider presets (OpenAI-compatible)
# -------------------------
PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    # Alibaba Cloud Model Studio / DashScope (China)
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    # Alibaba Cloud Model Studio / DashScope (US, Virginia)
    "dashscope_us": {
        "base_url": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    # DeepSeek official (OpenAI-compatible /v1)
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    # Z.AI (GLM)
    "zai": {
        "base_url": "https://api.z.ai/api/paas/v4",
        "api_key_env": "ZAI_API_KEY",
    },
    # OpenRouter
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    # Mistral (if using their OpenAI-compatible endpoint)
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
    },
    # Moonshot (Kimi)
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    # MiniMax (varies by account; keep as placeholder)
    "minimax": {
        "base_url": os.getenv("MINIMAX_BASE_URL", ""),
        "api_key_env": "MINIMAX_API_KEY",
    },
    # Custom: user provides --base_url and --api_key (or env)
    "custom": {
        "base_url": "",
        "api_key_env": "OPENAI_API_KEY",
    },
}


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
    cands = sorted(sample_dir.glob("*_extracted.md"))
    return cands[0] if cands else None


# -------------------------
# Model wrappers
# -------------------------
class BaseModel:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class HFModel(BaseModel):
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not available")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.device = device

    def generate(self, prompt: str, system_prompt: str = None, max_new_tokens: int = 1024) -> str:
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT_NODES
        full_prompt = system_prompt + "\n\n" + prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if full_text.startswith(full_prompt):
            return full_text[len(full_prompt):].strip()
        return full_text.strip()


class APIModel(BaseModel):
    """
    OpenAI-compatible API wrapper with:
    - optional streaming (required for many DashScope/Qwen3 endpoints)
    - DashScope thinking passthrough via extra_body (enable_thinking, thinking_budget)
    - DeepSeek reasoning_content fallback when content is empty
    - read timeouts + retry/backoff so evaluation doesn't hang forever
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        *,
        stream: Optional[bool] = None,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        stream_read_timeout_s: Optional[float] = None,
        nonstream_read_timeout_s: Optional[float] = None,
        connect_timeout_s: Optional[float] = None,
        write_timeout_s: Optional[float] = None,
        pool_timeout_s: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        if not API_AVAILABLE:
            raise RuntimeError("openai sdk not available (pip install openai)")

        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)

        self.is_dashscope = self._looks_like_dashscope(base_url)
        self.is_deepseek = self._looks_like_deepseek(base_url)

        if stream is None:
            stream = True if self._is_thinking_like_model(model_name) else False
        self.stream = bool(stream)

        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget

        if self.is_dashscope and self._is_thinking_like_model(model_name):
            if self.enable_thinking is None:
                self.enable_thinking = True
            # DashScope commonly requires thinking params via streaming.
            self.stream = True

        self.extra_headers = extra_headers or {}

        # Timeouts (env defaults)
        def _env_float(name: str, default: str) -> float:
            try:
                return float(os.getenv(name, default))
            except Exception:
                return float(default)

        stream_read_timeout_s = float(stream_read_timeout_s) if stream_read_timeout_s is not None else _env_float("STREAM_READ_TIMEOUT_S", "120")
        nonstream_read_timeout_s = float(nonstream_read_timeout_s) if nonstream_read_timeout_s is not None else _env_float("NONSTREAM_READ_TIMEOUT_S", "600")
        connect_timeout_s = float(connect_timeout_s) if connect_timeout_s is not None else _env_float("CONNECT_TIMEOUT_S", "10")
        write_timeout_s = float(write_timeout_s) if write_timeout_s is not None else _env_float("WRITE_TIMEOUT_S", "30")
        pool_timeout_s = float(pool_timeout_s) if pool_timeout_s is not None else _env_float("POOL_TIMEOUT_S", "30")

        self._timeout_stream = None
        self._timeout_nonstream = None
        if HTTPX_AVAILABLE:
            self._timeout_stream = httpx.Timeout(connect=connect_timeout_s, read=stream_read_timeout_s, write=write_timeout_s, pool=pool_timeout_s)
            self._timeout_nonstream = httpx.Timeout(connect=connect_timeout_s, read=nonstream_read_timeout_s, write=write_timeout_s, pool=pool_timeout_s)

        # Client
        if OPENAI_SDK_NEW:
            kwargs = {"api_key": api_key, "base_url": base_url}
            if self._timeout_nonstream is not None:
                # client-level default; per-request timeout still used below
                kwargs["timeout"] = self._timeout_nonstream
            self.client = OpenAI(**kwargs)
        else:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _looks_like_dashscope(base_url: str) -> bool:
        u = (base_url or "").lower()
        return "dashscope" in u or "aliyuncs.com" in u

    @staticmethod
    def _looks_like_deepseek(base_url: str) -> bool:
        u = (base_url or "").lower()
        return "deepseek" in u

    @staticmethod
    def _is_thinking_like_model(model_name: str) -> bool:
        if not model_name:
            return False
        m = model_name.lower()
        if "thinking" in m:
            return True
        if "deepseek-r1" in m or m.endswith("-r1") or m == "deepseek-r1":
            return True
        if "reasoner" in m:
            return True
        if "qwq" in m:
            return True
        if "qwen3" in m:
            # Qwen3 endpoints often require/benefit from streaming output.
            return True
        return False

    def _build_extra_body(self) -> Optional[Dict[str, Any]]:
        extra_body: Dict[str, Any] = {}
        if self.is_dashscope:
            if self.enable_thinking is not None:
                extra_body["enable_thinking"] = bool(self.enable_thinking)
            if self.thinking_budget is not None:
                extra_body["thinking_budget"] = int(self.thinking_budget)
        return extra_body or None

    @staticmethod
    def _get_reasoning_content(msg_obj: Any) -> str:
        rc = getattr(msg_obj, "reasoning_content", None)
        if isinstance(rc, str) and rc.strip():
            return rc.strip()
        extra = getattr(msg_obj, "model_extra", None)
        if isinstance(extra, dict):
            rc2 = extra.get("reasoning_content")
            if isinstance(rc2, str) and rc2.strip():
                return rc2.strip()
        return ""

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT_NODES

        extra_body = self._build_extra_body()

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.stream:
                    stream_resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                        max_tokens=self.max_tokens,
                        stream=True,
                        extra_body=extra_body,
                        extra_headers=self.extra_headers or None,
                        timeout=self._timeout_stream,
                    )

                    parts: List[str] = []
                    rparts: List[str] = []
                    try:
                        for chunk in stream_resp:
                            if not getattr(chunk, "choices", None):
                                continue
                            delta = chunk.choices[0].delta
                            c = getattr(delta, "content", None)
                            if isinstance(c, str) and c:
                                parts.append(c)

                            rc = getattr(delta, "reasoning_content", None)
                            if isinstance(rc, str) and rc:
                                rparts.append(rc)
                            d_extra = getattr(delta, "model_extra", None)
                            if isinstance(d_extra, dict):
                                rc2 = d_extra.get("reasoning_content")
                                if isinstance(rc2, str) and rc2:
                                    rparts.append(rc2)
                    finally:
                        close = getattr(stream_resp, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception:
                                pass

                    out = "".join(parts).strip()
                    if not out:
                        out = "".join(rparts).strip()
                    return out

                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=self.max_tokens,
                    stream=False,
                    extra_body=extra_body,
                    extra_headers=self.extra_headers or None,
                    timeout=self._timeout_nonstream,
                )
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                if not content:
                    content = self._get_reasoning_content(msg)
                return content.strip()

            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                sleep_s = (self.retry_backoff ** attempt) + (0.05 * attempt)
                time.sleep(sleep_s)

        raise RuntimeError(f"APIModel.generate failed after retries: {last_err}")


# -------------------------
# Prompt
# -------------------------
def build_prompt_nodes(extracted_text: str, gt_nodes: List[Dict[str, Any]], gt_links: List[Dict[str, Any]]) -> str:
    """Build prompt for node extraction (stage 1) given graph structure."""
    # Build the graph structure to show the model (only source and target, no label)
    links_without_label = [
        {"source": link.get("source", ""), "target": link.get("target", "")}
        for link in gt_links
    ]
    links_json = json.dumps({"links": links_without_label}, indent=2, ensure_ascii=False)
    
    # Extract unique node IDs from links
    node_ids = set()
    for link in gt_links:
        node_ids.add(link.get("source", ""))
        node_ids.add(link.get("target", ""))
    # Also include nodes that may not have links
    for node in gt_nodes:
        node_ids.add(node.get("id", ""))
    node_ids = sorted([nid for nid in node_ids if nid])
    
    node_structure = [{"id": nid} for nid in node_ids]
    nodes_json = json.dumps({"nodes": node_structure}, indent=2, ensure_ascii=False)
    
    prompt = f"""TEXT PARAGRAPH:
{extracted_text}

GRAPH STRUCTURE (Links between nodes):
{links_json}

NODE IDs to label:
{nodes_json}

Based on the text and the graph structure above, provide a concise label (single word or short phrase) for each node ID that best represents its meaning.

{FORMAT_INSTRUCTION_NODES}
"""
    return prompt.strip()


def build_prompt_links(extracted_text: str, nodes: List[Dict[str, Any]]) -> str:
    """Build prompt for link extraction (stage 2) given nodes."""
    nodes_json = json.dumps({"nodes": nodes}, indent=2, ensure_ascii=False)
    prompt = f"""TEXT PARAGRAPH:
{extracted_text}

GIVEN NODES:
{nodes_json}

{FORMAT_INSTRUCTION_LINKS}
"""
    return prompt.strip()


# -------------------------
# Parsing
# -------------------------
def extract_json_from_response(response: str, marker: str = "[Structure]") -> Optional[Dict[str, Any]]:
    """Extract JSON object after marker."""
    if not response:
        return None
    # Find marker
    idx = response.find(marker)
    if idx >= 0:
        response = response[idx + len(marker):]
    # Find first JSON object
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    json_str = response[start:end + 1]
    try:
        return json.loads(json_str)
    except Exception:
        # Attempt to clean trailing code fences
        json_str = json_str.strip()
        json_str = re.sub(r"```.*?```", "", json_str, flags=re.DOTALL)
        try:
            return json.loads(json_str)
        except Exception:
            return None


def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return ""
    label = label.strip().lower()
    label = re.sub(r"\s+", " ", label)
    return label


# -------------------------
# SBERT utils
# -------------------------
class SbertScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not SBERT_AVAILABLE:
            raise RuntimeError("sentence-transformers not available")
        self.model = SentenceTransformer(model_name)
        self.cache = {}
        self._lock = None
        try:
            import threading
            self._lock = threading.Lock()
        except Exception:
            self._lock = None

    def embed(self, texts: List[str]) -> np.ndarray:
        # Cache embeddings per text
        if self._lock:
            with self._lock:
                to_compute = [t for t in texts if t not in self.cache]
                if to_compute:
                    embs = self.model.encode(to_compute, convert_to_numpy=True, show_progress_bar=False)
                    for t, e in zip(to_compute, embs):
                        self.cache[t] = e
                return np.array([self.cache[t] for t in texts])
        # Fallback without lock
        to_compute = [t for t in texts if t not in self.cache]
        if to_compute:
            embs = self.model.encode(to_compute, convert_to_numpy=True, show_progress_bar=False)
            for t, e in zip(to_compute, embs):
                self.cache[t] = e
        return np.array([self.cache[t] for t in texts])

    def cosine_sim_matrix(self, a: List[str], b: List[str]) -> np.ndarray:
        if not a or not b:
            return np.zeros((len(a), len(b)))
        emb_a = self.embed(a)
        emb_b = self.embed(b)
        # Normalize
        emb_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
        emb_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
        return np.matmul(emb_a, emb_b.T)


# -------------------------
# Metrics
# -------------------------
def best_node_mapping(sim_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """Return list of matched (pred_idx, gt_idx, sim)."""
    if sim_matrix.size == 0:
        return []
    num_pred, num_gt = sim_matrix.shape
    # Hungarian to maximize similarity
    if SCIPY_AVAILABLE:
        # Convert to cost (negative for max)
        cost = -sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = [(r, c, sim_matrix[r, c]) for r, c in zip(row_ind, col_ind)]
    else:
        # Greedy fallback
        matches = []
        used_gt = set()
        for r in range(num_pred):
            best_c = None
            best_sim = -1.0
            for c in range(num_gt):
                if c in used_gt:
                    continue
                if sim_matrix[r, c] > best_sim:
                    best_sim = sim_matrix[r, c]
                    best_c = c
            if best_c is not None:
                used_gt.add(best_c)
                matches.append((r, best_c, best_sim))
    return matches


def compute_node_f1(pred_labels: List[str], gt_labels: List[str], scorer: SbertScorer, threshold: float) -> Tuple[float, int, int, int]:
    """Compute Node F1 using SBERT similarity threshold."""
    if not pred_labels and not gt_labels:
        return 1.0, 0, 0, 0
    if not pred_labels or not gt_labels:
        return 0.0, len(pred_labels), len(gt_labels), 0
    sim = scorer.cosine_sim_matrix(pred_labels, gt_labels)
    matches = best_node_mapping(sim)
    tp = sum(1 for _, _, s in matches if s >= threshold)
    fp = len(pred_labels) - tp
    fn = len(gt_labels) - tp
    if tp == 0:
        return 0.0, fp, fn, tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, fp, fn, tp


def compute_node_similarity_direct(
    pred_labels_by_id: Dict[str, str],
    gt_labels_by_id: Dict[str, str],
    scorer: SbertScorer
) -> Tuple[float, Dict[str, float]]:
    """
    Compute average node similarity with direct correspondence by node ID.
    Returns (avg_similarity, per_node_similarity).
    """
    if not gt_labels_by_id:
        return 1.0, {}
    
    per_node_sim = {}
    all_sims = []
    
    for node_id, gt_label in gt_labels_by_id.items():
        pred_label = pred_labels_by_id.get(node_id, "")
        if not pred_label or not gt_label:
            sim = 0.0
        else:
            # Compute similarity between single pair
            sim_matrix = scorer.cosine_sim_matrix([pred_label], [gt_label])
            sim = float(sim_matrix[0, 0])
        per_node_sim[node_id] = sim
        all_sims.append(sim)
    
    avg_sim = float(np.mean(all_sims)) if all_sims else 0.0
    return avg_sim, per_node_sim


def build_triples_by_idx(
    id_to_idx: Dict[str, int],
    links: List[Dict[str, Any]]
) -> List[Tuple[int, str, int]]:
    """Build triples (src_idx, link_label, tgt_idx) using node id mapping."""
    triples = []
    for link in links:
        src_id = link.get("source", "")
        tgt_id = link.get("target", "")
        if src_id not in id_to_idx or tgt_id not in id_to_idx:
            continue
        src_idx = id_to_idx[src_id]
        tgt_idx = id_to_idx[tgt_id]
        lbl = normalize_label(link.get("label", ""))
        triples.append((src_idx, lbl, tgt_idx))
    return triples


def compute_semantic_smatch(
    pred_labels: List[str],
    gt_labels: List[str],
    pred_triples: List[Tuple[int, str, int]],
    gt_triples: List[Tuple[int, str, int]],
    scorer: SbertScorer,
    threshold: float,
) -> Tuple[float, int]:
    """Compute semantic-smatch via best node mapping and triple F1. Returns (f1, edge_matches)."""
    if not pred_triples and not gt_triples:
        return 1.0, 0
    if not pred_labels or not gt_labels:
        return 0.0, 0
    sim = scorer.cosine_sim_matrix(pred_labels, gt_labels)
    matches = best_node_mapping(sim)
    # Build mapping pred_idx -> gt_idx if above threshold
    mapping = {p: g for p, g, s in matches if s >= threshold}
    # Map pred triples to gt indices
    mapped_pred = []
    for s_idx, l, t_idx in pred_triples:
        if s_idx in mapping and t_idx in mapping:
            mapped_pred.append((mapping[s_idx], l, mapping[t_idx]))
    # Count overlap
    gt_set = set(gt_triples)
    pred_set = set(mapped_pred)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    if tp == 0:
        return 0.0, 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, tp


def compute_ged_similarity(node_matches: int, pred_node_count: int, gt_node_count: int,
                           pred_edge_count: int, gt_edge_count: int, edge_matches: int) -> float:
    """Approximate GED similarity using matched nodes/edges."""
    node_edit = max(pred_node_count, gt_node_count) - node_matches
    edge_edit = max(pred_edge_count, gt_edge_count) - edge_matches
    ged = node_edit + edge_edit
    denom = max(pred_node_count + gt_node_count + pred_edge_count + gt_edge_count, 1)
    similarity = 1.0 - (ged / denom)
    return max(0.0, similarity)


def compute_link_f1(
    pred_triples: List[Tuple[int, str, int]],
    gt_triples: List[Tuple[int, str, int]],
) -> Tuple[float, int, int, int]:
    """
    Compute Link F1 by comparing connections (source, target) only.
    Ignores link labels - only checks if the connection exists.
    Returns (f1, fp, fn, tp).
    """
    if not pred_triples and not gt_triples:
        return 1.0, 0, 0, 0
    if not pred_triples or not gt_triples:
        return 0.0, len(pred_triples), len(gt_triples), 0
    
    # Extract (source, target) pairs, ignoring labels
    pred_set = set((src, tgt) for src, _, tgt in pred_triples)
    gt_set = set((src, tgt) for src, _, tgt in gt_triples)
    
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    if tp == 0:
        return 0.0, fp, fn, tp
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, fp, fn, tp


# -------------------------
# Helpers
# -------------------------
def extract_major_class(folder_name: str) -> str:
    parts = folder_name.split("_")
    if parts:
        major = parts[0]
        if major == "Life-Sciences":
            major = "Life-Science"
        return major
    return "Unknown"


# -------------------------
# Evaluation
# -------------------------
def process_sample(
    folder: Path,
    model: BaseModel,
    scorer: SbertScorer,
    node_threshold: float,
    verbose: bool = False,
    *,
    cache_path: Optional[Path] = None,
    resume: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Two-stage evaluation:
    1. Extract nodes from text -> compute node_f1
    2. Given text + GT nodes, extract links -> compute link_f1
    """
    if resume and cache_path and cache_path.exists():
        try:
            return load_json(cache_path)
        except Exception:
            pass

    info_path = folder / "information.json"
    if not info_path.exists():
        return None
    info = load_json(info_path)
    key_frame = info.get("key_frame")
    if not key_frame:
        return None
    md_path = find_extracted_md(folder)
    if not md_path:
        return None
    extracted_text = read_text(md_path)
    
    gt_nodes_raw = key_frame.get("nodes", [])
    gt_links_raw = key_frame.get("links", [])
    gt_labels = [normalize_label(n.get("label", "")) for n in gt_nodes_raw]

    # =========================================================================
    # Stage 1: Extract node labels given graph structure (links)
    # =========================================================================
    try:
        prompt_nodes = build_prompt_nodes(extracted_text, gt_nodes_raw, gt_links_raw)
        response_nodes = model.generate(prompt_nodes, system_prompt=SYSTEM_PROMPT_NODES)
    except Exception as e:
        result = {
            "sample_name": folder.name,
            "major_class": extract_major_class(folder.name),
            "error": f"stage1_generate_failed: {e}",
            "node_f1": 0.0,
            "link_f1": 0.0,
            "per_node_similarity": {},
            "pred_nodes": [],
            "pred_links": [],
            "gt_nodes": gt_nodes_raw,
            "gt_links": gt_links_raw,
        }
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                save_json(cache_path, result)
            except Exception:
                pass
        return result
    
    pred_nodes_json = extract_json_from_response(response_nodes, marker="[Nodes]") or {"nodes": []}
    pred_nodes_raw = pred_nodes_json.get("nodes", [])
    
    # Build dictionaries: node_id -> label
    gt_labels_by_id = {n.get("id", ""): normalize_label(n.get("label", "")) for n in gt_nodes_raw}
    pred_labels_by_id = {n.get("id", ""): normalize_label(n.get("label", "")) for n in pred_nodes_raw}
    
    # Compute average node similarity (direct correspondence by ID)
    node_avg_sim, per_node_sim = compute_node_similarity_direct(pred_labels_by_id, gt_labels_by_id, scorer)
    
    # For backwards compatibility, store as node_f1
    node_f1 = node_avg_sim
    node_tp = sum(1 for s in per_node_sim.values() if s >= node_threshold)
    node_fp = 0  # Not applicable in direct correspondence
    node_fn = len(gt_labels_by_id) - node_tp

    # =========================================================================
    # Stage 2: Extract links given text + GT nodes
    # =========================================================================
    try:
        prompt_links = build_prompt_links(extracted_text, gt_nodes_raw)
        response_links = model.generate(prompt_links, system_prompt=SYSTEM_PROMPT_LINKS)
    except Exception as e:
        result = {
            "sample_name": folder.name,
            "major_class": extract_major_class(folder.name),
            "error": f"stage2_generate_failed: {e}",
            "node_f1": node_f1,
            "link_f1": 0.0,
            "per_node_similarity": per_node_sim,
            "pred_nodes": pred_nodes_raw,
            "pred_links": [],
            "gt_nodes": gt_nodes_raw,
            "gt_links": gt_links_raw,
        }
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                save_json(cache_path, result)
            except Exception:
                pass
        return result
    
    pred_links_json = extract_json_from_response(response_links, marker="[Links]") or {"links": []}
    pred_links_raw = pred_links_json.get("links", [])
    
    # Build triples using GT node IDs (since we give GT nodes to model)
    gt_id_to_idx = {n.get("id", f"g{i}"): i for i, n in enumerate(gt_nodes_raw)}
    pred_triples = build_triples_by_idx(gt_id_to_idx, pred_links_raw)
    gt_triples = build_triples_by_idx(gt_id_to_idx, gt_links_raw)
    
    # Compute Link F1
    link_f1, link_fp, link_fn, link_tp = compute_link_f1(pred_triples, gt_triples)
    
    # =========================================================================
    # Result
    # =========================================================================
    result = {
        "sample_name": folder.name,
        "major_class": extract_major_class(folder.name),
        "node_f1": node_f1,  # Actually avg similarity now
        "link_f1": link_f1,
        "per_node_similarity": per_node_sim,
        "pred_nodes": pred_nodes_raw,
        "pred_links": pred_links_raw,
        "gt_nodes": gt_nodes_raw,
        "gt_links": gt_links_raw,
    }

    if verbose:
        print(f"\n[{folder.name}]")
        print(f"  Node Avg Similarity: {node_f1:.4f} (Matched@{node_threshold}={node_tp}/{len(gt_labels_by_id)})")
        print(f"  Link F1: {link_f1:.4f} (TP={link_tp}, FP={link_fp}, FN={link_fn})")

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(cache_path, result)
        except Exception:
            pass

    return result


# -------------------------
# Process dict-based sample (for parquet data)
# -------------------------
def process_sample_from_dict(
    sample_dict: Dict[str, Any],
    model: BaseModel,
    scorer: SbertScorer,
    node_threshold: float,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Process a single E2E sample loaded from parquet data."""
    sample_name = sample_dict['sample_name']
    extracted_text = sample_dict['text']
    key_frame = sample_dict['key_frame']
    if not key_frame:
        return None

    gt_nodes_raw = key_frame.get("nodes", [])
    gt_links_raw = key_frame.get("links", [])
    gt_labels = [normalize_label(n.get("label", "")) for n in gt_nodes_raw]

    try:
        prompt_nodes = build_prompt_nodes(extracted_text, gt_nodes_raw, gt_links_raw)
        response_nodes = model.generate(prompt_nodes, system_prompt=SYSTEM_PROMPT_NODES)
    except Exception as e:
        return {
            "sample_name": sample_name,
            "major_class": sample_dict.get('major_class', 'Unknown'),
            "error": f"stage1_generate_failed: {e}",
            "node_f1": 0.0, "link_f1": 0.0,
            "per_node_similarity": {},
            "pred_nodes": [], "pred_links": [],
            "gt_nodes": gt_nodes_raw, "gt_links": gt_links_raw,
        }

    pred_nodes_json = extract_json_from_response(response_nodes, marker="[Nodes]") or {"nodes": []}
    pred_nodes_raw = pred_nodes_json.get("nodes", [])

    gt_labels_by_id = {n.get("id", ""): normalize_label(n.get("label", "")) for n in gt_nodes_raw}
    pred_labels_by_id = {n.get("id", ""): normalize_label(n.get("label", "")) for n in pred_nodes_raw}
    node_avg_sim, per_node_sim = compute_node_similarity_direct(pred_labels_by_id, gt_labels_by_id, scorer)
    node_f1 = node_avg_sim
    node_tp = sum(1 for s in per_node_sim.values() if s >= node_threshold)
    node_fn = len(gt_labels_by_id) - node_tp

    try:
        prompt_links = build_prompt_links(extracted_text, gt_nodes_raw)
        response_links = model.generate(prompt_links, system_prompt=SYSTEM_PROMPT_LINKS)
    except Exception as e:
        return {
            "sample_name": sample_name,
            "major_class": sample_dict.get('major_class', 'Unknown'),
            "error": f"stage2_generate_failed: {e}",
            "node_f1": node_f1, "link_f1": 0.0,
            "per_node_similarity": per_node_sim,
            "pred_nodes": pred_nodes_raw, "pred_links": [],
            "gt_nodes": gt_nodes_raw, "gt_links": gt_links_raw,
        }

    pred_links_json = extract_json_from_response(response_links, marker="[Links]") or {"links": []}
    pred_links_raw = pred_links_json.get("links", [])

    gt_id_to_idx = {n.get("id", f"g{i}"): i for i, n in enumerate(gt_nodes_raw)}
    pred_triples = build_triples_by_idx(gt_id_to_idx, pred_links_raw)
    gt_triples = build_triples_by_idx(gt_id_to_idx, gt_links_raw)
    link_f1, link_fp, link_fn, link_tp = compute_link_f1(pred_triples, gt_triples)

    result = {
        "sample_name": sample_name,
        "major_class": sample_dict.get('major_class', 'Unknown'),
        "node_f1": node_f1,
        "link_f1": link_f1,
        "per_node_similarity": per_node_sim,
        "pred_nodes": pred_nodes_raw,
        "pred_links": pred_links_raw,
        "gt_nodes": gt_nodes_raw,
        "gt_links": gt_links_raw,
    }

    if verbose:
        print(f"\n[{sample_name}]")
        print(f"  Node Avg Similarity: {node_f1:.4f} (Matched@{node_threshold}={node_tp}/{len(gt_labels_by_id)})")
        print(f"  Link F1: {link_f1:.4f} (TP={link_tp}, FP={link_fp}, FN={link_fn})")

    return result


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate structure extraction against key_frame")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset_dir", default="", help="Dataset directory (folder-based format)")
    data_group.add_argument("--data_dir", default="", help="Local HuggingFace data directory (e.g. data/T2S-Bench-E2E)")
    parser.add_argument("--model_type", required=True, choices=["hf", "api"], help="Model type")
    parser.add_argument("--model_name_or_path", default="", help="HF model name or path")
    parser.add_argument("--model_name", default="", help="API model name")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""), help="API key (prefer env)")
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", ""), help="API base URL (prefer env)")
    parser.add_argument("--provider", default="custom", choices=sorted(PROVIDER_PRESETS.keys()), help="Provider preset (optional)")
    parser.add_argument("--stream", action="store_true", help="Force stream=True (recommended for Qwen3/thinking models)")
    parser.add_argument("--no_stream", action="store_true", help="Force stream=False")
    parser.add_argument("--enable_thinking", action="store_true", help="DashScope: enable thinking mode (extra_body)")
    parser.add_argument("--thinking_budget", type=int, default=None, help="DashScope: thinking_budget (extra_body)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max output tokens for API models")
    parser.add_argument("--max_retries", type=int, default=3, help="API retry attempts")
    parser.add_argument("--retry_backoff", type=float, default=1.5, help="Retry backoff base")
    parser.add_argument("--cache_dir", type=str, default="", help="Optional per-sample cache directory")
    parser.add_argument("--resume", action="store_true", help="Resume from cache_dir if available")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for API model")
    parser.add_argument("--node_sim_threshold", type=float, default=0.75, help="SBERT similarity threshold")
    parser.add_argument("--sbert_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SBERT model")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample results")
    args = parser.parse_args()

    # Initialize model
    if args.model_type == "hf":
        if not args.model_name_or_path:
            raise SystemExit("--model_name_or_path required for HF")
        model = HFModel(args.model_name_or_path)
    else:
        if not args.model_name:
            raise SystemExit("--model_name required for API")

        # Resolve provider preset (optional)
        base_url = args.base_url
        api_key = args.api_key
        if args.provider and args.provider != "custom":
            preset = PROVIDER_PRESETS.get(args.provider, {})
            base_url = base_url or preset.get("base_url", "")
            key_env = preset.get("api_key_env", "")
            if not api_key and key_env:
                api_key = os.getenv(key_env, "")

        if not api_key or not base_url:
            raise SystemExit("API requires --api_key and --base_url (or set --provider with env key)")

        if args.stream and args.no_stream:
            raise SystemExit("Cannot set both --stream and --no_stream.")

        stream: Optional[bool]
        if args.stream:
            stream = True
        elif args.no_stream:
            stream = False
        else:
            stream = None  # auto

        enable_thinking: Optional[bool] = True if args.enable_thinking else None

        model = APIModel(
            args.model_name,
            api_key,
            base_url,
            stream=stream,
            enable_thinking=enable_thinking,
            thinking_budget=args.thinking_budget,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )

    # SBERT scorer
    scorer = SbertScorer(args.sbert_model)

    results = []
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.data_dir:
        # Load from local HuggingFace parquet data
        data_dir = Path(args.data_dir)
        print(f"Loading data from parquet: {data_dir}")
        parquet_samples = load_samples_from_parquet_e2e(data_dir)
        print(f"Found {len(parquet_samples)} samples")

        if args.model_type == "api" and args.num_workers > 1:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(
                        process_sample_from_dict,
                        s, model, scorer, args.node_sim_threshold, args.verbose,
                    )
                    for s in parquet_samples
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                    res = future.result()
                    if res:
                        results.append(res)
        else:
            for s in tqdm(parquet_samples, desc="Evaluating"):
                res = process_sample_from_dict(
                    s, model, scorer, args.node_sim_threshold, args.verbose,
                )
                if res:
                    results.append(res)
    else:
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.exists():
            raise SystemExit(f"Dataset directory not found: {dataset_dir}")

        folders = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
        print(f"Found {len(folders)} samples")

        if args.model_type == "api" and args.num_workers > 1:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(
                        process_sample,
                        f,
                        model,
                        scorer,
                        args.node_sim_threshold,
                        args.verbose,
                        cache_path=(cache_dir / f"{f.name}.json") if cache_dir else None,
                        resume=args.resume,
                    )
                    for f in folders
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                    res = future.result()
                    if res:
                        results.append(res)
        else:
            for f in tqdm(folders, desc="Evaluating"):
                res = process_sample(
                    f,
                    model,
                    scorer,
                    args.node_sim_threshold,
                    args.verbose,
                    cache_path=(cache_dir / f"{f.name}.json") if cache_dir else None,
                    resume=args.resume,
                )
                if res:
                    results.append(res)

    # Aggregate by major class
    by_major = defaultdict(lambda: {"node_f1": [], "link_f1": []})
    for r in results:
        m = r["major_class"]
        by_major[m]["node_f1"].append(r["node_f1"])
        by_major[m]["link_f1"].append(r["link_f1"])

    major_summary = {}
    for m, vals in by_major.items():
        major_summary[m] = {
            "node_f1": float(np.mean(vals["node_f1"])) if vals["node_f1"] else 0.0,
            "link_f1": float(np.mean(vals["link_f1"])) if vals["link_f1"] else 0.0,
            "num_samples": len(vals["node_f1"]),
        }

    # Overall
    overall = {
        "node_f1": float(np.mean([r["node_f1"] for r in results])) if results else 0.0,
        "link_f1": float(np.mean([r["link_f1"] for r in results])) if results else 0.0,
        "num_samples": len(results),
    }

    output = {
        "overall": overall,
        "by_major_class": major_summary,
        "all_samples": results,
    }

    save_json(Path(args.output), output)

    # Print summary
    print("\n" + "=" * 80)
    print("Overall Metrics")
    print("=" * 80)
    print(f"Node Avg Similarity: {overall['node_f1']:.4f}")
    print(f"Link F1: {overall['link_f1']:.4f}")
    print(f"Num Samples: {overall['num_samples']}")

    print("\n" + "=" * 80)
    print("By Major Class")
    print("=" * 80)
    print(f"{'Major Class':<35} {'Node Sim':<12} {'Link F1':<12} {'Num':<8}")
    print("-" * 80)
    for m, v in sorted(major_summary.items()):
        print(f"{m:<35} {v['node_f1']:<12.4f} {v['link_f1']:<12.4f} {v['num_samples']:<8}")

    print("\n✅ Results saved to", args.output)


if __name__ == "__main__":
    main()
