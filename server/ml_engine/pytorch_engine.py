#!/usr/bin/env python3
"""
OMNIBRAIN AI — PyTorch Foundation Model Training Backend
Zero external AI APIs, zero external pretrained weights.
Implements a decoder-only Transformer matching the Pure-Python reference engine:
- Token & Positional Embeddings (nn.Embedding)
- Multi-Head Causal Self-Attention (explicit upper-triangular causal mask)
- LayerNorm (pre-norm + final LayerNorm)
- Feed-Forward Network (ReLU) + Residual Connections
- Autoregressive next-token Cross-Entropy Loss (pad_id=0 excluded via ignore_index)
- AdamW with decoupled weight decay (excluding biases & LayerNorm params)
- Cosine LR schedule with linear warmup + global L2 gradient clipping
- CPU & CUDA device selection with strict RAM/VRAM safety guards
- Versioned checkpoints (checkpoint.pt + manifest.json) with tokenizer validation
"""

import os
import sys
import json
import math
import time
import shutil
import hashlib
import resource
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
    PYTORCH_VERSION = torch.__version__
    PYTORCH_IMPORT_ERROR = None
except Exception as exc:  # ModuleNotFoundError / ImportError / OSError
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    PYTORCH_AVAILABLE = False
    PYTORCH_VERSION = None
    PYTORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

PYTORCH_SOFTWARE_VERSION = "OMNIBRAIN-PyTorch-0.5.2"
EXPECTED_TOKENIZER_VERSION = "omnibrain-tok-v1.1.0"
EXPECTED_SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}


def compute_sha256(file_path):
    """Computes SHA-256 hash of a file on disk."""
    if not os.path.exists(file_path):
        return ""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _get_peak_ram_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(usage / (1024 * 1024), 3)
    return round(usage / 1024.0, 3)


def get_cuda_info():
    if not PYTORCH_AVAILABLE:
        return {
            "cudaAvailable": False,
            "gpuName": None,
            "gpuMemoryMB": 0.0,
            "gpuCount": 0,
        }
    cuda_ok = bool(torch.cuda.is_available())
    if not cuda_ok:
        return {
            "cudaAvailable": False,
            "gpuName": None,
            "gpuMemoryMB": 0.0,
            "gpuCount": 0,
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "cudaAvailable": True,
        "gpuName": torch.cuda.get_device_name(0),
        "gpuMemoryMB": round(props.total_memory / (1024 * 1024), 1),
        "gpuCount": torch.cuda.device_count(),
    }


def resolve_pytorch_device(requested_device="auto"):
    if not PYTORCH_AVAILABLE:
        raise RuntimeError(f"PyTorch is not installed in this environment ({PYTORCH_IMPORT_ERROR}).")
    cuda_ok = bool(torch.cuda.is_available())
    req = (requested_device or "auto").lower()
    if req == "cuda":
        if not cuda_ok:
            raise RuntimeError("CUDA device was explicitly requested, but torch.cuda.is_available() is False.")
        return "cuda"
    if req == "cpu":
        return "cpu"
    return "cuda" if cuda_ok else "cpu"


def validate_tokenizer_compatibility(tok_meta, model_config):
    """Strict validation between tokenizer_v1.json, model embedding dims, and special token IDs."""
    if not tok_meta:
        return True, None
    tok_ver = tok_meta.get("version")
    if tok_ver and tok_ver != EXPECTED_TOKENIZER_VERSION:
        return False, (
            f"Tokenizer version mismatch: expected '{EXPECTED_TOKENIZER_VERSION}', got '{tok_ver}'."
        )
    tok_vocab = tok_meta.get("vocabSize")
    if tok_vocab is not None and int(tok_vocab) != int(model_config["vocabSize"]):
        return False, (
            f"Tokenizer vocabSize ({tok_vocab}) does not match model vocabSize ({model_config['vocabSize']})."
        )
    specials = tok_meta.get("specialTokens")
    if specials:
        for k, expected_id in EXPECTED_SPECIAL_TOKENS.items():
            if specials.get(k) != expected_id:
                return False, (
                    f"Special token '{k}' ID mismatch: expected {expected_id}, got {specials.get(k)}."
                )
    return True, None


if PYTORCH_AVAILABLE:

    class CausalSelfAttentionBlock(nn.Module):
        def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.0):
            super().__init__()
            if d_model % num_heads != 0:
                raise ValueError(f"dModel ({d_model}) must be divisible by numHeads ({num_heads})")
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
            self.q_proj = nn.Linear(d_model, d_model, bias=True)
            self.k_proj = nn.Linear(d_model, d_model, bias=True)
            self.v_proj = nn.Linear(d_model, d_model, bias=True)
            self.o_proj = nn.Linear(d_model, d_model, bias=True)

            self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
            self.fc1 = nn.Linear(d_model, d_ff, bias=True)
            self.fc2 = nn.Linear(d_ff, d_model, bias=True)
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

            mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, D = x.shape
            h1 = self.ln1(x)

            q = self.q_proj(h1).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            k = self.k_proj(h1).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            v = self.v_proj(h1).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
            x = x + self.dropout(self.o_proj(attn_out))

            h2 = self.ln2(x)
            ffn = self.fc2(F.relu(self.fc1(h2)))
            x = x + self.dropout(ffn)
            return x

    class OmnibrainPyTorchTransformer(nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.config = dict(config)
            V = int(config["vocabSize"])
            D = int(config["dModel"])
            L = int(config["numLayers"])
            H = int(config["numHeads"])
            dff = int(config["dFf"])
            S = int(config["maxSeqLen"])
            dropout = float(config.get("dropout", 0.0))

            self.w_tok = nn.Embedding(V, D)
            self.w_pos = nn.Embedding(S, D)
            self.emb_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

            self.layers = nn.ModuleList(
                [CausalSelfAttentionBlock(D, H, dff, S, dropout=dropout) for _ in range(L)]
            )
            self.ln_f = nn.LayerNorm(D, eps=1e-5)
            self.lm_head = nn.Linear(D, V, bias=True)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.xavier_uniform_(m.weight)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            B, T = token_ids.shape
            if T > self.config["maxSeqLen"]:
                raise ValueError(f"Sequence length {T} exceeds maxSeqLen {self.config['maxSeqLen']}")
            positions = torch.arange(T, device=token_ids.device, dtype=torch.long).unsqueeze(0).expand(B, T)
            h = self.emb_drop(self.w_tok(token_ids) + self.w_pos(positions))
            for block in self.layers:
                h = block(h)
            h = self.ln_f(h)
            logits = self.lm_head(h)
            return logits

        def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor, pad_id: int = 0):
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)
            logits = self.forward(inputs)
            active_mask = targets != pad_id
            active_tokens = int(active_mask.sum().item())
            if active_tokens == 0:
                loss = logits.sum() * 0.0
                return loss, 0
            loss = F.cross_entropy(
                logits.view(-1, self.config["vocabSize"]),
                targets.view(-1),
                ignore_index=pad_id,
                reduction="mean",
            )
            return loss, active_tokens

        def load_from_reference_params(self, ref_params: dict):
            """
            Copies exact parameter matrices from the Pure-Python DecoderTransformer
            into this PyTorch module (accounting for Linear weight transpose [out, in]).
            Used for exact numerical verification tests.
            """
            with torch.no_grad():
                self.w_tok.weight.copy_(torch.tensor(ref_params["w_tok"], dtype=self.w_tok.weight.dtype))
                self.w_pos.weight.copy_(torch.tensor(ref_params["w_pos"], dtype=self.w_pos.weight.dtype))

                for l, block in enumerate(self.layers):
                    block.ln1.weight.copy_(torch.tensor(ref_params[f"l{l}_ln1_g"][0], dtype=block.ln1.weight.dtype))
                    block.ln1.bias.copy_(torch.tensor(ref_params[f"l{l}_ln1_b"][0], dtype=block.ln1.bias.dtype))
                    block.q_proj.weight.copy_(torch.tensor(ref_params[f"l{l}_wq"], dtype=block.q_proj.weight.dtype).t())
                    block.q_proj.bias.copy_(torch.tensor(ref_params[f"l{l}_bq"][0], dtype=block.q_proj.bias.dtype))
                    block.k_proj.weight.copy_(torch.tensor(ref_params[f"l{l}_wk"], dtype=block.k_proj.weight.dtype).t())
                    block.k_proj.bias.copy_(torch.tensor(ref_params[f"l{l}_bk"][0], dtype=block.k_proj.bias.dtype))
                    block.v_proj.weight.copy_(torch.tensor(ref_params[f"l{l}_wv"], dtype=block.v_proj.weight.dtype).t())
                    block.v_proj.bias.copy_(torch.tensor(ref_params[f"l{l}_bv"][0], dtype=block.v_proj.bias.dtype))
                    block.o_proj.weight.copy_(torch.tensor(ref_params[f"l{l}_wo"], dtype=block.o_proj.weight.dtype).t())
                    block.o_proj.bias.copy_(torch.tensor(ref_params[f"l{l}_bo"][0], dtype=block.o_proj.bias.dtype))

                    block.ln2.weight.copy_(torch.tensor(ref_params[f"l{l}_ln2_g"][0], dtype=block.ln2.weight.dtype))
                    block.ln2.bias.copy_(torch.tensor(ref_params[f"l{l}_ln2_b"][0], dtype=block.ln2.bias.dtype))
                    block.fc1.weight.copy_(torch.tensor(ref_params[f"l{l}_w1"], dtype=block.fc1.weight.dtype).t())
                    block.fc1.bias.copy_(torch.tensor(ref_params[f"l{l}_b1"][0], dtype=block.fc1.bias.dtype))
                    block.fc2.weight.copy_(torch.tensor(ref_params[f"l{l}_w2"], dtype=block.fc2.weight.dtype).t())
                    block.fc2.bias.copy_(torch.tensor(ref_params[f"l{l}_b2"][0], dtype=block.fc2.bias.dtype))

                self.ln_f.weight.copy_(torch.tensor(ref_params["lnf_g"][0], dtype=self.ln_f.weight.dtype))
                self.ln_f.bias.copy_(torch.tensor(ref_params["lnf_b"][0], dtype=self.ln_f.bias.dtype))
                self.lm_head.weight.copy_(torch.tensor(ref_params["w_out"], dtype=self.lm_head.weight.dtype).t())
                self.lm_head.bias.copy_(torch.tensor(ref_params["b_out"][0], dtype=self.lm_head.bias.dtype))


class PyTorchFoundationEngine:
    def __init__(self, preset="tiny_cpu", config=None, seed=42, requested_device="auto"):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError(f"PyTorch backend cannot initialize: {PYTORCH_IMPORT_ERROR}")

        from engine import MODEL_PRESETS, get_available_ram_bytes, estimate_config_resources

        self._presets = MODEL_PRESETS
        self._get_available_ram_bytes = get_available_ram_bytes
        self._estimate_config_resources = estimate_config_resources

        self.active_preset = preset
        self.config = dict(config or MODEL_PRESETS[preset])
        self.seed = seed
        self.device = resolve_pytorch_device(requested_device)

        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.model = OmnibrainPyTorchTransformer(self.config).to(self.device)
        self.optimizer = self._build_optimizer(lr=0.008, weight_decay=0.01)
        self.step = 0
        self.epoch = 0
        self.cumulative_tokens = 0
        self.seq_cursor = 0
        self.best_val_loss = float("inf")
        self.best_val_checkpoint_dir = None
        self.phase7_history = []
        self.is_trained = False

        self.tokenizer_vocab = None
        self.tokenizer_version = EXPECTED_TOKENIZER_VERSION
        self.tokenizer_checksum = ""
        self.tokenizer_meta = None
        self.last_metrics = None
        self.last_checkpoint_dir = None
        self.cancel_flag_path = os.path.abspath("./checkpoints/.cancel_training")
        self._load_tokenizer_file("./checkpoints/tokenizer_v1.json")

    def _build_optimizer(self, lr=0.008, weight_decay=0.01):
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or ".ln" in name or name.startswith("ln_f"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _load_tokenizer_file(self, tok_path):
        if os.path.exists(tok_path):
            try:
                with open(tok_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.tokenizer_meta = data
                self.tokenizer_version = data.get("version", self.tokenizer_version)
                self.tokenizer_checksum = data.get("checksum", "")
                self.tokenizer_vocab = data.get("idToToken", {})
                return True
            except Exception:
                pass
        return False

    def _check_resource_budget(self, cfg):
        est = self._estimate_config_resources(cfg)
        avail_ram = self._get_available_ram_bytes()
        safe_ram = int(avail_ram * 0.70)
        if est["totalEstimatedBytes"] > safe_ram:
            avail_mb = round(avail_ram / (1024 * 1024), 1)
            return False, (
                f"Insufficient system RAM for '{cfg['name']}': requires ~{est['estimatedMemoryMB']} MB, "
                f"exceeding 70% of available {avail_mb} MB."
            ), est, avail_mb

        if self.device == "cuda" and torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory
            if est["totalEstimatedBytes"] > int(gpu_total * 0.85):
                gpu_mb = round(gpu_total / (1024 * 1024), 1)
                return False, (
                    f"Insufficient GPU VRAM for '{cfg['name']}': requires ~{est['estimatedMemoryMB']} MB, "
                    f"exceeding 85% of {gpu_mb} MB GPU memory."
                ), est, round(avail_ram / (1024 * 1024), 1)

        return True, None, est, round(avail_ram / (1024 * 1024), 1)

    def set_config(self, config_name, seed=42):
        if config_name not in self._presets:
            return {"success": False, "error": f"Unknown preset: {config_name}"}
        cfg = dict(self._presets[config_name])

        ok, err_msg, est, avail_mb = self._check_resource_budget(cfg)
        if not ok:
            return {
                "success": False,
                "rejected": True,
                "backend": "pytorch",
                "selectedDevice": self.device,
                "error": err_msg,
                "estimates": est,
                "availableRamMB": avail_mb,
            }

        tok_ok, tok_err = validate_tokenizer_compatibility(self.tokenizer_meta, cfg)
        if not tok_ok:
            return {"success": False, "rejected": True, "error": tok_err}

        self.active_preset = config_name
        self.config = cfg
        self.seed = seed
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.model = OmnibrainPyTorchTransformer(cfg).to(self.device)
        self.optimizer = self._build_optimizer(lr=0.008, weight_decay=0.01)
        self.step = 0
        self.epoch = 0
        self.cumulative_tokens = 0
        self.seq_cursor = 0
        self.best_val_loss = float("inf")
        self.best_val_checkpoint_dir = None
        self.phase7_history = []
        self.is_trained = False
        return {
            "success": True,
            "backend": "pytorch",
            "selectedDevice": self.device,
            "preset": config_name,
            "config": self.config,
            "estimates": est,
            "parameterCount": self.model.count_parameters(),
        }

    @staticmethod
    def _cosine_lr(step, total_steps, base_lr, min_lr=1e-4):
        warmup = max(1, int(total_steps * 0.15))
        if step < warmup:
            return base_lr * float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def evaluate_sequences(self, sequences):
        if not sequences:
            return None, 0, 0.0
        was_training = self.model.training
        self.model.eval()
        t0 = time.perf_counter()
        total_loss = 0.0
        total_tokens = 0
        seqs_used = 0
        with torch.no_grad():
            for seq in sequences:
                if len(seq) < 2:
                    continue
                inp = torch.tensor(seq[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
                tgt = torch.tensor(seq[1:], dtype=torch.long, device=self.device).unsqueeze(0)
                loss, n_toks = self.model.compute_loss(inp, tgt, pad_id=0)
                if n_toks > 0:
                    total_loss += float(loss.item())
                    total_tokens += n_toks
                    seqs_used += 1
        elapsed = max(1e-6, time.perf_counter() - t0)
        if was_training:
            self.model.train()
        avg_loss = total_loss / max(1, seqs_used)
        return round(avg_loss, 4), total_tokens, round(total_tokens / elapsed, 2)

    def save_checkpoint(self, base_dir="./checkpoints", reason="periodic"):
        os.makedirs(base_dir, exist_ok=True)
        disk = shutil.disk_usage(base_dir)
        if disk.free < 5 * 1024 * 1024:
            raise RuntimeError("Insufficient disk space (< 5 MB free) to save PyTorch checkpoint.")

        run_dir = os.path.join(
            base_dir,
            f"ckpt_pt_{self.active_preset}_step_{self.step}_{int(time.time() * 1000)}",
        )
        os.makedirs(run_dir, exist_ok=True)

        ckpt_file = os.path.join(run_dir, "checkpoint.pt")
        manifest_file = os.path.join(run_dir, "manifest.json")

        payload = {
            "backend": "pytorch",
            "softwareVersion": PYTORCH_SOFTWARE_VERSION,
            "pytorchVersion": PYTORCH_VERSION,
            "device": self.device,
            "preset": self.active_preset,
            "config": self.config,
            "seed": self.seed,
            "step": self.step,
            "epoch": self.epoch,
            "isTrained": self.is_trained,
            "tokenizerVersion": self.tokenizer_version,
            "tokenizerChecksum": self.tokenizer_checksum,
            "specialTokens": EXPECTED_SPECIAL_TOKENS,
            "datasetVersion": self.last_metrics.get("datasetVersion") if self.last_metrics else "unknown",
            "modelStateDict": self.model.state_dict(),
            "optimizerStateDict": self.optimizer.state_dict(),
            "rngState": torch.get_rng_state(),
        }
        torch.save(payload, ckpt_file)
        ckpt_size_bytes = os.path.getsize(ckpt_file)

        manifest = {
            "backend": "pytorch",
            "softwareVersion": PYTORCH_SOFTWARE_VERSION,
            "pytorchVersion": PYTORCH_VERSION,
            "device": self.device,
            "reason": reason,
            "savedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preset": self.active_preset,
            "config": self.config,
            "parameterCount": self.model.count_parameters(),
            "tokenizerVersion": self.tokenizer_version,
            "tokenizerChecksum": self.tokenizer_checksum,
            "specialTokens": EXPECTED_SPECIAL_TOKENS,
            "datasetVersion": self.last_metrics.get("datasetVersion") if self.last_metrics else "unknown",
            "seed": self.seed,
            "step": self.step,
            "epoch": self.epoch,
            "metrics": self.last_metrics,
            "checkpointSizeBytes": ckpt_size_bytes,
            "checkpointFile": ckpt_file,
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.last_checkpoint_dir = run_dir
        return {
            "backend": "pytorch",
            "checkpointDir": run_dir,
            "checkpointFile": ckpt_file,
            "manifestFile": manifest_file,
            "checkpointSizeBytes": ckpt_size_bytes,
        }

    def load_checkpoint(self, ckpt_dir_or_file):
        ckpt_file = (
            os.path.join(ckpt_dir_or_file, "checkpoint.pt")
            if os.path.isdir(ckpt_dir_or_file)
            else ckpt_dir_or_file
        )
        if not os.path.exists(ckpt_file):
            return {"success": False, "error": f"PyTorch checkpoint file not found: {ckpt_file}"}

        data = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        if not isinstance(data, dict) or data.get("backend") != "pytorch":
            return {
                "success": False,
                "error": f"Incompatible checkpoint backend '{data.get('backend', 'unknown')}' for PyTorch engine.",
            }

        saved_tok_ver = data.get("tokenizerVersion")
        if saved_tok_ver and saved_tok_ver != EXPECTED_TOKENIZER_VERSION:
            return {
                "success": False,
                "error": f"Tokenizer version mismatch: checkpoint uses '{saved_tok_ver}', expected '{EXPECTED_TOKENIZER_VERSION}'.",
            }

        saved_tok_sum = data.get("tokenizerChecksum", "")
        if self.tokenizer_checksum and saved_tok_sum and saved_tok_sum != self.tokenizer_checksum:
            return {
                "success": False,
                "error": (
                    f"Tokenizer checksum mismatch! Checkpoint expects {saved_tok_sum[:12]}, "
                    f"active tokenizer is {self.tokenizer_checksum[:12]}."
                ),
            }

        cfg = data["config"]
        ok, err_msg, _, _ = self._check_resource_budget(cfg)
        if not ok:
            return {"success": False, "rejected": True, "error": err_msg}

        self.config = cfg
        self.active_preset = data.get("preset", self.active_preset)
        self.seed = data.get("seed", 42)
        self.model = OmnibrainPyTorchTransformer(cfg).to(self.device)
        self.model.load_state_dict(data["modelStateDict"], strict=True)
        self.optimizer = self._build_optimizer(lr=0.008, weight_decay=0.01)
        if "optimizerStateDict" in data:
            self.optimizer.load_state_dict(data["optimizerStateDict"])
        self.step = int(data.get("step", 0))
        self.epoch = int(data.get("epoch", 0))
        self.cumulative_tokens = int(data.get("cumulativeTokens", 0))
        self.seq_cursor = int(data.get("seqCursor", 0))
        self.best_val_loss = float(data.get("bestValLoss", float("inf")))
        if "rngState" in data:
            try:
                torch.set_rng_state(data["rngState"])
            except Exception:
                pass
        self.is_trained = bool(data.get("isTrained", True))
        self.last_checkpoint_dir = os.path.dirname(ckpt_file)

        return {
            "success": True,
            "backend": "pytorch",
            "selectedDevice": self.device,
            "step": self.step,
            "epoch": self.epoch,
            "isTrained": self.is_trained,
            "config": self.config,
            "parameterCount": self.model.count_parameters(),
            "tokenizerVersion": saved_tok_ver,
            "tokenizerChecksum": saved_tok_sum,
            "datasetVersion": data.get("datasetVersion", "unknown"),
        }

    def train_packed(
        self,
        splits_path="./datasets/packed_splits.json",
        epochs=2,
        base_lr=0.008,
        max_steps=None,
        eval_interval=5,
        checkpoint_interval=10,
        resume_from=None,
    ):
        self._load_tokenizer_file("./checkpoints/tokenizer_v1.json")
        tok_ok, tok_err = validate_tokenizer_compatibility(self.tokenizer_meta, self.config)
        if not tok_ok:
            return {"success": False, "error": tok_err}

        if resume_from:
            load_res = self.load_checkpoint(resume_from)
            if not load_res.get("success"):
                return load_res

        if not os.path.exists(splits_path):
            return {"success": False, "error": f"Packed splits file not found: {splits_path}"}

        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        ds_tok_ver = splits.get("tokenizerVersion", "")
        if ds_tok_ver and ds_tok_ver != EXPECTED_TOKENIZER_VERSION:
            return {"success": False, "error": f"Dataset tokenizer version '{ds_tok_ver}' incompatible."}

        ds_tok_sum = splits.get("tokenizerChecksum", "")
        if self.tokenizer_checksum and ds_tok_sum and ds_tok_sum != self.tokenizer_checksum:
            return {"success": False, "error": "Dataset tokenizer checksum does not match active tokenizer."}

        ok, err_msg, est, _ = self._check_resource_budget(self.config)
        if not ok:
            return {"success": False, "rejected": True, "error": err_msg}

        train_seqs = splits.get("trainSequences", [])
        val_seqs = splits.get("valSequences", [])
        if not train_seqs:
            return {"success": False, "error": "No training sequences in packed splits."}

        max_seq = min(self.config["maxSeqLen"], splits.get("contextLength", 48))
        train_seqs = [s[: max_seq + 1] for s in train_seqs]
        val_seqs = [s[: max_seq + 1] for s in val_seqs]

        tracemalloc.start()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        eval_train_seqs = train_seqs[:16]
        init_train_loss, _, _ = self.evaluate_sequences(eval_train_seqs)
        init_val_loss, val_tokens, val_tps = self.evaluate_sequences(val_seqs)

        total_planned_steps = (epochs * len(train_seqs)) if max_steps is None else max_steps
        self.model.train()
        train_t0 = time.perf_counter()
        total_train_tokens = 0
        optimizer_updates = 0
        last_val_loss = init_val_loss
        last_lr = base_lr
        cancelled = False
        history = []

        for ep in range(epochs):
            if cancelled:
                break
            self.epoch = ep + 1
            for seq in train_seqs:
                if max_steps is not None and optimizer_updates >= max_steps:
                    break
                if os.path.exists(self.cancel_flag_path):
                    os.remove(self.cancel_flag_path)
                    cancelled = True
                    break

                inp = torch.tensor(seq[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
                tgt = torch.tensor(seq[1:], dtype=torch.long, device=self.device).unsqueeze(0)

                lr = self._cosine_lr(self.step, max(1, self.step + total_planned_steps), base_lr)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                self.optimizer.zero_grad(set_to_none=True)
                loss, n_toks = self.model.compute_loss(inp, tgt, pad_id=0)
                if n_toks == 0:
                    continue
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item())
                self.optimizer.step()

                self.step += 1
                optimizer_updates += 1
                total_train_tokens += n_toks
                last_lr = lr
                self.is_trained = True

                if optimizer_updates % eval_interval == 0:
                    v_loss, _, _ = self.evaluate_sequences(val_seqs)
                    if v_loss is not None:
                        last_val_loss = v_loss
                    history.append({
                        "step": self.step,
                        "epoch": self.epoch,
                        "trainLoss": round(float(loss.item()), 4),
                        "valLoss": last_val_loss,
                        "lr": round(lr, 6),
                        "gradNorm": round(grad_norm, 4),
                    })

                if optimizer_updates % checkpoint_interval == 0:
                    self.save_checkpoint(reason="periodic")

            if max_steps is not None and optimizer_updates >= max_steps:
                break

        train_elapsed_sec = max(1e-6, time.perf_counter() - train_t0)
        final_train_loss, _, _ = self.evaluate_sequences(eval_train_seqs)
        final_val_loss, val_tokens, val_tps = self.evaluate_sequences(val_seqs)
        _, traced_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gpu_peak_mb = 0.0
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)

        self.last_metrics = {
            "backend": "pytorch",
            "pytorchVersion": PYTORCH_VERSION,
            "selectedDevice": self.device,
            "datasetVersion": splits.get("version", "unknown"),
            "manifestHash": splits.get("manifestHash", ""),
            "cancelled": cancelled,
            "epochsCompleted": self.epoch,
            "optimizerSteps": optimizer_updates,
            "totalStepCounter": self.step,
            "parameterCount": self.model.count_parameters(),
            "tokensProcessed": total_train_tokens,
            "valTokensEvaluated": val_tokens,
            "initialTrainLoss": init_train_loss,
            "finalTrainLoss": final_train_loss,
            "initialValLoss": init_val_loss,
            "finalValLoss": final_val_loss,
            "lastLearningRate": round(last_lr, 6),
            "elapsedSec": round(train_elapsed_sec, 4),
            "trainTokensPerSec": round(total_train_tokens / train_elapsed_sec, 2),
            "valTokensPerSec": val_tps,
            "peakProcessRamMB": _get_peak_ram_mb(),
            "tracedPeakMB": round(traced_peak_bytes / (1024 * 1024), 3),
            "peakGpuMemoryMB": gpu_peak_mb,
            "weightMemoryBytes": est["weightMemoryBytes"],
            "optimizerGradMemoryBytes": est["optimizerGradMemoryBytes"],
            "history": history,
        }

        ckpt_info = self.save_checkpoint(reason="cancelled" if cancelled else "completed")
        self.last_metrics["checkpointSizeBytes"] = ckpt_info["checkpointSizeBytes"]
        self.last_metrics["checkpointDir"] = ckpt_info["checkpointDir"]

        return {
            "success": True,
            "backend": "pytorch",
            "selectedDevice": self.device,
            "cancelled": cancelled,
            "metrics": self.last_metrics,
            "checkpoint": ckpt_info,
        }

    def generate(self, prompt, max_new_tokens=32, temperature=0.8, allow_untrained=False):
        if not self.is_trained and not allow_untrained:
            return {"success": True, "backend": "pytorch", "isTrained": False, "text": "", "error": "Model is not trained yet."}

        if self.tokenizer_vocab:
            tok_to_id = {v: int(k) for k, v in self.tokenizer_vocab.items()}
            tokens = [2] + [tok_to_id.get(ch, 1) for ch in prompt]
        else:
            tokens = [2] + [ord(c) % self.config["vocabSize"] for c in prompt]

        max_ctx = self.config["maxSeqLen"]
        generated = list(tokens)
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ctx = torch.tensor(generated[-max_ctx:], dtype=torch.long, device=self.device).unsqueeze(0)
                logits = self.model.forward(ctx)[0, -1]
                probs = F.softmax(logits / max(0.1, float(temperature)), dim=-1)
                next_tok = int(torch.argmax(probs).item())
                if next_tok == 3:
                    break
                generated.append(next_tok)

        if self.tokenizer_vocab:
            decoded = "".join(self.tokenizer_vocab.get(str(t), "") for t in generated if t not in (0, 2, 3))
        else:
            decoded = "".join(chr(t) if 32 <= t < 127 else "" for t in generated[1:])

        return {
            "success": True,
            "backend": "pytorch",
            "selectedDevice": self.device,
            "isTrained": True,
            "text": decoded,
            "tokensGenerated": len(generated) - len(tokens),
        }

    def generate_sample_completions(self, prompts=None):
        if prompts is None:
            prompts = [
                "Neural networks learn",
                "Science and mathematics",
                "Computer programming algorithms",
            ]
        samples = []
        for p in prompts:
            out = self.generate(p, max_new_tokens=24, temperature=0.7, allow_untrained=True)
            samples.append({
                "prompt": p,
                "completion": out.get("text", ""),
            })
        return samples

    def save_phase7_checkpoint(self, base_dir="./checkpoints/phase7", stage="stageA", reason="stage_complete"):
        os.makedirs(base_dir, exist_ok=True)
        disk = shutil.disk_usage(base_dir)
        if disk.free < 5 * 1024 * 1024:
            raise RuntimeError("Insufficient disk space (< 5 MB free) to save PyTorch Phase 7 checkpoint.")

        if reason == "best_val":
            run_dir = os.path.join(base_dir, "best_val_checkpoint")
        else:
            run_dir = os.path.join(base_dir, f"ckpt_{stage}_{int(self.cumulative_tokens)}tok")
        os.makedirs(run_dir, exist_ok=True)

        ckpt_file = os.path.join(run_dir, "checkpoint.pt")
        manifest_file = os.path.join(run_dir, "manifest.json")

        payload = {
            "backend": "pytorch",
            "phase": "7",
            "stage": stage,
            "softwareVersion": PYTORCH_SOFTWARE_VERSION,
            "pytorchVersion": PYTORCH_VERSION,
            "device": self.device,
            "preset": self.active_preset,
            "config": self.config,
            "seed": self.seed,
            "step": self.step,
            "epoch": self.epoch,
            "cumulativeTokens": self.cumulative_tokens,
            "seqCursor": self.seq_cursor,
            "bestValLoss": self.best_val_loss,
            "isTrained": self.is_trained,
            "tokenizerVersion": self.tokenizer_version,
            "tokenizerChecksum": self.tokenizer_checksum,
            "specialTokens": EXPECTED_SPECIAL_TOKENS,
            "datasetVersion": self.last_metrics.get("datasetVersion") if self.last_metrics else "omnibrain-ds-v2.0.0",
            "manifestHash": self.last_metrics.get("manifestHash") if self.last_metrics else "f080a58dbfad98c2",
            "modelStateDict": self.model.state_dict(),
            "optimizerStateDict": self.optimizer.state_dict(),
            "rngState": torch.get_rng_state(),
        }
        torch.save(payload, ckpt_file)
        ckpt_size_bytes = os.path.getsize(ckpt_file)
        ckpt_sha256 = compute_sha256(ckpt_file)

        manifest = {
            "phase": "7",
            "stage": stage,
            "backend": "pytorch",
            "softwareVersion": PYTORCH_SOFTWARE_VERSION,
            "pytorchVersion": PYTORCH_VERSION,
            "device": self.device,
            "reason": reason,
            "savedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preset": self.active_preset,
            "config": self.config,
            "parameterCount": self.model.count_parameters(),
            "tokenizerVersion": self.tokenizer_version,
            "tokenizerChecksum": self.tokenizer_checksum,
            "specialTokens": EXPECTED_SPECIAL_TOKENS,
            "manifestHash": self.last_metrics.get("manifestHash") if self.last_metrics else "f080a58dbfad98c2",
            "datasetVersion": self.last_metrics.get("datasetVersion") if self.last_metrics else "omnibrain-ds-v2.0.0",
            "quarantineExcluded": ["FineWeb-Edu", "Tatoeba-Indic"],
            "seed": self.seed,
            "step": self.step,
            "epoch": self.epoch,
            "cumulativeTokens": self.cumulative_tokens,
            "seqCursor": self.seq_cursor,
            "bestValLoss": self.best_val_loss,
            "metrics": self.last_metrics,
            "checkpointFile": "checkpoint.pt",
            "checkpointSha256": ckpt_sha256,
            "checkpointSizeBytes": ckpt_size_bytes,
            "qualitativeSamples": self.generate_sample_completions() if self.is_trained else [],
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.last_checkpoint_dir = run_dir
        if reason == "best_val":
            self.best_val_checkpoint_dir = run_dir

        return {
            "backend": "pytorch",
            "checkpointDir": run_dir,
            "checkpointFile": ckpt_file,
            "manifestFile": manifest_file,
            "checkpointSha256": ckpt_sha256,
            "checkpointSizeBytes": ckpt_size_bytes,
        }

    def load_phase7_checkpoint(self, ckpt_dir_or_file):
        if os.path.isdir(ckpt_dir_or_file):
            ckpt_file = os.path.join(ckpt_dir_or_file, "checkpoint.pt")
            manifest_file = os.path.join(ckpt_dir_or_file, "manifest.json")
        else:
            if ckpt_dir_or_file.endswith("manifest.json"):
                manifest_file = ckpt_dir_or_file
                ckpt_file = os.path.join(os.path.dirname(ckpt_dir_or_file), "checkpoint.pt")
            else:
                ckpt_file = ckpt_dir_or_file
                manifest_file = os.path.join(os.path.dirname(ckpt_dir_or_file), "manifest.json")

        if not os.path.exists(ckpt_file):
            return {"success": False, "error": f"Checkpoint file not found: {ckpt_file}"}

        if os.path.exists(manifest_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as mf:
                    manifest_data = json.load(mf)
                expected_sha = manifest_data.get("checkpointSha256")
                if expected_sha:
                    actual_sha = compute_sha256(ckpt_file)
                    if actual_sha != expected_sha:
                        return {
                            "success": False,
                            "error": f"Checkpoint SHA-256 integrity check failed! Expected {expected_sha}, got {actual_sha}",
                        }
            except Exception as e:
                return {"success": False, "error": f"Failed reading manifest: {e}"}

        data = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        if not isinstance(data, dict) or data.get("backend") != "pytorch":
            return {
                "success": False,
                "error": f"Incompatible checkpoint backend '{data.get('backend', 'unknown')}' for PyTorch engine.",
            }

        saved_tok_ver = data.get("tokenizerVersion")
        if saved_tok_ver and saved_tok_ver != EXPECTED_TOKENIZER_VERSION:
            return {
                "success": False,
                "error": f"Tokenizer version mismatch: checkpoint uses '{saved_tok_ver}', expected '{EXPECTED_TOKENIZER_VERSION}'.",
            }

        saved_tok_sum = data.get("tokenizerChecksum", "")
        if self.tokenizer_checksum and saved_tok_sum and saved_tok_sum != self.tokenizer_checksum:
            return {
                "success": False,
                "error": f"Tokenizer checksum mismatch! Checkpoint expects {saved_tok_sum[:12]}, active tokenizer is {self.tokenizer_checksum[:12]}.",
            }

        cfg = data["config"]
        self.config = cfg
        self.active_preset = data.get("preset", self.active_preset)
        self.seed = data.get("seed", 42)
        self.model = OmnibrainPyTorchTransformer(cfg).to(self.device)
        self.model.load_state_dict(data["modelStateDict"], strict=True)
        self.optimizer = self._build_optimizer(lr=0.005, weight_decay=0.01)
        if "optimizerStateDict" in data:
            self.optimizer.load_state_dict(data["optimizerStateDict"])
        if "rngState" in data:
            try:
                torch.set_rng_state(data["rngState"])
            except Exception:
                pass

        self.step = int(data.get("step", 0))
        self.epoch = int(data.get("epoch", 0))
        self.cumulative_tokens = int(data.get("cumulativeTokens", 0))
        self.seq_cursor = int(data.get("seqCursor", 0))
        self.best_val_loss = float(data.get("bestValLoss", float("inf")))
        self.is_trained = bool(data.get("isTrained", True))
        self.last_checkpoint_dir = os.path.dirname(ckpt_file)

        return {
            "success": True,
            "backend": "pytorch",
            "phase": "7",
            "stage": data.get("stage", "unknown"),
            "selectedDevice": self.device,
            "step": self.step,
            "epoch": self.epoch,
            "cumulativeTokens": self.cumulative_tokens,
            "seqCursor": self.seq_cursor,
            "bestValLoss": self.best_val_loss,
            "isTrained": self.is_trained,
            "config": self.config,
            "parameterCount": self.model.count_parameters(),
            "tokenizerVersion": saved_tok_ver,
            "tokenizerChecksum": saved_tok_sum,
            "datasetVersion": data.get("datasetVersion", "unknown"),
            "checkpointSha256": compute_sha256(ckpt_file),
            "checkpointDir": self.last_checkpoint_dir,
        }

    def train_phase7_extended(
        self,
        splits_path="./datasets/packed_splits.json",
        stage="stageA",
        target_tokens=None,
        batch_size=32,
        base_lr=0.005,
        min_lr=0.0002,
        eval_interval_tokens=48000,
        resume_from=None,
        checkpoint_base_dir="./checkpoints/phase7",
        fixed_seed=42,
    ):
        self._load_tokenizer_file("./checkpoints/tokenizer_v1.json")
        tok_ok, tok_err = validate_tokenizer_compatibility(self.tokenizer_meta, self.config)
        if not tok_ok:
            return {"success": False, "error": tok_err}

        stage_token_budgets = {
            "stageA": 500000,
            "stageB": 1000000,
            "stageC": 2000000,
        }
        if target_tokens is None:
            target_tokens = stage_token_budgets.get(stage, 500000)
        target_tokens = int(target_tokens)

        # Resume handling
        if resume_from:
            load_res = self.load_phase7_checkpoint(resume_from)
            if not load_res.get("success"):
                return load_res
        elif stage in ("stageB", "stageC") and self.cumulative_tokens == 0:
            prev_stage = "stageA" if stage == "stageB" else "stageB"
            if os.path.exists(checkpoint_base_dir):
                candidates = [d for d in os.listdir(checkpoint_base_dir) if d.startswith(f"ckpt_{prev_stage}_")]
                if candidates:
                    candidates.sort()
                    prev_ckpt = os.path.join(checkpoint_base_dir, candidates[-1])
                    load_res = self.load_phase7_checkpoint(prev_ckpt)
                    if not load_res.get("success"):
                        return load_res

        if not os.path.exists(splits_path):
            return {"success": False, "error": f"Packed splits file not found: {splits_path}"}

        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        ds_tok_ver = splits.get("tokenizerVersion", "")
        if ds_tok_ver and ds_tok_ver != EXPECTED_TOKENIZER_VERSION:
            return {"success": False, "error": f"Dataset tokenizer version '{ds_tok_ver}' incompatible."}

        ds_tok_sum = splits.get("tokenizerChecksum", "")
        if self.tokenizer_checksum and ds_tok_sum and ds_tok_sum != self.tokenizer_checksum:
            return {"success": False, "error": "Dataset tokenizer checksum does not match active tokenizer."}

        manifest_hash = splits.get("manifestHash", "")
        if manifest_hash != "f080a58dbfad98c2":
            return {"success": False, "error": f"Unexpected manifest hash: {manifest_hash}"}

        if not splits.get("leakageVerifiedZeroOverlap", False):
            return {"success": False, "error": "Corpus does not have verified zero leakage/quarantine exclusion."}

        train_seqs = splits.get("trainSequences", [])
        val_seqs = splits.get("valSequences", [])
        if not train_seqs:
            return {"success": False, "error": "No training sequences in packed splits."}

        max_seq = min(self.config["maxSeqLen"], splits.get("contextLength", 48))
        train_seqs = [s[: max_seq + 1] for s in train_seqs]
        val_seqs = [s[: max_seq + 1] for s in val_seqs]

        if self.cumulative_tokens >= target_tokens:
            return {
                "success": True,
                "backend": "pytorch",
                "phase": "7",
                "stage": stage,
                "message": f"Target token budget of {target_tokens} already satisfied (cumulative={self.cumulative_tokens}).",
                "cumulativeTokens": self.cumulative_tokens,
                "targetTokens": target_tokens,
                "step": self.step,
                "lastMetrics": self.last_metrics,
            }

        tokens_to_process = target_tokens - self.cumulative_tokens
        tokens_per_seq = max_seq
        tokens_per_step = batch_size * tokens_per_seq
        planned_steps = max(1, math.ceil(tokens_to_process / tokens_per_step))

        baseline_samples = self.generate_sample_completions() if not self.is_trained else []

        tracemalloc.start()
        t0 = time.perf_counter()

        val_eval_seqs = val_seqs[:min(100, len(val_seqs))]
        init_val_loss, _, _ = self.evaluate_sequences(val_eval_seqs)
        if math.isinf(self.best_val_loss) and init_val_loss is not None:
            self.best_val_loss = init_val_loss

        self.model.train()
        stage_train_tokens = 0
        stage_steps = 0
        last_val_loss = init_val_loss
        last_train_loss = None
        last_lr = base_lr
        cancelled = False
        diverged = False
        divergence_reason = None
        history = list(self.phase7_history)
        total_train_seqs = len(train_seqs)

        while self.cumulative_tokens < target_tokens:
            if self.seq_cursor >= total_train_seqs:
                break

            if os.path.exists(self.cancel_flag_path):
                os.remove(self.cancel_flag_path)
                cancelled = True
                break

            batch_end = min(self.seq_cursor + batch_size, total_train_seqs)
            batch = train_seqs[self.seq_cursor : batch_end]
            if not batch:
                break

            actual_b = len(batch)
            batch_tensor = torch.tensor(batch, dtype=torch.long, device=self.device)
            inputs = batch_tensor[:, :-1].contiguous()
            targets = batch_tensor[:, 1:].contiguous()

            total_target_steps = max(1, math.ceil(target_tokens / tokens_per_step))
            warmup_steps = max(1, int(total_target_steps * 0.10))
            if self.step < warmup_steps:
                lr = base_lr * float(self.step + 1) / float(warmup_steps)
            else:
                progress = float(self.step - warmup_steps) / float(max(1, total_target_steps - warmup_steps))
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, self.config["vocabSize"]),
                targets.reshape(-1),
                ignore_index=0,
                reduction="mean",
            )

            loss_val = float(loss.item())
            if math.isnan(loss_val) or math.isinf(loss_val) or loss_val > 50.0:
                diverged = True
                divergence_reason = f"Non-finite or extreme loss detected: {loss_val} at step {self.step+1}"
                break

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item())
            self.optimizer.step()

            step_tokens = actual_b * tokens_per_seq
            self.step += 1
            stage_steps += 1
            self.seq_cursor = batch_end
            self.cumulative_tokens += step_tokens
            stage_train_tokens += step_tokens
            last_train_loss = round(loss_val, 4)
            last_lr = lr
            self.is_trained = True

            if stage_train_tokens % eval_interval_tokens < step_tokens or self.cumulative_tokens >= target_tokens:
                v_loss, _, _ = self.evaluate_sequences(val_eval_seqs)
                if v_loss is not None:
                    last_val_loss = v_loss
                    if v_loss < self.best_val_loss:
                        self.best_val_loss = v_loss
                        self.save_phase7_checkpoint(
                            base_dir=checkpoint_base_dir,
                            stage=stage,
                            reason="best_val",
                        )

                history.append({
                    "step": self.step,
                    "stageSteps": stage_steps,
                    "cumulativeTokens": self.cumulative_tokens,
                    "trainLoss": last_train_loss,
                    "valLoss": last_val_loss,
                    "bestValLoss": self.best_val_loss,
                    "lr": round(lr, 6),
                    "gradNorm": round(grad_norm, 4),
                })

        elapsed_sec = max(1e-6, time.perf_counter() - t0)
        _, traced_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        throughput_tps = round(stage_train_tokens / elapsed_sec, 2)
        peak_ram_mb = _get_peak_ram_mb()

        final_val_loss, _, _ = self.evaluate_sequences(val_eval_seqs)
        if final_val_loss is not None and final_val_loss < self.best_val_loss:
            self.best_val_loss = final_val_loss
            self.save_phase7_checkpoint(base_dir=checkpoint_base_dir, stage=stage, reason="best_val")

        post_samples = self.generate_sample_completions()
        self.phase7_history = history

        self.last_metrics = {
            "phase": "7",
            "stage": stage,
            "backend": "pytorch",
            "pytorchVersion": PYTORCH_VERSION,
            "selectedDevice": self.device,
            "datasetVersion": splits.get("version", "omnibrain-ds-v2.0.0"),
            "manifestHash": manifest_hash,
            "tokenizerVersion": self.tokenizer_version,
            "tokenizerChecksum": self.tokenizer_checksum,
            "targetTokens": target_tokens,
            "tokensProcessedThisStage": stage_train_tokens,
            "cumulativeTokens": self.cumulative_tokens,
            "sequencesConsumed": self.seq_cursor,
            "optimizerStepsThisStage": stage_steps,
            "totalStepCounter": self.step,
            "parameterCount": self.model.count_parameters(),
            "initialValLoss": init_val_loss,
            "finalTrainLoss": last_train_loss,
            "finalValLoss": final_val_loss or last_val_loss,
            "bestValLoss": self.best_val_loss,
            "lastLearningRate": round(last_lr, 6),
            "elapsedSec": round(elapsed_sec, 4),
            "trainTokensPerSec": throughput_tps,
            "peakProcessRamMB": peak_ram_mb,
            "tracedPeakMB": round(traced_peak_bytes / (1024 * 1024), 3),
            "cancelled": cancelled,
            "diverged": diverged,
            "divergenceReason": divergence_reason,
            "history": history,
            "baselineSamples": baseline_samples,
            "qualitativeSamples": post_samples,
        }

        reason = "cancelled" if cancelled else ("diverged" if diverged else "stage_complete")
        ckpt_info = self.save_phase7_checkpoint(
            base_dir=checkpoint_base_dir,
            stage=stage,
            reason=reason,
        )
        self.last_metrics["checkpointSha256"] = ckpt_info["checkpointSha256"]
        self.last_metrics["checkpointSizeBytes"] = ckpt_info["checkpointSizeBytes"]
        self.last_metrics["checkpointDir"] = ckpt_info["checkpointDir"]

        return {
            "success": not diverged,
            "backend": "pytorch",
            "phase": "7",
            "stage": stage,
            "cancelled": cancelled,
            "diverged": diverged,
            "divergenceReason": divergence_reason,
            "metrics": self.last_metrics,
            "checkpoint": ckpt_info,
            "bestValCheckpoint": os.path.join(checkpoint_base_dir, "best_val_checkpoint"),
        }


def _matrix_diff_stats(mat_a, mat_b):
    max_abs = 0.0
    max_rel = 0.0
    for r in range(len(mat_a)):
        for c in range(len(mat_a[r])):
            a = float(mat_a[r][c])
            b = float(mat_b[r][c])
            diff = abs(a - b)
            denom = max(abs(a), abs(b), 1e-8)
            rel = diff / denom if (abs(a) > 1e-6 or abs(b) > 1e-6) else 0.0
            if diff > max_abs:
                max_abs = diff
            if rel > max_rel:
                max_rel = rel
    return round(max_abs, 10), round(max_rel, 10)


def run_cross_backend_numerical_verification(config=None, seed=42):
    """
    Compares Pure-Python Reference DecoderTransformer against OmnibrainPyTorchTransformer
    using identical initial weights and fixed seeds.
    """
    from engine import DecoderTransformer

    cfg = config or {
        "vocabSize": 32,
        "dModel": 8,
        "numLayers": 2,
        "numHeads": 2,
        "dFf": 16,
        "maxSeqLen": 16,
        "contextLength": 12,
        "dropout": 0.0,
    }

    if not PYTORCH_AVAILABLE:
        return {
            "success": False,
            "pytorchAvailable": False,
            "error": f"Cannot run PyTorch numerical comparison: {PYTORCH_IMPORT_ERROR}",
        }

    ref_model = DecoderTransformer(cfg, seed=seed)
    pt_model = OmnibrainPyTorchTransformer(cfg)
    pt_model.load_from_reference_params(ref_model.params)
    pt_model.eval()

    inputs = [2, 5, 10, 15, 20]
    targets = [5, 10, 15, 20, 0]  # includes <pad>=0 at last step

    # 1. Forward Logits Comparison
    ref_logits = ref_model.forward(inputs)
    with torch.no_grad():
        pt_logits = pt_model.forward(torch.tensor(inputs, dtype=torch.long).unsqueeze(0))[0].tolist()

    max_logit_diff, max_logit_rel_diff = _matrix_diff_stats(ref_logits, pt_logits)

    # 2. Loss Comparison (padding excluded)
    ref_loss, ref_grads, ref_active = ref_model.compute_loss_and_grads(inputs, targets, pad_id=0)
    pt_model.zero_grad(set_to_none=True)
    pt_loss_t, pt_active = pt_model.compute_loss(
        torch.tensor(inputs, dtype=torch.long).unsqueeze(0),
        torch.tensor(targets, dtype=torch.long).unsqueeze(0),
        pad_id=0,
    )
    pt_loss = float(pt_loss_t.item())
    loss_diff = round(abs(ref_loss - pt_loss), 10)
    loss_rel_diff = round(loss_diff / max(abs(ref_loss), abs(pt_loss), 1e-8), 10)

    # 3. All-Padding Comparison
    pad_targets = [0, 0, 0, 0, 0]
    ref_pad_loss, _, ref_pad_active = ref_model.compute_loss_and_grads(inputs, pad_targets, pad_id=0)
    with torch.no_grad():
        pt_pad_loss_t, pt_pad_active = pt_model.compute_loss(
            torch.tensor(inputs, dtype=torch.long).unsqueeze(0),
            torch.tensor(pad_targets, dtype=torch.long).unsqueeze(0),
            pad_id=0,
        )
    pt_pad_loss = float(pt_pad_loss_t.item())

    # 4. Selected Gradient Comparisons
    pt_loss_t.backward()
    grad_comparisons = {
        "w_out": _matrix_diff_stats(ref_grads["w_out"], pt_model.lm_head.weight.grad.t().tolist()),
        "w_tok": _matrix_diff_stats(ref_grads["w_tok"], pt_model.w_tok.weight.grad.tolist()),
        "l0_wq": _matrix_diff_stats(ref_grads["l0_wq"], pt_model.layers[0].q_proj.weight.grad.t().tolist()),
        "l0_w1": _matrix_diff_stats(ref_grads["l0_w1"], pt_model.layers[0].fc1.weight.grad.t().tolist()),
        "l0_ln1_g": _matrix_diff_stats(ref_grads["l0_ln1_g"], [pt_model.layers[0].ln1.weight.grad.tolist()]),
    }
    max_grad_abs = max(v[0] for v in grad_comparisons.values())
    max_grad_rel = max(v[1] for v in grad_comparisons.values())

    return {
        "success": True,
        "pytorchAvailable": True,
        "pytorchVersion": PYTORCH_VERSION,
        "parameterCountRef": ref_model.count_parameters(),
        "parameterCountPyTorch": pt_model.count_parameters(),
        "activeTokensRef": ref_active,
        "activeTokensPyTorch": pt_active,
        "refLoss": round(ref_loss, 8),
        "pytorchLoss": round(pt_loss, 8),
        "maxLogitDiff": max_logit_diff,
        "maxLogitRelDiff": max_logit_rel_diff,
        "lossDiff": loss_diff,
        "lossRelDiff": loss_rel_diff,
        "paddingCheck": {
            "refPadLoss": ref_pad_loss,
            "pytorchPadLoss": pt_pad_loss,
            "refPadActive": ref_pad_active,
            "pytorchPadActive": pt_pad_active,
        },
        "maxOutputGradDiff": grad_comparisons["w_out"][0],
        "maxGradAbsDiff": max_grad_abs,
        "maxGradRelDiff": max_grad_rel,
        "gradientDetails": {
            k: {"maxAbsDiff": v[0], "maxRelDiff": v[1]} for k, v in grad_comparisons.items()
        },
        "withinFloat32Tolerance": max_logit_diff < 1e-4 and loss_diff < 1e-4 and max_grad_abs < 1e-4,
    }
