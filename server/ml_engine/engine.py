#!/usr/bin/env python3
"""
OMNIBRAIN AI — Independent Foundation Model Engine (Pure Python 3.11)
Zero external AI APIs, zero third-party weights.
Implements decoder-only Transformer with multi-head causal self-attention,
LayerNorm, FFN, full analytical backpropagation, AdamW, cosine LR schedule,
global gradient norm clipping, resource safety checks, and versioned checkpoints.
"""

import sys
import os
import json
import math
import time
import random
import shutil
import hashlib
import resource
import tracemalloc
import platform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_engine import (
    PYTORCH_AVAILABLE,
    PYTORCH_VERSION,
    PYTORCH_IMPORT_ERROR,
    EXPECTED_TOKENIZER_VERSION,
    EXPECTED_SPECIAL_TOKENS,
    compute_sha256,
    get_cuda_info,
    validate_tokenizer_compatibility,
    run_cross_backend_numerical_verification,
    PyTorchFoundationEngine,
)

SOFTWARE_VERSION = "OMNIBRAIN-Core-0.5.2"

MODEL_PRESETS = {
    "tiny_cpu": {
        "name": "Tiny CPU Dev",
        "vocabSize": 1024,
        "dModel": 24,
        "numLayers": 2,
        "numHeads": 2,
        "dFf": 48,
        "maxSeqLen": 64,
        "contextLength": 48,
        "dropout": 0.0,
    },
    "small_exp": {
        "name": "Small Experimental",
        "vocabSize": 1024,
        "dModel": 64,
        "numLayers": 4,
        "numHeads": 4,
        "dFf": 128,
        "maxSeqLen": 128,
        "contextLength": 64,
        "dropout": 0.0,
    },
    "medium_res": {
        "name": "Medium Research",
        "vocabSize": 1024,
        "dModel": 256,
        "numLayers": 8,
        "numHeads": 8,
        "dFf": 1024,
        "maxSeqLen": 256,
        "contextLength": 128,
        "dropout": 0.1,
    },
    "large_gpu": {
        "name": "Large GPU Future (1.2B Scale)",
        "vocabSize": 32000,
        "dModel": 2048,
        "numLayers": 24,
        "numHeads": 16,
        "dFf": 8192,
        "maxSeqLen": 2048,
        "contextLength": 1024,
        "dropout": 0.1,
    },
}


def get_available_ram_bytes():
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            mem_avail_kb = None
            mem_free_kb = None
            mem_total_kb = None
            for line in lines:
                if line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
                elif line.startswith("MemFree:"):
                    mem_free_kb = int(line.split()[1])
                elif line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
            kb = mem_avail_kb or mem_free_kb or mem_total_kb or 0
            return kb * 1024
        except Exception:
            pass
    # Windows fallback via ctypes
    if os.name == "nt":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullAvailPhys)
        except Exception:
            pass
    return 1024 * 1024 * 1024


def get_peak_ram_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(usage / (1024 * 1024), 3)
    return round(usage / 1024.0, 3)


def get_total_ram_bytes():
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return get_available_ram_bytes()


def get_cpu_model():
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        val = line.split(":", 1)[1].strip()
                        if val and val.lower() != "unknown":
                            return val
        except Exception:
            pass
    proc = platform.processor()
    if proc and proc.lower() != "unknown":
        return proc
    return f"{platform.machine() or 'x86_64'} (virtualized)"


def get_cpu_usage_pct():
    try:
        load1, _, _ = os.getloadavg()
        cpus = os.cpu_count() or 1
        return round(min(100.0, (load1 / cpus) * 100.0), 2)
    except Exception:
        return None


def detect_hardware(selected_device="cpu"):
    avail_ram = get_available_ram_bytes()
    total_ram = get_total_ram_bytes()
    disk = shutil.disk_usage(".")
    cuda = get_cuda_info()
    # Never claim GPU acceleration when running on CPU
    actual_device = "cuda" if (selected_device == "cuda" and cuda["cudaAvailable"]) else "cpu"
    return {
        "os": platform.platform(),
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version.split()[0],
        "pytorchAvailable": PYTORCH_AVAILABLE,
        "pytorchVersion": PYTORCH_VERSION,
        "pytorchStatus": f"PyTorch {PYTORCH_VERSION}" if PYTORCH_AVAILABLE else f"Unavailable ({PYTORCH_IMPORT_ERROR})",
        "cpuModel": get_cpu_model(),
        "cpuCount": os.cpu_count() or 1,
        "cpuUsagePct": get_cpu_usage_pct(),
        "totalRamMB": round(total_ram / (1024 * 1024), 1),
        "availableRamMB": round(avail_ram / (1024 * 1024), 1),
        "peakProcessRamMB": get_peak_ram_mb(),
        "freeDiskMB": round(disk.free / (1024 * 1024), 1),
        "cudaAvailable": cuda["cudaAvailable"],
        "gpuName": cuda["gpuName"],
        "gpuMemoryMB": cuda["gpuMemoryMB"],
        "gpuAvailable": cuda["cudaAvailable"],
        "device": actual_device,
        "selectedDevice": actual_device,
    }


def estimate_config_resources(cfg, batch_size=2):
    v = cfg["vocabSize"]
    d = cfg["dModel"]
    L = cfg["numLayers"]
    dff = cfg["dFf"]
    s = cfg["maxSeqLen"]

    tok_params = v * d
    pos_params = s * d
    layer_params = L * (
        4 * (d * d) + 4 * d +   # Wq, Wk, Wv, Wo + biases
        2 * (d * dff) + dff + d + # W1, b1, W2, b2
        4 * d                     # LN1 (gamma, beta) + LN2 (gamma, beta)
    )
    final_ln = 2 * d
    out_head = d * v + v
    total_params = tok_params + pos_params + layer_params + final_ln + out_head

    # Float32 = 4 bytes per param; Python list floats take ~32 bytes overhead,
    # but standard tensor/float32 estimate is 4 bytes/param.
    weight_bytes = total_params * 4
    # AdamW stores m + v + gradient = 3 * weight_bytes
    optimizer_grad_bytes = weight_bytes * 3
    seq = cfg.get("contextLength", s)
    activation_bytes = batch_size * seq * d * L * 16 * 4
    total_estimated_bytes = weight_bytes + optimizer_grad_bytes + activation_bytes

    return {
        "parameterCount": total_params,
        "weightMemoryBytes": weight_bytes,
        "optimizerGradMemoryBytes": optimizer_grad_bytes,
        "activationMemoryBytes": activation_bytes,
        "totalEstimatedBytes": total_estimated_bytes,
        "estimatedMemoryMB": round(total_estimated_bytes / (1024 * 1024), 3),
    }


class DecoderTransformer:
    def __init__(self, config, seed=42):
        self.config = dict(config)
        self.seed = seed
        self.rng = random.Random(seed)
        self.is_trained = False
        self.step = 0
        self.epoch = 0
        self.adam_t = 0

        d = self.config["dModel"]
        v = self.config["vocabSize"]
        s = self.config["maxSeqLen"]
        dff = self.config["dFf"]
        L = self.config["numLayers"]

        if d % self.config["numHeads"] != 0:
            raise ValueError(f"dModel ({d}) must be divisible by numHeads ({self.config['numHeads']})")

        self.params = {}
        self.params["w_tok"] = self._xavier(v, d)
        self.params["w_pos"] = self._xavier(s, d)

        for l in range(L):
            self.params[f"l{l}_ln1_g"] = [[1.0] * d]
            self.params[f"l{l}_ln1_b"] = [[0.0] * d]
            self.params[f"l{l}_wq"] = self._xavier(d, d)
            self.params[f"l{l}_bq"] = [[0.0] * d]
            self.params[f"l{l}_wk"] = self._xavier(d, d)
            self.params[f"l{l}_bk"] = [[0.0] * d]
            self.params[f"l{l}_wv"] = self._xavier(d, d)
            self.params[f"l{l}_bv"] = [[0.0] * d]
            self.params[f"l{l}_wo"] = self._xavier(d, d)
            self.params[f"l{l}_bo"] = [[0.0] * d]
            self.params[f"l{l}_ln2_g"] = [[1.0] * d]
            self.params[f"l{l}_ln2_b"] = [[0.0] * d]
            self.params[f"l{l}_w1"] = self._xavier(d, dff)
            self.params[f"l{l}_b1"] = [[0.0] * dff]
            self.params[f"l{l}_w2"] = self._xavier(dff, d)
            self.params[f"l{l}_b2"] = [[0.0] * d]

        self.params["lnf_g"] = [[1.0] * d]
        self.params["lnf_b"] = [[0.0] * d]
        self.params["w_out"] = self._xavier(d, v)
        self.params["b_out"] = [[0.0] * v]

        # AdamW moment states
        self.m = {k: [[0.0] * len(row) for row in mat] for k, mat in self.params.items()}
        self.v = {k: [[0.0] * len(row) for row in mat] for k, mat in self.params.items()}

    def _xavier(self, rows, cols):
        scale = math.sqrt(2.0 / (rows + cols))
        return [[self.rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    def count_parameters(self):
        return sum(len(row) for mat in self.params.values() for row in mat)

    def _zeros_like_params(self):
        return {k: [[0.0] * len(row) for row in mat] for k, mat in self.params.items()}

    @staticmethod
    def _layernorm_fwd(x, gamma_row, beta_row, eps=1e-5):
        T = len(x)
        D = len(x[0])
        out = []
        cache = []
        for t in range(T):
            row = x[t]
            mean = sum(row) / D
            var = sum((v - mean) ** 2 for v in row) / D
            inv_std = 1.0 / math.sqrt(var + eps)
            x_hat = [(v - mean) * inv_std for v in row]
            y = [gamma_row[d] * x_hat[d] + beta_row[d] for d in range(D)]
            out.append(y)
            cache.append((x_hat, inv_std))
        return out, cache

    @staticmethod
    def _layernorm_bwd(dout, cache, gamma_row):
        T = len(dout)
        D = len(dout[0])
        dx = []
        dgamma = [0.0] * D
        dbeta = [0.0] * D
        for t in range(T):
            dy = dout[t]
            x_hat, inv_std = cache[t]
            for d in range(D):
                dbeta[d] += dy[d]
                dgamma[d] += dy[d] * x_hat[d]
            dx_hat = [dy[d] * gamma_row[d] for d in range(D)]
            sum_dxh = sum(dx_hat)
            sum_dxh_xh = sum(dx_hat[d] * x_hat[d] for d in range(D))
            dx_row = [
                (inv_std / D) * (D * dx_hat[d] - sum_dxh - x_hat[d] * sum_dxh_xh)
                for d in range(D)
            ]
            dx.append(dx_row)
        return dx, dgamma, dbeta

    @staticmethod
    def _linear_fwd(x, w, b_row):
        # x: [T, in_d], w: [in_d, out_d], b_row: [out_d]
        T = len(x)
        in_d = len(w)
        out_d = len(w[0])
        out = []
        for t in range(T):
            xt = x[t]
            row = [b_row[j] for j in range(out_d)]
            for i in range(in_d):
                xi = xt[i]
                if xi == 0.0:
                    continue
                wi = w[i]
                for j in range(out_d):
                    row[j] += xi * wi[j]
            out.append(row)
        return out

    @staticmethod
    def _linear_bwd(dout, x, w):
        T = len(dout)
        in_d = len(w)
        out_d = len(w[0])
        dx = [[0.0] * in_d for _ in range(T)]
        dw = [[0.0] * out_d for _ in range(in_d)]
        db = [0.0] * out_d
        for t in range(T):
            dy = dout[t]
            xt = x[t]
            for j in range(out_d):
                dyj = dy[j]
                db[j] += dyj
            for i in range(in_d):
                xi = xt[i]
                wi = w[i]
                dwi = dw[i]
                acc = 0.0
                for j in range(out_d):
                    dyj = dy[j]
                    acc += dyj * wi[j]
                    dwi[j] += xi * dyj
                dx[t][i] = acc
        return dx, dw, db

    def forward_with_cache(self, token_ids):
        T = len(token_ids)
        D = self.config["dModel"]
        V = self.config["vocabSize"]
        L = self.config["numLayers"]
        H = self.config["numHeads"]
        d_k = D // H

        if T > self.config["maxSeqLen"]:
            raise ValueError(f"Sequence length {T} exceeds maxSeqLen {self.config['maxSeqLen']}")

        w_tok = self.params["w_tok"]
        w_pos = self.params["w_pos"]

        h = []
        tok_indices = []
        for t in range(T):
            tid = token_ids[t] % V
            tok_indices.append(tid)
            h.append([w_tok[tid][d] + w_pos[t][d] for d in range(D)])

        layer_caches = []
        for l in range(L):
            h_in = h
            ln1_out, ln1_cache = self._layernorm_fwd(h_in, self.params[f"l{l}_ln1_g"][0], self.params[f"l{l}_ln1_b"][0])

            q = self._linear_fwd(ln1_out, self.params[f"l{l}_wq"], self.params[f"l{l}_bq"][0])
            k = self._linear_fwd(ln1_out, self.params[f"l{l}_wk"], self.params[f"l{l}_bk"][0])
            v = self._linear_fwd(ln1_out, self.params[f"l{l}_wv"], self.params[f"l{l}_bv"][0])

            # Multi-head causal self-attention
            attn_concat = [[0.0] * D for _ in range(T)]
            attn_weights_all = [] # [H][T][T]
            inv_sqrt_dk = 1.0 / math.sqrt(d_k)

            for head in range(H):
                s_d = head * d_k
                head_weights = []
                for i in range(T):
                    scores = []
                    for j in range(i + 1): # Causal mask: j <= i ONLY
                        dot = sum(q[i][s_d + u] * k[j][s_d + u] for u in range(d_k))
                        scores.append(dot * inv_sqrt_dk)
                    max_s = max(scores)
                    exps = [math.exp(sc - max_s) for sc in scores]
                    sum_e = sum(exps)
                    probs = [e / sum_e for e in exps]
                    # Pad remaining future weights with 0.0 for full T vector
                    full_probs = probs + [0.0] * (T - i - 1)
                    head_weights.append(full_probs)

                    for u in range(d_k):
                        val = sum(probs[j] * v[j][s_d + u] for j in range(i + 1))
                        attn_concat[i][s_d + u] = val
                attn_weights_all.append(head_weights)

            attn_proj = self._linear_fwd(attn_concat, self.params[f"l{l}_wo"], self.params[f"l{l}_bo"][0])
            h_mid = [[h_in[t][d] + attn_proj[t][d] for d in range(D)] for t in range(T)]

            ln2_out, ln2_cache = self._layernorm_fwd(h_mid, self.params[f"l{l}_ln2_g"][0], self.params[f"l{l}_ln2_b"][0])
            ffn_pre = self._linear_fwd(ln2_out, self.params[f"l{l}_w1"], self.params[f"l{l}_b1"][0])
            ffn_act = [[max(0.0, val) for val in row] for row in ffn_pre]
            ffn_out = self._linear_fwd(ffn_act, self.params[f"l{l}_w2"], self.params[f"l{l}_b2"][0])

            h = [[h_mid[t][d] + ffn_out[t][d] for d in range(D)] for t in range(T)]

            layer_caches.append({
                "h_in": h_in,
                "ln1_out": ln1_out,
                "ln1_cache": ln1_cache,
                "q": q,
                "k": k,
                "v": v,
                "attn_weights": attn_weights_all,
                "attn_concat": attn_concat,
                "h_mid": h_mid,
                "ln2_out": ln2_out,
                "ln2_cache": ln2_cache,
                "ffn_pre": ffn_pre,
                "ffn_act": ffn_act,
            })

        lnf_out, lnf_cache = self._layernorm_fwd(h, self.params["lnf_g"][0], self.params["lnf_b"][0])
        logits = self._linear_fwd(lnf_out, self.params["w_out"], self.params["b_out"][0])

        cache = {
            "tok_indices": tok_indices,
            "h_pre_lnf": h,
            "lnf_out": lnf_out,
            "lnf_cache": lnf_cache,
            "layer_caches": layer_caches,
        }
        return logits, cache

    def forward(self, token_ids):
        logits, _ = self.forward_with_cache(token_ids)
        return logits

    def compute_loss(self, inputs, targets, pad_id=0):
        logits, _ = self.forward_with_cache(inputs)
        T = len(inputs)
        V = self.config["vocabSize"]
        active_positions = [t for t in range(T) if targets[t] != pad_id]
        if not active_positions:
            return 0.0, 0
        total_loss = 0.0
        for t in active_positions:
            row = logits[t]
            tgt = targets[t] % V
            max_l = max(row)
            exps = [math.exp(x - max_l) for x in row]
            sum_e = sum(exps)
            prob_tgt = exps[tgt] / sum_e
            total_loss += -math.log(max(1e-15, prob_tgt))
        return total_loss / len(active_positions), len(active_positions)

    def compute_loss_and_grads(self, inputs, targets, pad_id=0):
        logits, cache = self.forward_with_cache(inputs)
        T = len(inputs)
        V = self.config["vocabSize"]
        D = self.config["dModel"]
        L = self.config["numLayers"]
        H = self.config["numHeads"]
        d_k = D // H

        # Count non-pad tokens
        active_positions = [t for t in range(T) if targets[t] != pad_id]
        if not active_positions:
            return 0.0, self._zeros_like_params(), 0

        total_loss = 0.0
        dlogits = [[0.0] * V for _ in range(T)]
        inv_n = 1.0 / len(active_positions)

        for t in active_positions:
            row = logits[t]
            tgt = targets[t] % V
            max_l = max(row)
            exps = [math.exp(x - max_l) for x in row]
            sum_e = sum(exps)
            probs = [e / sum_e for e in exps]
            total_loss += -math.log(max(1e-15, probs[tgt]))
            for v_i in range(V):
                dlogits[t][v_i] = (probs[v_i] - (1.0 if v_i == tgt else 0.0)) * inv_n

        grads = self._zeros_like_params()

        # Backward through LM head
        dh_lnf, dw_out, db_out = self._linear_bwd(dlogits, cache["lnf_out"], self.params["w_out"])
        grads["w_out"] = dw_out
        grads["b_out"] = [db_out]

        # Backward through Final LayerNorm
        dh, dlnf_g, dlnf_b = self._layernorm_bwd(dh_lnf, cache["lnf_cache"], self.params["lnf_g"][0])
        grads["lnf_g"] = [dlnf_g]
        grads["lnf_b"] = [dlnf_b]

        # Backward through Transformer layers
        for l in range(L - 1, -1, -1):
            lc = cache["layer_caches"][l]

            # Residual 2: h = h_mid + ffn_out
            dffn_out = dh
            dh_mid_from_res = dh

            dffn_act, dw2, db2 = self._linear_bwd(dffn_out, lc["ffn_act"], self.params[f"l{l}_w2"])
            grads[f"l{l}_w2"] = dw2
            grads[f"l{l}_b2"] = [db2]

            # ReLU backward
            dffn_pre = [
                [dffn_act[t][j] if lc["ffn_pre"][t][j] > 0.0 else 0.0 for j in range(len(dffn_act[0]))]
                for t in range(T)
            ]

            dln2_out, dw1, db1 = self._linear_bwd(dffn_pre, lc["ln2_out"], self.params[f"l{l}_w1"])
            grads[f"l{l}_w1"] = dw1
            grads[f"l{l}_b1"] = [db1]

            dh_mid_from_ln2, dln2_g, dln2_b = self._layernorm_bwd(dln2_out, lc["ln2_cache"], self.params[f"l{l}_ln2_g"][0])
            grads[f"l{l}_ln2_g"] = [dln2_g]
            grads[f"l{l}_ln2_b"] = [dln2_b]

            dh_mid = [[dh_mid_from_res[t][d] + dh_mid_from_ln2[t][d] for d in range(D)] for t in range(T)]

            # Residual 1: h_mid = h_in + attn_proj
            dattn_proj = dh_mid
            dh_in_from_res = dh_mid

            dattn_concat, dwo, dbo = self._linear_bwd(dattn_proj, lc["attn_concat"], self.params[f"l{l}_wo"])
            grads[f"l{l}_wo"] = dwo
            grads[f"l{l}_bo"] = [dbo]

            # Multi-head causal self-attention backward
            dq = [[0.0] * D for _ in range(T)]
            dk = [[0.0] * D for _ in range(T)]
            dv = [[0.0] * D for _ in range(T)]
            inv_sqrt_dk = 1.0 / math.sqrt(d_k)

            q = lc["q"]
            k = lc["k"]
            v = lc["v"]

            for head in range(H):
                s_d = head * d_k
                hw = lc["attn_weights"][head]
                for i in range(T):
                    d_probs = [0.0] * (i + 1)
                    for j in range(i + 1):
                        w_ij = hw[i][j]
                        dp = 0.0
                        for u in range(d_k):
                            dout_iu = dattn_concat[i][s_d + u]
                            dp += dout_iu * v[j][s_d + u]
                            dv[j][s_d + u] += w_ij * dout_iu
                        d_probs[j] = dp

                    # Softmax backward for row i over j <= i
                    dot_p_dp = sum(hw[i][j] * d_probs[j] for j in range(i + 1))
                    for j in range(i + 1):
                        dscore = hw[i][j] * (d_probs[j] - dot_p_dp) * inv_sqrt_dk
                        for u in range(d_k):
                            dq[i][s_d + u] += dscore * k[j][s_d + u]
                            dk[j][s_d + u] += dscore * q[i][s_d + u]

            dln1_q, dwq, dbq = self._linear_bwd(dq, lc["ln1_out"], self.params[f"l{l}_wq"])
            dln1_k, dwk, dbk = self._linear_bwd(dk, lc["ln1_out"], self.params[f"l{l}_wk"])
            dln1_v, dwv, dbv = self._linear_bwd(dv, lc["ln1_out"], self.params[f"l{l}_wv"])

            grads[f"l{l}_wq"] = dwq
            grads[f"l{l}_bq"] = [dbq]
            grads[f"l{l}_wk"] = dwk
            grads[f"l{l}_bk"] = [dbk]
            grads[f"l{l}_wv"] = dwv
            grads[f"l{l}_bv"] = [dbv]

            dln1_out = [
                [dln1_q[t][d] + dln1_k[t][d] + dln1_v[t][d] for d in range(D)]
                for t in range(T)
            ]

            dh_in_from_ln1, dln1_g, dln1_b = self._layernorm_bwd(dln1_out, lc["ln1_cache"], self.params[f"l{l}_ln1_g"][0])
            grads[f"l{l}_ln1_g"] = [dln1_g]
            grads[f"l{l}_ln1_b"] = [dln1_b]

            dh = [[dh_in_from_res[t][d] + dh_in_from_ln1[t][d] for d in range(D)] for t in range(T)]

        # Embeddings backward
        for t in range(T):
            tid = cache["tok_indices"][t]
            for d in range(D):
                grads["w_tok"][tid][d] += dh[t][d]
                grads["w_pos"][t][d] += dh[t][d]

        avg_loss = total_loss * inv_n
        return avg_loss, grads, len(active_positions)

    def adamw_step(self, grads, lr, max_grad_norm=1.0, weight_decay=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        # Global L2 gradient norm clipping
        sq_norm = 0.0
        for mat in grads.values():
            for row in mat:
                for g in row:
                    sq_norm += g * g
        grad_norm = math.sqrt(sq_norm)
        clip_coef = min(1.0, max_grad_norm / (grad_norm + 1e-6))

        self.adam_t += 1
        bc1 = 1.0 - (beta1 ** self.adam_t)
        bc2 = 1.0 - (beta2 ** self.adam_t)

        updated_params = 0
        for name, mat in self.params.items():
            gmat = grads[name]
            mmat = self.m[name]
            vmat = self.v[name]
            apply_wd = not (name.endswith("_b") or "_ln" in name or name.startswith("b_"))

            for i in range(len(mat)):
                row = mat[i]
                grow = gmat[i]
                mrow = mmat[i]
                vrow = vmat[i]
                for j in range(len(row)):
                    g = grow[j] * clip_coef
                    mrow[j] = beta1 * mrow[j] + (1.0 - beta1) * g
                    vrow[j] = beta2 * vrow[j] + (1.0 - beta2) * (g * g)
                    m_hat = mrow[j] / bc1
                    v_hat = vrow[j] / bc2
                    wd_term = weight_decay * row[j] if apply_wd else 0.0
                    row[j] -= lr * (m_hat / (math.sqrt(v_hat) + eps) + wd_term)
                    updated_params += 1

        self.step += 1
        return grad_norm, updated_params


class PythonFoundationEngine:
    def __init__(self):
        self.active_preset = "tiny_cpu"
        self.config = dict(MODEL_PRESETS["tiny_cpu"])
        self.seed = 42
        self.model = DecoderTransformer(self.config, seed=self.seed)
        self.cumulative_tokens = 0
        self.seq_cursor = 0
        self.best_val_loss = float("inf")
        self.best_val_checkpoint_dir = None
        self.phase7_history = []
        self.tokenizer_vocab = None
        self.tokenizer_version = EXPECTED_TOKENIZER_VERSION
        self.tokenizer_checksum = ""
        self.tokenizer_meta = None
        self.last_metrics = None
        self.last_checkpoint_dir = None
        self.cancel_flag_path = os.path.abspath("./checkpoints/.cancel_training")
        self._load_tokenizer_file("./checkpoints/tokenizer_v1.json")

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

    def set_config(self, config_name, seed=42):
        if config_name not in MODEL_PRESETS:
            return {"success": False, "error": f"Unknown preset: {config_name}"}

        cfg = dict(MODEL_PRESETS[config_name])
        est = estimate_config_resources(cfg)
        avail_ram = get_available_ram_bytes()
        safe_limit = int(avail_ram * 0.70)

        if est["totalEstimatedBytes"] > safe_limit:
            avail_mb = round(avail_ram / (1024 * 1024), 1)
            need_mb = est["estimatedMemoryMB"]
            return {
                "success": False,
                "rejected": True,
                "error": (
                    f"Insufficient RAM for '{cfg['name']}': requires ~{need_mb} MB "
                    f"(weights + AdamW + activations), exceeding 70% of available {avail_mb} MB system RAM."
                ),
                "estimates": est,
                "availableRamMB": avail_mb,
            }

        tok_ok, tok_err = validate_tokenizer_compatibility(self.tokenizer_meta, cfg)
        if not tok_ok:
            return {"success": False, "rejected": True, "error": tok_err}

        self.active_preset = config_name
        self.config = cfg
        self.seed = seed
        self.model = DecoderTransformer(self.config, seed=seed)
        self.cumulative_tokens = 0
        self.seq_cursor = 0
        self.best_val_loss = float("inf")
        self.best_val_checkpoint_dir = None
        self.phase7_history = []
        return {
            "success": True,
            "backend": "python_ref",
            "selectedDevice": "cpu",
            "preset": config_name,
            "config": self.config,
            "estimates": est,
            "parameterCount": self.model.count_parameters(),
        }

    def get_telemetry(self):
        est = estimate_config_resources(self.config)
        return {
            "success": True,
            "backend": "python_ref",
            "softwareVersion": SOFTWARE_VERSION,
            "preset": self.active_preset,
            "config": self.config,
            "isTrained": self.model.is_trained,
            "step": self.model.step,
            "epoch": self.model.epoch,
            "seed": self.seed,
            "tokenizerVersion": self.tokenizer_version,
            "tokenizerChecksum": self.tokenizer_checksum,
            "estimates": est,
            "hardware": detect_hardware(selected_device="cpu"),
            "lastMetrics": self.last_metrics,
            "lastCheckpointDir": self.last_checkpoint_dir,
        }

    def _cosine_lr(self, step, total_steps, base_lr, min_lr=1e-4):
        warmup = max(1, int(total_steps * 0.15))
        if step < warmup:
            return base_lr * float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def evaluate_sequences(self, sequences):
        if not sequences:
            return None, 0, 0.0
        t0 = time.perf_counter()
        total_loss = 0.0
        total_tokens = 0
        seqs_used = 0
        for seq in sequences:
            if len(seq) < 2:
                continue
            inputs = seq[:-1]
            targets = seq[1:]
            loss, n_toks = self.model.compute_loss(inputs, targets, pad_id=0)
            if n_toks > 0:
                total_loss += loss
                total_tokens += n_toks
                seqs_used += 1
        elapsed = max(1e-6, time.perf_counter() - t0)
        avg_loss = total_loss / max(1, seqs_used)
        tps = total_tokens / elapsed
        return round(avg_loss, 4), total_tokens, round(tps, 2)

    def save_checkpoint(self, base_dir="./checkpoints", reason="periodic"):
        os.makedirs(base_dir, exist_ok=True)
        disk = shutil.disk_usage(base_dir)
        if disk.free < 5 * 1024 * 1024:
            raise RuntimeError("Insufficient disk space (< 5 MB free) to save checkpoint safely.")

        run_dir = os.path.join(
            base_dir,
            f"ckpt_{self.active_preset}_step_{self.model.step}_{int(time.time() * 1000)}"
        )
        os.makedirs(run_dir, exist_ok=True)

        ckpt_file = os.path.join(run_dir, "checkpoint.json")
        manifest_file = os.path.join(run_dir, "manifest.json")

        payload = {
            "backend": "python_ref",
            "softwareVersion": SOFTWARE_VERSION,
            "preset": self.active_preset,
            "config": self.config,
            "seed": self.seed,
            "step": self.model.step,
            "epoch": self.model.epoch,
            "adamT": self.model.adam_t,
            "isTrained": self.model.is_trained,
            "tokenizerVersion": self.tokenizer_version,
            "tokenizerChecksum": self.tokenizer_checksum,
            "specialTokens": EXPECTED_SPECIAL_TOKENS,
            "datasetVersion": self.last_metrics.get("datasetVersion") if self.last_metrics else "unknown",
            "params": self.model.params,
            "optimizerState": {"m": self.model.m, "v": self.model.v},
        }
        with open(ckpt_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        ckpt_size_bytes = os.path.getsize(ckpt_file)

        manifest = {
            "backend": "python_ref",
            "softwareVersion": SOFTWARE_VERSION,
            "device": "cpu",
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
            "step": self.model.step,
            "epoch": self.model.epoch,
            "metrics": self.last_metrics,
            "checkpointSizeBytes": ckpt_size_bytes,
            "checkpointFile": ckpt_file,
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.last_checkpoint_dir = run_dir
        return {
            "backend": "python_ref",
            "checkpointDir": run_dir,
            "checkpointFile": ckpt_file,
            "manifestFile": manifest_file,
            "checkpointSizeBytes": ckpt_size_bytes,
        }

    def load_checkpoint(self, ckpt_dir_or_file):
        if os.path.isdir(ckpt_dir_or_file):
            pt_file = os.path.join(ckpt_dir_or_file, "checkpoint.pt")
            json_file = os.path.join(ckpt_dir_or_file, "checkpoint.json")
            if os.path.exists(pt_file) and not os.path.exists(json_file):
                return {
                    "success": False,
                    "error": "Checkpoint belongs to the 'pytorch' backend (checkpoint.pt); active backend is 'python_ref'. Switch backend to 'pytorch' to load it.",
                }
            ckpt_file = json_file
        else:
            if ckpt_dir_or_file.endswith(".pt"):
                return {
                    "success": False,
                    "error": "PyTorch (.pt) checkpoints cannot be loaded by the 'python_ref' backend without switching to 'pytorch'.",
                }
            ckpt_file = ckpt_dir_or_file

        if not os.path.exists(ckpt_file):
            return {"success": False, "error": f"Checkpoint file not found: {ckpt_file}"}

        with open(ckpt_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ckpt_backend = data.get("backend", "python_ref")
        if ckpt_backend != "python_ref":
            return {
                "success": False,
                "error": f"Checkpoint backend mismatch: checkpoint is '{ckpt_backend}', active backend is 'python_ref'.",
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

        self.config = data["config"]
        self.active_preset = data.get("preset", self.active_preset)
        self.seed = data.get("seed", 42)
        self.model = DecoderTransformer(self.config, seed=self.seed)
        self.model.params = data["params"]
        self.model.m = data["optimizerState"]["m"]
        self.model.v = data["optimizerState"]["v"]
        self.model.step = data["step"]
        self.model.epoch = data["epoch"]
        self.model.adam_t = data.get("adamT", data["step"])
        self.model.is_trained = data.get("isTrained", True)
        self.last_checkpoint_dir = os.path.dirname(ckpt_file)

        return {
            "success": True,
            "step": self.model.step,
            "epoch": self.model.epoch,
            "isTrained": self.model.is_trained,
            "config": self.config,
            "parameterCount": self.model.count_parameters(),
            "tokenizerChecksum": saved_tok_sum,
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

        if resume_from:
            load_res = self.load_checkpoint(resume_from)
            if not load_res.get("success"):
                return load_res

        if not os.path.exists(splits_path):
            return {"success": False, "error": f"Packed splits file not found: {splits_path}"}

        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        # Verify tokenizer compatibility with packed dataset
        ds_tok_sum = splits.get("tokenizerChecksum", "")
        if self.tokenizer_checksum and ds_tok_sum and ds_tok_sum != self.tokenizer_checksum:
            return {
                "success": False,
                "error": "Dataset tokenizer checksum does not match active tokenizer.",
            }

        # Resource safety check before starting
        est = estimate_config_resources(self.config)
        avail_ram = get_available_ram_bytes()
        if est["totalEstimatedBytes"] > int(avail_ram * 0.70):
            return {
                "success": False,
                "error": "Model memory requirement exceeds 70% of available system RAM.",
            }

        train_seqs = splits.get("trainSequences", [])
        val_seqs = splits.get("valSequences", [])
        if not train_seqs:
            return {"success": False, "error": "No training sequences in packed splits."}

        # Trim sequences to model contextLength/maxSeqLen
        max_seq = min(self.config["maxSeqLen"], splits.get("contextLength", 48))
        train_seqs = [s[: max_seq + 1] for s in train_seqs]
        val_seqs = [s[: max_seq + 1] for s in val_seqs]

        tracemalloc.start()

        # Measure initial train (deterministic subset) & full validation loss before optimizer steps
        eval_train_seqs = train_seqs[:16]
        init_train_loss, _, _ = self.evaluate_sequences(eval_train_seqs)
        init_val_loss, val_tokens, val_tps = self.evaluate_sequences(val_seqs)

        total_planned_steps = (epochs * len(train_seqs)) if max_steps is None else max_steps
        train_t0 = time.perf_counter()
        total_train_tokens = 0
        optimizer_updates = 0
        last_train_loss = init_train_loss
        last_val_loss = init_val_loss
        last_lr = base_lr
        cancelled = False
        history = []

        for ep in range(epochs):
            if cancelled:
                break
            self.model.epoch = ep + 1
            for seq in train_seqs:
                if max_steps is not None and optimizer_updates >= max_steps:
                    break
                if os.path.exists(self.cancel_flag_path):
                    os.remove(self.cancel_flag_path)
                    cancelled = True
                    break

                inputs = seq[:-1]
                targets = seq[1:]
                loss, grads, n_toks = self.model.compute_loss_and_grads(inputs, targets, pad_id=0)
                if n_toks == 0:
                    continue

                lr = self._cosine_lr(self.model.step, max(1, self.model.step + total_planned_steps), base_lr)
                grad_norm, _ = self.model.adamw_step(grads, lr=lr, max_grad_norm=1.0)

                optimizer_updates += 1
                total_train_tokens += n_toks
                last_train_loss = loss
                last_lr = lr
                self.model.is_trained = True

                if optimizer_updates % eval_interval == 0:
                    v_loss, _, _ = self.evaluate_sequences(val_seqs)
                    if v_loss is not None:
                        last_val_loss = v_loss
                    history.append({
                        "step": self.model.step,
                        "epoch": self.model.epoch,
                        "trainLoss": round(loss, 4),
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

        train_tps = round(total_train_tokens / train_elapsed_sec, 2)

        self.last_metrics = {
            "backend": "python_ref",
            "selectedDevice": "cpu",
            "datasetVersion": splits.get("version", "unknown"),
            "manifestHash": splits.get("manifestHash", ""),
            "cancelled": cancelled,
            "epochsCompleted": self.model.epoch,
            "optimizerSteps": optimizer_updates,
            "totalStepCounter": self.model.step,
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
            "peakProcessRamMB": get_peak_ram_mb(),
            "tracedPeakMB": round(traced_peak_bytes / (1024 * 1024), 3),
            "peakGpuMemoryMB": 0.0,
            "weightMemoryBytes": est["weightMemoryBytes"],
            "optimizerGradMemoryBytes": est["optimizerGradMemoryBytes"],
            "history": history,
        }

        ckpt_info = self.save_checkpoint(reason="cancelled" if cancelled else "completed")
        self.last_metrics["checkpointSizeBytes"] = ckpt_info["checkpointSizeBytes"]
        self.last_metrics["checkpointDir"] = ckpt_info["checkpointDir"]

        return {
            "success": True,
            "backend": "python_ref",
            "selectedDevice": "cpu",
            "cancelled": cancelled,
            "metrics": self.last_metrics,
            "checkpoint": ckpt_info,
            "hardware": self.get_telemetry()["hardware"],
        }

    def generate(self, prompt, max_new_tokens=32, temperature=0.8, allow_untrained=False):
        if not self.model.is_trained and not allow_untrained:
            return {
                "success": True,
                "backend": "python_ref",
                "isTrained": False,
                "text": "",
                "error": "Model is not trained yet.",
            }

        # Encode using active tokenizer vocab if loaded, otherwise ASCII fallback
        if self.tokenizer_vocab:
            tok_to_id = {v: int(k) for k, v in self.tokenizer_vocab.items()}
            tokens = [2] + [tok_to_id.get(ch, 1) for ch in prompt]
        else:
            tokens = [2] + [ord(c) % self.config["vocabSize"] for c in prompt]

        max_ctx = self.config["maxSeqLen"]
        generated = list(tokens)

        for _ in range(max_new_tokens):
            ctx = generated[-max_ctx:]
            logits = self.model.forward(ctx)
            last_row = logits[-1]

            scaled = [x / max(0.1, temperature) for x in last_row]
            m = max(scaled)
            exps = [math.exp(x - m) for x in scaled]
            s = sum(exps)
            probs = [e / s for e in exps]

            # Argmax/top-sample deterministic-friendly choice
            next_tok = max(range(len(probs)), key=lambda i: probs[i])
            if next_tok == 3:
                break
            generated.append(next_tok)

        if self.tokenizer_vocab:
            out_chars = [
                self.tokenizer_vocab.get(str(t), "")
                for t in generated
                if t not in (0, 2, 3)
            ]
            decoded = "".join(out_chars)
        else:
            decoded = "".join(chr(t) if 32 <= t < 127 else "" for t in generated[1:])

        return {
            "success": True,
            "backend": "python_ref",
            "selectedDevice": "cpu",
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
            raise RuntimeError("Insufficient disk space (< 5 MB free) to save Python Ref Phase 7 checkpoint.")

        if reason == "best_val":
            run_dir = os.path.join(base_dir, "best_val_checkpoint_py")
        else:
            run_dir = os.path.join(base_dir, f"ckpt_py_{stage}_{int(self.cumulative_tokens)}tok")
        os.makedirs(run_dir, exist_ok=True)

        ckpt_file = os.path.join(run_dir, "checkpoint.json")
        manifest_file = os.path.join(run_dir, "manifest.json")

        payload = {
            "backend": "python_ref",
            "phase": "7",
            "stage": stage,
            "softwareVersion": SOFTWARE_VERSION,
            "preset": self.active_preset,
            "config": self.config,
            "seed": self.seed,
            "step": self.model.step,
            "epoch": self.model.epoch,
            "adamT": self.model.adam_t,
            "cumulativeTokens": self.cumulative_tokens,
            "seqCursor": self.seq_cursor,
            "bestValLoss": self.best_val_loss,
            "isTrained": self.model.is_trained,
            "tokenizerVersion": self.tokenizer_version,
            "tokenizerChecksum": self.tokenizer_checksum,
            "specialTokens": EXPECTED_SPECIAL_TOKENS,
            "datasetVersion": self.last_metrics.get("datasetVersion") if self.last_metrics else "omnibrain-ds-v2.0.0",
            "manifestHash": self.last_metrics.get("manifestHash") if self.last_metrics else "f080a58dbfad98c2",
            "params": self.model.params,
            "adamM": self.model.adam_m,
            "adamV": self.model.adam_v,
        }
        with open(ckpt_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        ckpt_size_bytes = os.path.getsize(ckpt_file)
        ckpt_sha256 = compute_sha256(ckpt_file)

        manifest = {
            "phase": "7",
            "stage": stage,
            "backend": "python_ref",
            "softwareVersion": SOFTWARE_VERSION,
            "device": "cpu",
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
            "step": self.model.step,
            "epoch": self.model.epoch,
            "cumulativeTokens": self.cumulative_tokens,
            "seqCursor": self.seq_cursor,
            "bestValLoss": self.best_val_loss,
            "metrics": self.last_metrics,
            "checkpointFile": "checkpoint.json",
            "checkpointSha256": ckpt_sha256,
            "checkpointSizeBytes": ckpt_size_bytes,
            "qualitativeSamples": self.generate_sample_completions() if self.model.is_trained else [],
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.last_checkpoint_dir = run_dir
        if reason == "best_val":
            self.best_val_checkpoint_dir = run_dir

        return {
            "backend": "python_ref",
            "checkpointDir": run_dir,
            "checkpointFile": ckpt_file,
            "manifestFile": manifest_file,
            "checkpointSha256": ckpt_sha256,
            "checkpointSizeBytes": ckpt_size_bytes,
        }

    def load_phase7_checkpoint(self, ckpt_dir_or_file):
        if os.path.isdir(ckpt_dir_or_file):
            ckpt_file = os.path.join(ckpt_dir_or_file, "checkpoint.json")
            manifest_file = os.path.join(ckpt_dir_or_file, "manifest.json")
        else:
            if ckpt_dir_or_file.endswith("manifest.json"):
                manifest_file = ckpt_dir_or_file
                ckpt_file = os.path.join(os.path.dirname(ckpt_dir_or_file), "checkpoint.json")
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

        with open(ckpt_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("backend") != "python_ref":
            return {"success": False, "error": f"Incompatible checkpoint backend '{data.get('backend')}' for pure-Python engine."}

        saved_tok_ver = data.get("tokenizerVersion")
        if saved_tok_ver and saved_tok_ver != EXPECTED_TOKENIZER_VERSION:
            return {"success": False, "error": f"Tokenizer version mismatch: expected '{EXPECTED_TOKENIZER_VERSION}', got '{saved_tok_ver}'."}

        saved_tok_sum = data.get("tokenizerChecksum", "")
        if self.tokenizer_checksum and saved_tok_sum and saved_tok_sum != self.tokenizer_checksum:
            return {"success": False, "error": "Tokenizer checksum mismatch!"}

        self.config = data["config"]
        self.active_preset = data.get("preset", self.active_preset)
        self.seed = data.get("seed", 42)
        self.model = DecoderTransformer(self.config, seed=self.seed)
        self.model.params = data["params"]
        self.model.adam_m = data.get("adamM", self.model.adam_m)
        self.model.adam_v = data.get("adamV", self.model.adam_v)
        self.model.adam_t = int(data.get("adamT", 0))
        self.model.step = int(data.get("step", 0))
        self.model.epoch = int(data.get("epoch", 0))
        self.cumulative_tokens = int(data.get("cumulativeTokens", 0))
        self.seq_cursor = int(data.get("seqCursor", 0))
        self.best_val_loss = float(data.get("bestValLoss", float("inf")))
        self.model.is_trained = bool(data.get("isTrained", True))
        self.last_checkpoint_dir = os.path.dirname(ckpt_file)

        return {
            "success": True,
            "backend": "python_ref",
            "phase": "7",
            "stage": data.get("stage", "unknown"),
            "selectedDevice": "cpu",
            "step": self.model.step,
            "epoch": self.model.epoch,
            "cumulativeTokens": self.cumulative_tokens,
            "seqCursor": self.seq_cursor,
            "bestValLoss": self.best_val_loss,
            "isTrained": self.model.is_trained,
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
        batch_size=1,
        base_lr=0.005,
        min_lr=0.0002,
        eval_interval_tokens=4800,
        resume_from=None,
        checkpoint_base_dir="./checkpoints/phase7",
        fixed_seed=42,
    ):
        self._load_tokenizer_file("./checkpoints/tokenizer_v1.json")
        stage_token_budgets = {"stageA": 500000, "stageB": 1000000, "stageC": 2000000}
        if target_tokens is None:
            target_tokens = stage_token_budgets.get(stage, 500000)
        target_tokens = int(target_tokens)

        if resume_from:
            load_res = self.load_phase7_checkpoint(resume_from)
            if not load_res.get("success"):
                return load_res

        if not os.path.exists(splits_path):
            return {"success": False, "error": f"Packed splits file not found: {splits_path}"}

        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

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
                "backend": "python_ref",
                "phase": "7",
                "stage": stage,
                "message": f"Target token budget of {target_tokens} already satisfied (cumulative={self.cumulative_tokens}).",
                "cumulativeTokens": self.cumulative_tokens,
                "targetTokens": target_tokens,
                "step": self.model.step,
                "lastMetrics": self.last_metrics,
            }

        tokens_to_process = target_tokens - self.cumulative_tokens
        tokens_per_seq = max_seq
        planned_steps = max(1, math.ceil(tokens_to_process / tokens_per_seq))

        baseline_samples = self.generate_sample_completions() if not self.model.is_trained else []
        tracemalloc.start()
        t0 = time.perf_counter()

        val_eval_seqs = val_seqs[:min(16, len(val_seqs))]
        init_val_loss, _, _ = self.evaluate_sequences(val_eval_seqs)
        if math.isinf(self.best_val_loss) and init_val_loss is not None:
            self.best_val_loss = init_val_loss

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

        while self.cumulative_tokens < target_tokens and self.seq_cursor < total_train_seqs:
            if os.path.exists(self.cancel_flag_path):
                os.remove(self.cancel_flag_path)
                cancelled = True
                break

            seq = train_seqs[self.seq_cursor]
            inputs = seq[:-1]
            targets = seq[1:]

            current_rel_step = stage_steps
            warmup_steps = max(1, int(planned_steps * 0.10))
            if current_rel_step < warmup_steps:
                lr = base_lr * float(current_rel_step + 1) / float(warmup_steps)
            else:
                progress = float(current_rel_step - warmup_steps) / float(max(1, planned_steps - warmup_steps))
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

            loss, grads, n_toks = self.model.compute_loss_and_grads(inputs, targets, pad_id=0)
            if math.isnan(loss) or math.isinf(loss) or loss > 50.0:
                diverged = True
                divergence_reason = f"Non-finite loss detected: {loss}"
                break

            grad_norm, _ = self.model.adamw_step(grads, lr=lr, max_grad_norm=1.0)
            self.seq_cursor += 1
            self.cumulative_tokens += n_toks
            stage_train_tokens += n_toks
            stage_steps += 1
            last_train_loss = round(loss, 4)
            last_lr = lr
            self.model.is_trained = True

            if stage_train_tokens % eval_interval_tokens < n_toks or self.cumulative_tokens >= target_tokens:
                v_loss, _, _ = self.evaluate_sequences(val_eval_seqs)
                if v_loss is not None:
                    last_val_loss = v_loss
                    if v_loss < self.best_val_loss:
                        self.best_val_loss = v_loss
                        self.save_phase7_checkpoint(base_dir=checkpoint_base_dir, stage=stage, reason="best_val")
                history.append({
                    "step": self.model.step,
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
        peak_ram_mb = get_peak_ram_mb()

        final_val_loss, _, _ = self.evaluate_sequences(val_eval_seqs)
        if final_val_loss is not None and final_val_loss < self.best_val_loss:
            self.best_val_loss = final_val_loss
            self.save_phase7_checkpoint(base_dir=checkpoint_base_dir, stage=stage, reason="best_val")

        post_samples = self.generate_sample_completions()
        self.phase7_history = history

        self.last_metrics = {
            "phase": "7",
            "stage": stage,
            "backend": "python_ref",
            "selectedDevice": "cpu",
            "datasetVersion": splits.get("version", "omnibrain-ds-v2.0.0"),
            "manifestHash": splits.get("manifestHash", ""),
            "tokenizerVersion": self.tokenizer_version,
            "tokenizerChecksum": self.tokenizer_checksum,
            "targetTokens": target_tokens,
            "tokensProcessedThisStage": stage_train_tokens,
            "cumulativeTokens": self.cumulative_tokens,
            "sequencesConsumed": self.seq_cursor,
            "optimizerStepsThisStage": stage_steps,
            "totalStepCounter": self.model.step,
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
        ckpt_info = self.save_phase7_checkpoint(base_dir=checkpoint_base_dir, stage=stage, reason=reason)
        self.last_metrics["checkpointSha256"] = ckpt_info["checkpointSha256"]
        self.last_metrics["checkpointSizeBytes"] = ckpt_info["checkpointSizeBytes"]
        self.last_metrics["checkpointDir"] = ckpt_info["checkpointDir"]

        return {
            "success": not diverged,
            "backend": "python_ref",
            "phase": "7",
            "stage": stage,
            "cancelled": cancelled,
            "diverged": diverged,
            "divergenceReason": divergence_reason,
            "metrics": self.last_metrics,
            "checkpoint": ckpt_info,
            "bestValCheckpoint": os.path.join(checkpoint_base_dir, "best_val_checkpoint_py"),
        }


class EngineBackendManager:
    """
    Coordinates the Pure-Python Reference Engine and the PyTorch Optimized Engine.
    Strictly enforces explicit backend selection and NEVER silently switches backends.
    """

    def __init__(self):
        self.python_engine = PythonFoundationEngine()
        self.pytorch_engine = None
        if PYTORCH_AVAILABLE:
            try:
                self.pytorch_engine = PyTorchFoundationEngine()
                self.active_backend = "pytorch"
            except Exception:
                self.active_backend = "python_ref"
        else:
            self.active_backend = "python_ref"

    def get_available_backends(self):
        cuda = get_cuda_info()
        return [
            {
                "id": "python_ref",
                "name": "Pure-Python Reference Engine",
                "available": True,
                "version": SOFTWARE_VERSION,
                "devices": ["cpu"],
                "reason": None,
            },
            {
                "id": "pytorch",
                "name": "PyTorch Optimized Engine",
                "available": PYTORCH_AVAILABLE,
                "version": PYTORCH_VERSION,
                "devices": (["cpu", "cuda"] if cuda["cudaAvailable"] else ["cpu"]) if PYTORCH_AVAILABLE else [],
                "reason": None if PYTORCH_AVAILABLE else PYTORCH_IMPORT_ERROR,
            },
        ]

    def _active_engine(self):
        if self.active_backend == "pytorch" and self.pytorch_engine is not None:
            return self.pytorch_engine
        return self.python_engine

    def _verify_requested_backend(self, requested_backend):
        if requested_backend is None or requested_backend == self.active_backend:
            return True, None
        return False, (
            f"Requested backend '{requested_backend}' does not match active backend '{self.active_backend}'. "
            f"Silent backend switching is disabled; select '{requested_backend}' explicitly first."
        )

    def set_backend(self, backend_name, requested_device="auto"):
        if backend_name not in ("python_ref", "pytorch"):
            return {
                "success": False,
                "rejected": True,
                "activeBackend": self.active_backend,
                "error": f"Unknown backend '{backend_name}'. Allowed: 'python_ref', 'pytorch'.",
            }

        if backend_name == "pytorch":
            if not PYTORCH_AVAILABLE:
                return {
                    "success": False,
                    "rejected": True,
                    "activeBackend": self.active_backend,
                    "requestedBackend": "pytorch",
                    "error": f"Cannot activate PyTorch backend (no silent fallback): {PYTORCH_IMPORT_ERROR}",
                }
            try:
                if self.pytorch_engine is None:
                    self.pytorch_engine = PyTorchFoundationEngine(requested_device=requested_device)
                elif requested_device and requested_device != "auto":
                    from pytorch_engine import resolve_pytorch_device
                    dev = resolve_pytorch_device(requested_device)
                    self.pytorch_engine.device = dev
                    self.pytorch_engine.model = self.pytorch_engine.model.to(dev)
            except Exception as exc:
                return {
                    "success": False,
                    "rejected": True,
                    "activeBackend": self.active_backend,
                    "error": str(exc),
                }

        self.active_backend = backend_name
        return {
            "success": True,
            "activeBackend": self.active_backend,
            "selectedDevice": getattr(self._active_engine(), "device", "cpu"),
            "availableBackends": self.get_available_backends(),
            "telemetry": self.get_telemetry(),
        }

    def get_telemetry(self):
        eng = self._active_engine()
        if self.active_backend == "pytorch" and self.pytorch_engine is not None:
            est = self.pytorch_engine._estimate_config_resources(self.pytorch_engine.config)
            hw = detect_hardware(selected_device=self.pytorch_engine.device)
            base = {
                "success": True,
                "backend": "pytorch",
                "softwareVersion": SOFTWARE_VERSION,
                "preset": self.pytorch_engine.active_preset,
                "config": self.pytorch_engine.config,
                "isTrained": self.pytorch_engine.is_trained,
                "step": self.pytorch_engine.step,
                "epoch": self.pytorch_engine.epoch,
                "seed": self.pytorch_engine.seed,
                "tokenizerVersion": self.pytorch_engine.tokenizer_version,
                "tokenizerChecksum": self.pytorch_engine.tokenizer_checksum,
                "estimates": est,
                "hardware": hw,
                "lastMetrics": self.pytorch_engine.last_metrics,
                "lastCheckpointDir": self.pytorch_engine.last_checkpoint_dir,
            }
        else:
            base = self.python_engine.get_telemetry()

        base["activeBackend"] = self.active_backend
        base["selectedDevice"] = base["hardware"]["selectedDevice"]
        base["availableBackends"] = self.get_available_backends()
        return base

    def handle_request(self, req):
        action = req.get("action")
        ok, err = self._verify_requested_backend(req.get("backend"))
        if not ok and action not in ("backend", "telemetry", "estimates", "numerical_verify"):
            return {"success": False, "rejected": True, "activeBackend": self.active_backend, "error": err}

        eng = self._active_engine()
        if action == "backend":
            return self.set_backend(req.get("backend", self.active_backend), req.get("device", "auto"))
        if action == "config":
            res = eng.set_config(req.get("configName", "tiny_cpu"), seed=req.get("seed", 42))
            res["activeBackend"] = self.active_backend
            return res
        if action in ("estimates", "telemetry"):
            return self.get_telemetry()
        if action == "train_packed":
            res = eng.train_packed(
                splits_path=req.get("splitsPath", "./datasets/packed_splits.json"),
                epochs=req.get("epochs", 2),
                base_lr=req.get("learningRate", 0.008),
                max_steps=req.get("maxSteps"),
                eval_interval=req.get("evalInterval", 5),
                checkpoint_interval=req.get("checkpointInterval", 10),
                resume_from=req.get("resumeFrom"),
            )
            res["activeBackend"] = self.active_backend
            if "hardware" not in res:
                res["hardware"] = self.get_telemetry()["hardware"]
            return res
        if action == "train_phase7":
            res = eng.train_phase7_extended(
                splits_path=req.get("splitsPath", "./datasets/packed_splits.json"),
                stage=req.get("stage", "stageA"),
                target_tokens=req.get("targetTokens"),
                batch_size=req.get("batchSize", 32),
                base_lr=req.get("learningRate", 0.005),
                min_lr=req.get("minLr", 0.0002),
                eval_interval_tokens=req.get("evalIntervalTokens", 48000),
                resume_from=req.get("resumeFrom"),
                checkpoint_base_dir=req.get("checkpointBaseDir", "./checkpoints/phase7"),
            )
            res["activeBackend"] = self.active_backend
            if "hardware" not in res:
                res["hardware"] = self.get_telemetry()["hardware"]
            return res
        if action == "inference":
            return eng.generate(
                req.get("prompt", ""),
                max_new_tokens=req.get("maxNewTokens", 32),
                temperature=req.get("temperature", 0.8),
            )
        if action == "save":
            return {"success": True, **eng.save_checkpoint(reason="manual")}
        if action == "load":
            return eng.load_checkpoint(req.get("path", ""))
        if action == "load_phase7":
            if hasattr(eng, "load_phase7_checkpoint"):
                return eng.load_phase7_checkpoint(req.get("path", ""))
            return eng.load_checkpoint(req.get("path", ""))
        if action == "numerical_verify":
            return run_cross_backend_numerical_verification()

        return {"success": False, "error": f"Unknown action: {action}"}


def main():
    manager = EngineBackendManager()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            req = json.loads(line)
            res = manager.handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"success": False, "error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
