#!/usr/bin/env python3
"""
OMNIBRAIN AI — Phase 5.1 Deep ML & Engine Verification Suite (Python)
"""
import os
import sys
import json
import tempfile
import shutil
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (
    DecoderTransformer,
    PythonFoundationEngine,
    MODEL_PRESETS,
    estimate_config_resources,
)


class TestOmnibrainPythonEngine(unittest.TestCase):
    def setUp(self):
        self.small_cfg = {
            "vocabSize": 32,
            "dModel": 8,
            "numLayers": 1,
            "numHeads": 2,
            "dFf": 16,
            "maxSeqLen": 16,
            "contextLength": 12,
            "dropout": 0.0,
        }

    def test_01_causal_masking_zero_future_leakage(self):
        model = DecoderTransformer(self.small_cfg, seed=42)
        seq_a = [2, 5, 10, 15, 20]
        seq_b = [2, 5, 10, 28, 30]  # Differ only at positions 3 and 4

        logits_a = model.forward(seq_a)
        logits_b = model.forward(seq_b)

        # Positions 0, 1, 2 must be IDENTICAL (zero leakage from positions 3, 4)
        for pos in [0, 1, 2]:
            for v in range(self.small_cfg["vocabSize"]):
                self.assertAlmostEqual(
                    logits_a[pos][v],
                    logits_b[pos][v],
                    places=12,
                    msg=f"Causal leakage detected at pos {pos}, vocab {v}",
                )

        # Position 3 must differ because token at pos 3 changed
        diff_pos3 = max(abs(logits_a[3][v] - logits_b[3][v]) for v in range(self.small_cfg["vocabSize"]))
        self.assertGreater(diff_pos3, 1e-6)

    def test_02_padding_excluded_from_loss(self):
        model = DecoderTransformer(self.small_cfg, seed=7)
        inputs = [2, 5, 6, 7]
        targets_pad = [5, 6, 0, 0]  # Last two tokens are <pad>=0
        loss_pad, _, active_count = model.compute_loss_and_grads(inputs, targets_pad, pad_id=0)
        self.assertEqual(active_count, 2)
        self.assertGreater(loss_pad, 0.0)

        all_pad_targets = [0, 0, 0, 0]
        loss_zero, _, active_zero = model.compute_loss_and_grads(inputs, all_pad_targets, pad_id=0)
        self.assertEqual(active_zero, 0)
        self.assertEqual(loss_zero, 0.0)

    def test_03_full_finite_difference_gradient_check_all_params(self):
        model = DecoderTransformer(self.small_cfg, seed=99)
        inputs = [2, 4, 8, 12]
        targets = [4, 8, 12, 3]
        eps = 1e-5

        _, grads, _ = model.compute_loss_and_grads(inputs, targets, pad_id=0)

        # Check every parameter tensor (weights AND biases AND LayerNorm scale/bias)
        max_abs_err = 0.0
        max_rel_err = 0.0
        checked_tensors = 0

        for name, mat in model.params.items():
            # Pick a deterministic coordinate (r, c) inside active indices
            r = 2 if name == "w_tok" else 0
            c = 0
            orig = mat[r][c]

            mat[r][c] = orig + eps
            loss_plus, _, _ = model.compute_loss_and_grads(inputs, targets, pad_id=0)

            mat[r][c] = orig - eps
            loss_minus, _, _ = model.compute_loss_and_grads(inputs, targets, pad_id=0)

            mat[r][c] = orig

            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            ana_grad = grads[name][r][c]

            abs_err = abs(num_grad - ana_grad)
            rel_err = abs_err / (abs(num_grad) + abs(ana_grad) + 1e-8)

            max_abs_err = max(max_abs_err, abs_err)
            if abs(num_grad) > 1e-5 or abs(ana_grad) > 1e-5:
                max_rel_err = max(max_rel_err, rel_err)

            self.assertLess(abs_err, 1e-5, f"Gradient abs error too high for {name}: {abs_err}")
            if abs(num_grad) > 1e-5:
                self.assertLess(rel_err, 1e-3, f"Gradient rel error too high for {name}: {rel_err}")
            checked_tensors += 1

        print(
            f"\n[Gradient Check] Verified {checked_tensors} tensors (weights + biases + LN): "
            f"max_abs_err={max_abs_err:.2e}, max_rel_err={max_rel_err:.2e}"
        )

    def test_04_adamw_updates_and_reproducibility(self):
        m1 = DecoderTransformer(self.small_cfg, seed=1234)
        m2 = DecoderTransformer(self.small_cfg, seed=1234)

        inputs = [2, 5, 9, 14, 18]
        targets = [5, 9, 14, 18, 3]

        losses_1 = []
        losses_2 = []
        for _ in range(4):
            l1, g1, _ = m1.compute_loss_and_grads(inputs, targets, pad_id=0)
            m1.adamw_step(g1, lr=0.02)
            losses_1.append(l1)

            l2, g2, _ = m2.compute_loss_and_grads(inputs, targets, pad_id=0)
            m2.adamw_step(g2, lr=0.02)
            losses_2.append(l2)

        self.assertEqual(losses_1, losses_2)
        self.assertLess(losses_1[-1], losses_1[0])

    def test_05_resource_rejection_on_large_gpu_config(self):
        engine = PythonFoundationEngine()
        res = engine.set_config("large_gpu")
        self.assertFalse(res["success"])
        self.assertTrue(res.get("rejected", False))
        self.assertIn("Insufficient RAM", res["error"])

    def test_06_checkpoint_save_compatibility_and_resume(self):
        tmp_dir = tempfile.mkdtemp(prefix="omnibrain_ckpt_test_")
        try:
            splits_file = os.path.join(tmp_dir, "splits.json")
            with open(splits_file, "w") as f:
                json.dump({
                    "version": "test-ds-v1",
                    "manifestHash": "abc123",
                    "contextLength": 8,
                    "tokenizerChecksum": "",
                    "trainSequences": [[2, 10, 11, 12, 13, 14, 15, 3, 0]],
                    "valSequences": [[2, 20, 21, 22, 23, 24, 25, 3, 0]],
                }, f)

            engine1 = PythonFoundationEngine()
            engine1.tokenizer_checksum = "tok_sum_valid"
            res1 = engine1.train_packed(
                splits_path=splits_file,
                epochs=1,
                max_steps=1,
                eval_interval=1,
                checkpoint_interval=10,
            )
            self.assertTrue(res1["success"])
            ckpt_dir = res1["checkpoint"]["checkpointDir"]
            # Override saved checkpoint into tmp_dir by copying or testing directly
            self.assertTrue(os.path.exists(ckpt_dir))

            # Verify incompatible tokenizer checksum is rejected
            engine_bad = PythonFoundationEngine()
            engine_bad.tokenizer_checksum = "different_checksum_xyz"
            bad_load = engine_bad.load_checkpoint(ckpt_dir)
            self.assertFalse(bad_load["success"])
            self.assertIn("Tokenizer checksum mismatch", bad_load["error"])

            # Verify valid resume increments step counter
            engine2 = PythonFoundationEngine()
            engine2.tokenizer_checksum = "tok_sum_valid"
            res2 = engine2.train_packed(
                splits_path=splits_file,
                epochs=1,
                max_steps=2,
                resume_from=ckpt_dir,
            )
            self.assertTrue(res2["success"])
            self.assertEqual(res2["metrics"]["totalStepCounter"], res1["metrics"]["totalStepCounter"] + 1)

            # Cleanup created checkpoint dirs from test
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            shutil.rmtree(res2["checkpoint"]["checkpointDir"], ignore_errors=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_07_graceful_cancellation(self):
        tmp_dir = tempfile.mkdtemp(prefix="omnibrain_cancel_test_")
        try:
            splits_file = os.path.join(tmp_dir, "splits.json")
            with open(splits_file, "w") as f:
                json.dump({
                    "version": "test-ds-cancel",
                    "manifestHash": "def456",
                    "contextLength": 8,
                    "tokenizerChecksum": "",
                    "trainSequences": [
                        [2, 10, 11, 12, 13, 14, 15, 3, 0],
                        [2, 11, 12, 13, 14, 15, 16, 3, 0],
                    ],
                    "valSequences": [[2, 20, 21, 22, 23, 24, 25, 3, 0]],
                }, f)

            engine = PythonFoundationEngine()
            engine.tokenizer_checksum = ""
            # Create cancel flag BEFORE calling train_packed so loop stops cleanly at first check
            os.makedirs(os.path.dirname(engine.cancel_flag_path), exist_ok=True)
            with open(engine.cancel_flag_path, "w") as f:
                f.write("cancel")

            res = engine.train_packed(splits_path=splits_file, epochs=2)
            self.assertTrue(res["success"])
            self.assertTrue(res["cancelled"])
            self.assertEqual(res["metrics"]["optimizerSteps"], 0)
            shutil.rmtree(res["checkpoint"]["checkpointDir"], ignore_errors=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
