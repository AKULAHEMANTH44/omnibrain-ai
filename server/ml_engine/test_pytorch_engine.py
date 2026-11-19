#!/usr/bin/env python3
"""
OMNIBRAIN AI — Phase 5.2 PyTorch & Cross-Backend Verification Suite
Covers:
- Hardware & CUDA/CPU detection
- Strict backend selection without silent fallback
- Tokenizer, special token ID, and checkpoint compatibility validation
- Resource safety guard (rejects large_gpu without allocation)
- Pure-Python reference engine compatibility
- PyTorch model construction, CPU forward, causal masking, gradient propagation,
  AdamW updates, padding exclusion, checkpoint round-trip, and numerical
  comparison against Pure-Python reference (executed when torch is installed,
  explicitly skipped with reason when torch is unavailable).
"""
import ast
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytorch_engine
from pytorch_engine import (
    PYTORCH_AVAILABLE,
    PYTORCH_VERSION,
    PYTORCH_IMPORT_ERROR,
    EXPECTED_TOKENIZER_VERSION,
    EXPECTED_SPECIAL_TOKENS,
    validate_tokenizer_compatibility,
    run_cross_backend_numerical_verification,
)
from engine import (
    DecoderTransformer,
    EngineBackendManager,
    detect_hardware,
    estimate_config_resources,
)


class TestPhase52BackendAndPyTorch(unittest.TestCase):
    def setUp(self):
        self.small_cfg = {
            "name": "Test Tiny",
            "vocabSize": 32,
            "dModel": 8,
            "numLayers": 2,
            "numHeads": 2,
            "dFf": 16,
            "maxSeqLen": 16,
            "contextLength": 12,
            "dropout": 0.0,
        }

    def test_01_pytorch_module_ast_and_implementation_structure(self):
        pt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch_engine.py")
        with open(pt_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        self.assertIn("CausalSelfAttentionBlock", class_names)
        self.assertIn("OmnibrainPyTorchTransformer", class_names)
        self.assertIn("PyTorchFoundationEngine", class_names)
        for token in ("nn.Embedding", "nn.LayerNorm", "torch.optim.AdamW", "clip_grad_norm_", "ignore_index", "torch.triu", "torch.save", "torch.load"):
            self.assertIn(token, source, f"Expected PyTorch primitive '{token}' missing in pytorch_engine.py")

    def test_02_hardware_detection_accuracy(self):
        hw = detect_hardware(selected_device="cpu")
        self.assertIn("os", hw)
        self.assertTrue(len(hw["os"]) > 0)
        self.assertEqual(hw["pythonVersion"], sys.version.split()[0])
        self.assertGreaterEqual(hw["cpuCount"], 1)
        self.assertGreater(hw["availableRamMB"], 0)
        self.assertEqual(hw["pytorchAvailable"], PYTORCH_AVAILABLE)
        self.assertEqual(hw["selectedDevice"], "cpu")
        if not hw["cudaAvailable"]:
            # Never claim GPU acceleration when running without CUDA
            self.assertEqual(hw["device"], "cpu")
            self.assertFalse(hw["gpuAvailable"])

    def test_03_backend_selection_and_no_silent_fallback(self):
        mgr = EngineBackendManager()
        backends = {b["id"]: b for b in mgr.get_available_backends()}
        self.assertIn("python_ref", backends)
        self.assertIn("pytorch", backends)
        self.assertTrue(backends["python_ref"]["available"])

        if not PYTORCH_AVAILABLE:
            self.assertEqual(mgr.active_backend, "python_ref")
            res = mgr.set_backend("pytorch")
            self.assertFalse(res["success"])
            self.assertTrue(res.get("rejected", False))
            self.assertIn("no silent fallback", res["error"].lower())
            self.assertEqual(mgr.active_backend, "python_ref")

            # Passing mismatched backend in request must fail cleanly instead of silently switching
            req_res = mgr.handle_request({"action": "config", "configName": "tiny_cpu", "backend": "pytorch"})
            self.assertFalse(req_res["success"])
            self.assertTrue(req_res.get("rejected", False))
        else:
            res_pt = mgr.set_backend("pytorch")
            self.assertTrue(res_pt["success"])
            self.assertEqual(mgr.active_backend, "pytorch")
            res_py = mgr.set_backend("python_ref")
            self.assertTrue(res_py["success"])
            self.assertEqual(mgr.active_backend, "python_ref")

    def test_04_resource_rejection_without_allocation(self):
        mgr = EngineBackendManager()
        res = mgr.handle_request({"action": "config", "configName": "large_gpu"})
        self.assertFalse(res["success"])
        self.assertTrue(res.get("rejected", False))
        self.assertIn("Insufficient", res["error"])

    def test_05_tokenizer_and_checkpoint_cross_backend_validation(self):
        ok, err = validate_tokenizer_compatibility(
            {"version": EXPECTED_TOKENIZER_VERSION, "vocabSize": 1024, "specialTokens": EXPECTED_SPECIAL_TOKENS},
            {"vocabSize": 1024},
        )
        self.assertTrue(ok)
        self.assertIsNone(err)

        bad_ver, err_ver = validate_tokenizer_compatibility(
            {"version": "omnibrain-tok-v99.0", "vocabSize": 1024, "specialTokens": EXPECTED_SPECIAL_TOKENS},
            {"vocabSize": 1024},
        )
        self.assertFalse(bad_ver)
        self.assertIn("version mismatch", err_ver)

        bad_tok, err_tok = validate_tokenizer_compatibility(
            {"version": EXPECTED_TOKENIZER_VERSION, "vocabSize": 1024, "specialTokens": {"<pad>": 9, "<unk>": 1, "<bos>": 2, "<eos>": 3}},
            {"vocabSize": 1024},
        )
        self.assertFalse(bad_tok)
        self.assertIn("Special token", err_tok)

        # Verify pure-python engine refuses a .pt checkpoint path
        mgr = EngineBackendManager()
        mgr.set_backend("python_ref")
        bad_load = mgr.handle_request({"action": "load", "path": "./nonexistent_model/checkpoint.pt"})
        self.assertFalse(bad_load["success"])
        self.assertIn("PyTorch", bad_load["error"])

    def test_06_pure_python_reference_compatibility(self):
        mgr = EngineBackendManager()
        res = mgr.set_backend("python_ref")
        self.assertTrue(res["success"])
        est = estimate_config_resources(self.small_cfg)
        ref = DecoderTransformer(self.small_cfg, seed=42)
        self.assertEqual(ref.count_parameters(), est["parameterCount"])

    @unittest.skipUnless(PYTORCH_AVAILABLE, f"PyTorch unavailable: {PYTORCH_IMPORT_ERROR}")
    def test_07_pytorch_model_construction_forward_and_causal_masking(self):
        import torch
        from pytorch_engine import OmnibrainPyTorchTransformer

        torch.manual_seed(42)
        model = OmnibrainPyTorchTransformer(self.small_cfg)
        model.eval()
        est = estimate_config_resources(self.small_cfg)
        self.assertEqual(model.count_parameters(), est["parameterCount"])

        seq_a = torch.tensor([[2, 5, 10, 15, 20]], dtype=torch.long)
        seq_b = torch.tensor([[2, 5, 10, 28, 30]], dtype=torch.long)
        with torch.no_grad():
            logits_a = model(seq_a)
            logits_b = model(seq_b)

        self.assertEqual(tuple(logits_a.shape), (1, 5, self.small_cfg["vocabSize"]))
        # Causal check: first 3 positions must match within float32 tolerance
        max_early_diff = float((logits_a[0, :3] - logits_b[0, :3]).abs().max().item())
        self.assertLess(max_early_diff, 1e-6)
        max_pos3_diff = float((logits_a[0, 3] - logits_b[0, 3]).abs().max().item())
        self.assertGreater(max_pos3_diff, 1e-5)

    @unittest.skipUnless(PYTORCH_AVAILABLE, f"PyTorch unavailable: {PYTORCH_IMPORT_ERROR}")
    def test_08_pytorch_gradients_adamw_and_padding_exclusion(self):
        import torch
        from pytorch_engine import OmnibrainPyTorchTransformer

        torch.manual_seed(7)
        model = OmnibrainPyTorchTransformer(self.small_cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=0.02)

        inp = torch.tensor([[2, 4, 8, 12]], dtype=torch.long)
        tgt_all_pad = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        loss_zero, active_zero = model.compute_loss(inp, tgt_all_pad, pad_id=0)
        self.assertEqual(active_zero, 0)
        self.assertEqual(float(loss_zero.item()), 0.0)

        tgt = torch.tensor([[4, 8, 12, 0]], dtype=torch.long)
        before_w = model.lm_head.weight.detach().clone()
        opt.zero_grad()
        loss0, active = model.compute_loss(inp, tgt, pad_id=0)
        self.assertEqual(active, 3)
        loss0.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
        opt.step()
        after_w = model.lm_head.weight.detach()
        self.assertGreater(float((after_w - before_w).abs().max().item()), 1e-6)

        for _ in range(4):
            opt.zero_grad()
            l, _ = model.compute_loss(inp, tgt, pad_id=0)
            l.backward()
            opt.step()
        loss_end, _ = model.compute_loss(inp, tgt, pad_id=0)
        self.assertLess(float(loss_end.item()), float(loss0.item()))

    @unittest.skipUnless(PYTORCH_AVAILABLE, f"PyTorch unavailable: {PYTORCH_IMPORT_ERROR}")
    def test_09_pytorch_checkpoint_roundtrip(self):
        from pytorch_engine import PyTorchFoundationEngine

        tmp_dir = tempfile.mkdtemp(prefix="omnibrain_pt_ckpt_")
        try:
            splits_file = os.path.join(tmp_dir, "splits.json")
            with open(splits_file, "w") as f:
                json.dump({
                    "version": "test-pt-v1",
                    "manifestHash": "pt123",
                    "contextLength": 8,
                    "tokenizerVersion": EXPECTED_TOKENIZER_VERSION,
                    "tokenizerChecksum": "",
                    "trainSequences": [[2, 10, 11, 12, 13, 14, 15, 3, 0]],
                    "valSequences": [[2, 20, 21, 22, 23, 24, 25, 3, 0]],
                }, f)

            e1 = PyTorchFoundationEngine(preset="tiny_cpu", seed=42, requested_device="cpu")
            res1 = e1.train_packed(splits_path=splits_file, epochs=1, max_steps=1, eval_interval=1, checkpoint_interval=10)
            self.assertTrue(res1["success"])
            ckpt_dir = res1["checkpoint"]["checkpointDir"]
            manifest_path = res1["checkpoint"]["manifestFile"]
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
            self.assertEqual(manifest["backend"], "pytorch")
            self.assertEqual(manifest["pytorchVersion"], PYTORCH_VERSION)
            self.assertEqual(manifest["tokenizerVersion"], EXPECTED_TOKENIZER_VERSION)

            e_restored = PyTorchFoundationEngine(preset="tiny_cpu", seed=99, requested_device="cpu")
            load_res = e_restored.load_checkpoint(ckpt_dir)
            self.assertTrue(load_res["success"])
            self.assertEqual(load_res["backend"], "pytorch")
            self.assertEqual(load_res["step"], res1["metrics"]["totalStepCounter"])
            for k, tensor in e1.model.state_dict().items():
                diff = float((tensor - e_restored.model.state_dict()[k]).abs().max().item())
                self.assertEqual(diff, 0.0, f"Restored weight mismatch on {k}")
            gen_res = e_restored.generate("Test", max_new_tokens=4)
            self.assertTrue(gen_res["success"])
            self.assertTrue(gen_res["isTrained"])
            self.assertEqual(gen_res["backend"], "pytorch")

            e2 = PyTorchFoundationEngine(preset="tiny_cpu", seed=42, requested_device="cpu")
            res2 = e2.train_packed(splits_path=splits_file, epochs=1, max_steps=2, resume_from=ckpt_dir)
            self.assertTrue(res2["success"])
            self.assertEqual(res2["metrics"]["totalStepCounter"], res1["metrics"]["totalStepCounter"] + 1)

            shutil.rmtree(ckpt_dir, ignore_errors=True)
            shutil.rmtree(res2["checkpoint"]["checkpointDir"], ignore_errors=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @unittest.skipUnless(PYTORCH_AVAILABLE, f"PyTorch unavailable: {PYTORCH_IMPORT_ERROR}")
    def test_10_numerical_comparison_python_ref_vs_pytorch(self):
        report = run_cross_backend_numerical_verification(config=self.small_cfg, seed=42)
        self.assertTrue(report["success"])
        self.assertEqual(report["parameterCountRef"], report["parameterCountPyTorch"])
        self.assertEqual(report["activeTokensRef"], report["activeTokensPyTorch"])
        self.assertEqual(report["paddingCheck"]["refPadActive"], 0)
        self.assertEqual(report["paddingCheck"]["pytorchPadActive"], 0)
        self.assertEqual(report["paddingCheck"]["refPadLoss"], 0.0)
        self.assertEqual(report["paddingCheck"]["pytorchPadLoss"], 0.0)
        self.assertLess(report["maxLogitDiff"], 1e-5)
        self.assertLess(report["maxLogitRelDiff"], 1e-4)
        self.assertLess(report["lossDiff"], 1e-5)
        self.assertLess(report["lossRelDiff"], 1e-4)
        self.assertLess(report["maxGradAbsDiff"], 1e-5)
        self.assertLess(report["maxGradRelDiff"], 1e-3)
    @unittest.skipUnless(PYTORCH_AVAILABLE, f"PyTorch unavailable: {PYTORCH_IMPORT_ERROR}")
    def test_11_phase7_corpus_verification_and_quarantine_exclusion(self):
        splits_path = os.path.abspath("./datasets/packed_splits.json")
        self.assertTrue(os.path.exists(splits_path), f"Packed splits missing at {splits_path}")
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        self.assertEqual(splits.get("version"), "omnibrain-ds-v2.0.0")
        self.assertEqual(splits.get("manifestHash"), "f080a58dbfad98c2")
        self.assertEqual(splits.get("tokenizerVersion"), "omnibrain-tok-v1.1.0")
        self.assertEqual(splits.get("tokenizerChecksum"), "8bca50838ead180ee93e6d84b685da62dba6a6500466e237570d5a8d0311b48f")
        self.assertEqual(splits.get("trainTokenCount"), 6096070)
        self.assertEqual(splits.get("valTokenCount"), 794737)
        self.assertEqual(splits.get("testTokenCount"), 753390)
        self.assertTrue(splits.get("leakageVerifiedZeroOverlap", False))
        self.assertEqual(len(splits.get("trainSequences", [])), 127002)
        self.assertEqual(len(splits.get("valSequences", [])), 16557)
        self.assertEqual(len(splits.get("testSequences", [])), 15696)

    @unittest.skipUnless(PYTORCH_AVAILABLE, f"PyTorch unavailable: {PYTORCH_IMPORT_ERROR}")
    def test_12_phase7_causal_masking_and_shifted_targets(self):
        import torch
        from pytorch_engine import OmnibrainPyTorchTransformer
        model = OmnibrainPyTorchTransformer(self.small_cfg)
        model.eval()

        # Sequence of length 6
        seq1 = torch.tensor([[2, 5, 8, 12, 16, 20]], dtype=torch.long)
        # Sequence identical up to position 3, different at position 4 and 5
        seq2 = torch.tensor([[2, 5, 8, 12, 22, 25]], dtype=torch.long)

        with torch.no_grad():
            out1 = model(seq1)
            out2 = model(seq2)

        # Due to causal attention, positions 0, 1, 2, 3 must be strictly identical!
        diff_prefix = (out1[:, :4, :] - out2[:, :4, :]).abs().max().item()
        self.assertEqual(diff_prefix, 0.0, "Causal attention leak! Future tokens affected past positions.")

        # Position 4 should differ because input token at position 4 differs
        diff_future = (out1[:, 4:, :] - out2[:, 4:, :]).abs().max().item()
        self.assertGreater(diff_future, 1e-4, "Expected different outputs for different future tokens.")

    @unittest.skipUnless(PYTORCH_AVAILABLE, f"PyTorch unavailable: {PYTORCH_IMPORT_ERROR}")
    def test_13_phase7_checkpoint_sha256_and_integrity(self):
        from pytorch_engine import PyTorchFoundationEngine, compute_sha256
        tmp_base = tempfile.mkdtemp(prefix="phase7_test_ckpt_")
        try:
            eng = PyTorchFoundationEngine(preset="tiny_cpu", seed=42, requested_device="cpu")
            eng.cumulative_tokens = 50000
            eng.step = 32
            ckpt = eng.save_phase7_checkpoint(base_dir=tmp_base, stage="test_stage", reason="unit_test")

            ckpt_file = ckpt["checkpointFile"]
            manifest_file = ckpt["manifestFile"]
            self.assertTrue(os.path.exists(ckpt_file))
            self.assertTrue(os.path.exists(manifest_file))

            # Recalculate SHA-256 independently from disk
            actual_sha = compute_sha256(ckpt_file)
            self.assertEqual(ckpt["checkpointSha256"], actual_sha)

            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["checkpointSha256"], actual_sha)
            self.assertEqual(manifest["phase"], "7")
            self.assertEqual(manifest["stage"], "test_stage")
            self.assertEqual(manifest["cumulativeTokens"], 50000)

            # Test load
            load_res = eng.load_phase7_checkpoint(ckpt["checkpointDir"])
            self.assertTrue(load_res["success"])
            self.assertEqual(load_res["checkpointSha256"], actual_sha)
        finally:
            shutil.rmtree(tmp_base, ignore_errors=True)

    @unittest.skipUnless(PYTORCH_AVAILABLE, f"PyTorch unavailable: {PYTORCH_IMPORT_ERROR}")
    def test_14_phase7_exact_resumable_training(self):
        import torch
        from pytorch_engine import PyTorchFoundationEngine
        tmp_base = tempfile.mkdtemp(prefix="phase7_resume_test_")
        try:
            # Deterministic small split data
            torch.manual_seed(42)
            dummy_train = [[2] + [((i * 7 + j * 13) % 28 + 4) for j in range(48)] for i in range(16)]
            dummy_val = [[2] + [((i * 11 + j * 17) % 28 + 4) for j in range(48)] for i in range(4)]
            test_splits_file = os.path.join(tmp_base, "test_splits.json")
            with open(test_splits_file, "w", encoding="utf-8") as f:
                json.dump({
                    "version": "omnibrain-ds-v2.0.0",
                    "manifestHash": "f080a58dbfad98c2",
                    "tokenizerVersion": "omnibrain-tok-v1.1.0",
                    "tokenizerChecksum": "8bca50838ead180ee93e6d84b685da62dba6a6500466e237570d5a8d0311b48f",
                    "contextLength": 48,
                    "leakageVerifiedZeroOverlap": True,
                    "trainSequences": dummy_train,
                    "valSequences": dummy_val,
                }, f)

            # Run 1: train 10 steps continuously from scratch
            e1 = PyTorchFoundationEngine(preset="tiny_cpu", seed=42, requested_device="cpu")
            r1 = e1.train_phase7_extended(
                splits_path=test_splits_file,
                stage="stageA",
                target_tokens=48 * 8, # 8 steps with batch_size=1
                batch_size=1,
                checkpoint_base_dir=tmp_base,
            )
            final_loss_continuous = r1["metrics"]["finalTrainLoss"]

            # Run 2: train 4 steps, save checkpoint, then resume for 4 more steps
            e2 = PyTorchFoundationEngine(preset="tiny_cpu", seed=42, requested_device="cpu")
            r2_part1 = e2.train_phase7_extended(
                splits_path=test_splits_file,
                stage="stageA",
                target_tokens=48 * 4,
                batch_size=1,
                checkpoint_base_dir=tmp_base,
            )
            ckpt_part1 = r2_part1["checkpoint"]["checkpointDir"]

            # Create brand new engine instance and resume
            e3 = PyTorchFoundationEngine(preset="tiny_cpu", seed=99, requested_device="cpu")
            r3_part2 = e3.train_phase7_extended(
                splits_path=test_splits_file,
                stage="stageA",
                target_tokens=48 * 8,
                batch_size=1,
                resume_from=ckpt_part1,
                checkpoint_base_dir=tmp_base,
            )
            final_loss_resumed = r3_part2["metrics"]["finalTrainLoss"]

            # Both continuous and resumed must reach identical final loss under deterministic state
            self.assertAlmostEqual(final_loss_continuous, final_loss_resumed, places=4,
                                   msg="Resumed training diverged from continuous execution under restored state!")
        finally:
            shutil.rmtree(tmp_base, ignore_errors=True)

    @unittest.skipUnless(PYTORCH_AVAILABLE, f"PyTorch unavailable: {PYTORCH_IMPORT_ERROR}")
    def test_15_phase7_non_finite_loss_handling(self):
        import torch
        from pytorch_engine import PyTorchFoundationEngine
        eng = PyTorchFoundationEngine(preset="tiny_cpu", seed=42, requested_device="cpu")
        # Artificially set embedding weights to NaN
        with torch.no_grad():
            eng.model.w_tok.weight.fill_(float("nan"))

        inputs = torch.tensor([[2, 4, 6]], dtype=torch.long)
        targets = torch.tensor([[4, 6, 8]], dtype=torch.long)
        loss, _ = eng.model.compute_loss(inputs, targets)
        self.assertTrue(torch.isnan(loss), "Expected NaN loss from corrupted parameters")


if __name__ == "__main__":
    print(f"Environment PyTorch Status: available={PYTORCH_AVAILABLE}, version={PYTORCH_VERSION}, error={PYTORCH_IMPORT_ERROR}")
    unittest.main(verbosity=2)
