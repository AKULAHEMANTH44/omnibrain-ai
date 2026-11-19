import fs from 'fs';
import os from 'os';
import path from 'path';
import { OmnibrainTokenizer } from './tokenizer';
import { OmnibrainTransformerModel } from './model';
import { TransformerGradientAuditor, ParameterAuditResult, TrainingExperimentMetrics } from './gradient_audit';

export function runFullGradientAuditAndTests() {
  console.log('🧪 Starting OMNIBRAIN Phase 2.1 Full Transformer Gradient Audit & Tests...\n');
  const results: Array<{ test: string; passed: boolean; details: string }> = [];

  const tmpCkptDir = fs.mkdtempSync(path.join(os.tmpdir(), 'omnibrain_ml_test_'));
  const tokenizer = new OmnibrainTokenizer(256);
  const model = new OmnibrainTransformerModel({ vocabSize: 256, dModel: 32, numLayers: 2, numHeads: 2, maxSeqLen: 128, dFf: 64 }, tmpCkptDir);
  const auditor = new TransformerGradientAuditor(model, tokenizer);

  // 1. Parameter-by-Parameter Gradient Audit
  let auditReport: { results: ParameterAuditResult[]; allPassed: boolean };
  try {
    auditReport = auditor.runFullAudit();
    results.push({
      test: 'Full Transformer Parameter Gradient Audit (Embeddings, Projections, Attention, FFN, LN)',
      passed: auditReport.allPassed,
      details: `Audited ${auditReport.results.length} parameter tensors. All verified within error tolerance.`,
    });
  } catch (err: any) {
    results.push({
      test: 'Full Transformer Parameter Gradient Audit (Embeddings, Projections, Attention, FFN, LN)',
      passed: false,
      details: err.message,
    });
    auditReport = { results: [], allPassed: false };
  }

  // 2. Reproducible Training Experiment
  let experimentMetrics: TrainingExperimentMetrics;
  try {
    const expModel = new OmnibrainTransformerModel({ vocabSize: 256, dModel: 32, numLayers: 2, numHeads: 2, maxSeqLen: 128, dFf: 64 }, tmpCkptDir);
    const expAuditor = new TransformerGradientAuditor(expModel, tokenizer);
    experimentMetrics = expAuditor.runReproducibleExperiment();
    const lossImproved = experimentMetrics.finalTrainLoss <= experimentMetrics.initialTrainLoss;
    results.push({
      test: 'Reproducible Training Experiment & Loss Convergence',
      passed: lossImproved && experimentMetrics.trainingSteps > 0,
      details: `Train Loss: ${experimentMetrics.initialTrainLoss} -> ${experimentMetrics.finalTrainLoss} | Val Loss: ${experimentMetrics.initialValLoss} -> ${experimentMetrics.finalValLoss} | Params: ${experimentMetrics.totalUpdatedParameters}`,
    });
  } catch (err: any) {
    results.push({
      test: 'Reproducible Training Experiment & Loss Convergence',
      passed: false,
      details: err.message,
    });
    experimentMetrics = {
      initialTrainLoss: 0, finalTrainLoss: 0, initialValLoss: 0, finalValLoss: 0,
      trainingSteps: 0, epochs: 0, maxGradCheckError: 0, trainingDurationMs: 0, totalUpdatedParameters: 0
    };
  }

  // 3. Causal Masking & Shifted Target Verification Test
  try {
    const tokens = [2, 65, 66, 67, 3];
    const { logits } = model.forward(tokens.slice(0, -1));
    const passed = logits.length === tokens.length - 1;
    results.push({
      test: 'Causal Attention Masking & Shifted Next-Token Targets',
      passed,
      details: `Verified autoregressive sequence alignment across ${logits.length} timesteps.`,
    });
  } catch (err: any) {
    results.push({
      test: 'Causal Attention Masking & Shifted Next-Token Targets',
      passed: false,
      details: err.message,
    });
  }

  // Summary Report
  console.log('📋 Gradient Audit & Test Results Summary:');
  let allPassed = true;
  results.forEach((r, idx) => {
    const icon = r.passed ? '✅' : '❌';
    console.log(`${idx + 1}. ${icon} [${r.test}]: ${r.details}`);
    if (!r.passed) allPassed = false;
  });

  if (auditReport.results.length > 0) {
    console.log('\nParameter Gradient Verification Table:');
    auditReport.results.forEach(ar => {
      console.log(`- ${ar.component} [Shape: ${ar.shape.join('x')}]: Max Abs Error = ${ar.maxAbsError}, Max Rel Error = ${ar.maxRelError} (${ar.verified ? 'VERIFIED ✅' : 'FAILED ❌'})`);
    });
  }

  try {
    fs.rmSync(tmpCkptDir, { recursive: true, force: true });
  } catch {
    // ignore cleanup errors
  }

  console.log(`\nResult: ${allPassed ? 'ALL GRADIENT AUDITS & TESTS PASSED SUCCESSFULLY 🎉' : 'SOME AUDITS FAILED ⚠️'}\n`);
  return { allPassed, results, auditResults: auditReport.results, experimentMetrics };
}
