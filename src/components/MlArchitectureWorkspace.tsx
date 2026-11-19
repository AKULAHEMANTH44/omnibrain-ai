import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Database,
  Play,
  Square,
  RefreshCw,
  ShieldCheck,
  Settings,
  Save,
  FolderOpen,
  AlertTriangle,
} from 'lucide-react';

export const MlArchitectureWorkspace: React.FC = () => {
  const [mlStatus, setMlStatus] = useState<any | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [loading, setLoading] = useState(true);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [selectedConfig, setSelectedConfig] = useState('tiny_cpu');
  const [resumeDir, setResumeDir] = useState('');
  const [numericalReport, setNumericalReport] = useState<any | null>(null);
  const [busyArchiveId, setBusyArchiveId] = useState<string | null>(null);
  const [benchmarkingTok, setBenchmarkingTok] = useState(false);
  const [runningAudit, setRunningAudit] = useState(false);

  const fetchStatus = () => {
    fetch('/api/ml/status')
      .then((res) => res.json())
      .then((data) => {
        setMlStatus(data);
        if (data.preset) setSelectedConfig(data.preset);
        if (data.lastCheckpointDir && !resumeDir) setResumeDir(data.lastCheckpointDir);
        setLoading(false);
      })
      .catch((err) => {
        setErrorBanner(err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleBackendChange = async (backendId: string) => {
    setErrorBanner(null);
    setStatusMessage(null);
    try {
      const res = await fetch('/api/ml/backend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: backendId, device: 'auto' }),
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        setErrorBanner(data.error || `Backend '${backendId}' is unavailable.`);
        return;
      }
      setStatusMessage(`Active training backend set to '${data.activeBackend}' on device '${data.selectedDevice?.toUpperCase()}'.`);
      fetchStatus();
    } catch (err: any) {
      setErrorBanner('Backend switch error: ' + err.message);
    }
  };

  const handleNumericalVerify = async () => {
    setErrorBanner(null);
    try {
      const res = await fetch('/api/ml/verify-numerical', { method: 'POST' });
      const data = await res.json();
      setNumericalReport(data);
      if (!data.success) {
        setErrorBanner(data.error || 'Numerical verification could not run PyTorch comparison.');
      } else {
        setStatusMessage(
          `Numerical comparison passed: maxLogitDiff=${data.maxLogitDiff}, lossDiff=${data.lossDiff}, maxOutputGradDiff=${data.maxOutputGradDiff}`
        );
      }
    } catch (err: any) {
      setErrorBanner('Verification error: ' + err.message);
    }
  };

  const handleConfigChange = async (configName: string) => {
    setErrorBanner(null);
    setStatusMessage(null);
    try {
      const res = await fetch('/api/ml/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ configName, backend: mlStatus?.activeBackend }),
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        setErrorBanner(data.error || 'Configuration rejected due to hardware limits.');
        return;
      }
      setSelectedConfig(configName);
      setStatusMessage(`Model initialized on '${data.activeBackend || 'python_ref'}' with '${configName}' (${data.parameterCount?.toLocaleString()} parameters).`);
      fetchStatus();
    } catch (err: any) {
      setErrorBanner('Config change error: ' + err.message);
    }
  };

  const handleTrainLocalModel = async (resume = false) => {
    setIsTraining(true);
    setErrorBanner(null);
    const activeBackend = mlStatus?.activeBackend || 'python_ref';
    setStatusMessage(`Running real pretraining on '${activeBackend}' (${mlStatus?.selectedDevice?.toUpperCase() || 'CPU'})...`);
    try {
      const res = await fetch('/api/ml/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          backend: activeBackend,
          epochs: 1,
          learningRate: 0.01,
          maxSteps: 8,
          evalInterval: 4,
          checkpointInterval: 4,
          resumeFrom: resume && resumeDir ? resumeDir : undefined,
        }),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        const m = data.metrics;
        setStatusMessage(
          `${data.message} Train Loss: ${m.initialTrainLoss} -> ${m.finalTrainLoss} | Val Loss: ${m.initialValLoss} -> ${m.finalValLoss} | Speed: ${m.trainTokensPerSec} tok/s`
        );
        if (data.checkpoint?.checkpointDir) {
          setResumeDir(data.checkpoint.checkpointDir);
        }
        fetchStatus();
      } else {
        setErrorBanner(data.error || 'Training failed.');
      }
    } catch (err: any) {
      setErrorBanner('Training error: ' + err.message);
    } finally {
      setIsTraining(false);
    }
  };

  const handleCancelTraining = async () => {
    try {
      await fetch('/api/ml/cancel', { method: 'POST' });
      setStatusMessage('Cancellation requested. Saving current checkpoint cleanly...');
    } catch (err: any) {
      setErrorBanner('Cancel failed: ' + err.message);
    }
  };

  const handleSaveCheckpoint = async () => {
    setErrorBanner(null);
    try {
      const res = await fetch('/api/ml/checkpoint/save', { method: 'POST' });
      const data = await res.json();
      if (res.ok && data.success) {
        setResumeDir(data.checkpointDir);
        setStatusMessage(`Checkpoint saved to ${data.checkpointDir} (${(data.checkpointSizeBytes / 1024).toFixed(1)} KB).`);
        fetchStatus();
      } else {
        setErrorBanner(data.error || 'Failed to save checkpoint.');
      }
    } catch (err: any) {
      setErrorBanner(err.message);
    }
  };

  const handleLoadCheckpoint = async () => {
    if (!resumeDir.trim()) return;
    setErrorBanner(null);
    try {
      const res = await fetch('/api/ml/checkpoint/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: resumeDir.trim() }),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        setStatusMessage(`Restored checkpoint at step ${data.step} (epoch ${data.epoch}).`);
        fetchStatus();
      } else {
        setErrorBanner(data.error || 'Failed to restore checkpoint.');
      }
    } catch (err: any) {
      setErrorBanner(err.message);
    }
  };

  const handleArchiveAction = async (archiveId: string, action: 'approve' | 'reject' | 'ingest') => {
    setBusyArchiveId(archiveId);
    setErrorBanner(null);
    setStatusMessage(null);
    try {
      const res = await fetch(`/api/datasets/archives/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ archiveId }),
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        setErrorBanner(data.error || `Archive ${action} failed for '${archiveId}'.`);
      } else {
        setStatusMessage(
          action === 'ingest'
            ? `Ingested ${data.documentsIngestedThisPass} docs (${data.cleanedTokensThisPass} clean tokens) from '${archiveId}'.`
            : `Archive '${archiveId}' marked as ${action}d.`
        );
        fetchStatus();
      }
    } catch (err: any) {
      setErrorBanner(err.message);
    } finally {
      setBusyArchiveId(null);
    }
  };

  const handleRunTokenizerComparison = async () => {
    setBenchmarkingTok(true);
    setErrorBanner(null);
    try {
      const res = await fetch('/api/datasets/tokenizer-comparison/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ targetVocabSize: 2048 }),
      });
      const data = await res.json();
      if (res.ok && data.experimentalBpeTokenizer) {
        setStatusMessage(
          `Tokenizer benchmark updated: ${data.experimentalBpeTokenizer.version} (${data.experimentalBpeTokenizer.charsPerToken} chars/tok, ${(data.experimentalBpeTokenizer.unkRate * 100).toFixed(3)}% <unk>).`
        );
        fetchStatus();
      } else {
        setErrorBanner(data.error || 'Tokenizer comparison failed.');
      }
    } catch (err: any) {
      setErrorBanner(err.message);
    } finally {
      setBenchmarkingTok(false);
    }
  };

  const handleRunCorpusAudit = async () => {
    setRunningAudit(true);
    setErrorBanner(null);
    try {
      const res = await fetch('/api/datasets/audit/run', { method: 'POST' });
      const data = await res.json();
      if (res.ok && data.summary) {
        setStatusMessage(
          `Forensic corpus audit complete: ${data.summary.totalVerifiedDocsAfterQuarantine.toLocaleString()} verified docs (${data.summary.totalVerifiedCleanedTokens.toLocaleString()} tokens), ${data.summary.quarantinedDocs.toLocaleString()} quarantined docs across ${data.summary.quarantinedArchives} archives.`
        );
        fetchStatus();
      } else {
        setErrorBanner(data.error || 'Corpus audit failed.');
      }
    } catch (err: any) {
      setErrorBanner(err.message);
    } finally {
      setRunningAudit(false);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#07080a] text-slate-400 text-sm">
        <RefreshCw className="w-5 h-5 animate-spin text-cyan-400 mr-2" />
        <span>Loading OMNIBRAIN Foundation Model & Dataset Analytics...</span>
      </div>
    );
  }

  const estimates = mlStatus?.estimates || { parameterCount: 0, estimatedMemoryMB: 0 };
  const hardware = mlStatus?.hardware || {};
  const availableBackends = mlStatus?.availableBackends || [];
  const activeBackend = mlStatus?.activeBackend || 'python_ref';
  const selectedDevice = mlStatus?.selectedDevice || hardware.selectedDevice || 'cpu';
  const datasets = mlStatus?.datasets || {};
  const storage = datasets.storageBreakdown || {};
  const qualityBreakdown = datasets.qualityBreakdown || {};
  const provenance = mlStatus?.provenance || [];
  const quarantinedProvenance = mlStatus?.quarantinedProvenance || [];
  const rejectedProvenance = mlStatus?.rejectedProvenance || [];
  const corpusAudit = mlStatus?.corpusAudit || null;
  const rawArchives = mlStatus?.rawArchives || [];
  const researchCatalog = mlStatus?.researchCatalog || { entries: [], summary: {} };
  const tokenizer = mlStatus?.tokenizer || {};
  const tokenizerComparison = mlStatus?.tokenizerComparison || null;
  const lastMetrics = mlStatus?.lastMetrics;

  return (
    <div className="flex-1 flex flex-col h-screen bg-[#07080a] overflow-y-auto p-6 md:p-10">
      <div className="max-w-6xl mx-auto w-full space-y-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-6">
          <div>
            <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono uppercase tracking-widest mb-1">
              <Cpu className="w-4 h-4" />
              <span>OMNIBRAIN Independent Foundation Model Core</span>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              ML Architecture & Dataset Quality Console
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Dual-backend training architecture (PyTorch + Pure-Python reference), hardware telemetry, tokenizer validation, and versioned checkpoints.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => handleTrainLocalModel(false)}
              disabled={isTraining}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-xs transition-all whitespace-nowrap shrink-0 ${
                isTraining
                  ? 'bg-amber-500/40 text-black cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black cursor-pointer'
              }`}
            >
              {isTraining ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-black" />}
              <span>{isTraining ? 'Pretraining Running...' : 'Run Pretraining'}</span>
            </button>

            {isTraining && (
              <button
                onClick={handleCancelTraining}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-rose-500/20 border border-rose-500/40 text-rose-300 hover:bg-rose-500/30 font-bold text-xs transition-all whitespace-nowrap shrink-0"
              >
                <Square className="w-3.5 h-3.5 fill-rose-300" />
                <span>Cancel Safely</span>
              </button>
            )}

            <button
              onClick={handleSaveCheckpoint}
              disabled={isTraining}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 text-slate-200 font-medium text-xs transition-all whitespace-nowrap shrink-0"
            >
              <Save className="w-3.5 h-3.5" />
              <span>Save Checkpoint</span>
            </button>
          </div>
        </div>

        {statusMessage && (
          <div className="p-4 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-200 text-xs font-mono">
            {statusMessage}
          </div>
        )}

        {errorBanner && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-mono flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{errorBanner}</span>
          </div>
        )}

        {/* Backend Architecture & Hardware Detection */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <span className="font-bold text-white text-sm">Training Backend Selector (No Silent Fallback)</span>
              <span className="text-xs font-mono text-cyan-400">Active: {activeBackend}</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {availableBackends.map((b: any) => (
                <button
                  key={b.id}
                  onClick={() => handleBackendChange(b.id)}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    activeBackend === b.id
                      ? 'bg-cyan-500/10 border-cyan-500 text-white'
                      : 'bg-[#0a0c10] border-white/10 text-slate-300 hover:border-white/25'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-xs text-white">{b.name}</span>
                    <span
                      className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                        b.available ? 'bg-emerald-500/15 text-emerald-400' : 'bg-rose-500/15 text-rose-300'
                      }`}
                    >
                      {b.available ? 'AVAILABLE' : 'UNAVAILABLE'}
                    </span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-400">
                    {b.available
                      ? `Version: ${b.version} · Devices: ${(b.devices || []).join(', ').toUpperCase()}`
                      : b.reason || 'Module not installed'}
                  </div>
                </button>
              ))}
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-white/10 text-xs font-mono">
              <span className="text-slate-400">Selected Execution Device: <strong className="text-white">{selectedDevice.toUpperCase()}</strong></span>
              <button
                onClick={handleNumericalVerify}
                className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 text-cyan-300 text-xs font-mono"
              >
                Verify Cross-Backend Numerics
              </button>
            </div>
            {numericalReport && (
              <div className="p-3 rounded-xl bg-[#0a0c10] border border-white/10 text-[11px] font-mono text-slate-300">
                {numericalReport.success
                  ? `Verified (${numericalReport.pytorchVersion}): Logits abs=${numericalReport.maxLogitDiff} rel=${numericalReport.maxLogitRelDiff} · Loss abs=${numericalReport.lossDiff} rel=${numericalReport.lossRelDiff} · Grads abs=${numericalReport.maxGradAbsDiff} rel=${numericalReport.maxGradRelDiff}`
                  : `PyTorch Comparison Skipped: ${numericalReport.error}`}
              </div>
            )}
          </div>

          <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 space-y-3 font-mono text-xs tabular-nums">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <span className="font-bold text-white text-sm font-sans">Detected Hardware & Environment</span>
              <span className="text-emerald-400">{hardware.selectedDevice?.toUpperCase() || 'CPU'}</span>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <div className="p-2.5 rounded-xl bg-[#0a0c10] border border-white/5">
                <div className="text-slate-400">Operating System</div>
                <div className="text-white truncate mt-0.5" title={hardware.os}>{hardware.os || 'Unknown'}</div>
              </div>
              <div className="p-2.5 rounded-xl bg-[#0a0c10] border border-white/5">
                <div className="text-slate-400">Python / PyTorch</div>
                <div className="text-white truncate mt-0.5">
                  Py {hardware.pythonVersion || 'N/A'} · {hardware.pytorchAvailable ? `Torch ${hardware.pytorchVersion}` : 'No PyTorch'}
                </div>
              </div>
              <div className="p-2.5 rounded-xl bg-[#0a0c10] border border-white/5">
                <div className="text-slate-400">CPU & Logical Cores</div>
                <div className="text-white truncate mt-0.5" title={hardware.cpuModel}>
                  {hardware.cpuCount || 1} cores · {hardware.cpuModel || 'Unknown'}
                </div>
              </div>
              <div className="p-2.5 rounded-xl bg-[#0a0c10] border border-white/5">
                <div className="text-slate-400">CUDA / GPU Acceleration</div>
                <div className={hardware.cudaAvailable ? 'text-emerald-400 mt-0.5' : 'text-amber-300 mt-0.5'}>
                  {hardware.cudaAvailable ? `${hardware.gpuName} (${hardware.gpuMemoryMB} MB)` : 'Unavailable (CPU Mode)'}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 pt-1">
              System RAM: <span className="text-white">{hardware.availableRamMB || 0} MB free</span> / {hardware.totalRamMB || 0} MB total · Peak RSS:{' '}
              <span className="text-emerald-400">{hardware.peakProcessRamMB || 0} MB</span> · Free Disk: {hardware.freeDiskMB || 0} MB
            </div>
          </div>
        </div>

        {/* Model Configuration Selector */}
        <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-white/10 pb-4">
            <div className="flex items-center gap-2 text-white font-bold">
              <Settings className="w-5 h-5 text-cyan-400" />
              <span>Model Size Configurations & Resource Guard</span>
            </div>
            <span className="text-xs text-slate-400 font-mono">
              Backend: {activeBackend} · Device: {selectedDevice.toUpperCase()} · Est. Memory: {estimates.estimatedMemoryMB || 0} MB
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { id: 'tiny_cpu', name: 'Tiny CPU Dev', desc: '1024 vocab · 24 dim · 2 layers' },
              { id: 'small_exp', name: 'Small Experimental', desc: '1024 vocab · 64 dim · 4 layers' },
              { id: 'medium_res', name: 'Medium Research', desc: '1024 vocab · 256 dim · 8 layers' },
              { id: 'large_gpu', name: 'Large GPU Future', desc: '32k vocab · 2048 dim · 24 layers' },
            ].map((cfg) => (
              <button
                key={cfg.id}
                onClick={() => handleConfigChange(cfg.id)}
                className={`p-4 rounded-xl border text-left transition-all ${
                  selectedConfig === cfg.id
                    ? 'bg-cyan-500/10 border-cyan-500 text-white'
                    : 'bg-[#0a0c10] border-white/10 text-slate-400 hover:text-slate-200'
                }`}
              >
                <div className="font-bold text-sm text-white mb-1">{cfg.name}</div>
                <div className="text-[11px] text-slate-400">{cfg.desc}</div>
              </button>
            ))}
          </div>

          {/* Checkpoint Resume Controls */}
          <div className="pt-3 border-t border-white/10 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <input
              type="text"
              value={resumeDir}
              onChange={(e) => setResumeDir(e.target.value)}
              placeholder="./checkpoints/ckpt_..."
              className="flex-1 bg-[#0a0c10] border border-white/10 rounded-xl px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500/50"
            />
            <button
              onClick={handleLoadCheckpoint}
              className="flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 text-xs text-slate-200 font-medium whitespace-nowrap shrink-0"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              <span>Restore Checkpoint</span>
            </button>
            <button
              onClick={() => handleTrainLocalModel(true)}
              disabled={isTraining || !resumeDir.trim()}
              className="flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-violet-500/20 border border-violet-500/30 hover:bg-violet-500/30 text-xs text-violet-200 font-medium whitespace-nowrap shrink-0 disabled:opacity-40"
            >
              <Play className="w-3.5 h-3.5" />
              <span>Resume Training</span>
            </button>
          </div>
        </div>

        {/* Model & Hardware Telemetry Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 tabular-nums">
          <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-5 space-y-1.5">
            <span className="text-xs text-slate-400 font-mono">MODEL PARAMETERS</span>
            <div className="text-lg font-bold text-white">{estimates.parameterCount?.toLocaleString()}</div>
            <span className="text-xs text-cyan-400 font-mono">
              Weights: {((estimates.weightMemoryBytes || 0) / 1024).toFixed(1)} KB · Opt: {((estimates.optimizerGradMemoryBytes || 0) / 1024).toFixed(1)} KB
            </span>
          </div>

          <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-5 space-y-1.5">
            <span className="text-xs text-slate-400 font-mono">MEMORY & DEVICE</span>
            <div className="text-lg font-bold text-emerald-400">{hardware.peakProcessRamMB || 0} MB Peak RAM</div>
            <span className="text-xs text-slate-400 font-mono">
              Device: {selectedDevice.toUpperCase()} · GPU: {hardware.cudaAvailable ? hardware.gpuName : 'Unavailable'}
            </span>
          </div>

          <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-5 space-y-1.5">
            <span className="text-xs text-slate-400 font-mono">TRAINING STATUS</span>
            <div className={`text-lg font-bold ${mlStatus?.isTrained ? 'text-emerald-400' : 'text-amber-400'}`}>
              {mlStatus?.trainingState?.status}
            </div>
            <span className="text-xs text-slate-400 font-mono">
              Step {mlStatus?.step || 0} · Epoch {mlStatus?.epoch || 0} · Seed {mlStatus?.seed || 42}
            </span>
          </div>

          <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-5 space-y-1.5">
            <span className="text-xs text-slate-400 font-mono">DATASET CORPUS</span>
            <div className="text-lg font-bold text-cyan-400">{datasets.totalDocuments || 0} Docs</div>
            <span className="text-xs text-slate-400 font-mono">
              Raw: {datasets.rawTokenCount || 0} · Clean: {datasets.cleanedTokenCount || 0} tok
            </span>
          </div>
        </div>

        {/* Measured Pretraining Metrics (Only shown when actual run exists) */}
        {lastMetrics && (
          <div className="bg-[#12151c]/90 border border-cyan-500/20 rounded-2xl p-6 space-y-4 tabular-nums">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <span className="font-bold text-white text-sm">
                Measured Pretraining Run Telemetry ({lastMetrics.backend || activeBackend} · {(lastMetrics.selectedDevice || selectedDevice).toUpperCase()})
              </span>
              <span className="text-xs font-mono text-cyan-400">{lastMetrics.checkpointDir}</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
              <div className="p-3 rounded-xl bg-[#0a0c10] border border-white/5">
                <div className="text-slate-400">Train Loss (Init → Final)</div>
                <div className="text-white font-bold mt-1">{lastMetrics.initialTrainLoss} → {lastMetrics.finalTrainLoss}</div>
              </div>
              <div className="p-3 rounded-xl bg-[#0a0c10] border border-white/5">
                <div className="text-slate-400">Validation Loss</div>
                <div className="text-white font-bold mt-1">{lastMetrics.initialValLoss} → {lastMetrics.finalValLoss}</div>
              </div>
              <div className="p-3 rounded-xl bg-[#0a0c10] border border-white/5">
                <div className="text-slate-400">Measured Throughput</div>
                <div className="text-emerald-400 font-bold mt-1">{lastMetrics.trainTokensPerSec} tok/s (Val: {lastMetrics.valTokensPerSec})</div>
              </div>
              <div className="p-3 rounded-xl bg-[#0a0c10] border border-white/5">
                <div className="text-slate-400">Steps / Elapsed / Ckpt Size</div>
                <div className="text-white font-bold mt-1">
                  {lastMetrics.optimizerSteps} steps · {lastMetrics.elapsedSec}s · {((lastMetrics.checkpointSizeBytes || 0) / 1024).toFixed(1)} KB
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Dataset Quality Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 tabular-nums">
          {/* Domain & Language Token Distribution */}
          <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
              <div className="flex items-center gap-2 text-white font-bold">
                <Database className="w-5 h-5 text-cyan-400" />
                <span>Tokens & Docs Across 11 Domains</span>
              </div>
              <span className="text-xs text-emerald-400 font-mono">
                {datasets.domainsCovered || 0}/11 Covered · {datasets.datasetVersion}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 font-mono text-xs">
              {datasets.domainDistribution &&
                Object.entries(datasets.domainDistribution).map(([domain, count]: [string, any]) => (
                  <div key={domain} className="flex items-center justify-between p-2.5 rounded-xl bg-[#0a0c10] border border-white/5">
                    <span className="text-slate-300 capitalize truncate">{domain.replace('_', ' ')}</span>
                    <span className="text-cyan-400 font-bold shrink-0">
                      {count}d · {datasets.tokensPerDomain?.[domain] || 0}t
                    </span>
                  </div>
                ))}
            </div>

            <div className="pt-2 border-t border-white/10 text-xs font-mono space-y-2">
              <div className="text-slate-400">Language Distribution (Docs / Tokens):</div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-slate-200">
                {datasets.languageDistribution &&
                  Object.entries(datasets.languageDistribution).map(([lang, cnt]: [string, any]) => (
                    <span key={lang}>
                      {lang}: <strong className="text-cyan-400">{cnt}d</strong> ({datasets.tokensPerLanguage?.[lang] || 0}t)
                    </span>
                  ))}
              </div>
              <div className="text-slate-400 pt-1">
                Split Tokens — Train: <span className="text-white">{datasets.trainTokenCount || 0}</span> · Val:{' '}
                <span className="text-white">{datasets.valTokenCount || 0}</span> · Test:{' '}
                <span className="text-white">{datasets.testTokenCount || 0}</span> · Exact/Near Dup:{' '}
                <span className="text-amber-400">
                  {datasets.duplicateCount || 0}/{datasets.nearDuplicateCount || 0}
                </span>{' '}
                · Rejected: <span className="text-rose-300">{datasets.rejectedDocuments || 0}</span>
              </div>
              <div className="text-[11px] text-slate-400 pt-1">
                Seed Docs: <span className="text-white">{datasets.seedDocumentCount || 0}</span> · Expanded Docs:{' '}
                <span className="text-emerald-400">{datasets.expandedDocumentCount || 0}</span> · Versions:{' '}
                <span className="text-cyan-400">{(datasets.availableManifestVersions || [datasets.datasetVersion]).join(', ')}</span>
              </div>
            </div>
          </div>

          {/* Provenance, License Distribution, Storage & Tokenizer Checksum */}
          <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
              <div className="flex items-center gap-2 text-white font-bold">
                <ShieldCheck className="w-5 h-5 text-emerald-400" />
                <span>Provenance, Licenses & Storage Consumption</span>
              </div>
              <span className="text-xs text-slate-400 font-mono">{datasets.manifestPath}</span>
            </div>

            <div className="p-3.5 rounded-xl bg-[#0a0c10] border border-white/10 space-y-1 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">Tokenizer Version:</span>
                <span className="text-white">{tokenizer.version} ({tokenizer.activeTokens}/{tokenizer.vocabSize} active)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">SHA-256 Checksum:</span>
                <span className="text-cyan-400">{tokenizer.checksum?.slice(0, 24)}...</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">License Distribution:</span>
                <span className="text-emerald-400 text-right">
                  {datasets.licenseDistribution
                    ? Object.entries(datasets.licenseDistribution)
                        .map(([lic, cnt]) => `${lic}:${cnt}`)
                        .join(' · ')
                    : 'None'}
                </span>
              </div>
              <div className="flex justify-between pt-1 border-t border-white/5">
                <span className="text-slate-400">Storage (raw/proc/tok):</span>
                <span className="text-slate-200">
                  {((storage.rawBytes || 0) / 1024 / 1024).toFixed(2)} MB / {((storage.processedBytes || 0) / 1024 / 1024).toFixed(2)} MB /{' '}
                  {((storage.tokenizedBytes || 0) / 1024 / 1024).toFixed(2)} MB
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Total Corpus Disk Usage:</span>
                <span className="text-cyan-400">
                  {((storage.totalCorpusBytes || datasets.storageBytes || 0) / 1024 / 1024).toFixed(2)} MB (Corrupt/LowQ:{' '}
                  {(qualityBreakdown.corruptedEncoding || 0) + (qualityBreakdown.tooShort || 0) + (qualityBreakdown.excessiveRepetition || 0)})
                </span>
              </div>
            </div>

            <div className="space-y-3 max-h-60 overflow-y-auto">
              {provenance.map((prov: any, idx: number) => (
                <div key={idx} className="p-3.5 rounded-xl bg-[#0a0c10] border border-white/10 space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-xs">{prov.datasetName}</span>
                    <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                      {prov.license}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                    <span className="truncate max-w-[60%]">{prov.source}</span>
                    <span>
                      {prov.documentCount}d · {prov.cleanedTokenCount} tok
                    </span>
                  </div>
                </div>
              ))}
              {quarantinedProvenance.length > 0 && (
                <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/25 text-xs font-mono text-amber-300 space-y-1">
                  <div className="font-bold">Quarantined Sources ({quarantinedProvenance.length} — Excluded from Verified Splits, Raw Files Preserved):</div>
                  <div>{quarantinedProvenance.map((r: any) => `${r.datasetName} (${r.documentCount}d)`).join(' · ')}</div>
                </div>
              )}
              {rejectedProvenance.length > 0 && (
                <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs font-mono text-rose-300">
                  Rejected Datasets ({rejectedProvenance.length}):{' '}
                  {rejectedProvenance.map((r: any) => `${r.datasetName} (${r.license})`).join(', ')}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Controlled Local Acquisition & Resumable Ingestion Workflow */}
        <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 space-y-4 tabular-nums">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-white/10 pb-3">
            <div>
              <span className="font-bold text-white text-sm">Controlled Local Dataset Acquisition & Resumable Ingestion</span>
              <p className="text-xs text-slate-400 mt-0.5">
                Offline-first local archives in <span className="font-mono">datasets/raw/</span> with SHA-256 verification and explicit approval before ingestion.
              </p>
            </div>
            <div className="text-[11px] font-mono text-slate-300">
              Proposed: {researchCatalog.summary?.proposed || 0} · Downloaded: {researchCatalog.summary?.downloaded || 0} · Approved:{' '}
              {researchCatalog.summary?.verified_approved || 0} · Ingested: <span className="text-emerald-400">{researchCatalog.summary?.ingested || 0}</span> · Quarantined:{' '}
              <span className="text-amber-300">{researchCatalog.summary?.quarantined || 0}</span> · Rejected:{' '}
              <span className="text-rose-300">{researchCatalog.summary?.rejected || 0}</span>
            </div>
          </div>

          {rawArchives.length === 0 ? (
            <div className="text-xs font-mono text-slate-400 p-4 rounded-xl bg-[#0a0c10] border border-white/5">
              No archives currently registered in datasets/raw/.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-80 overflow-y-auto">
              {rawArchives.map((arch: any) => (
                <div key={arch.archiveId} className="p-3.5 rounded-xl bg-[#0a0c10] border border-white/10 space-y-2 text-xs font-mono">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-bold text-white font-sans truncate">{arch.datasetName}</span>
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded shrink-0 ${
                        arch.statusStage === 'ingested'
                          ? 'bg-emerald-500/15 text-emerald-400'
                          : arch.statusStage === 'verified_approved'
                            ? 'bg-cyan-500/15 text-cyan-300'
                            : arch.statusStage === 'quarantined'
                              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                              : arch.statusStage === 'rejected'
                                ? 'bg-rose-500/15 text-rose-300'
                                : 'bg-amber-500/15 text-amber-300'
                      }`}
                    >
                      {arch.statusStage.toUpperCase()}
                    </span>
                  </div>
                  {arch.quarantineReason && (
                    <div className="text-[10px] text-amber-300/90 leading-snug">{arch.quarantineReason}</div>
                  )}
                  <div className="text-[11px] text-slate-400 flex flex-wrap gap-x-3 gap-y-0.5">
                    <span>Lic: {arch.license}</span>
                    <span>Size: {(arch.fileSizeBytes / 1024).toFixed(1)} KB</span>
                    <span>SHA256: {arch.sha256?.slice(0, 10)}...</span>
                  </div>
                  <div className="text-[11px] text-slate-300">
                    Progress: {arch.progress?.ingestedDocuments || 0} ingested ({arch.progress?.cleanedTokens || 0} tok) ·{' '}
                    {arch.progress?.processedRecords || 0} records scanned
                  </div>
                  <div className="flex items-center gap-2 pt-1">
                    {arch.approvalStatus === 'pending_approval' && (
                      <>
                        <button
                          onClick={() => handleArchiveAction(arch.archiveId, 'approve')}
                          disabled={busyArchiveId === arch.archiveId}
                          className="px-2.5 py-1 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/30"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() => handleArchiveAction(arch.archiveId, 'reject')}
                          disabled={busyArchiveId === arch.archiveId}
                          className="px-2.5 py-1 rounded bg-rose-500/20 border border-rose-500/40 text-rose-300 hover:bg-rose-500/30"
                        >
                          Reject
                        </button>
                      </>
                    )}
                    {arch.approvalStatus === 'approved' && !arch.progress?.completed && (
                      <button
                        onClick={() => handleArchiveAction(arch.archiveId, 'ingest')}
                        disabled={busyArchiveId === arch.archiveId}
                        className="px-2.5 py-1 rounded bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/30"
                      >
                        {busyArchiveId === arch.archiveId ? 'Streaming...' : 'Ingest Stream'}
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Tokenizer Research Evaluation: Production vs Experimental Byte-Level BPE */}
        <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 space-y-4 tabular-nums">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/10 pb-3">
            <div>
              <span className="font-bold text-white text-sm">Tokenizer Research: Production (omnibrain-tok-v1.1.0) vs Experimental Byte-Level BPE</span>
              <p className="text-xs text-slate-400 mt-0.5">
                Production token IDs stay locked for checkpoint compatibility while BPE subword compression is benchmarked across scripts, math, and code.
              </p>
            </div>
            <button
              onClick={handleRunTokenizerComparison}
              disabled={benchmarkingTok}
              className="px-3.5 py-2 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 text-xs font-mono text-cyan-300 whitespace-nowrap shrink-0"
            >
              {benchmarkingTok ? 'Benchmarking BPE...' : 'Run BPE Evaluation'}
            </button>
          </div>

          {tokenizerComparison ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
                <div className="p-4 rounded-xl bg-[#0a0c10] border border-white/10 space-y-1">
                  <div className="text-cyan-400 font-bold">{tokenizerComparison.productionTokenizer.version} (Production Locked)</div>
                  <div className="text-slate-300">
                    Vocab: {tokenizerComparison.productionTokenizer.activeTokens}/{tokenizerComparison.productionTokenizer.vocabSize} ·{' '}
                    Compression: {tokenizerComparison.productionTokenizer.charsPerToken} ch/tok ({tokenizerComparison.productionTokenizer.bytesPerToken} B/tok)
                  </div>
                  <div className="text-slate-400">
                    &lt;unk&gt; Rate: {(tokenizerComparison.productionTokenizer.unkRate * 100).toFixed(4)}% · Speed:{' '}
                    {tokenizerComparison.productionTokenizer.speedTokensPerSec.toLocaleString()} tok/s
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-[#0a0c10] border border-emerald-500/20 space-y-1">
                  <div className="text-emerald-400 font-bold">{tokenizerComparison.experimentalBpeTokenizer.version} (Isolated Research)</div>
                  <div className="text-slate-300">
                    Vocab: {tokenizerComparison.experimentalBpeTokenizer.vocabSize} ({tokenizerComparison.experimentalBpeTokenizer.learnedMerges} merges) ·{' '}
                    Compression: {tokenizerComparison.experimentalBpeTokenizer.charsPerToken} ch/tok ({tokenizerComparison.experimentalBpeTokenizer.bytesPerToken} B/tok)
                  </div>
                  <div className="text-slate-400">
                    &lt;unk&gt; Rate: {(tokenizerComparison.experimentalBpeTokenizer.unkRate * 100).toFixed(4)}% · Round-Trip Exact:{' '}
                    {tokenizerComparison.experimentalBpeTokenizer.roundTripExactMatch ? 'TRUE' : 'FALSE'} · Speed:{' '}
                    {tokenizerComparison.experimentalBpeTokenizer.speedTokensPerSec.toLocaleString()} tok/s
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono border-collapse">
                  <thead>
                    <tr className="border-b border-white/10 text-slate-400">
                      <th className="py-2 pr-3">Category</th>
                      <th className="py-2 pr-3">Samples</th>
                      <th className="py-2 pr-3">Prod Tok</th>
                      <th className="py-2 pr-3">Prod &lt;unk&gt;%</th>
                      <th className="py-2 pr-3">BPE Tok</th>
                      <th className="py-2 pr-3">BPE &lt;unk&gt;%</th>
                      <th className="py-2">Compression Gain</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(tokenizerComparison.categoryComparisons || []).map((row: any) => (
                      <tr key={row.category} className="border-b border-white/5 text-slate-200">
                        <td className="py-1.5 pr-3 text-white">{row.category}</td>
                        <td className="py-1.5 pr-3">{row.samples}</td>
                        <td className="py-1.5 pr-3">{row.prodTokens.toLocaleString()}</td>
                        <td className="py-1.5 pr-3">{(row.prodUnkRate * 100).toFixed(3)}%</td>
                        <td className="py-1.5 pr-3 text-emerald-400">{row.bpeTokens.toLocaleString()}</td>
                        <td className="py-1.5 pr-3 text-emerald-400">{(row.bpeUnkRate * 100).toFixed(3)}%</td>
                        <td className="py-1.5 text-cyan-400">{row.bpeCompressionGainVsProd}x</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="text-xs font-mono text-slate-400">Click "Run BPE Evaluation" to measure production vs experimental Byte-Level BPE metrics on the active corpus.</div>
          )}
        </div>

        {/* Phase 6.1 Independent Forensic Corpus Audit Report */}
        <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 space-y-5 tabular-nums">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/10 pb-3">
            <div>
              <span className="font-bold text-white text-sm">Independent Corpus Audit Report (Phase 6.1 Pretraining Readiness)</span>
              <p className="text-xs text-slate-400 mt-0.5">
                Recalculated independently from raw archives in <span className="font-mono">datasets/raw/</span>, processed JSONL, and packed splits without trusting summary caches.
              </p>
            </div>
            <button
              onClick={handleRunCorpusAudit}
              disabled={runningAudit}
              className="px-3.5 py-2 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 text-xs font-mono text-cyan-300 whitespace-nowrap shrink-0"
            >
              {runningAudit ? 'Auditing Disk Corpus...' : 'Re-Run Forensic Audit'}
            </button>
          </div>

          {corpusAudit ? (
            <div className="space-y-5">
              {/* Top KPI Verification Grid */}
              <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 font-mono text-xs">
                <div className="p-3.5 rounded-xl bg-[#0a0c10] border border-white/10">
                  <div className="text-[10px] text-slate-400 uppercase">Verified Archives</div>
                  <div className="text-base font-bold text-emerald-400 mt-1">
                    {corpusAudit.summary.verifiedApprovedArchives}/{corpusAudit.summary.totalRawArchivesOnDisk} + Seed
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    SHA-256: {corpusAudit.summary.archiveSha256AllMatched ? '100% Matched' : 'Mismatch'}
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-[#0a0c10] border border-amber-500/20">
                  <div className="text-[10px] text-slate-400 uppercase">Quarantined Sources</div>
                  <div className="text-base font-bold text-amber-300 mt-1">
                    {corpusAudit.summary.quarantinedArchives} archives
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    {corpusAudit.summary.quarantinedDocs.toLocaleString()} docs excluded ({corpusAudit.summary.quarantinedTokens.toLocaleString()} tok)
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-[#0a0c10] border border-white/10">
                  <div className="text-[10px] text-slate-400 uppercase">Verified Unique Docs</div>
                  <div className="text-base font-bold text-white mt-1">
                    {corpusAudit.summary.totalVerifiedDocsAfterQuarantine.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Pre-quarantine: {corpusAudit.summary.totalIngestedDocsBeforeQuarantine.toLocaleString()}
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-[#0a0c10] border border-white/10">
                  <div className="text-[10px] text-slate-400 uppercase">Verified Clean Tokens</div>
                  <div className="text-base font-bold text-cyan-400 mt-1">
                    {corpusAudit.summary.totalVerifiedCleanedTokens.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    &lt;unk&gt; in splits: {(corpusAudit.integrityAudit.packedSplitUnkRate * 100).toFixed(3)}%
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-[#0a0c10] border border-white/10">
                  <div className="text-[10px] text-slate-400 uppercase">Cross-Split Leakage</div>
                  <div className="text-base font-bold text-emerald-400 mt-1">
                    Exact: {corpusAudit.summary.crossSplitExactOverlapCount} · Prefix: {corpusAudit.summary.crossSplitPrefixOverlapCount}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Audit Hash: {corpusAudit.auditContentSha256}
                  </div>
                </div>
              </div>

              {/* Quality, PII & Contamination Summary */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 text-xs font-mono">
                <div className="p-4 rounded-xl bg-[#0a0c10] border border-white/10 space-y-2">
                  <div className="text-white font-bold font-sans">Quality, Boilerplate & PII Findings</div>
                  <div className="text-slate-300 space-y-1 text-[11px]">
                    <div>Exact Duplicates Remaining: {corpusAudit.qualityAndContaminationAudit.exactDuplicatesRemainingInVerified} · Corrupted: {corpusAudit.qualityAndContaminationAudit.corruptedOrMalformedDocs}</div>
                    <div>Repeated Paragraphs: {corpusAudit.qualityAndContaminationAudit.docsWithRepeatedParagraphs} docs · Wiki Footers: {corpusAudit.qualityAndContaminationAudit.boilerplateHits.wikipediaFooterSections} · Web Nav Boilerplate: {corpusAudit.qualityAndContaminationAudit.boilerplateHits.webNavigationOrDownload}</div>
                    <div>PII Scan (Post-Quarantine): Emails={corpusAudit.qualityAndContaminationAudit.piiFindings.docsWithEmailAddresses} (OSS commit/example emails), Phones={corpusAudit.qualityAndContaminationAudit.piiFindings.docsWithPhoneNumbers}, API Secrets={corpusAudit.qualityAndContaminationAudit.piiFindings.docsWithApiSecretsOrTokens}</div>
                    <div className="text-slate-400 pt-1">{corpusAudit.qualityAndContaminationAudit.crossSplitLeakage.notes}</div>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-[#0a0c10] border border-amber-500/20 space-y-2">
                  <div className="text-amber-300 font-bold font-sans">Benchmark Contamination & Imbalance Alerts</div>
                  <div className="space-y-1 text-[11px] text-slate-300">
                    {(corpusAudit.qualityAndContaminationAudit.benchmarkContamination || []).map((b: any) => (
                      <div key={b.archiveId}>
                        <span className="text-amber-300">{b.benchmarkName}</span> [{b.splitContaminated}]: {b.documentCount}d ({b.cleanedTokenCount.toLocaleString()} tok)
                      </div>
                    ))}
                    <div className="pt-1 text-slate-400">
                      Script Balance: Indic {corpusAudit.domainAndLanguageBalance.scriptGroupBreakdown.indicTokenPct}% · English {corpusAudit.domainAndLanguageBalance.scriptGroupBreakdown.englishTokenPct}% · Code {corpusAudit.domainAndLanguageBalance.scriptGroupBreakdown.codeTokenPct}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Pretraining Readiness Matrix */}
              <div className="space-y-2">
                <div className="text-xs font-bold text-white">Pretraining Readiness & Unique-Token Scaling Assessment</div>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 text-xs font-mono">
                  {(corpusAudit.pretrainingReadiness?.regimes || []).map((r: any) => (
                    <div key={r.regime} className="p-3.5 rounded-xl bg-[#0a0c10] border border-white/10 space-y-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-sans font-bold text-white text-[12px]">{r.regime}</span>
                      </div>
                      <div className={`text-[10px] inline-block px-1.5 py-0.5 rounded ${
                        r.status === 'READY_WITH_LIMITATIONS'
                          ? 'bg-emerald-500/15 text-emerald-400'
                          : r.status === 'NEEDS_EXPANSION_OR_HIGH_REPETITION'
                            ? 'bg-amber-500/15 text-amber-300'
                            : 'bg-rose-500/15 text-rose-300'
                      }`}>
                        {r.status} ({r.corpusCoveragePct}% of Chinchilla target)
                      </div>
                      <div className="text-[11px] text-slate-400 leading-relaxed">{r.assessment}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* License Verification Status Table */}
              <div className="overflow-x-auto max-h-72 overflow-y-auto border border-white/10 rounded-xl">
                <table className="w-full text-left text-[11px] font-mono border-collapse">
                  <thead className="sticky top-0 bg-[#0a0c10]">
                    <tr className="border-b border-white/10 text-slate-400">
                      <th className="py-2 px-3">Source</th>
                      <th className="py-2 px-3">License</th>
                      <th className="py-2 px-3">Commercial</th>
                      <th className="py-2 px-3">Attr OK</th>
                      <th className="py-2 px-3">Docs / Tok</th>
                      <th className="py-2 px-3">Audit Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(corpusAudit.licenseAndProvenanceAudit?.records || []).map((rec: any) => (
                      <tr key={rec.archiveId} className="border-b border-white/5 text-slate-300">
                        <td className="py-1.5 px-3 text-white font-sans">{rec.datasetName}</td>
                        <td className="py-1.5 px-3">{rec.license}</td>
                        <td className="py-1.5 px-3">{rec.commercialUsePermitted ? 'YES' : 'NO/BLOCKED'}</td>
                        <td className="py-1.5 px-3">{rec.attributionSatisfied ? 'YES' : 'MISSING'}</td>
                        <td className="py-1.5 px-3">{rec.documentCount} / {rec.cleanedTokenCount.toLocaleString()}</td>
                        <td className="py-1.5 px-3">
                          <span className={rec.auditStatus === 'verified_approved' ? 'text-emerald-400' : 'text-amber-300'}>
                            {rec.auditStatus.toUpperCase()}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="text-xs font-mono text-slate-400">Click "Re-Run Forensic Audit" to compute corpus verification metrics.</div>
          )}
        </div>
      </div>
    </div>
  );
};
