import fs from 'fs';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import dotenv from 'dotenv';
import { globalTokenizer } from './server/core/tokenizer';
import { evaluateTokenizerComparison } from './server/core/bpe_research';
import { runFullGradientAuditAndTests } from './server/core/tests_ml';
import { globalDatasetPipeline } from './server/data/pipeline';
import { globalAcquisitionManager } from './server/data/acquisition';
import { globalCorpusAuditor } from './server/data/audit';
import { runDataEngineeringAndAuditTests } from './server/data/tests_data';
import { globalPythonBridge } from './server/ml_engine/bridge';
import { globalMemoryEngine } from './server/memory';
import { SPECIALIZED_MODULES, classifyQuestionCategory } from './server/modules/registry';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === 'production';

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '10mb' }));

  const PORT = Number(process.env.PORT) || 3000;

  // 1. Local AI Generation Endpoint (Python Foundation Model Engine Only — Zero External APIs)
  app.post('/api/chat', async (req, res) => {
    try {
      const { messages, conversationId = 'default_conv' } = req.body;

      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        return res.status(400).json({ error: 'Messages array is required.' });
      }

      const lastMessage = messages[messages.length - 1].content;
      const category = classifyQuestionCategory(lastMessage);
      const module = SPECIALIZED_MODULES[category] || SPECIALIZED_MODULES.general;

      globalMemoryEngine.saveConversation(conversationId, lastMessage.substring(0, 30) + '...');
      globalMemoryEngine.saveMessage({
        id: Date.now().toString(),
        conversationId,
        role: 'user',
        content: lastMessage,
        timestamp: new Date().toISOString(),
      });

      const startTime = Date.now();
      const pyRes = await globalPythonBridge.inference(lastMessage, 48, 0.8);
      const latencyMs = Date.now() - startTime;

      let responseText = '';
      if (!pyRes.isTrained) {
        responseText =
          `[OMNIBRAIN FOUNDATION MODEL // UNTRAINED STATE]\n\n` +
          `The local decoder-only Transformer checkpoint has not been trained yet. ` +
          `Navigate to the **ML Architecture** workspace and click **"Run Pretraining"** to train from randomly initialized weights.\n\n` +
          `Query received: "${lastMessage}"\nRouted Module: ${module.name}`;
      } else {
        responseText = `[OMNIBRAIN LOCAL FOUNDATION MODEL // ${module.name}]\n\n${pyRes.text}`;
      }

      globalMemoryEngine.saveMessage({
        id: (Date.now() + 1).toString(),
        conversationId,
        role: 'assistant',
        content: responseText,
        timestamp: new Date().toISOString(),
      });

      res.json({
        text: responseText,
        modelUsed: 'omnibrain-transformer-python-v1',
        provider: 'PythonFoundationEngine',
        isTrained: Boolean(pyRes.isTrained),
        category,
        moduleName: module.name,
        latencyMs,
        timestamp: new Date().toISOString(),
      });
    } catch (err: any) {
      console.error('Inference Error:', err);
      res.status(500).json({ error: err.message || 'Local inference failure.' });
    }
  });

  // 2. ML Telemetry, Resource Safety & Dataset Analytics Endpoint
  app.get('/api/ml/status', async (req, res) => {
    try {
      const telemetry = await globalPythonBridge.getTelemetry();
      const datasetStats = globalDatasetPipeline.getStats();
      const provenance = globalDatasetPipeline.getProvenanceList();
      const quarantinedProvenance = globalDatasetPipeline.getQuarantinedProvenanceList();
      const rejectedProvenance = globalDatasetPipeline.getRejectedProvenanceList();
      const tokenizerConfig = globalTokenizer.getConfig();
      const corpusAudit = globalCorpusAuditor.getCachedOrRun(globalDatasetPipeline);

      const tokenizerComparisonPath = path.resolve('./datasets/reports/tokenizer_comparison.json');
      let tokenizerComparison = null;
      if (fs.existsSync(tokenizerComparisonPath)) {
        try {
          tokenizerComparison = JSON.parse(fs.readFileSync(tokenizerComparisonPath, 'utf8'));
        } catch {
          tokenizerComparison = null;
        }
      }

      res.json({
        modelName: 'OMNIBRAIN-Decoder-Transformer',
        softwareVersion: telemetry.softwareVersion,
        activeBackend: telemetry.activeBackend || telemetry.backend || 'python_ref',
        availableBackends: telemetry.availableBackends || [],
        selectedDevice: telemetry.selectedDevice || telemetry.hardware?.selectedDevice || 'cpu',
        preset: telemetry.preset,
        config: telemetry.config,
        isTrained: telemetry.isTrained,
        step: telemetry.step,
        epoch: telemetry.epoch,
        seed: telemetry.seed,
        estimates: telemetry.estimates,
        hardware: telemetry.hardware,
        lastMetrics: telemetry.lastMetrics,
        lastCheckpointDir: telemetry.lastCheckpointDir,
        tokenizer: {
          ...tokenizerConfig,
          activeTokens: globalTokenizer.getActiveTokenCount(),
          experimentalSubword: globalTokenizer.getExperimentalSubwordStatus(),
        },
        tokenizerComparison,
        datasets: datasetStats,
        provenance,
        quarantinedProvenance,
        rejectedProvenance,
        corpusAudit,
        rawArchives: globalAcquisitionManager.listArchivesWithProgress(),
        researchCatalog: globalAcquisitionManager.getResearchCatalogStatus(),
        trainingState: {
          status: telemetry.isTrained ? 'Trained & Operational' : 'Not Trained (Training Required)',
          checkpointDir: telemetry.lastCheckpointDir || 'None saved yet',
        },
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to read Python ML telemetry.' });
    }
  });

  app.post('/api/ml/backend', async (req, res) => {
    try {
      const { backend, device = 'auto' } = req.body || {};
      if (!backend) {
        return res.status(400).json({ success: false, error: 'backend is required (python_ref | pytorch).' });
      }
      const pyRes = await globalPythonBridge.setBackend(backend, device);
      if (!pyRes.success) {
        return res.status(400).json(pyRes);
      }
      res.json(pyRes);
    } catch (err: any) {
      res.status(500).json({ success: false, error: err.message });
    }
  });

  app.post('/api/ml/verify-numerical', async (_req, res) => {
    try {
      const pyRes = await globalPythonBridge.runNumericalVerify();
      res.json(pyRes);
    } catch (err: any) {
      res.status(500).json({ success: false, error: err.message });
    }
  });

  app.post('/api/ml/config', async (req, res) => {
    try {
      const { configName, seed = 42, backend } = req.body;
      const pyRes = await globalPythonBridge.setConfig(configName, seed, backend);
      if (!pyRes.success) {
        return res.status(400).json(pyRes);
      }
      res.json(pyRes);
    } catch (err: any) {
      res.status(500).json({ success: false, error: err.message });
    }
  });

  // 3. Dataset Analytics, Streaming Ingestion & License Validation Endpoints
  app.get('/api/datasets/stats', (req, res) => {
    res.json(globalDatasetPipeline.getStats());
  });

  app.get('/api/datasets/provenance', (req, res) => {
    res.json({
      accepted: globalDatasetPipeline.getProvenanceList(),
      quarantined: globalDatasetPipeline.getQuarantinedProvenanceList(),
      rejected: globalDatasetPipeline.getRejectedProvenanceList(),
    });
  });

  app.get('/api/datasets/audit', (_req, res) => {
    res.json(globalCorpusAuditor.getCachedOrRun(globalDatasetPipeline));
  });

  app.post('/api/datasets/audit/run', (_req, res) => {
    const report = globalCorpusAuditor.runFullAudit(globalDatasetPipeline);
    res.json(report);
  });

  app.post('/api/datasets/ingest-stream', async (req, res) => {
    try {
      const { filePath, datasetName, source, license, domain, language, maxBytes, maxDocuments } = req.body;
      if (!filePath || !datasetName || !license) {
        return res.status(400).json({ error: 'filePath, datasetName, and license are required.' });
      }
      const result = await globalDatasetPipeline.ingestFileStream(filePath, {
        datasetName,
        source: source || filePath,
        license,
        domain,
        language,
        maxBytes,
        maxDocuments,
      });
      res.json({ ...result, stats: globalDatasetPipeline.getStats() });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get('/api/datasets/research-catalog', (_req, res) => {
    res.json(globalAcquisitionManager.getResearchCatalogStatus());
  });

  app.get('/api/datasets/raw-archives', (_req, res) => {
    res.json({ archives: globalAcquisitionManager.listArchivesWithProgress() });
  });

  app.post('/api/datasets/archives/approve', (req, res) => {
    const { archiveId, approvedBy = 'operator' } = req.body || {};
    if (!archiveId) return res.status(400).json({ success: false, error: 'archiveId is required.' });
    const result = globalAcquisitionManager.approveArchive(archiveId, approvedBy);
    if (!result.success) return res.status(400).json(result);
    res.json(result);
  });

  app.post('/api/datasets/archives/reject', (req, res) => {
    const { archiveId, reason = 'Rejected during review' } = req.body || {};
    if (!archiveId) return res.status(400).json({ success: false, error: 'archiveId is required.' });
    const result = globalAcquisitionManager.rejectArchive(archiveId, reason);
    if (!result.success) return res.status(400).json(result);
    res.json(result);
  });

  app.post('/api/datasets/archives/quarantine', (req, res) => {
    const { archiveId, reason = 'Quarantined pending copyright/attribution verification' } = req.body || {};
    if (!archiveId) return res.status(400).json({ success: false, error: 'archiveId is required.' });
    const result = globalAcquisitionManager.quarantineArchive(archiveId, reason);
    if (!result.success) return res.status(400).json(result);
    res.json(result);
  });

  app.post('/api/datasets/archives/ingest', async (req, res) => {
    try {
      const { archiveId, maxRecordsThisPass, maxDocumentsTotal } = req.body || {};
      if (!archiveId) return res.status(400).json({ success: false, error: 'archiveId is required.' });
      const result = await globalAcquisitionManager.ingestApprovedArchive(archiveId, {
        maxRecordsThisPass,
        maxDocumentsTotal,
      });
      if (!result.success) return res.status(400).json(result);
      res.json({ ...result, stats: globalDatasetPipeline.getStats() });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err.message });
    }
  });

  app.get('/api/datasets/tokenizer-comparison', (_req, res) => {
    const reportPath = path.resolve('./datasets/reports/tokenizer_comparison.json');
    if (fs.existsSync(reportPath)) {
      return res.json(JSON.parse(fs.readFileSync(reportPath, 'utf8')));
    }
    const docs = globalDatasetPipeline.getCorpusDocuments().map((d) => ({
      text: d.cleanedText,
      language: d.language,
      domain: d.domain,
    }));
    res.json(evaluateTokenizerComparison(docs, globalTokenizer, 2048));
  });

  app.post('/api/datasets/tokenizer-comparison/run', (req, res) => {
    const targetVocabSize = Number(req.body?.targetVocabSize) || 2048;
    const docs = globalDatasetPipeline.getCorpusDocuments().map((d) => ({
      text: d.cleanedText,
      language: d.language,
      domain: d.domain,
    }));
    res.json(evaluateTokenizerComparison(docs, globalTokenizer, targetVocabSize));
  });

  // 4. Genuine Foundation Model Pretraining & Checkpoint Management
  app.post('/api/ml/train', async (req, res) => {
    try {
      const { backend, epochs = 1, learningRate = 0.008, maxSteps = 12, evalInterval = 4, checkpointInterval = 6, resumeFrom } = req.body || {};

      // Rebuild packed splits and persist tokenizer vocab + manifest before training
      const splits = globalDatasetPipeline.buildPretrainingSplits(48, 42);
      globalDatasetPipeline.writeManifest();

      const pyRes = await globalPythonBridge.trainPacked({
        backend,
        splitsPath: globalDatasetPipeline.packedSplitsPath,
        epochs,
        learningRate,
        maxSteps,
        evalInterval,
        checkpointInterval,
        resumeFrom,
      });

      if (!pyRes.success) {
        return res.status(400).json(pyRes);
      }

      res.json({
        success: true,
        backendUsed: pyRes.backend || pyRes.activeBackend,
        selectedDevice: pyRes.selectedDevice || pyRes.hardware?.selectedDevice || 'cpu',
        cancelled: pyRes.cancelled,
        message: pyRes.cancelled
          ? `Training (${pyRes.backend}) gracefully cancelled; interrupted checkpoint saved.`
          : `Pretraining via ${pyRes.backend} (${pyRes.selectedDevice || 'cpu'}) completed (${pyRes.metrics.optimizerSteps} optimizer steps across ${splits.trainTokenCount} training tokens).`,
        metrics: pyRes.metrics,
        checkpoint: pyRes.checkpoint,
        hardware: pyRes.hardware,
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Training failed.' });
    }
  });

  app.post('/api/ml/cancel', (req, res) => {
    const cancelRes = globalPythonBridge.cancelTraining();
    res.json(cancelRes);
  });

  // 4.1 Phase 7 Extended Foundation Pretraining Endpoints
  app.post('/api/ml/phase7/train', async (req, res) => {
    try {
      const {
        stage = 'stageA',
        targetTokens,
        backend,
        batchSize = 32,
        learningRate = 0.005,
        minLr = 0.0002,
        evalIntervalTokens = 48000,
        resumeFrom,
      } = req.body || {};

      const pyRes = await globalPythonBridge.trainPhase7({
        stage,
        targetTokens: targetTokens ? Number(targetTokens) : undefined,
        backend,
        batchSize: Number(batchSize),
        learningRate: Number(learningRate),
        minLr: Number(minLr),
        evalIntervalTokens: Number(evalIntervalTokens),
        resumeFrom,
      });

      if (!pyRes.success && !pyRes.cancelled) {
        return res.status(400).json(pyRes);
      }

      res.json(pyRes);
    } catch (err: any) {
      res.status(500).json({ success: false, error: err.message || 'Phase 7 training failed.' });
    }
  });

  app.get('/api/ml/phase7/status', async (_req, res) => {
    try {
      const phase7Dir = path.resolve('./checkpoints/phase7');
      const checkpoints: any[] = [];
      if (fs.existsSync(phase7Dir)) {
        const entries = fs.readdirSync(phase7Dir);
        for (const entry of entries) {
          const entryPath = path.join(phase7Dir, entry);
          const manifestPath = path.join(entryPath, 'manifest.json');
          if (fs.existsSync(manifestPath)) {
            try {
              const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
              checkpoints.push({
                dirName: entry,
                dirPath: entryPath,
                ...manifest,
              });
            } catch {
              // Ignore corrupt manifest
            }
          }
        }
      }

      checkpoints.sort((a, b) => (b.cumulativeTokens || 0) - (a.cumulativeTokens || 0));

      const telemetry = await globalPythonBridge.getTelemetry();
      const packedSplitsPath = path.resolve('./datasets/packed_splits.json');
      let splitInfo: any = null;
      if (fs.existsSync(packedSplitsPath)) {
        try {
          const rawSplits = JSON.parse(fs.readFileSync(packedSplitsPath, 'utf8'));
          splitInfo = {
            version: rawSplits.version,
            manifestHash: rawSplits.manifestHash,
            tokenizerVersion: rawSplits.tokenizerVersion,
            tokenizerChecksum: rawSplits.tokenizerChecksum,
            trainTokenCount: rawSplits.trainTokenCount,
            valTokenCount: rawSplits.valTokenCount,
            testTokenCount: rawSplits.testTokenCount,
            leakageVerifiedZeroOverlap: rawSplits.leakageVerifiedZeroOverlap,
            quarantinedExcluded: ['FineWeb-Edu', 'Tatoeba-Indic'],
          };
        } catch {
          splitInfo = null;
        }
      }

      res.json({
        success: true,
        phase: '7',
        activeBackend: telemetry.activeBackend || telemetry.backend || 'pytorch',
        selectedDevice: telemetry.selectedDevice || telemetry.hardware?.selectedDevice || 'cpu',
        preset: telemetry.preset || 'tiny_cpu',
        isTrained: telemetry.isTrained,
        cumulativeTokens: telemetry.lastMetrics?.cumulativeTokens || (checkpoints[0]?.cumulativeTokens || 0),
        bestValLoss: telemetry.lastMetrics?.bestValLoss || (checkpoints[0]?.bestValLoss || null),
        lastMetrics: telemetry.lastMetrics,
        checkpoints,
        splitInfo,
        hardware: telemetry.hardware,
      });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err.message });
    }
  });

  app.post('/api/ml/phase7/restore', async (req, res) => {
    try {
      const { path: ckptPath } = req.body || {};
      if (!ckptPath) {
        return res.status(400).json({ success: false, error: 'Checkpoint path is required.' });
      }
      const loadRes = await globalPythonBridge.loadPhase7Checkpoint(ckptPath);
      if (!loadRes.success) {
        return res.status(400).json(loadRes);
      }
      res.json(loadRes);
    } catch (err: any) {
      res.status(500).json({ success: false, error: err.message });
    }
  });

  app.post('/api/ml/checkpoint/save', async (req, res) => {
    try {
      const saveRes = await globalPythonBridge.saveCheckpoint();
      res.json(saveRes);
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.post('/api/ml/checkpoint/load', async (req, res) => {
    try {
      const { path: ckptPath } = req.body;
      if (!ckptPath) return res.status(400).json({ error: 'Checkpoint path is required.' });
      const loadRes = await globalPythonBridge.loadCheckpoint(ckptPath);
      if (!loadRes.success) return res.status(400).json(loadRes);
      res.json(loadRes);
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // 5. Automated Architecture Test Suite Endpoint
  app.get('/api/tests', async (req, res) => {
    try {
      const mlTests = runFullGradientAuditAndTests();
      const dataTests = await runDataEngineeringAndAuditTests();
      res.json({
        allPassed: mlTests.allPassed && dataTests.allPassed,
        mlTests,
        dataTests,
      });
    } catch (err: any) {
      res.status(500).json({ allPassed: false, error: err.message });
    }
  });

  // AI Create Studio endpoint (uses Python local engine)
  app.post('/api/create-studio', async (req, res) => {
    try {
      const { type, topic, format } = req.body;
      const pyRes = await globalPythonBridge.inference(`Generate ${format || 'report'} about ${topic}`, 64, 0.8);

      res.json({
        content: pyRes.isTrained
          ? pyRes.text
          : `[OMNIBRAIN Studio - Model Untrained]\n\nPlease train the local foundation model in the ML Architecture workspace before generating text.`,
        title: topic,
        type,
        format,
        timestamp: new Date().toISOString(),
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // Vite integration for development
  if (!isProduction) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 OMNIBRAIN Phase 5.1 Server running at http://localhost:${PORT}`);
  });
}

startServer().catch(console.error);
