import fs from 'fs';
import os from 'os';
import path from 'path';
import zlib from 'zlib';
import { DatasetPipeline, ALL_DOMAINS, globalDatasetPipeline } from './pipeline';
import { LocalDatasetAcquisitionManager } from './acquisition';
import { DataQualitySystem } from './quality';
import { ProvenanceRegistry } from './provenance';
import { globalCorpusAuditor } from './audit';
import { OmnibrainTransformerModel } from '../core/model';
import { OmnibrainTokenizer, globalTokenizer } from '../core/tokenizer';
import { ExperimentalByteLevelBpeTokenizer } from '../core/bpe_research';
import { TransformerGradientAuditor } from '../core/gradient_audit';

export async function runDataEngineeringAndAuditTests() {
  console.log('🧪 Starting OMNIBRAIN Phase 6 Data Engineering, Acquisition & Tokenizer Validation Tests...\n');
  const results: Array<{ test: string; passed: boolean; details: string }> = [];

  // 1. Quality Filtering, Near-Duplicate & Corruption Detection
  try {
    const qSys = new DataQualitySystem();
    const res1 = qSys.cleanAndFilter('Quantum mechanics describes nature at the atomic scale accurately.');
    const resExact = qSys.cleanAndFilter('Quantum mechanics describes nature at the atomic scale accurately.');
    const resNear = qSys.cleanAndFilter('Quantum mechanics describes nature at the atomic scale precisely.');
    const resCorrupt = qSys.cleanAndFilter('Invalid\u0000corrupted\u0000binary\u0000payload');
    const resEmpty = qSys.cleanAndFilter('   ');
    const passed = res1.passed && !resExact.passed && !resNear.passed && !resCorrupt.passed && !resEmpty.passed;
    results.push({
      test: 'Quality Filtering, Exact/Near-Duplicate & Corruption Detection',
      passed,
      details: `Valid=${res1.passed}, ExactDupBlocked=${!resExact.passed}, NearDupBlocked=${!resNear.passed}, CorruptBlocked=${!resCorrupt.passed}`,
    });
  } catch (err: any) {
    results.push({ test: 'Quality Filtering, Exact/Near-Duplicate & Corruption Detection', passed: false, details: err.message });
  }

  // 2. All 11 Dataset Domains Coverage
  try {
    const stats = globalDatasetPipeline.getStats();
    const missing = ALL_DOMAINS.filter((d) => (stats.domainDistribution[d] || 0) === 0);
    const passed = stats.all11DomainsCovered && missing.length === 0;
    results.push({
      test: 'All 11 Foundation Domains Represented',
      passed,
      details: `Domains covered: ${stats.domainsCovered}/${ALL_DOMAINS.length}, Total docs: ${stats.totalDocuments}, Total tokens: ${stats.totalTokens}`,
    });
  } catch (err: any) {
    results.push({ test: 'All 11 Foundation Domains Represented', passed: false, details: err.message });
  }

  // 3. Streaming File Ingestion (.jsonl, .csv, .txt.gz) & License Rejection
  try {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'omnibrain_stream_'));
    const isolatedQuality = new DataQualitySystem();
    const isolatedProvenance = new ProvenanceRegistry();
    const pipeline = new DatasetPipeline(isolatedQuality, isolatedProvenance, false);

    const gzPath = path.join(tmpDir, 'sample_corpus.txt.gz');
    const txtContent =
      'Streaming test paragraph one discussing algorithmic complexity and asymptotic notation.\n\n' +
      'Streaming test paragraph two explaining thermodynamic entropy and statistical ensembles.\n';
    fs.writeFileSync(gzPath, zlib.gzipSync(Buffer.from(txtContent, 'utf8')));

    const streamRes = await pipeline.ingestFileStream(gzPath, {
      datasetName: 'Test-Gzip-Stream',
      source: gzPath,
      license: 'CC-BY-4.0',
      domain: 'computer_science',
      maxBytes: 1024 * 1024,
    });

    const rejectedRes = await pipeline.ingestFileStream(gzPath, {
      datasetName: 'Test-Restricted-Dataset',
      source: gzPath,
      license: 'CC-BY-NC-SA-4.0',
      domain: 'computer_science',
    });

    const rejectedList = pipeline.getRejectedProvenanceList();
    fs.rmSync(tmpDir, { recursive: true, force: true });

    const passed =
      streamRes.accepted &&
      streamRes.documentsIngested === 2 &&
      !rejectedRes.accepted &&
      rejectedList.some((r) => r.datasetName === 'Test-Restricted-Dataset');
    results.push({
      test: 'Streaming Gzip/Text Ingestion & License Rejection',
      passed,
      details: `Streamed docs=${streamRes.documentsIngested}, Bytes=${streamRes.bytesProcessed}, NC rejected=${!rejectedRes.accepted}, RejectedTracked=${rejectedList.length}`,
    });
  } catch (err: any) {
    results.push({ test: 'Streaming Gzip/Text Ingestion & License Rejection', passed: false, details: err.message });
  }

  // 4. Tokenizer Checksum Stability & Multilingual Unknown-Token Rates
  try {
    const t1 = new OmnibrainTokenizer(1024);
    const t2 = new OmnibrainTokenizer(1024);
    const checksumMatch = t1.getChecksum() === t2.getChecksum();

    const samples: Record<string, string> = {
      English: 'Transformer models learn contextual representations.',
      Hindi: 'विज्ञान और गणित की शिक्षा आवश्यक है।',
      Tamil: 'கணிதம் மற்றும் கணினி அறிவியல்.',
      Telugu: 'శాస్త్రీయ పరిశోధన మరియు డేటా విశ్లేషణ కీలకం.',
      Kannada: 'ತಂತ್ರಜ್ಞಾನ ಮತ್ತು ಶಿಕ್ಷಣದ ಸಮನ್ವಯವು ಮುಖ್ಯವಾಗಿದೆ.',
      Bengali: 'যুক্তিবাদী চিন্তা এবং বৈজ্ঞানিক অনুসন্ধান গুরুত্বপূর্ণ।',
      Math: '∫_a^b f(x) dx = ΔF ≈ 0',
      Code: 'def f(x): return [i * 2 for i in x]',
    };

    const rates: string[] = [];
    let allLowUnk = true;
    for (const [lang, sample] of Object.entries(samples)) {
      const stats = t1.encodeWithStats(sample);
      const decoded = t1.decode(stats.tokens);
      if (decoded !== sample) allLowUnk = false;
      rates.push(`${lang}:${(stats.unkRate * 100).toFixed(1)}%`);
    }

    const emojiStats = t1.encodeWithStats('Hello 🚀🤖');
    const unkDetectedForEmoji = emojiStats.unkCount === 2;

    results.push({
      test: 'Tokenizer Checksum Stability & Multilingual <unk> Rates',
      passed: checksumMatch && allLowUnk && unkDetectedForEmoji,
      details: `Checksum=${t1.getChecksum().slice(0, 12)}, UnkRates=[${rates.join(', ')}], EmojiUnkCount=${emojiStats.unkCount}`,
    });
  } catch (err: any) {
    results.push({ test: 'Tokenizer Checksum Stability & Multilingual <unk> Rates', passed: false, details: err.message });
  }

  // 5. Sequence Packing & Train/Val/Test Leakage Prevention + Manifest Reproducibility
  try {
    const splits1 = globalDatasetPipeline.getPackedSplits();
    const splits2 = globalDatasetPipeline.buildPretrainingSplits(48, 42);
    const reproducible = splits1.manifestHash === splits2.manifestHash && splits1.trainTokenCount === splits2.trainTokenCount;
    const passed =
      splits1.leakageVerifiedZeroOverlap &&
      reproducible &&
      splits1.trainSequences.length > 0 &&
      splits1.valSequences.length > 0 &&
      splits1.tokenizerChecksum === globalTokenizer.getChecksum();
    results.push({
      test: 'Sequence Packing, Zero-Leakage Split & Manifest Reproducibility',
      passed,
      details: `TrainSeqs=${splits1.trainSequences.length} (${splits1.trainTokenCount} tok), ValSeqs=${splits1.valSequences.length} (${splits1.valTokenCount} tok), Hash=${splits1.manifestHash}, Reproducible=${reproducible}`,
    });
  } catch (err: any) {
    results.push({ test: 'Sequence Packing, Zero-Leakage Split & Manifest Reproducibility', passed: false, details: err.message });
  }

  // 6. Phase 6 Resumable Ingestion, Explicit Approval Gate, Duplicate Archive & Tamper Checks
  try {
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'omnibrain_acq_'));
    const isoPipeline = new DatasetPipeline(new DataQualitySystem(), new ProvenanceRegistry(), false);
    // Point isolated pipeline directories cleanly inside tmpRoot
    (isoPipeline as any).dirs = {
      root: tmpRoot,
      raw: path.join(tmpRoot, 'raw'),
      processed: path.join(tmpRoot, 'processed'),
      tokenized: path.join(tmpRoot, 'tokenized'),
      manifests: path.join(tmpRoot, 'manifests'),
      rejected: path.join(tmpRoot, 'rejected'),
      reports: path.join(tmpRoot, 'reports'),
    };
    isoPipeline.ensureDirectoryLayout();

    const acq = new LocalDatasetAcquisitionManager(isoPipeline);
    const jsonlGz = path.join(tmpRoot, 'raw', 'test_records.jsonl.gz');
    const lines = [
      JSON.stringify({ domain: 'physics_chemistry', language: 'English', text: 'Thermodynamic entropy quantifies microscopic state multiplicity cleanly in equilibrium systems.' }),
      JSON.stringify({ domain: 'indian_languages', language: 'Telugu', text: 'శాస్త్రీయ పద్ధతిలో పరిశీలన మరియు ప్రయోగం ద్వారా విజ్ఞానం అభివృద్ధి చెందుతుంది.' }),
      JSON.stringify({ domain: 'mathematics', language: 'English', text: 'Eigenvalues of real symmetric matrices remain strictly real under orthogonal transformations.' }),
      JSON.stringify({ domain: 'biology', language: 'English', text: 'Mitochondrial ATP synthase couples proton gradients to adenosine triphosphate synthesis.' }),
    ];
    fs.writeFileSync(jsonlGz, zlib.gzipSync(Buffer.from(lines.join('\n') + '\n', 'utf8')));

    // Register -> must block ingestion until explicitly approved
    const reg = acq.registerLocalArchive({
      filePath: jsonlGz,
      archiveId: 'test_resumable_gz',
      datasetName: 'Test Resumable Corpus',
      officialSourceUrl: 'local://test',
      publisher: 'OMNIBRAIN Test',
      license: 'MIT',
      attributionRequirements: 'MIT attribution',
      language: 'Multi',
      defaultDomain: 'general_knowledge',
    });
    const blockedUnapproved = await acq.ingestApprovedArchive('test_resumable_gz');

    // Duplicate registration by same content SHA-256 under new ID must fail
    const dupCopy = path.join(tmpRoot, 'raw', 'test_records_dup.jsonl.gz');
    fs.copyFileSync(jsonlGz, dupCopy);
    const dupReg = acq.registerLocalArchive({
      filePath: dupCopy,
      archiveId: 'test_dup_gz',
      datasetName: 'Duplicate Archive',
      officialSourceUrl: 'local://test',
      publisher: 'OMNIBRAIN Test',
      license: 'MIT',
      attributionRequirements: 'MIT attribution',
      language: 'Multi',
      defaultDomain: 'general_knowledge',
    });

    // Explicitly approve, ingest 2 records, then resume remaining 2 records
    const approveRes = acq.approveArchive('test_resumable_gz', 'unit-tester');
    const pass1 = await acq.ingestApprovedArchive('test_resumable_gz', { maxRecordsThisPass: 2 });
    const pass2 = await acq.ingestApprovedArchive('test_resumable_gz', { maxRecordsThisPass: 5 });
    const dupReingest = await acq.ingestApprovedArchive('test_resumable_gz');

    // Verify tamper/integrity rejection when NC license archive tries approval
    const ncFile = path.join(tmpRoot, 'raw', 'nc_test.jsonl.gz');
    fs.writeFileSync(ncFile, zlib.gzipSync(Buffer.from(lines[0] + '\n', 'utf8')));
    acq.registerLocalArchive({
      filePath: ncFile,
      archiveId: 'nc_blocked',
      datasetName: 'NC Blocked Corpus',
      officialSourceUrl: 'local://nc',
      publisher: 'Test',
      license: 'CC-BY-NC-3.0',
      attributionRequirements: 'NC',
      language: 'English',
      defaultDomain: 'biology',
    });
    const ncApprove = acq.approveArchive('nc_blocked', 'unit-tester');

    fs.rmSync(tmpRoot, { recursive: true, force: true });

    const passed =
      reg.registered &&
      !blockedUnapproved.success &&
      !dupReg.registered &&
      approveRes.success &&
      pass1.success &&
      pass1.documentsIngestedThisPass === 2 &&
      !pass1.completed &&
      pass2.success &&
      pass2.resumedFromRecord === 2 &&
      pass2.documentsIngestedThisPass === 2 &&
      pass2.completed &&
      !dupReingest.success &&
      !ncApprove.success;

    results.push({
      test: 'Controlled Acquisition: Approval Gate, SHA-256 Integrity, Resumable Stream & Duplicate Prevention',
      passed,
      details: `UnapprovedBlocked=${!blockedUnapproved.success}, DupArchiveBlocked=${!dupReg.registered}, Pass1=${pass1.documentsIngestedThisPass}, ResumedFrom=${pass2.resumedFromRecord}, Pass2=${pass2.documentsIngestedThisPass}, NCBlocked=${!ncApprove.success}`,
    });
  } catch (err: any) {
    results.push({
      test: 'Controlled Acquisition: Approval Gate, SHA-256 Integrity, Resumable Stream & Duplicate Prevention',
      passed: false,
      details: err.message,
    });
  }

  // 7. Phase 6 Experimental Byte-Level BPE Verification (Long Docs, Rare Unicode, Zero-<unk> & Production Locked)
  try {
    const bpe = new ExperimentalByteLevelBpeTokenizer();
    const longDocSample =
      'Transformer self-attention computes queries, keys, and values with causal masking and residual connections. '.repeat(30);
    const rareUnicodeSample =
      'Rare Unicode & Conjuncts: ∫_a^b ∑_{k=1}^∞ ∂f/∂x dx ≈ ΔF, ∀x ∈ ℤ^n, €99 · ₹420 — తెలుగు క్షత్ర · हिन्दी कृत्रिम · தமிழ் · ಕನ್ನಡ · বাংলা 🚀✅';
    const corpus = [
      longDocSample,
      rareUnicodeSample,
      'శాస్త్రీయ పరిశోధన మరియు డేటా విశ్లేషణ ఆధునిక ఇంజనీరింగ్ సమస్యలను పరిష్కరిస్తాయి.',
      'विज्ञान और गणित की शिक्षा महत्वपूर्ण है। 🚀∑∫∇',
      'def attention(q, k, v): return softmax(q @ k.T) @ v',
    ];
    const trainStats = bpe.train(corpus, 512);
    let exactRoundTrip = true;
    for (const s of corpus) {
      const encoded = bpe.encode(s);
      if (encoded.includes(bpe.specialTokens['<unk>'])) exactRoundTrip = false;
      if (bpe.decode(encoded) !== s) exactRoundTrip = false;
    }
    const passed =
      trainStats.learnedMerges > 10 &&
      exactRoundTrip &&
      globalTokenizer.version === 'omnibrain-tok-v1.1.0';
    results.push({
      test: 'Experimental BPE Long-Doc & Rare-Unicode Round-Trip + Production Lock',
      passed,
      details: `BPEVocab=${bpe.getVocabSize()} (${trainStats.learnedMerges} merges), LongDocLen=${longDocSample.length}, ExactRoundTrip=${exactRoundTrip}, ProdLocked=${globalTokenizer.version}`,
    });
  } catch (err: any) {
    results.push({
      test: 'Experimental BPE Long-Doc & Rare-Unicode Round-Trip + Production Lock',
      passed: false,
      details: err.message,
    });
  }

  // 8. Phase 2.1 Gradient & Bias Verification
  try {
    const tmpCkptDir = fs.mkdtempSync(path.join(os.tmpdir(), 'omnibrain_data_ml_'));
    const tokenizer = new OmnibrainTokenizer(256);
    const model = new OmnibrainTransformerModel({
      vocabSize: 256,
      dModel: 32,
      numLayers: 2,
      numHeads: 2,
      maxSeqLen: 128,
      dFf: 64,
    }, tmpCkptDir);
    const auditor = new TransformerGradientAuditor(model, tokenizer);
    const auditRes = auditor.runFullAudit();
    fs.rmSync(tmpCkptDir, { recursive: true, force: true });
    results.push({
      test: 'Phase 2.1 Gradient & Bias Verification',
      passed: auditRes.allPassed,
      details: `Verified ${auditRes.results.length} parameter tensors (maxAbsErr threshold < 0.05).`,
    });
  } catch (err: any) {
    results.push({ test: 'Phase 2.1 Gradient & Bias Verification', passed: false, details: err.message });
  }

  // 9. Phase 6.1 Independent Forensic Corpus Audit, Archive Hashes, Quarantine & Reproducibility
  try {
    const audit1 = globalCorpusAuditor.runFullAudit(globalDatasetPipeline);
    const audit2 = globalCorpusAuditor.runFullAudit(globalDatasetPipeline);

    const hashesOk =
      audit1.summary.archiveSha256AllMatched &&
      audit1.integrityAudit.archiveHashChecksPassed === audit1.integrityAudit.archiveHashChecksTotal;
    const quarantineEnforced =
      audit1.summary.quarantinedArchives === 6 &&
      audit1.summary.quarantinedDocs === 1016 &&
      audit1.summary.totalVerifiedDocsAfterQuarantine === 4726 &&
      audit1.summary.all11DomainsCoveredAfterQuarantine;
    const provenanceComplete = audit1.licenseAndProvenanceAudit.records.every(
      (r) => Boolean(r.publisher && r.officialSourceUrl && r.license && r.evidenceArtifactPath && r.sha256)
    );
    const zeroCrossSplitLeak =
      audit1.qualityAndContaminationAudit.crossSplitLeakage.exactHashOverlaps === 0 &&
      audit1.qualityAndContaminationAudit.crossSplitLeakage.prefix80Overlaps === 0 &&
      audit1.qualityAndContaminationAudit.exactDuplicatesRemainingInVerified === 0;
    const reproducibleAudit = audit1.auditContentSha256 === audit2.auditContentSha256;

    const passed = hashesOk && quarantineEnforced && provenanceComplete && zeroCrossSplitLeak && reproducibleAudit;
    results.push({
      test: 'Phase 6.1 Forensic Audit: Archive SHA-256, Quarantine, Provenance, Zero Prefix Leakage & Reproducibility',
      passed,
      details: `ArchivesVerified=${audit1.integrityAudit.archiveHashChecksPassed}/${audit1.integrityAudit.archiveHashChecksTotal}, VerifiedDocs=${audit1.summary.totalVerifiedDocsAfterQuarantine}, Quarantined=${audit1.summary.quarantinedDocs}, PrefixLeak=${audit1.qualityAndContaminationAudit.crossSplitLeakage.prefix80Overlaps}, AuditHashMatch=${reproducibleAudit}`,
    });
  } catch (err: any) {
    results.push({
      test: 'Phase 6.1 Forensic Audit: Archive SHA-256, Quarantine, Provenance, Zero Prefix Leakage & Reproducibility',
      passed: false,
      details: err.message,
    });
  }

  console.log('📋 Phase 6.1 Data Engineering, Acquisition & Corpus Audit Test Results:');
  let allPassed = true;
  results.forEach((r, idx) => {
    const icon = r.passed ? '✅' : '❌';
    console.log(`${idx + 1}. ${icon} [${r.test}]: ${r.details}`);
    if (!r.passed) allPassed = false;
  });

  console.log(`\nResult: ${allPassed ? 'ALL DATA ENGINEERING & AUDIT TESTS PASSED 🎉' : 'SOME TESTS FAILED ⚠️'}\n`);
  return { allPassed, results };
}
