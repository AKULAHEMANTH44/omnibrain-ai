import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import readline from 'readline';
import zlib from 'zlib';
import { execFileSync, spawn } from 'child_process';
import { ALL_DOMAINS, DatasetDocument, DatasetDomain, DatasetPipeline, globalDatasetPipeline } from './pipeline';
import { OFFICIAL_DATASET_CATALOG, CatalogDatasetEntry } from './catalog';

export interface LocalArchiveMetadata {
  archiveId: string;
  fileName: string;
  relativePath: string;
  datasetName: string;
  officialSourceUrl: string;
  publisher: string;
  license: string;
  attributionRequirements: string;
  language: string;
  defaultDomain: DatasetDomain;
  fileSizeBytes: number;
  sha256: string;
  downloadedAt: string;
  approvalStatus: 'pending_approval' | 'approved' | 'rejected' | 'quarantined';
  approvedBy?: string;
  approvedAt?: string;
  quarantinedAt?: string;
  quarantineReason?: string;
  rejectionReason?: string;
  statusStage: 'downloaded' | 'verified_approved' | 'ingested' | 'rejected' | 'quarantined';
}

export interface IngestionCheckpoint {
  archiveId: string;
  sha256: string;
  processedRecords: number;
  ingestedDocuments: number;
  rejectedDocuments: number;
  rawTokens: number;
  cleanedTokens: number;
  completed: boolean;
  lastUpdatedAt: string;
}

export interface IngestProgressOptions {
  maxRecordsThisPass?: number;
  maxDocumentsTotal?: number;
}

function sha256File(filePath: string): string {
  const hash = crypto.createHash('sha256');
  const fd = fs.openSync(filePath, 'r');
  const buf = Buffer.allocUnsafe(64 * 1024);
  try {
    let bytesRead = 0;
    while ((bytesRead = fs.readSync(fd, buf, 0, buf.length, null)) > 0) {
      hash.update(buf.subarray(0, bytesRead));
    }
  } finally {
    fs.closeSync(fd);
  }
  return hash.digest('hex');
}

function resolvePythonBin(): string {
  const venvUnix = path.resolve('./.venv/bin/python');
  if (fs.existsSync(venvUnix)) return venvUnix;
  const venvWin = path.resolve('./.venv/Scripts/python.exe');
  if (fs.existsSync(venvWin)) return venvWin;
  return 'python3';
}

function mapMmluOrArxivCategoryToDomain(hint: string, fallback: DatasetDomain): DatasetDomain {
  const h = (hint || '').toLowerCase();
  if (/math|algebra|calculus|geometry|statistics|number_theory/.test(h)) return 'mathematics';
  if (/phys|chem|quant-ph|cond-mat|thermodynamics|astronomy/.test(h)) return 'physics_chemistry';
  if (/bio|anatomy|medicine|genetics|nutrition|virology|q-bio|pubmed/.test(h)) return 'biology';
  if (/computer|cs\.|algorithm|machine_learning|operating_system|security/.test(h)) return 'computer_science';
  if (/program|python|rust|code|software/.test(h)) return 'programming';
  if (/eess|electr|mechan|civil|engineering/.test(h)) return 'engineering';
  if (/edu|curriculum|course|pedagog/.test(h)) return 'education';
  if (/stat\.ml|research|preprint/.test(h)) return 'research';
  if (/telugu|hindi|tamil|kannada|bengali|indic/.test(h)) return 'indian_languages';
  return fallback;
}

export class LocalDatasetAcquisitionManager {
  private pipeline: DatasetPipeline;
  private rawDir: string;
  private checkpointsPath: string;
  private statusReportPath: string;

  constructor(pipeline: DatasetPipeline = globalDatasetPipeline) {
    this.pipeline = pipeline;
    this.rawDir = pipeline.dirs.raw;
    this.checkpointsPath = path.join(pipeline.dirs.manifests, 'ingestion_checkpoints.json');
    this.statusReportPath = path.join(pipeline.dirs.reports, 'acquisition_status.json');
    fs.mkdirSync(this.rawDir, { recursive: true });
    fs.mkdirSync(pipeline.dirs.manifests, { recursive: true });
    fs.mkdirSync(pipeline.dirs.reports, { recursive: true });
  }

  private loadCheckpoints(): Record<string, IngestionCheckpoint> {
    if (!fs.existsSync(this.checkpointsPath)) return {};
    try {
      return JSON.parse(fs.readFileSync(this.checkpointsPath, 'utf8'));
    } catch {
      return {};
    }
  }

  private saveCheckpoints(map: Record<string, IngestionCheckpoint>): void {
    fs.writeFileSync(this.checkpointsPath, JSON.stringify(map, null, 2), 'utf8');
  }

  public registerLocalArchive(params: {
    filePath: string;
    archiveId: string;
    datasetName: string;
    officialSourceUrl: string;
    publisher: string;
    license: string;
    attributionRequirements: string;
    language: string;
    defaultDomain: DatasetDomain;
  }): { registered: boolean; reason?: string; metadata?: LocalArchiveMetadata } {
    const absFile = path.resolve(params.filePath);
    if (!fs.existsSync(absFile)) {
      return { registered: false, reason: `Local archive not found: ${absFile}` };
    }

    const hash = sha256File(absFile);
    const stat = fs.statSync(absFile);

    // Check duplicate registration by SHA-256 across existing metadata files
    const existingList = this.listArchives();
    const dupByHash = existingList.find((m) => m.sha256 === hash && m.archiveId !== params.archiveId);
    if (dupByHash) {
      return {
        registered: false,
        reason: `Duplicate archive blocked: SHA-256 ${hash.slice(0, 12)} already registered under '${dupByHash.archiveId}'.`,
      };
    }

    const metaPath = `${absFile}.meta.json`;
    const compatible = this.pipeline.getProvenanceRegistry().isLicenseCompatible(params.license);

    const metadata: LocalArchiveMetadata = {
      archiveId: params.archiveId,
      fileName: path.basename(absFile),
      relativePath: path.relative(process.cwd(), absFile),
      datasetName: params.datasetName,
      officialSourceUrl: params.officialSourceUrl,
      publisher: params.publisher,
      license: params.license,
      attributionRequirements: params.attributionRequirements,
      language: params.language,
      defaultDomain: ALL_DOMAINS.includes(params.defaultDomain) ? params.defaultDomain : 'general_knowledge',
      fileSizeBytes: stat.size,
      sha256: hash,
      downloadedAt: new Date().toISOString(),
      approvalStatus: compatible ? 'pending_approval' : 'rejected',
      rejectionReason: compatible ? undefined : `License '${params.license}' is incompatible or unverified for model training.`,
      statusStage: compatible ? 'downloaded' : 'rejected',
    };

    fs.writeFileSync(metaPath, JSON.stringify(metadata, null, 2), 'utf8');
    this.writeStatusReport();
    return { registered: true, metadata };
  }

  public listArchives(): LocalArchiveMetadata[] {
    if (!fs.existsSync(this.rawDir)) return [];
    const checkpoints = this.loadCheckpoints();
    const metas: LocalArchiveMetadata[] = [];

    for (const file of fs.readdirSync(this.rawDir).sort()) {
      if (!file.endsWith('.meta.json')) continue;
      try {
        const meta: LocalArchiveMetadata = JSON.parse(fs.readFileSync(path.join(this.rawDir, file), 'utf8'));
        const cp = checkpoints[meta.archiveId];
        if (meta.approvalStatus === 'quarantined') {
          meta.statusStage = 'quarantined';
        } else if (meta.approvalStatus === 'rejected') {
          meta.statusStage = 'rejected';
        } else if (cp?.completed) {
          meta.statusStage = 'ingested';
        } else if (meta.approvalStatus === 'approved') {
          meta.statusStage = 'verified_approved';
        } else {
          meta.statusStage = 'downloaded';
        }
        metas.push(meta);
      } catch {
        // Ignore unreadable metadata file
      }
    }
    return metas;
  }

  public listArchivesWithProgress() {
    const archives = this.listArchives();
    const checkpoints = this.loadCheckpoints();
    return archives.map((a) => ({
      ...a,
      progress: checkpoints[a.archiveId] || {
        archiveId: a.archiveId,
        sha256: a.sha256,
        processedRecords: 0,
        ingestedDocuments: 0,
        rejectedDocuments: 0,
        rawTokens: 0,
        cleanedTokens: 0,
        completed: false,
        lastUpdatedAt: '',
      },
    }));
  }

  private findMetaPath(archiveId: string): string | null {
    if (!fs.existsSync(this.rawDir)) return null;
    for (const file of fs.readdirSync(this.rawDir)) {
      if (!file.endsWith('.meta.json')) continue;
      const full = path.join(this.rawDir, file);
      try {
        const meta: LocalArchiveMetadata = JSON.parse(fs.readFileSync(full, 'utf8'));
        if (meta.archiveId === archiveId) return full;
      } catch {
        continue;
      }
    }
    return null;
  }

  public approveArchive(
    archiveId: string,
    approvedBy = 'local-operator'
  ): { success: boolean; error?: string; metadata?: LocalArchiveMetadata } {
    const metaPath = this.findMetaPath(archiveId);
    if (!metaPath) {
      return { success: false, error: `Archive '${archiveId}' metadata not found in datasets/raw/.` };
    }
    const meta: LocalArchiveMetadata = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    const archivePath = path.resolve(meta.relativePath);
    if (!fs.existsSync(archivePath)) {
      return { success: false, error: `Raw archive file missing at ${meta.relativePath}.` };
    }

    // 1. Verify license compatibility (no silent approval of incompatible licenses)
    if (!this.pipeline.getProvenanceRegistry().isLicenseCompatible(meta.license)) {
      meta.approvalStatus = 'rejected';
      meta.statusStage = 'rejected';
      meta.rejectionReason = `Cannot approve '${archiveId}': license '${meta.license}' is incompatible or unknown.`;
      fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), 'utf8');
      this.pipeline.getProvenanceRegistry().register({
        datasetName: meta.datasetName,
        source: meta.relativePath,
        officialSourceUrl: meta.officialSourceUrl,
        publisher: meta.publisher,
        license: meta.license,
        attributionRequired: true,
        approvalStatus: 'rejected',
        version: 'rejected',
        downloadDate: meta.downloadedAt,
        language: meta.language,
        domain: meta.defaultDomain,
        documentCount: 0,
        rawTokenCount: 0,
        cleanedTokenCount: 0,
        processingHistory: ['RejectedAtApprovalGate'],
      });
      this.pipeline.writeManifest();
      this.writeStatusReport();
      return { success: false, error: meta.rejectionReason };
    }

    // 2. Verify SHA-256 file integrity
    const currentHash = sha256File(archivePath);
    if (currentHash !== meta.sha256) {
      return {
        success: false,
        error: `SHA-256 integrity mismatch for '${archiveId}': expected ${meta.sha256.slice(0, 16)}, got ${currentHash.slice(0, 16)}.`,
      };
    }

    meta.approvalStatus = 'approved';
    meta.approvedBy = approvedBy;
    meta.approvedAt = new Date().toISOString();
    meta.statusStage = 'verified_approved';
    meta.rejectionReason = undefined;
    fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), 'utf8');
    this.writeStatusReport();

    return { success: true, metadata: meta };
  }

  public rejectArchive(archiveId: string, reason: string): { success: boolean; error?: string; metadata?: LocalArchiveMetadata } {
    const metaPath = this.findMetaPath(archiveId);
    if (!metaPath) {
      return { success: false, error: `Archive '${archiveId}' not found.` };
    }
    const meta: LocalArchiveMetadata = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    meta.approvalStatus = 'rejected';
    meta.statusStage = 'rejected';
    meta.rejectionReason = reason || 'Manually rejected during dataset review.';
    fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), 'utf8');
    this.pipeline.getProvenanceRegistry().register({
      datasetName: meta.datasetName,
      source: meta.relativePath,
      officialSourceUrl: meta.officialSourceUrl,
      publisher: meta.publisher,
      license: meta.license,
      attributionRequired: true,
      approvalStatus: 'rejected',
      version: 'rejected',
      downloadDate: meta.downloadedAt,
      language: meta.language,
      domain: meta.defaultDomain,
      documentCount: 0,
      rawTokenCount: 0,
      cleanedTokenCount: 0,
      processingHistory: ['ManuallyRejected'],
    });
    this.pipeline.writeManifest();
    this.writeStatusReport();
    return { success: true, metadata: meta };
  }

  public quarantineArchive(
    archiveId: string,
    reason: string
  ): { success: boolean; error?: string; metadata?: LocalArchiveMetadata } {
    const metaPath = this.findMetaPath(archiveId);
    if (!metaPath) {
      return { success: false, error: `Archive '${archiveId}' not found.` };
    }
    const meta: LocalArchiveMetadata = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    meta.approvalStatus = 'quarantined';
    meta.statusStage = 'quarantined';
    meta.quarantinedAt = new Date().toISOString();
    meta.quarantineReason = reason;
    fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), 'utf8');
    this.pipeline.getProvenanceRegistry().register({
      datasetName: meta.datasetName,
      source: meta.relativePath,
      officialSourceUrl: meta.officialSourceUrl,
      publisher: meta.publisher,
      license: meta.license,
      attributionRequired: true,
      approvalStatus: 'quarantined',
      quarantineReason: reason,
      version: this.pipeline.expandedVersion,
      downloadDate: meta.downloadedAt,
      language: meta.language,
      domain: meta.defaultDomain,
      documentCount: 0,
      rawTokenCount: 0,
      cleanedTokenCount: 0,
      processingHistory: ['QuarantinedAtLicenseAudit'],
    });
    this.writeStatusReport();
    return { success: true, metadata: meta };
  }

  private async *streamRecordsFromArchive(archivePath: string, meta: LocalArchiveMetadata): AsyncGenerator<{
    text: string;
    domain?: DatasetDomain;
    language?: string;
  }> {
    const lower = archivePath.toLowerCase();

    if (lower.endsWith('.jsonl') || lower.endsWith('.jsonl.gz')) {
      const fileStream = fs.createReadStream(archivePath, { highWaterMark: 64 * 1024 });
      const stream: NodeJS.ReadableStream = lower.endsWith('.gz') ? fileStream.pipe(zlib.createGunzip()) : fileStream;
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      try {
        for await (const line of rl) {
          const t = line.trim();
          if (!t) continue;
          try {
            const rec = JSON.parse(t);
            const text = rec.text || rec.content || '';
            if (text) {
              yield {
                text,
                domain: rec.domain ? mapMmluOrArxivCategoryToDomain(rec.domain, meta.defaultDomain) : meta.defaultDomain,
                language: rec.language || meta.language,
              };
            }
          } catch {
            yield { text: '\u0000corrupt_jsonl' };
          }
        }
      } finally {
        rl.close();
        fileStream.destroy();
      }
      return;
    }

    if (lower.endsWith('.tsv.bz2') || lower.endsWith('.tsv') || lower.endsWith('.tar.gz') || lower.endsWith('.tgz') || lower.endsWith('.zip') || lower.endsWith('.xml')) {
      // Stream decoded JSON lines from a small Python stdlib extractor without loading archive into memory
      const pyScript = `
import sys, json, os, bz2, tarfile, zipfile, xml.etree.ElementTree as ET

path = sys.argv[1]
lower = path.lower()
default_lang = sys.argv[2]

def emit(text, domain_hint=""):
    sys.stdout.write(json.dumps({"text": text, "domain_hint": domain_hint}, ensure_ascii=False) + "\\n")

if lower.endswith(".tsv.bz2") or lower.endswith(".tsv"):
    opener = bz2.open if lower.endswith(".bz2") else open
    buf = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\\n").split("\\t")
            sent = parts[-1].strip() if parts else ""
            if len(sent) >= 12:
                buf.append(sent)
                if len(buf) >= 5:
                    emit(" ".join(buf))
                    buf = []
        if buf:
            emit(" ".join(buf))
elif lower.endswith(".xml"):
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    for entry in entries:
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        cats = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns)]
        hint = " ".join(cats)
        if title and summary:
            emit(f"Title: {title}\\nCategory: {', '.join(cats)}\\nAbstract: {summary}", hint)
elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
    with tarfile.open(path, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or member.size < 80 or member.size > 200000:
                continue
            name = member.name.lower()
            if not name.endswith((".md", ".py", ".rs", ".ts", ".txt")):
                continue
            if any(skip in name for skip in ("test", "__init__", "license", "changelog")):
                continue
            f = tf.extractfile(member)
            if not f:
                continue
            raw = f.read().decode("utf-8", errors="replace")
            emit(raw[:8000], name)
elif lower.endswith(".zip"):
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir() or info.file_size < 80 or info.file_size > 200000:
                continue
            name = info.filename.lower()
            if not name.endswith((".md", ".py", ".rs", ".ts", ".txt")):
                continue
            raw = zf.read(info).decode("utf-8", errors="replace")
            emit(raw[:8000], name)
`;
      const pyBin = resolvePythonBin();
      const proc = spawn(pyBin, ['-c', pyScript, archivePath, meta.language], {
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      const rl = readline.createInterface({ input: proc.stdout, crlfDelay: Infinity });
      try {
        for await (const line of rl) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const item = JSON.parse(trimmed);
            yield {
              text: item.text,
              domain: mapMmluOrArxivCategoryToDomain(item.domain_hint || '', meta.defaultDomain),
              language: meta.language,
            };
          } catch {
            // ignore malformed line
          }
        }
      } finally {
        rl.close();
        if (!proc.killed) proc.kill();
      }
      return;
    }

    // Fallback: TXT / Markdown / CSV
    const fileStream = fs.createReadStream(archivePath, { highWaterMark: 64 * 1024 });
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });
    let buf: string[] = [];
    try {
      for await (const line of rl) {
        const trimmed = line.trim();
        if (!trimmed) {
          if (buf.length > 0) {
            yield { text: buf.join('\n'), domain: meta.defaultDomain, language: meta.language };
            buf = [];
          }
        } else {
          buf.push(line);
        }
      }
      if (buf.length > 0) {
        yield { text: buf.join('\n'), domain: meta.defaultDomain, language: meta.language };
      }
    } finally {
      rl.close();
      fileStream.destroy();
    }
  }

  public async ingestApprovedArchive(
    archiveId: string,
    options: IngestProgressOptions = {}
  ): Promise<{
    success: boolean;
    error?: string;
    archiveId: string;
    resumedFromRecord: number;
    recordsProcessedThisPass: number;
    documentsIngestedThisPass: number;
    documentsRejectedThisPass: number;
    rawTokensThisPass: number;
    cleanedTokensThisPass: number;
    completed: boolean;
    checkpoint: IngestionCheckpoint;
  }> {
    const metaPath = this.findMetaPath(archiveId);
    if (!metaPath) {
      throw new Error(`Archive '${archiveId}' not found in datasets/raw/.`);
    }
    const meta: LocalArchiveMetadata = JSON.parse(fs.readFileSync(metaPath, 'utf8'));

    // Enforce explicit user approval before ingestion
    if (meta.approvalStatus !== 'approved') {
      return {
        success: false,
        error: `Archive '${archiveId}' status is '${meta.approvalStatus}'. Explicit approval is required before ingestion.`,
        archiveId,
        resumedFromRecord: 0,
        recordsProcessedThisPass: 0,
        documentsIngestedThisPass: 0,
        documentsRejectedThisPass: 0,
        rawTokensThisPass: 0,
        cleanedTokensThisPass: 0,
        completed: false,
        checkpoint: {
          archiveId,
          sha256: meta.sha256,
          processedRecords: 0,
          ingestedDocuments: 0,
          rejectedDocuments: 0,
          rawTokens: 0,
          cleanedTokens: 0,
          completed: false,
          lastUpdatedAt: new Date().toISOString(),
        },
      };
    }

    const archivePath = path.resolve(meta.relativePath);
    if (!fs.existsSync(archivePath)) {
      throw new Error(`Raw archive file missing at ${archivePath}`);
    }

    // Verify integrity before ingesting
    const currentHash = sha256File(archivePath);
    if (currentHash !== meta.sha256) {
      return {
        success: false,
        error: `SHA-256 verification failed for '${archiveId}': archive file modified or corrupted.`,
        archiveId,
        resumedFromRecord: 0,
        recordsProcessedThisPass: 0,
        documentsIngestedThisPass: 0,
        documentsRejectedThisPass: 0,
        rawTokensThisPass: 0,
        cleanedTokensThisPass: 0,
        completed: false,
        checkpoint: {
          archiveId,
          sha256: currentHash,
          processedRecords: 0,
          ingestedDocuments: 0,
          rejectedDocuments: 0,
          rawTokens: 0,
          cleanedTokens: 0,
          completed: false,
          lastUpdatedAt: new Date().toISOString(),
        },
      };
    }

    const checkpoints = this.loadCheckpoints();
    const existingCp = checkpoints[archiveId];
    if (existingCp && existingCp.completed && existingCp.sha256 === meta.sha256) {
      return {
        success: false,
        error: `Duplicate import prevented: archive '${archiveId}' (SHA-256 ${meta.sha256.slice(0, 12)}) was already completely ingested.`,
        archiveId,
        resumedFromRecord: existingCp.processedRecords,
        recordsProcessedThisPass: 0,
        documentsIngestedThisPass: 0,
        documentsRejectedThisPass: 0,
        rawTokensThisPass: 0,
        cleanedTokensThisPass: 0,
        completed: true,
        checkpoint: existingCp,
      };
    }

    const startCursor = existingCp && existingCp.sha256 === meta.sha256 ? existingCp.processedRecords : 0;
    const maxThisPass = options.maxRecordsThisPass ?? Number.MAX_SAFE_INTEGER;
    const maxDocsTotal = options.maxDocumentsTotal ?? Number.MAX_SAFE_INTEGER;

    let recordIndex = 0;
    let processedThisPass = 0;
    let ingestedThisPass = 0;
    let rejectedThisPass = 0;
    let rawTokensThisPass = 0;
    let cleanedTokensThisPass = 0;
    let hitCap = false;

    const batch: Array<{
      doc: DatasetDocument;
      source: string;
      officialSourceUrl?: string;
      publisher?: string;
      attributionText?: string;
      archiveSha256?: string;
    }> = [];

    for await (const rec of this.streamRecordsFromArchive(archivePath, meta)) {
      if (recordIndex < startCursor) {
        recordIndex++;
        continue;
      }
      if (processedThisPass >= maxThisPass || (existingCp?.ingestedDocuments || 0) + ingestedThisPass >= maxDocsTotal) {
        hitCap = true;
        break;
      }

      const res = this.pipeline.ingestDocument(
        rec.domain || meta.defaultDomain,
        rec.text,
        rec.language === 'Multi' ? undefined : rec.language || (meta.language === 'Multi' ? undefined : meta.language),
        meta.datasetName,
        meta.license,
        false
      );

      recordIndex++;
      processedThisPass++;
      if (res.accepted && res.doc) {
        ingestedThisPass++;
        rawTokensThisPass += res.doc.rawTokenCount;
        cleanedTokensThisPass += res.doc.cleanedTokenCount;
        batch.push({
          doc: res.doc,
          source: meta.relativePath,
          officialSourceUrl: meta.officialSourceUrl,
          publisher: meta.publisher,
          attributionText: meta.attributionRequirements,
          archiveSha256: meta.sha256,
        });
      } else {
        rejectedThisPass++;
      }
    }

    this.pipeline.appendExpandedDocumentsToDisk(batch);

    const updatedCp: IngestionCheckpoint = {
      archiveId,
      sha256: meta.sha256,
      processedRecords: recordIndex,
      ingestedDocuments: (existingCp?.ingestedDocuments || 0) + ingestedThisPass,
      rejectedDocuments: (existingCp?.rejectedDocuments || 0) + rejectedThisPass,
      rawTokens: (existingCp?.rawTokens || 0) + rawTokensThisPass,
      cleanedTokens: (existingCp?.cleanedTokens || 0) + cleanedTokensThisPass,
      completed: !hitCap || (existingCp?.ingestedDocuments || 0) + ingestedThisPass >= maxDocsTotal,
      lastUpdatedAt: new Date().toISOString(),
    };

    checkpoints[archiveId] = updatedCp;
    this.saveCheckpoints(checkpoints);

    if (ingestedThisPass > 0 || updatedCp.completed) {
      this.pipeline.getProvenanceRegistry().register({
        datasetName: meta.datasetName,
        source: meta.relativePath,
        officialSourceUrl: meta.officialSourceUrl,
        publisher: meta.publisher,
        attributionText: meta.attributionRequirements,
        archiveSha256: meta.sha256,
        fileSizeBytes: meta.fileSizeBytes,
        license: meta.license,
        attributionRequired: this.pipeline.getProvenanceRegistry().requiresAttribution(meta.license),
        approvalStatus: 'approved',
        approvedBy: meta.approvedBy,
        approvedAt: meta.approvedAt,
        version: this.pipeline.expandedVersion,
        downloadDate: meta.downloadedAt,
        language: meta.language,
        domain: meta.defaultDomain,
        documentCount: ingestedThisPass,
        rawTokenCount: rawTokensThisPass,
        cleanedTokenCount: cleanedTokensThisPass,
        processingHistory: [
          'IntegrityVerifiedSHA256',
          'ExplicitUserApproval',
          'ResumableStreamed',
          'Normalized',
          'Deduplicated',
          'Tokenized',
        ],
      });
    }

    if (updatedCp.completed) {
      meta.statusStage = 'ingested';
      fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), 'utf8');
    }

    this.pipeline.finalizeCorpusUpdate();
    this.writeStatusReport();

    return {
      success: true,
      archiveId,
      resumedFromRecord: startCursor,
      recordsProcessedThisPass: processedThisPass,
      documentsIngestedThisPass: ingestedThisPass,
      documentsRejectedThisPass: rejectedThisPass,
      rawTokensThisPass,
      cleanedTokensThisPass,
      completed: updatedCp.completed,
      checkpoint: updatedCp,
    };
  }

  public getResearchCatalogStatus() {
    const archives = this.listArchivesWithProgress();
    const archiveById = new Map(archives.map((a) => [a.archiveId, a]));

    const entries = OFFICIAL_DATASET_CATALOG.map((cat: CatalogDatasetEntry) => {
      const local = archiveById.get(cat.id);
      let lifecycleStatus: 'proposed' | 'downloaded' | 'verified_approved' | 'ingested' | 'rejected' | 'quarantined' = 'proposed';
      if (!cat.licenseCompatible) {
        lifecycleStatus = 'rejected';
      } else if (local) {
        lifecycleStatus = local.statusStage;
      }
      return {
        ...cat,
        lifecycleStatus,
        quarantineReason: local?.quarantineReason,
        localFile: local ? local.relativePath : null,
        localSizeBytes: local ? local.fileSizeBytes : 0,
        localSha256: local ? local.sha256 : null,
        ingestedDocuments: local ? local.progress.ingestedDocuments : 0,
        cleanedTokens: local ? local.progress.cleanedTokens : 0,
      };
    });

    const summary = {
      proposed: entries.filter((e) => e.lifecycleStatus === 'proposed').length,
      downloaded: entries.filter((e) => e.lifecycleStatus === 'downloaded').length,
      verified_approved: entries.filter((e) => e.lifecycleStatus === 'verified_approved').length,
      ingested: entries.filter((e) => e.lifecycleStatus === 'ingested').length,
      rejected: entries.filter((e) => e.lifecycleStatus === 'rejected').length,
      quarantined: archives.filter((a) => a.statusStage === 'quarantined').length,
    };

    return { entries, summary };
  }

  public writeStatusReport(): void {
    const status = {
      generatedAt: new Date().toISOString(),
      archives: this.listArchivesWithProgress(),
      catalogSummary: this.getResearchCatalogStatus().summary,
    };
    fs.writeFileSync(this.statusReportPath, JSON.stringify(status, null, 2), 'utf8');
  }
}

export const globalAcquisitionManager = new LocalDatasetAcquisitionManager();
