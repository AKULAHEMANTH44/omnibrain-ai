export type LicenseType =
  | 'MIT'
  | 'Apache-2.0'
  | 'BSD-3-Clause'
  | 'CC-BY-4.0'
  | 'CC-BY-2.0-FR'
  | 'CC-BY-SA-3.0'
  | 'CC-BY-SA-4.0'
  | 'ODC-By-1.0'
  | 'CC0-1.0'
  | 'Public Domain'
  | 'CC-BY-NC-3.0'
  | 'CC-BY-NC-SA-4.0'
  | 'Unknown'
  | 'Restricted'
  | 'Proprietary';

export interface DatasetProvenance {
  datasetName: string;
  source: string;
  officialSourceUrl?: string;
  publisher?: string;
  license: string;
  compatible: boolean;
  attributionRequired: boolean;
  attributionText?: string;
  archiveSha256?: string;
  fileSizeBytes?: number;
  approvalStatus?: 'pending_approval' | 'approved' | 'rejected' | 'quarantined';
  approvedBy?: string;
  approvedAt?: string;
  quarantineReason?: string;
  redistributionTerms?: string;
  commercialUsePermitted?: boolean;
  licenseEvidencePath?: string;
  version: string;
  downloadDate: string;
  language: string;
  domain: string;
  documentCount: number;
  rawTokenCount: number;
  cleanedTokenCount: number;
  processingHistory: string[];
  rejectionReason?: string;
}

const ALLOWED_TRAINING_LICENSES = new Set([
  'MIT',
  'Apache-2.0',
  'BSD-3-Clause',
  'CC-BY-4.0',
  'CC-BY-2.0-FR',
  'CC-BY-SA-3.0',
  'CC-BY-SA-4.0',
  'ODC-By-1.0',
  'CC0-1.0',
  'Public Domain',
]);

const ATTRIBUTION_REQUIRED_LICENSES = new Set([
  'MIT',
  'Apache-2.0',
  'BSD-3-Clause',
  'CC-BY-4.0',
  'CC-BY-2.0-FR',
  'CC-BY-SA-3.0',
  'CC-BY-SA-4.0',
  'ODC-By-1.0',
]);

export class ProvenanceRegistry {
  private registry: Map<string, DatasetProvenance> = new Map();
  private quarantined: Map<string, DatasetProvenance> = new Map();
  private rejected: Map<string, DatasetProvenance> = new Map();
  private importedArchiveHashes: Map<string, string> = new Map();

  isLicenseCompatible(license: string): boolean {
    if (!license || typeof license !== 'string') return false;
    const norm = license.trim();
    if (!norm || /nc|non-?commercial|unknown|restricted|proprietary/i.test(norm)) {
      return false;
    }
    return ALLOWED_TRAINING_LICENSES.has(norm);
  }

  requiresAttribution(license: string): boolean {
    return ATTRIBUTION_REQUIRED_LICENSES.has((license || '').trim());
  }

  hasImportedArchiveHash(sha256: string): boolean {
    return Boolean(sha256 && this.importedArchiveHashes.has(sha256));
  }

  getImportedDatasetByHash(sha256: string): string | undefined {
    return this.importedArchiveHashes.get(sha256);
  }

  markArchiveHashImported(sha256: string, datasetName: string): void {
    if (sha256) {
      this.importedArchiveHashes.set(sha256, datasetName);
    }
  }

  register(provenance: Omit<DatasetProvenance, 'compatible'>): { accepted: boolean; record: DatasetProvenance } {
    const compatible = this.isLicenseCompatible(provenance.license);
    const isQuarantined = provenance.approvalStatus === 'quarantined';
    const record: DatasetProvenance = {
      ...provenance,
      compatible: compatible && !isQuarantined,
      attributionRequired: provenance.attributionRequired ?? this.requiresAttribution(provenance.license),
      rejectionReason: compatible
        ? undefined
        : `License '${provenance.license}' is unknown, non-commercial, or incompatible for foundation model training until reviewed.`,
    };

    if (isQuarantined) {
      const existing = this.quarantined.get(provenance.datasetName);
      if (existing) {
        this.quarantined.set(provenance.datasetName, {
          ...existing,
          ...record,
          documentCount: existing.documentCount + record.documentCount,
          rawTokenCount: existing.rawTokenCount + record.rawTokenCount,
          cleanedTokenCount: existing.cleanedTokenCount + record.cleanedTokenCount,
          processingHistory: Array.from(new Set([...existing.processingHistory, ...record.processingHistory])),
        });
      } else {
        this.quarantined.set(provenance.datasetName, record);
      }
      this.registry.delete(provenance.datasetName);
      if (record.archiveSha256) {
        this.importedArchiveHashes.set(record.archiveSha256, record.datasetName);
      }
      return { accepted: false, record };
    }

    if (compatible) {
      const existing = this.registry.get(provenance.datasetName);
      if (existing) {
        const mergedDocs = existing.documentCount + record.documentCount;
        const mergedRaw = existing.rawTokenCount + record.rawTokenCount;
        const mergedClean = existing.cleanedTokenCount + record.cleanedTokenCount;
        const mergedHistory = Array.from(new Set([...existing.processingHistory, ...record.processingHistory]));
        this.registry.set(provenance.datasetName, {
          ...existing,
          ...record,
          documentCount: mergedDocs,
          rawTokenCount: mergedRaw,
          cleanedTokenCount: mergedClean,
          processingHistory: mergedHistory,
        });
      } else {
        this.registry.set(provenance.datasetName, record);
      }
      if (record.archiveSha256) {
        this.importedArchiveHashes.set(record.archiveSha256, record.datasetName);
      }
    } else {
      this.rejected.set(provenance.datasetName, { ...record, approvalStatus: 'rejected' });
    }
    return { accepted: compatible, record };
  }

  get(datasetName: string): DatasetProvenance | undefined {
    return this.registry.get(datasetName) || this.quarantined.get(datasetName);
  }

  getAll(): DatasetProvenance[] {
    return Array.from(this.registry.values());
  }

  getQuarantined(): DatasetProvenance[] {
    return Array.from(this.quarantined.values());
  }

  getRejected(): DatasetProvenance[] {
    return Array.from(this.rejected.values());
  }

  getLicenseDistribution(): Record<string, number> {
    const dist: Record<string, number> = {};
    for (const rec of this.registry.values()) {
      dist[rec.license] = (dist[rec.license] || 0) + rec.documentCount;
    }
    return dist;
  }

  clear() {
    this.registry.clear();
    this.quarantined.clear();
    this.rejected.clear();
    this.importedArchiveHashes.clear();
  }
}

export const globalProvenanceRegistry = new ProvenanceRegistry();
