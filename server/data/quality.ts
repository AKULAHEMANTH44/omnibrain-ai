import crypto from 'crypto';

export interface QualityFilterResult {
  passed: boolean;
  reason?: string;
  rejectionCategory?:
    | 'empty_or_invalid'
    | 'corrupted_encoding'
    | 'too_short'
    | 'too_long'
    | 'exact_duplicate'
    | 'near_duplicate'
    | 'excessive_repetition'
    | 'low_information';
  cleanedText: string;
  exactHash: string;
  simHash: string;
  detectedLanguage: string;
  qualityScore: number;
}

export interface QualityBreakdown {
  accepted: number;
  emptyOrInvalid: number;
  corruptedEncoding: number;
  tooShort: number;
  tooLong: number;
  exactDuplicate: number;
  nearDuplicate: number;
  excessiveRepetition: number;
  lowInformation: number;
}

export class DataQualitySystem {
  private seenHashes: Set<string> = new Set();
  private shingleStore: Array<{ hash: string; shingles: Set<string> }> = [];
  private invertedIndex: Map<string, number[]> = new Map();
  private breakdown: QualityBreakdown = {
    accepted: 0,
    emptyOrInvalid: 0,
    corruptedEncoding: 0,
    tooShort: 0,
    tooLong: 0,
    exactDuplicate: 0,
    nearDuplicate: 0,
    excessiveRepetition: 0,
    lowInformation: 0,
  };

  computeSha256(text: string): string {
    return crypto.createHash('sha256').update(text, 'utf8').digest('hex');
  }

  private extractShingles(text: string): Set<string> {
    const normalized = text.toLowerCase().replace(/\s+/g, ' ').trim();
    const shingles = new Set<string>();
    const words = normalized.split(' ');
    if (words.length >= 3) {
      const cap = Math.min(words.length - 3, 120);
      for (let i = 0; i <= cap; i++) {
        shingles.add(`${words[i]} ${words[i + 1]} ${words[i + 2]}`);
      }
    } else {
      const cap = Math.min(normalized.length - 4, 120);
      for (let i = 0; i <= cap; i++) {
        shingles.add(normalized.slice(i, i + 4));
      }
    }
    return shingles;
  }

  private jaccardSimilarity(a: Set<string>, b: Set<string>): number {
    if (a.size === 0 || b.size === 0) return 0;
    let intersection = 0;
    const [smaller, larger] = a.size <= b.size ? [a, b] : [b, a];
    for (const item of smaller) {
      if (larger.has(item)) intersection++;
    }
    const union = a.size + b.size - intersection;
    return union === 0 ? 0 : intersection / union;
  }

  detectLanguage(text: string): string {
    let devanagari = 0;
    let tamil = 0;
    let telugu = 0;
    let kannada = 0;
    let bengali = 0;
    let malayalam = 0;
    let gujarati = 0;
    let latin = 0;

    const sample = text.length > 2000 ? text.slice(0, 2000) : text;
    for (let i = 0; i < sample.length; i++) {
      const c = sample.charCodeAt(i);
      if (c >= 0x0900 && c <= 0x097f) devanagari++;
      else if (c >= 0x0b80 && c <= 0x0bff) tamil++;
      else if (c >= 0x0c00 && c <= 0x0c7f) telugu++;
      else if (c >= 0x0c80 && c <= 0x0cff) kannada++;
      else if (c >= 0x0980 && c <= 0x09ff) bengali++;
      else if (c >= 0x0d00 && c <= 0x0d7f) malayalam++;
      else if (c >= 0x0a80 && c <= 0x0aff) gujarati++;
      else if ((c >= 65 && c <= 90) || (c >= 97 && c <= 122)) latin++;
    }

    const indicCounts: Array<[string, number]> = [
      ['Hindi', devanagari],
      ['Tamil', tamil],
      ['Telugu', telugu],
      ['Kannada', kannada],
      ['Bengali', bengali],
      ['Malayalam', malayalam],
      ['Gujarati', gujarati],
    ];
    indicCounts.sort((a, b) => b[1] - a[1]);
    if (indicCounts[0][1] >= 4 && indicCounts[0][1] >= latin * 0.25) {
      return indicCounts[0][0];
    }

    const codeMatches = sample.match(/\b(def |fn |func |function |export |import |return |class |const |let |pub struct |impl |async |await )\b/g);
    const symbolDensity = (sample.match(/[{}();=><]/g) || []).length / Math.max(1, sample.length);
    if ((codeMatches && codeMatches.length >= 2) || (codeMatches && codeMatches.length >= 1 && symbolDensity > 0.02)) {
      return 'Code';
    }
    return 'English';
  }

  private isCorrupted(rawText: string): boolean {
    if (rawText.includes('\u0000')) return true;
    const replacementMatches = rawText.match(/\uFFFD/g);
    if (replacementMatches && replacementMatches.length / rawText.length > 0.03) return true;

    let controlChars = 0;
    const limit = Math.min(rawText.length, 4000);
    for (let i = 0; i < limit; i++) {
      const code = rawText.charCodeAt(i);
      if (code < 32 && code !== 9 && code !== 10 && code !== 13) {
        controlChars++;
      }
    }
    return controlChars / Math.max(1, limit) > 0.05;
  }

  cleanAndFilter(rawText: string, nearDuplicateThreshold = 0.75): QualityFilterResult {
    if (!rawText || typeof rawText !== 'string') {
      this.breakdown.emptyOrInvalid++;
      return {
        passed: false,
        reason: 'Invalid or empty content',
        rejectionCategory: 'empty_or_invalid',
        cleanedText: '',
        exactHash: '',
        simHash: '',
        detectedLanguage: 'Unknown',
        qualityScore: 0,
      };
    }

    if (this.isCorrupted(rawText)) {
      this.breakdown.corruptedEncoding++;
      return {
        passed: false,
        reason: 'Corrupted text or invalid encoding detected',
        rejectionCategory: 'corrupted_encoding',
        cleanedText: '',
        exactHash: '',
        simHash: '',
        detectedLanguage: 'Unknown',
        qualityScore: 0,
      };
    }

    const cleaned = rawText
      .replace(/^\uFEFF/, '')
      .replace(/\r\n/g, '\n')
      .replace(/[ \t]+/g, ' ')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    if (cleaned.length < 10) {
      this.breakdown.tooShort++;
      return {
        passed: false,
        reason: 'Document too short (< 10 chars)',
        rejectionCategory: 'too_short',
        cleanedText: '',
        exactHash: '',
        simHash: '',
        detectedLanguage: 'Unknown',
        qualityScore: 0,
      };
    }

    if (cleaned.length > 200000) {
      this.breakdown.tooLong++;
      return {
        passed: false,
        reason: 'Document exceeds maximum single-record limit (200k chars)',
        rejectionCategory: 'too_long',
        cleanedText: '',
        exactHash: '',
        simHash: '',
        detectedLanguage: 'Unknown',
        qualityScore: 0,
      };
    }

    const detectedLanguage = this.detectLanguage(cleaned);
    const exactHash = this.computeSha256(cleaned.toLowerCase());
    if (this.seenHashes.has(exactHash)) {
      this.breakdown.exactDuplicate++;
      return {
        passed: false,
        reason: 'Exact duplicate document',
        rejectionCategory: 'exact_duplicate',
        cleanedText: '',
        exactHash,
        simHash: exactHash.slice(0, 16),
        detectedLanguage,
        qualityScore: 0,
      };
    }

    // Fast indexed near-duplicate lookup via shared shingle posting lists
    const shingles = this.extractShingles(cleaned);
    const candidateHits = new Map<number, number>();
    let checked = 0;
    for (const s of shingles) {
      if (checked++ > 48) break;
      const docIds = this.invertedIndex.get(s);
      if (!docIds) continue;
      for (const docIdx of docIds) {
        candidateHits.set(docIdx, (candidateHits.get(docIdx) || 0) + 1);
      }
    }

    const minHits = Math.max(1, Math.min(3, Math.floor(shingles.size * 0.25)));
    for (const [docIdx, hits] of candidateHits.entries()) {
      if (hits < minHits) continue;
      const prev = this.shingleStore[docIdx];
      const sim = this.jaccardSimilarity(shingles, prev.shingles);
      if (sim >= nearDuplicateThreshold) {
        this.breakdown.nearDuplicate++;
        return {
          passed: false,
          reason: `Near-duplicate document (similarity ${sim.toFixed(2)})`,
          rejectionCategory: 'near_duplicate',
          cleanedText: '',
          exactHash,
          simHash: prev.hash,
          detectedLanguage,
          qualityScore: 0,
        };
      }
    }

    // Excessive word repetition check
    const words = cleaned.split(/\s+/);
    if (words.length > 8) {
      let consecutiveRepeats = 0;
      for (let i = 1; i < words.length; i++) {
        if (words[i] === words[i - 1]) consecutiveRepeats++;
      }
      if (consecutiveRepeats / words.length > 0.4) {
        this.breakdown.excessiveRepetition++;
        return {
          passed: false,
          reason: 'Excessive repetition detected',
          rejectionCategory: 'excessive_repetition',
          cleanedText: '',
          exactHash,
          simHash: exactHash.slice(0, 16),
          detectedLanguage,
          qualityScore: 0,
        };
      }
    }

    // Low-information check (unique character ratio & alpha/script ratio)
    const uniqueChars = new Set(cleaned).size;
    if (cleaned.length > 30 && uniqueChars < 5) {
      this.breakdown.lowInformation++;
      return {
        passed: false,
        reason: 'Low-information content',
        rejectionCategory: 'low_information',
        cleanedText: '',
        exactHash,
        simHash: exactHash.slice(0, 16),
        detectedLanguage,
        qualityScore: 0,
      };
    }

    const uniqueWordsRatio = words.length > 0 ? new Set(words).size / words.length : 1;
    const qualityScore = Number(Math.min(1.0, 0.5 + uniqueWordsRatio * 0.5).toFixed(3));

    this.seenHashes.add(exactHash);
    const simHash = exactHash.slice(0, 16);
    if (this.shingleStore.length < 30000) {
      const storeIdx = this.shingleStore.length;
      this.shingleStore.push({ hash: simHash, shingles });
      let indexedCount = 0;
      for (const s of shingles) {
        if (indexedCount++ >= 24) break;
        const list = this.invertedIndex.get(s);
        if (list) {
          if (list.length < 64) list.push(storeIdx);
        } else {
          this.invertedIndex.set(s, [storeIdx]);
        }
      }
    }

    this.breakdown.accepted++;
    return {
      passed: true,
      cleanedText: cleaned,
      exactHash,
      simHash,
      detectedLanguage,
      qualityScore,
    };
  }

  getBreakdown(): QualityBreakdown {
    return { ...this.breakdown };
  }

  reset() {
    this.seenHashes.clear();
    this.shingleStore = [];
    this.invertedIndex.clear();
    this.breakdown = {
      accepted: 0,
      emptyOrInvalid: 0,
      corruptedEncoding: 0,
      tooShort: 0,
      tooLong: 0,
      exactDuplicate: 0,
      nearDuplicate: 0,
      excessiveRepetition: 0,
      lowInformation: 0,
    };
  }
}

export const globalQualitySystem = new DataQualitySystem();
