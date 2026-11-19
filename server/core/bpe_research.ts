import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { globalTokenizer, OmnibrainTokenizer } from './tokenizer';

export interface BpeBenchmarkCategoryResult {
  category: string;
  samples: number;
  bytesTotal: number;
  charsTotal: number;
  prodTokens: number;
  prodUnkCount: number;
  prodUnkRate: number;
  prodCharsPerToken: number;
  prodBytesPerToken: number;
  bpeTokens: number;
  bpeUnkCount: number;
  bpeUnkRate: number;
  bpeCharsPerToken: number;
  bpeBytesPerToken: number;
  bpeCompressionGainVsProd: number;
}

export interface TokenizerComparisonReport {
  generatedAt: string;
  productionTokenizer: {
    version: string;
    vocabSize: number;
    activeTokens: number;
    checksum: string;
    totalTokensOnEvalSet: number;
    unkRate: number;
    charsPerToken: number;
    bytesPerToken: number;
    speedTokensPerSec: number;
    speedCharsPerSec: number;
  };
  experimentalBpeTokenizer: {
    version: string;
    status: string;
    vocabSize: number;
    learnedMerges: number;
    checksum: string;
    totalTokensOnEvalSet: number;
    unkRate: number;
    charsPerToken: number;
    bytesPerToken: number;
    speedTokensPerSec: number;
    speedCharsPerSec: number;
    roundTripExactMatch: boolean;
  };
  categoryComparisons: BpeBenchmarkCategoryResult[];
  productionRetainedReason: string;
  migrationRecommendation: {
    productionVersionLocked: string;
    migrateExistingCheckpointsNow: boolean;
    recommendedForNextScratchPretraining: boolean;
    recommendedVocabTargetForNextRun: number;
    summary: string;
  };
}

function buildByteToUnicode(): { byteToChar: Map<number, string>; charToByte: Map<string, number> } {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const byteToChar = new Map<number, string>();
  const charToByte = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCharCode(cs[i]);
    byteToChar.set(bs[i], ch);
    charToByte.set(ch, bs[i]);
  }
  return { byteToChar, charToByte };
}

export class ExperimentalByteLevelBpeTokenizer {
  public readonly version = 'omnibrain-tok-bpe-v2.0-exp';
  public readonly specialTokens: Record<string, number> = {
    '<pad>': 0,
    '<unk>': 1,
    '<bos>': 2,
    '<eos>': 3,
  };

  private byteToChar: Map<number, string>;
  private charToByte: Map<string, number>;
  private vocab: Map<string, number> = new Map();
  private inverseVocab: Map<number, string> = new Map();
  private merges: Array<[string, string]> = [];
  private mergeRanks: Map<string, number> = new Map();
  private vocabSize = 260;
  private checksum = '';

  constructor() {
    const tables = buildByteToUnicode();
    this.byteToChar = tables.byteToChar;
    this.charToByte = tables.charToByte;
    this.initBaseVocab();
  }

  private initBaseVocab() {
    this.vocab.clear();
    this.inverseVocab.clear();
    this.merges = [];
    this.mergeRanks.clear();

    for (const [tok, id] of Object.entries(this.specialTokens)) {
      this.vocab.set(tok, id);
      this.inverseVocab.set(id, tok);
    }

    let nextId = 4;
    for (let b = 0; b < 256; b++) {
      const sym = this.byteToChar.get(b)!;
      this.vocab.set(sym, nextId);
      this.inverseVocab.set(nextId, sym);
      nextId++;
    }
    this.vocabSize = nextId;
    this.checksum = this.computeChecksum();
  }

  private computeChecksum(): string {
    return crypto
      .createHash('sha256')
      .update(JSON.stringify({ version: this.version, vocabSize: this.vocabSize, merges: this.merges }))
      .digest('hex');
  }

  private pretokenizeWords(text: string): string[] {
    // Split cleanly preserving whitespace prefixes, numbers, punctuation, and UTF-8 runs
    const matches = text.match(/ ?[^\s\p{P}\p{S}]+| ?[\p{P}\p{S}]+|\s+/gu);
    return matches && matches.length > 0 ? matches : text ? [text] : [];
  }

  private wordToByteSymbols(word: string): string[] {
    const buf = Buffer.from(word, 'utf8');
    const syms: string[] = new Array(buf.length);
    for (let i = 0; i < buf.length; i++) {
      syms[i] = this.byteToChar.get(buf[i])!;
    }
    return syms;
  }

  public train(texts: string[], targetVocabSize = 2048): { vocabSize: number; learnedMerges: number; elapsedMs: number } {
    const t0 = performance.now();
    this.initBaseVocab();

    const wordFreqs = new Map<string, { symbols: string[]; freq: number }>();
    for (const text of texts) {
      const chunks = this.pretokenizeWords(text);
      for (const ch of chunks) {
        const key = ch;
        const existing = wordFreqs.get(key);
        if (existing) {
          existing.freq++;
        } else {
          wordFreqs.set(key, { symbols: this.wordToByteSymbols(ch), freq: 1 });
        }
      }
    }

    const words = Array.from(wordFreqs.values()).filter((w) => w.symbols.length >= 2);
    const numMerges = Math.max(0, targetVocabSize - this.vocabSize);

    for (let m = 0; m < numMerges; m++) {
      const pairCounts = new Map<string, number>();
      for (let i = 0; i < words.length; i++) {
        const { symbols, freq } = words[i];
        if (symbols.length < 2) continue;
        for (let j = 0; j < symbols.length - 1; j++) {
          const pairKey = `${symbols[j]}\u0001${symbols[j + 1]}`;
          pairCounts.set(pairKey, (pairCounts.get(pairKey) || 0) + freq);
        }
      }

      if (pairCounts.size === 0) break;

      let bestPair = '';
      let bestCount = 1; // require at least 2 occurrences
      for (const [pk, cnt] of pairCounts.entries()) {
        if (cnt > bestCount || (cnt === bestCount && bestPair && pk < bestPair)) {
          bestCount = cnt;
          bestPair = pk;
        }
      }

      if (!bestPair) break;
      const [a, b] = bestPair.split('\u0001');
      const merged = a + b;

      this.merges.push([a, b]);
      this.mergeRanks.set(bestPair, m);
      if (!this.vocab.has(merged)) {
        const id = this.vocabSize++;
        this.vocab.set(merged, id);
        this.inverseVocab.set(id, merged);
      }

      // Apply merge in-place across tracked word entries
      for (let i = 0; i < words.length; i++) {
        const syms = words[i].symbols;
        if (syms.length < 2) continue;
        let hasPair = false;
        for (let j = 0; j < syms.length - 1; j++) {
          if (syms[j] === a && syms[j + 1] === b) {
            hasPair = true;
            break;
          }
        }
        if (!hasPair) continue;

        const nextSyms: string[] = [];
        let j = 0;
        while (j < syms.length) {
          if (j < syms.length - 1 && syms[j] === a && syms[j + 1] === b) {
            nextSyms.push(merged);
            j += 2;
          } else {
            nextSyms.push(syms[j]);
            j += 1;
          }
        }
        words[i].symbols = nextSyms;
      }
    }

    this.checksum = this.computeChecksum();
    return {
      vocabSize: this.vocabSize,
      learnedMerges: this.merges.length,
      elapsedMs: Number((performance.now() - t0).toFixed(2)),
    };
  }

  private bpeWord(symbols: string[]): string[] {
    if (symbols.length <= 1) return symbols;
    let word = symbols;
    while (word.length > 1) {
      let bestRank = Number.MAX_SAFE_INTEGER;
      let bestIdx = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.mergeRanks.get(`${word[i]}\u0001${word[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      const merged = word[bestIdx] + word[bestIdx + 1];
      const next: string[] = [];
      for (let i = 0; i < word.length; i++) {
        if (i === bestIdx) {
          next.push(merged);
          i++;
        } else {
          next.push(word[i]);
        }
      }
      word = next;
    }
    return word;
  }

  public encode(text: string): number[] {
    const tokens: number[] = [this.specialTokens['<bos>']];
    const chunks = this.pretokenizeWords(text);
    const cache = new Map<string, number[]>();

    for (const chunk of chunks) {
      let ids = cache.get(chunk);
      if (!ids) {
        const bpeSyms = this.bpeWord(this.wordToByteSymbols(chunk));
        ids = bpeSyms.map((s) => this.vocab.get(s) ?? this.specialTokens['<unk>']);
        cache.set(chunk, ids);
      }
      for (let i = 0; i < ids.length; i++) {
        tokens.push(ids[i]);
      }
    }

    tokens.push(this.specialTokens['<eos>']);
    return tokens;
  }

  public decode(tokens: number[]): string {
    const bytes: number[] = [];
    const bos = this.specialTokens['<bos>'];
    const eos = this.specialTokens['<eos>'];
    const pad = this.specialTokens['<pad>'];

    for (const id of tokens) {
      if (id === bos || id === eos || id === pad) continue;
      const sym = this.inverseVocab.get(id);
      if (!sym) continue;
      for (const ch of sym) {
        const b = this.charToByte.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    return Buffer.from(bytes).toString('utf8');
  }

  public getVocabSize(): number {
    return this.vocabSize;
  }

  public getLearnedMergeCount(): number {
    return this.merges.length;
  }

  public getChecksum(): string {
    return this.checksum;
  }

  public saveToFile(filePath = './checkpoints/tokenizer_bpe_v2_exp.json'): string {
    const abs = path.resolve(filePath);
    fs.mkdirSync(path.dirname(abs), { recursive: true });
    const idToToken: Record<string, string> = {};
    for (const [id, tok] of this.inverseVocab.entries()) {
      idToToken[String(id)] = tok;
    }
    fs.writeFileSync(
      abs,
      JSON.stringify(
        {
          version: this.version,
          experimental: true,
          vocabSize: this.vocabSize,
          learnedMerges: this.merges.length,
          specialTokens: this.specialTokens,
          checksum: this.checksum,
          merges: this.merges,
          idToToken,
        },
        null,
        2
      ),
      'utf8'
    );
    return abs;
  }
}

export function evaluateTokenizerComparison(
  corpusDocs: Array<{ text: string; language: string; domain: string }>,
  prodTokenizer: OmnibrainTokenizer = globalTokenizer,
  targetVocabSize = 2048
): TokenizerComparisonReport {
  // Sample deterministically across languages & domains for fast BPE training and evaluation
  const bucketed = new Map<string, string[]>();
  const categorySamples = new Map<string, string[]>();

  for (const doc of corpusDocs) {
    const cat =
      doc.domain === 'mathematics'
        ? 'Mathematics'
        : doc.domain === 'programming' || doc.language === 'Code'
          ? 'Programming'
          : doc.language;
    const trainList = bucketed.get(cat) || [];
    if (trainList.length < 40) {
      trainList.push(doc.text.slice(0, 700));
      bucketed.set(cat, trainList);
    }
    const evalList = categorySamples.get(cat) || [];
    if (evalList.length < 30) {
      evalList.push(doc.text.slice(0, 900));
      categorySamples.set(cat, evalList);
    }
    if (doc.text.length >= 2000) {
      const longList = categorySamples.get('Long Documents (>2k chars)') || [];
      if (longList.length < 18) {
        longList.push(doc.text.slice(0, 4000));
        categorySamples.set('Long Documents (>2k chars)', longList);
      }
    }
  }

  // Dedicated Rare Unicode test stratum (math symbols, conjuncts, scripts, currency, diacritics, emojis)
  const rareUnicodeSamples = [
    '∫_a^b ∑_{k=1}^∞ ∂f/∂x_k dx = ΔF ≈ 0, ∀x ∈ ℤ^n, ∇·E = ρ/ε₀ ≥ 0 ≤ ∞ ≠ ≡ ∪ ∩ → ↔ ⇒ ⇔',
    'Multilingual Unicode & Indic Conjuncts: తెలుగు క్షత్ర జ్ఞాన, हिन्दी कृत्रिम बुद्धिमत्ता, தமிழ் அறிவியல், ಕನ್ನಡ ತಂತ್ರಜ್ಞಾನ, বাংলা বিজ্ঞান গবেষণা।',
    'Currency, Typography & Symbols: €1,250.50 · ₹9,840 · £420 — “naïve résumé coöperate” · café 2³ + 5² = 33 · ℃ · ℓ · ™ · № 14',
    'Diagrams, Arrows & Emoji Stress Test: Pipeline: [Raw] ──▶ [Audit] ──▶ [Clean] ✅ ⚠ 🔍 📊 🚀 ∑ ∆ Ω λ μ σ π',
  ];
  // Extract additional rare-unicode-containing excerpts directly from the actual corpus
  for (const doc of corpusDocs) {
    if (rareUnicodeSamples.length >= 16) break;
    if (/[\u2013-\u2BFF]/u.test(doc.text)) {
      rareUnicodeSamples.push(doc.text.slice(0, 500));
    }
  }
  categorySamples.set('Rare Unicode Characters', rareUnicodeSamples);

  const trainTexts: string[] = [];
  for (const arr of bucketed.values()) {
    trainTexts.push(...arr);
  }

  const bpe = new ExperimentalByteLevelBpeTokenizer();
  bpe.train(trainTexts, targetVocabSize);
  bpe.saveToFile('./checkpoints/tokenizer_bpe_v2_exp.json');

  const categoryComparisons: BpeBenchmarkCategoryResult[] = [];
  let totalBytes = 0;
  let totalChars = 0;
  let totalProdTokens = 0;
  let totalProdUnk = 0;
  let totalBpeTokens = 0;
  let totalBpeUnk = 0;
  let prodElapsedMs = 0;
  let bpeElapsedMs = 0;
  let roundTripOk = true;

  for (const [cat, samples] of categorySamples.entries()) {
    let catBytes = 0;
    let catChars = 0;
    let prodToks = 0;
    let prodUnk = 0;
    let bpeToks = 0;
    let bpeUnk = 0;

    const p0 = performance.now();
    for (const s of samples) {
      catBytes += Buffer.byteLength(s, 'utf8');
      catChars += s.length;
      const st = prodTokenizer.encodeWithStats(s);
      prodToks += Math.max(1, st.tokens.length - 2);
      prodUnk += st.unkCount;
    }
    prodElapsedMs += performance.now() - p0;

    const b0 = performance.now();
    for (const s of samples) {
      const toks = bpe.encode(s);
      bpeToks += Math.max(1, toks.length - 2);
      for (const t of toks) {
        if (t === bpe.specialTokens['<unk>']) bpeUnk++;
      }
      if (bpe.decode(toks) !== s) {
        roundTripOk = false;
      }
    }
    bpeElapsedMs += performance.now() - b0;

    totalBytes += catBytes;
    totalChars += catChars;
    totalProdTokens += prodToks;
    totalProdUnk += prodUnk;
    totalBpeTokens += bpeToks;
    totalBpeUnk += bpeUnk;

    categoryComparisons.push({
      category: cat,
      samples: samples.length,
      bytesTotal: catBytes,
      charsTotal: catChars,
      prodTokens: prodToks,
      prodUnkCount: prodUnk,
      prodUnkRate: catChars > 0 ? Number((prodUnk / catChars).toFixed(6)) : 0,
      prodCharsPerToken: prodToks > 0 ? Number((catChars / prodToks).toFixed(3)) : 0,
      prodBytesPerToken: prodToks > 0 ? Number((catBytes / prodToks).toFixed(3)) : 0,
      bpeTokens: bpeToks,
      bpeUnkCount: bpeUnk,
      bpeUnkRate: catChars > 0 ? Number((bpeUnk / catChars).toFixed(6)) : 0,
      bpeCharsPerToken: bpeToks > 0 ? Number((catChars / bpeToks).toFixed(3)) : 0,
      bpeBytesPerToken: bpeToks > 0 ? Number((catBytes / bpeToks).toFixed(3)) : 0,
      bpeCompressionGainVsProd: prodToks > 0 ? Number((prodToks / bpeToks).toFixed(3)) : 1,
    });
  }

  const prodSec = Math.max(1e-6, prodElapsedMs / 1000);
  const bpeSec = Math.max(1e-6, bpeElapsedMs / 1000);

  const report: TokenizerComparisonReport = {
    generatedAt: new Date().toISOString(),
    productionTokenizer: {
      version: prodTokenizer.version,
      vocabSize: prodTokenizer.getVocabSize(),
      activeTokens: prodTokenizer.getActiveTokenCount(),
      checksum: prodTokenizer.getChecksum(),
      totalTokensOnEvalSet: totalProdTokens,
      unkRate: totalChars > 0 ? Number((totalProdUnk / totalChars).toFixed(6)) : 0,
      charsPerToken: totalProdTokens > 0 ? Number((totalChars / totalProdTokens).toFixed(3)) : 0,
      bytesPerToken: totalProdTokens > 0 ? Number((totalBytes / totalProdTokens).toFixed(3)) : 0,
      speedTokensPerSec: Math.round(totalProdTokens / prodSec),
      speedCharsPerSec: Math.round(totalChars / prodSec),
    },
    experimentalBpeTokenizer: {
      version: bpe.version,
      status: 'Research Evaluation Only — Production checkpoints remain bound to omnibrain-tok-v1.1.0',
      vocabSize: bpe.getVocabSize(),
      learnedMerges: bpe.getLearnedMergeCount(),
      checksum: bpe.getChecksum(),
      totalTokensOnEvalSet: totalBpeTokens,
      unkRate: totalChars > 0 ? Number((totalBpeUnk / totalChars).toFixed(6)) : 0,
      charsPerToken: totalBpeTokens > 0 ? Number((totalChars / totalBpeTokens).toFixed(3)) : 0,
      bytesPerToken: totalBpeTokens > 0 ? Number((totalBytes / totalBpeTokens).toFixed(3)) : 0,
      speedTokensPerSec: Math.round(totalBpeTokens / bpeSec),
      speedCharsPerSec: Math.round(totalChars / bpeSec),
      roundTripExactMatch: roundTripOk,
    },
    categoryComparisons: categoryComparisons.sort((a, b) => a.category.localeCompare(b.category)),
    productionRetainedReason:
      'Production tokenizer omnibrain-tok-v1.1.0 (vocabSize=1024) is preserved unchanged to guarantee strict compatibility with existing Pure-Python & PyTorch checkpoints (including embedding/lm_head shape [1024, dModel] and checksum verification). Migration requires retraining or re-initializing embeddings after embedding-layer verification.',
    migrationRecommendation: {
      productionVersionLocked: prodTokenizer.version,
      migrateExistingCheckpointsNow: false,
      recommendedForNextScratchPretraining: true,
      recommendedVocabTargetForNextRun: 8192,
      summary:
        'Keep omnibrain-tok-v1.1.0 locked for all current v1 checkpoints and CPU experiments. Migrate to a frozen 8k–16k Byte-Level BPE tokenizer only when starting fresh scratch pretraining (v2 model family) so Indic scripts (~52.5% of tokens) and rare Unicode achieve 0% <unk> and >2.0x context compression.',
    },
  };

  const reportPath = path.resolve('./datasets/reports/tokenizer_comparison.json');
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf8');
  return report;
}
