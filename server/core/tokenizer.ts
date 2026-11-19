import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

export interface TokenizerConfig {
  version: string;
  vocabSize: number;
  specialTokens: Record<string, number>;
  checksum: string;
}

export interface TokenizationStats {
  tokens: number[];
  totalChars: number;
  unkCount: number;
  unkRate: number;
  unknownChars: string[];
}

export class OmnibrainTokenizer {
  public readonly version = 'omnibrain-tok-v1.1.0';
  private vocab: Map<string, number> = new Map();
  private inverseVocab: Map<number, string> = new Map();
  private config: TokenizerConfig;

  constructor(vocabSize: number = 1024) {
    const specialTokens: Record<string, number> = {
      '<pad>': 0,
      '<unk>': 1,
      '<bos>': 2,
      '<eos>': 3,
    };

    for (const [token, id] of Object.entries(specialTokens)) {
      this.vocab.set(token, id);
      this.inverseVocab.set(id, token);
    }

    let nextId = 4;

    // 1. Whitespace & Control (newline, tab)
    for (const ch of ['\n', '\t', '\r']) {
      if (!this.vocab.has(ch) && nextId < vocabSize) {
        this.vocab.set(ch, nextId);
        this.inverseVocab.set(nextId, ch);
        nextId++;
      }
    }

    // 2. Printable ASCII (32..126)
    for (let i = 32; i < 127; i++) {
      const char = String.fromCharCode(i);
      if (!this.vocab.has(char) && nextId < vocabSize) {
        this.vocab.set(char, nextId);
        this.inverseVocab.set(nextId, char);
        nextId++;
      }
    }

    // 3. Extended Latin, Mathematical Symbols, Indian Scripts (Devanagari, Tamil, Telugu, Kannada, Bengali)
    const extendedSets = [
      // Extended Latin & punctuation
      'äöüßñáéíóúàèìòùâêîôûçÄÖÜÑ—–•“”‘’…°',
      // Mathematics & Logic
      '∑∫∂∇√∞≈≠≤≥±×÷∈∉⊂⊆∪∩∀∃πθλΔΩαβγδμσφψω²³⁰¹⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉→⇒⇔',
      // Devanagari (Hindi/Sanskrit/Marathi)
      'अआइईउऊऋएऐओऔअंअःकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञड़ढ़ािीुूृेैोौ्ंःँ।ॐ०१२३४५६७८९ऑऍ',
      // Tamil
      'அஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஷஸஹாிீுூெேைொோௌ்',
      // Telugu
      'అఆఇఈఉఊఋఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహాిీుూృెేైొోౌ్ంః',
      // Kannada
      'ಅಆಇಈಉಊಋಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲಳವಶಷಸಹಾಿೀುೂೃೆೇೈೊೋೌ್ಂಃ',
      // Bengali
      'অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহািীুূৃেৈোৌ্ংঃঁয়ড়ঢ়',
    ];

    for (const charSet of extendedSets) {
      for (const char of charSet) {
        if (!this.vocab.has(char) && nextId < vocabSize) {
          this.vocab.set(char, nextId);
          this.inverseVocab.set(nextId, char);
          nextId++;
        }
      }
    }

    this.config = {
      version: this.version,
      vocabSize,
      specialTokens,
      checksum: this.computeChecksum(),
    };
  }

  private computeChecksum(): string {
    const entries: Array<[number, string]> = [];
    for (const [id, tok] of this.inverseVocab.entries()) {
      entries.push([id, tok]);
    }
    entries.sort((a, b) => a[0] - b[0]);
    const payload = JSON.stringify({ version: this.version, entries });
    return crypto.createHash('sha256').update(payload).digest('hex');
  }

  encode(text: string): number[] {
    const tokens: number[] = [this.config.specialTokens['<bos>']];
    for (const char of text) {
      const id = this.vocab.get(char);
      if (id !== undefined) {
        tokens.push(id);
      } else {
        tokens.push(this.config.specialTokens['<unk>']);
      }
    }
    tokens.push(this.config.specialTokens['<eos>']);
    return tokens;
  }

  encodeWithStats(text: string): TokenizationStats {
    const tokens: number[] = [this.config.specialTokens['<bos>']];
    let unkCount = 0;
    const unknownSet = new Set<string>();
    let totalChars = 0;

    for (const char of text) {
      totalChars++;
      const id = this.vocab.get(char);
      if (id !== undefined) {
        tokens.push(id);
      } else {
        tokens.push(this.config.specialTokens['<unk>']);
        unkCount++;
        unknownSet.add(char);
      }
    }
    tokens.push(this.config.specialTokens['<eos>']);

    return {
      tokens,
      totalChars,
      unkCount,
      unkRate: totalChars > 0 ? Number((unkCount / totalChars).toFixed(6)) : 0,
      unknownChars: Array.from(unknownSet),
    };
  }

  encodeBatch(texts: string[]): number[][] {
    return texts.map((t) => this.encode(t));
  }

  decode(tokens: number[]): string {
    let result = '';
    const bos = this.config.specialTokens['<bos>'];
    const eos = this.config.specialTokens['<eos>'];
    const pad = this.config.specialTokens['<pad>'];
    for (const t of tokens) {
      if (t === bos || t === eos || t === pad) continue;
      const char = this.inverseVocab.get(t);
      if (char !== undefined) {
        result += char;
      }
    }
    return result;
  }

  getVocabSize(): number {
    return this.config.vocabSize;
  }

  getActiveTokenCount(): number {
    return this.vocab.size;
  }

  getChecksum(): string {
    return this.config.checksum;
  }

  getConfig(): TokenizerConfig {
    return { ...this.config };
  }

  serialize(): string {
    const idToToken: Record<string, string> = {};
    for (const [id, tok] of this.inverseVocab.entries()) {
      idToToken[String(id)] = tok;
    }
    return JSON.stringify(
      {
        version: this.version,
        vocabSize: this.config.vocabSize,
        activeTokens: this.vocab.size,
        specialTokens: this.config.specialTokens,
        checksum: this.config.checksum,
        idToToken,
      },
      null,
      2
    );
  }

  saveToFile(filePath = './checkpoints/tokenizer_v1.json'): string {
    const abs = path.resolve(filePath);
    fs.mkdirSync(path.dirname(abs), { recursive: true });
    fs.writeFileSync(abs, this.serialize(), 'utf8');
    return abs;
  }

  validateCompatibility(expected: {
    version?: string;
    vocabSize?: number;
    checksum?: string;
    specialTokens?: Record<string, number>;
  }): { compatible: boolean; reason?: string } {
    if (expected.version && expected.version !== this.version) {
      return { compatible: false, reason: `Tokenizer version mismatch: expected ${expected.version}, active ${this.version}` };
    }
    if (expected.vocabSize !== undefined && expected.vocabSize !== this.config.vocabSize) {
      return { compatible: false, reason: `Vocab size mismatch: expected ${expected.vocabSize}, active ${this.config.vocabSize}` };
    }
    if (expected.checksum && expected.checksum !== this.config.checksum) {
      return { compatible: false, reason: `Checksum mismatch: expected ${expected.checksum.slice(0, 12)}, active ${this.config.checksum.slice(0, 12)}` };
    }
    if (expected.specialTokens) {
      for (const [tok, id] of Object.entries(expected.specialTokens)) {
        if (this.config.specialTokens[tok] !== id) {
          return { compatible: false, reason: `Special token '${tok}' mismatch: expected ${id}, active ${this.config.specialTokens[tok]}` };
        }
      }
    }
    return { compatible: true };
  }

  getExperimentalSubwordStatus() {
    return {
      activeProductionVersion: this.version,
      experimentalCandidateVersion: 'omnibrain-tok-bpe-v2.0-exp',
      status: 'Isolated Research Investigation Only (Production token IDs locked at omnibrain-tok-v1.1.0)',
      notes: 'Future byte-level BPE will ship under an isolated version tag and gated migration test before replacing character-level IDs.',
    };
  }
}

export const globalTokenizer = new OmnibrainTokenizer(1024);
