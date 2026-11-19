import fs from 'fs';
import path from 'path';

export interface TransformerConfig {
  vocabSize: number;
  dModel: number;
  numLayers: number;
  numHeads: number;
  maxSeqLen: number;
  dFf: number;
}

export class TransformerLayerWeights {
  // Layer normalization weights
  ln1Gamma: number[];
  ln1Beta: number[];
  ln2Gamma: number[];
  ln2Beta: number[];

  // Attention weights [dModel x dModel]
  wq: number[][];
  wk: number[][];
  wv: number[][];
  wo: number[][];

  // Feed-forward weights [dModel x dFf] and [dFf x dModel]
  w1: number[][]; // [dModel x dFf]
  w2: number[][]; // [dFf x dModel]

  constructor(dModel: number, dFf: number) {
    this.ln1Gamma = new Array(dModel).fill(1);
    this.ln1Beta = new Array(dModel).fill(0);
    this.ln2Gamma = new Array(dModel).fill(1);
    this.ln2Beta = new Array(dModel).fill(0);

    this.wq = randomMatrix(dModel, dModel);
    this.wk = randomMatrix(dModel, dModel);
    this.wv = randomMatrix(dModel, dModel);
    this.wo = randomMatrix(dModel, dModel);

    this.w1 = randomMatrix(dModel, dFf);
    this.w2 = randomMatrix(dFf, dModel);
  }
}

function randomMatrix(rows: number, cols: number): number[][] {
  const scale = Math.sqrt(2.0 / (rows + cols));
  const mat: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) {
      row.push((Math.random() - 0.5) * 2 * scale);
    }
    mat.push(row);
  }
  return mat;
}

function matrixMultiply(a: number[][], b: number[][]): number[][] {
  const rowsA = a.length;
  const colsA = a[0].length;
  const rowsB = b.length;
  const colsB = b[0].length;

  if (colsA !== rowsB) {
    throw new Error(`Matrix multiplication dimension mismatch: [${rowsA}x${colsA}] x [${rowsB}x${colsB}]`);
  }

  const result: number[][] = Array.from({ length: rowsA }, () => new Array(colsB).fill(0));
  for (let i = 0; i < rowsA; i++) {
    for (let k = 0; k < colsA; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < colsB; j++) {
        result[i][j] += aik * b[k][j];
      }
    }
  }
  return result;
}

function layerNorm(x: number[][], gamma: number[], beta: number[], eps = 1e-5): number[][] {
  return x.map(row => {
    const mean = row.reduce((sum, val) => sum + val, 0) / row.length;
    const variance = row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / row.length;
    const std = Math.sqrt(variance + eps);
    return row.map((val, idx) => gamma[idx] * ((val - mean) / std) + beta[idx]);
  });
}

function relu(x: number[][]): number[][] {
  return x.map(row => row.map(val => Math.max(0, val)));
}

export class OmnibrainTransformerModel {
  config: TransformerConfig;
  tokenEmbeddings: number[][]; // [vocabSize x dModel]
  positionEmbeddings: number[][]; // [maxSeqLen x dModel]
  layers: TransformerLayerWeights[];
  outputProjection: number[][]; // [dModel x vocabSize]

  isTrained: boolean = false;
  checkpointPath: string;

  constructor(
    config: TransformerConfig = { vocabSize: 512, dModel: 64, numLayers: 2, numHeads: 4, maxSeqLen: 128, dFf: 256 },
    checkpointDir = './checkpoints'
  ) {
    this.config = config;
    this.checkpointPath = path.join(checkpointDir, 'omnibrain_v2_rigorous.bin');

    this.tokenEmbeddings = randomMatrix(config.vocabSize, config.dModel);
    this.positionEmbeddings = randomMatrix(config.maxSeqLen, config.dModel);
    this.outputProjection = randomMatrix(config.dModel, config.vocabSize);

    this.layers = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.layers.push(new TransformerLayerWeights(config.dModel, config.dFf));
    }

    this.loadCheckpoint();
  }

  forward(tokenIds: number[]): { logits: number[][]; hiddenStates: number[][][] } {
    const seqLen = tokenIds.length;
    if (seqLen > this.config.maxSeqLen) {
      throw new Error(`Sequence length ${seqLen} exceeds maxSeqLen ${this.config.maxSeqLen}`);
    }

    // 1. Token Embeddings + Positional Encodings
    let h: number[][] = [];
    for (let t = 0; t < seqLen; t++) {
      const tokenId = tokenIds[t] % this.config.vocabSize;
      const tokEmb = this.tokenEmbeddings[tokenId];
      const posEmb = this.positionEmbeddings[t];
      const combined = tokEmb.map((val, d) => val + posEmb[d]);
      h.push(combined);
    }

    const hiddenStates: number[][][] = [h];

    // 2. Transformer Layers
    for (let l = 0; l < this.config.numLayers; l++) {
      const layer = this.layers[l];

      // --- Sub-layer 1: Multi-Head Causal Self-Attention ---
      const norm1 = layerNorm(h, layer.ln1Gamma, layer.ln1Beta);

      // Q, K, V Projections [seqLen x dModel]
      const Q = matrixMultiply(norm1, layer.wq);
      const K = matrixMultiply(norm1, layer.wk);
      const V = matrixMultiply(norm1, layer.wv);

      // Multi-head split & causal attention
      const dK = this.config.dModel / this.config.numHeads;
      const headOutputs: number[][] = Array.from({ length: seqLen }, () => new Array(this.config.dModel).fill(0));

      for (let hIdx = 0; hIdx < this.config.numHeads; hIdx++) {
        const startD = hIdx * dK;

        // Extract head queries, keys, values
        const Q_h = Q.map(row => row.slice(startD, startD + dK));
        const K_h = K.map(row => row.slice(startD, startD + dK));
        const V_h = V.map(row => row.slice(startD, startD + dK));

        // Attention scores Q_h * K_h^T / sqrt(dK) with causal mask
        for (let i = 0; i < seqLen; i++) {
          const scores: number[] = new Array(seqLen).fill(0);
          for (let j = 0; j <= i; j++) { // Causal mask: j <= i
            let dot = 0;
            for (let d = 0; d < dK; d++) {
              dot += Q_h[i][d] * K_h[j][d];
            }
            scores[j] = dot / Math.sqrt(dK);
          }
          // Mask future tokens with -1e9
          for (let j = i + 1; j < seqLen; j++) {
            scores[j] = -1e9;
          }

          // Numerically stable softmax
          const maxScore = Math.max(...scores);
          let sumExp = 0;
          const expScores = scores.map(s => {
            const exp = Math.exp(s - maxScore);
            sumExp += exp;
            return exp;
          });
          const attnWeights = expScores.map(e => e / sumExp);

          // Weighted sum of V_h
          for (let d = 0; d < dK; d++) {
            let weightedSum = 0;
            for (let j = 0; j <= i; j++) {
              weightedSum += attnWeights[j] * V_h[j][d];
            }
            headOutputs[i][startD + d] = weightedSum;
          }
        }
      }

      // Output projection & Residual connection 1
      const attnOut = matrixMultiply(headOutputs, layer.wo);
      h = h.map((row, i) => row.map((val, d) => val + attnOut[i][d]));

      // --- Sub-layer 2: Feed-Forward Network ---
      const norm2 = layerNorm(h, layer.ln2Gamma, layer.ln2Beta);
      const ffnHidden = relu(matrixMultiply(norm2, layer.w1));
      const ffnOut = matrixMultiply(ffnHidden, layer.w2);

      // Residual connection 2
      h = h.map((row, i) => row.map((val, d) => val + ffnOut[i][d]));
      hiddenStates.push(h);
    }

    // 3. Final Output Logits [seqLen x vocabSize]
    const logits = matrixMultiply(h, this.outputProjection);
    return { logits, hiddenStates };
  }

  saveCheckpoint() {
    try {
      const dir = path.dirname(this.checkpointPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(this.checkpointPath, JSON.stringify({
        config: this.config,
        tokenEmbeddings: this.tokenEmbeddings,
        positionEmbeddings: this.positionEmbeddings,
        layers: this.layers,
        outputProjection: this.outputProjection,
        isTrained: this.isTrained,
        savedAt: new Date().toISOString(),
      }, null, 2));
    } catch (err) {
      console.error('Failed to save checkpoint:', err);
    }
  }

  loadCheckpoint(): boolean {
    try {
      if (fs.existsSync(this.checkpointPath)) {
        const data = JSON.parse(fs.readFileSync(this.checkpointPath, 'utf8'));
        this.config = data.config;
        this.tokenEmbeddings = data.tokenEmbeddings;
        this.positionEmbeddings = data.positionEmbeddings;
        this.layers = data.layers;
        this.outputProjection = data.outputProjection;
        this.isTrained = data.isTrained;
        return true;
      }
    } catch (err) {
      console.warn('No checkpoint found or load error:', err);
    }
    return false;
  }
}

export const globalModel = new OmnibrainTransformerModel();
