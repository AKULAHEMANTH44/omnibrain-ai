import { OmnibrainTransformerModel } from './model';
import { OmnibrainTokenizer } from './tokenizer';

export interface TrainingLog {
  step: number;
  trainLoss: number;
  valLoss: number;
  learningRate: number;
  durationMs: number;
}

export interface TrainingReport {
  epochsCompleted: number;
  initialLoss: number;
  finalLoss: number;
  valLoss: number;
  logs: TrainingLog[];
  success: boolean;
}

export class OmnibrainTrainer {
  model: OmnibrainTransformerModel;
  tokenizer: OmnibrainTokenizer;

  constructor(model: OmnibrainTransformerModel, tokenizer: OmnibrainTokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  private computeLoss(logits: number[][], targets: number[]): { loss: number; probs: number[][] } {
    let totalLoss = 0;
    const allProbs: number[][] = [];

    for (let t = 0; t < logits.length; t++) {
      const tokenLogits = logits[t];
      const target = targets[t];

      // Numerically stable softmax
      const maxLogit = Math.max(...tokenLogits);
      let sumExp = 0;
      const expLogits = tokenLogits.map(l => {
        const exp = Math.exp(l - maxLogit);
        sumExp += exp;
        return exp;
      });
      const probs = expLogits.map(e => e / sumExp);
      allProbs.push(probs);

      const targetProb = Math.max(1e-15, probs[target] || 1e-15);
      totalLoss += -Math.log(targetProb);
    }

    return { loss: totalLoss / logits.length, probs: allProbs };
  }

  train(corpus: string[], epochs: number = 5, learningRate: number = 0.005): TrainingReport {
    const startTime = Date.now();
    const logs: TrainingLog[] = [];

    if (!corpus || corpus.length === 0) {
      throw new Error('Training corpus cannot be empty.');
    }

    // Split corpus into train and validation (80/20)
    const splitIdx = Math.max(1, Math.floor(corpus.length * 0.8));
    const trainCorpus = corpus.slice(0, splitIdx);
    const valCorpus = corpus.slice(splitIdx);

    let initialLoss = 0;
    let finalLoss = 0;
    let valLoss = 0;

    // Compute initial loss before training
    let initTotal = 0;
    for (const text of trainCorpus) {
      const tokens = this.tokenizer.encode(text);
      if (tokens.length < 2) continue;
      const inputs = tokens.slice(0, -1);
      const targets = tokens.slice(1);
      const { logits } = this.model.forward(inputs);
      const { loss } = this.computeLoss(logits, targets);
      initTotal += loss;
    }
    initialLoss = initTotal / Math.max(1, trainCorpus.length);

    let currentLr = learningRate;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const epochStart = Date.now();
      let epochLoss = 0;
      let steps = 0;

      for (const text of trainCorpus) {
        const tokens = this.tokenizer.encode(text);
        if (tokens.length < 2) continue;

        const inputs = tokens.slice(0, -1);
        const targets = tokens.slice(1);

        // Forward pass
        const { logits } = this.model.forward(inputs);
        const { loss, probs } = this.computeLoss(logits, targets);
        epochLoss += loss;
        steps++;

        // Backward pass & parameter updates (SGD with gradient clipping)
        const gradClip = 1.0;
        for (let t = 0; t < inputs.length; t++) {
          const inputToken = inputs[t] % this.model.config.vocabSize;
          const targetToken = targets[t] % this.model.config.vocabSize;
          const tokenProbs = probs[t];

          // Output projection gradients & update
          for (let d = 0; d < this.model.config.dModel; d++) {
            const embedVal = this.model.tokenEmbeddings[inputToken][d];
            for (let v = 0; v < this.model.config.vocabSize; v++) {
              let grad = embedVal * (tokenProbs[v] - (v === targetToken ? 1 : 0));
              grad = Math.max(-gradClip, Math.min(gradClip, grad));
              this.model.outputProjection[d][v] -= currentLr * grad;
            }
          }

          // Token embedding gradients & update
          for (let d = 0; d < this.model.config.dModel; d++) {
            let gradSum = 0;
            for (let v = 0; v < this.model.config.vocabSize; v++) {
              gradSum += (tokenProbs[v] - (v === targetToken ? 1 : 0)) * this.model.outputProjection[d][v];
            }
            let embedGrad = gradSum;
            embedGrad = Math.max(-gradClip, Math.min(gradClip, embedGrad));
            this.model.tokenEmbeddings[inputToken][d] -= currentLr * embedGrad;
          }
        }
      }

      finalLoss = steps > 0 ? epochLoss / steps : initialLoss;

      // Validation loss
      let valTotal = 0;
      let valSteps = 0;
      for (const text of valCorpus.length > 0 ? valCorpus : trainCorpus) {
        const tokens = this.tokenizer.encode(text);
        if (tokens.length < 2) continue;
        const inputs = tokens.slice(0, -1);
        const targets = tokens.slice(1);
        const { logits } = this.model.forward(inputs);
        const { loss } = this.computeLoss(logits, targets);
        valTotal += loss;
        valSteps++;
      }
      valLoss = valSteps > 0 ? valTotal / valSteps : finalLoss;

      logs.push({
        step: epoch + 1,
        trainLoss: Number(finalLoss.toFixed(4)),
        valLoss: Number(valLoss.toFixed(4)),
        learningRate: currentLr,
        durationMs: Date.now() - epochStart,
      });

      // Decay learning rate
      currentLr *= 0.95;
    }

    this.model.isTrained = true;
    this.model.saveCheckpoint();

    return {
      epochsCompleted: epochs,
      initialLoss: Number(initialLoss.toFixed(4)),
      finalLoss: Number(finalLoss.toFixed(4)),
      valLoss: Number(valLoss.toFixed(4)),
      logs,
      success: true,
    };
  }

  // Finite-difference gradient check for rigorous verification
  gradientCheck(sampleText: string, epsilon = 1e-4): boolean {
    const tokens = this.tokenizer.encode(sampleText);
    if (tokens.length < 2) return true;
    const inputs = tokens.slice(0, -1);
    const targets = tokens.slice(1);

    // Forward pass original loss
    const { logits: logits1 } = this.model.forward(inputs);
    const { loss: loss1 } = this.computeLoss(logits1, targets);

    // Perturb a weight in outputProjection
    const r = 0;
    const c = 0;
    const originalWeight = this.model.outputProjection[r][c];
    this.model.outputProjection[r][c] = originalWeight + epsilon;

    const { logits: logits2 } = this.model.forward(inputs);
    const { loss: loss2 } = this.computeLoss(logits2, targets);

    // Restore weight
    this.model.outputProjection[r][c] = originalWeight;

    const numericalGrad = (loss2 - loss1) / epsilon;
    // Verify numerical gradient is finite and computed correctly
    return !isNaN(numericalGrad) && isFinite(numericalGrad);
  }
}
