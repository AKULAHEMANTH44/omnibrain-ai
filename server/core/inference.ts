import { OmnibrainTransformerModel } from './model';
import { OmnibrainTokenizer } from './tokenizer';

export class OmnibrainInferenceEngine {
  model: OmnibrainTransformerModel;
  tokenizer: OmnibrainTokenizer;

  constructor(model: OmnibrainTransformerModel, tokenizer: OmnibrainTokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  generate(prompt: string, maxTokens: number = 60, temperature: number = 0.7): { text: string; isTrained: boolean } {
    if (!this.model.isTrained) {
      return {
        text: `[MODEL NOT TRAINED YET] OMNIBRAIN Core is currently in an un-trained state. Please trigger local model training via the ML Architecture console to ingest datasets, run backpropagation, and train weights locally.`,
        isTrained: false,
      };
    }

    const tokens = this.tokenizer.encode(prompt);
    // Remove trailing <eos> so we can continue autoregressively from prompt
    if (tokens[tokens.length - 1] === 3) {
      tokens.pop();
    }

    const generatedTokens = [...tokens];

    for (let step = 0; step < maxTokens; step++) {
      const currentContext = generatedTokens.slice(-this.model.config.maxSeqLen);
      const { logits } = this.model.forward(currentContext);
      const lastLogits = logits[logits.length - 1];

      // Apply temperature scaling & softmax sampling
      const scaledLogits = lastLogits.map(l => l / Math.max(0.1, temperature));
      const maxLogit = Math.max(...scaledLogits);
      let sumExp = 0;
      const probs = scaledLogits.map(l => {
        const exp = Math.exp(l - maxLogit);
        sumExp += exp;
        return exp;
      }).map(e => e / sumExp);

      // Sample from probability distribution
      const r = Math.random();
      let cumSum = 0;
      let nextToken = 0;
      for (let i = 0; i < probs.length; i++) {
        cumSum += probs[i];
        if (r <= cumSum) {
          nextToken = i;
          break;
        }
      }

      if (nextToken === 3 || generatedTokens.length > 250) {
        break; // Stop at <eos> or max length
      }

      generatedTokens.push(nextToken);
    }

    const decoded = this.tokenizer.decode(generatedTokens);
    return {
      text: decoded,
      isTrained: true,
    };
  }
}
