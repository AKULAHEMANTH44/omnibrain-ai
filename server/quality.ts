export interface QualityReport {
  isValid: boolean;
  confidenceScore: number;
  warnings: string[];
  suggestions: string[];
}

export class AnswerQualityLayer {
  validate(answer: string): QualityReport {
    const warnings: string[] = [];
    const suggestions: string[] = [];
    let confidenceScore = 0.95;

    if (!answer || answer.trim().length < 10) {
      warnings.push('Response is extremely brief or potentially incomplete.');
      confidenceScore -= 0.3;
    }

    if (answer.includes('I am not sure') || answer.includes('possibly') || answer.includes('might be')) {
      warnings.push('Response contains epistemic uncertainty.');
      confidenceScore -= 0.1;
    }

    if (answer.length > 5000) {
      suggestions.push('Consider summarizing lengthy technical responses into structured bullet points.');
    }

    return {
      isValid: warnings.length === 0 || confidenceScore > 0.6,
      confidenceScore: Math.max(0.1, Math.min(1.0, confidenceScore)),
      warnings,
      suggestions,
    };
  }
}

export const globalQualityLayer = new AnswerQualityLayer();
