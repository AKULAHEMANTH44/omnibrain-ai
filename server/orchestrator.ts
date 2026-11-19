import { IModelProvider, GenerateOptions, ModelResponse } from './providers/types';
import { OmnibrainNativeProvider } from './providers/omnibrain';
import { SPECIALIZED_MODULES, classifyQuestionCategory } from './modules/registry';
import { globalKnowledgeEngine } from './knowledge';
import { globalQualityLayer } from './quality';

export class IntelligenceOrchestrator {
  private provider: IModelProvider;

  constructor() {
    this.provider = new OmnibrainNativeProvider();
  }

  async processQuery(userInput: string, modelName: string, conversationHistory: Array<{ role: string; content: string }> = []): Promise<{
    response: ModelResponse;
    category: string;
    moduleName: string;
    sources: Array<{ title: string; snippet: string }>;
    quality: any;
  }> {
    // 1. Identify category & specialized module
    const category = classifyQuestionCategory(userInput);
    const module = SPECIALIZED_MODULES[category] || SPECIALIZED_MODULES.general;

    // 2. Retrieve relevant knowledge
    const knowledgeChunks = globalKnowledgeEngine.search(userInput, 2);
    const sources = knowledgeChunks.map(c => ({
      title: c.documentTitle,
      snippet: c.content.substring(0, 120) + '...',
    }));

    // 3. Build enriched context
    let contextPrompt = module.systemInstruction;
    if (sources.length > 0) {
      contextPrompt += `\n\nRelevant Knowledge Base Context:\n` + sources.map(s => `- ${s.title}: ${s.snippet}`).join('\n');
    }

    const typedHistory = conversationHistory.map(m => ({
      role: (m.role === 'assistant' ? 'assistant' : 'user') as 'user' | 'assistant' | 'system',
      content: m.content
    }));

    const messages = [
      ...typedHistory,
      { role: 'user' as const, content: userInput },
    ];

    // 4. Generate response via local OMNIBRAIN Native provider (Zero External APIs)
    const response = await this.provider.generate({
      model: modelName || 'omnibrain-transformer-v1',
      messages,
      systemInstruction: contextPrompt,
    });

    // 5. Validate answer quality
    const quality = globalQualityLayer.validate(response.text);

    return {
      response,
      category,
      moduleName: module.name,
      sources,
      quality,
    };
  }
}

export const globalOrchestrator = new IntelligenceOrchestrator();
