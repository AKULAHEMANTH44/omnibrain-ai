export interface ModelMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface GenerateOptions {
  model: string;
  messages: ModelMessage[];
  systemInstruction?: string;
  thinkingLevel?: 'LOW' | 'MEDIUM' | 'HIGH' | 'XHIGH';
  temperature?: number;
  maxTokens?: number;
}

export interface ModelResponse {
  text: string;
  modelUsed: string;
  provider: string;
  tokensUsed?: {
    prompt: number;
    completion: number;
    total: number;
  };
  latencyMs: number;
}

export interface IModelProvider {
  name: string;
  isAvailable(): Promise<boolean>;
  generate(options: GenerateOptions): Promise<ModelResponse>;
  listModels(): string[];
}
