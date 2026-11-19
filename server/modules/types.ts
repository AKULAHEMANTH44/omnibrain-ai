export type ModuleCategory =
  | 'general'
  | 'mathematics'
  | 'programming'
  | 'science'
  | 'engineering'
  | 'education'
  | 'research'
  | 'creative'
  | 'document'
  | 'reasoning';

export interface IIntelligenceModule {
  category: ModuleCategory;
  name: string;
  description: string;
  systemInstruction: string;
  temperature: number;
}
