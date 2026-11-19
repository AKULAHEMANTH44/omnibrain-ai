export type NavigationTab = 'chat' | 'create' | 'learning' | 'code' | 'knowledge' | 'ml' | 'settings';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  modelUsed?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: string;
  model: string;
}

export interface StudioDocument {
  id: string;
  title: string;
  type: 'pdf' | 'powerpoint' | 'word' | 'diagram' | 'flowchart' | 'notes' | 'quiz' | 'report';
  content: string;
  createdAt: string;
  topic: string;
}

export interface MLStatus {
  modelName: string;
  architecture: {
    layers: number;
    attentionHeads: number;
    embeddingDim: number;
    contextWindow: string;
    vocabSize: string;
    parameterCount: string;
  };
  datasets: Array<{
    name: string;
    size: string;
    status: string;
  }>;
  trainingState: {
    status: string;
    currentEpoch: number;
    totalEpochs: number;
    globalStep: number;
    loss: number;
    learningRate: string;
    gpuUtilization: string;
    vramUsage: string;
  };
}
