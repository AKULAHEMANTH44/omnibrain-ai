export interface DocumentChunk {
  id: string;
  documentTitle: string;
  content: string;
  embedding: number[];
  metadata: {
    size: string;
    indexedAt: string;
  };
}

export class KnowledgeEngine {
  private chunks: DocumentChunk[] = [];

  constructor() {
    // Seed initial internal knowledge base
    this.ingestDocument(
      'OMNIBRAIN Technical Whitepaper v2.0',
      'OMNIBRAIN AI is a universal intelligence platform featuring a modular orchestrator, 10 specialized intelligence modules, and a native transformer training pipeline. It supports provider abstraction between Gemini and OMNIBRAIN Native Core.'
    );
    this.ingestDocument(
      'Transformer Scaling Laws',
      'Transformer model scaling follows predictable power laws relative to parameter count, dataset size, and compute budget. Mixture-of-experts (MoE) routing allows high capacity with reduced active inference compute.'
    );
  }

  private textToVector(text: string): number[] {
    // Lightweight local embedding vector representation (32-dim)
    const vec = new Array(32).fill(0);
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      vec[i % 32] += code / 1000;
    }
    const magnitude = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0)) || 1;
    return vec.map(v => v / magnitude);
  }

  ingestDocument(title: string, content: string): DocumentChunk {
    // Text chunking (simulate paragraph splitting)
    const embedding = this.textToVector(content);
    const chunk: DocumentChunk = {
      id: `chunk_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      documentTitle: title,
      content,
      embedding,
      metadata: {
        size: `${(content.length / 1024).toFixed(2)} KB`,
        indexedAt: new Date().toISOString(),
      },
    };
    this.chunks.push(chunk);
    return chunk;
  }

  search(query: string, topK = 2): DocumentChunk[] {
    if (this.chunks.length === 0) return [];
    const queryVec = this.textToVector(query);

    const scored = this.chunks.map(chunk => {
      // Cosine similarity
      let dot = 0;
      for (let i = 0; i < 32; i++) {
        dot += queryVec[i] * chunk.embedding[i];
      }
      return { chunk, score: dot };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map(s => s.chunk);
  }

  listDocuments() {
    return this.chunks.map(c => ({
      id: c.id,
      title: c.documentTitle,
      size: c.metadata.size,
      indexedAt: c.metadata.indexedAt,
    }));
  }
}

export const globalKnowledgeEngine = new KnowledgeEngine();
