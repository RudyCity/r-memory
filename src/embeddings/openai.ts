import { EmbeddingProvider } from '../types.js';

export interface OpenAIEmbeddingConfig {
  baseURL?: string;
  apiKey?: string;
  model?: string;
  dimensions?: number;
  // If true, will send the "dimensions" parameter in the request body
  sendDimensionsParam?: boolean;
}

export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  readonly dimensions: number;
  private baseURL: string;
  private apiKey: string;
  private model: string;
  private sendDimensionsParam: boolean;

  constructor(config: OpenAIEmbeddingConfig = {}) {
    this.baseURL = config.baseURL || 'https://api.openai.com/v1';
    this.apiKey = config.apiKey || process.env.OPENAI_API_KEY || '';
    this.model = config.model || 'text-embedding-3-small';
    this.dimensions = config.dimensions || 1536; // Default OpenAI dimensions
    this.sendDimensionsParam = config.sendDimensionsParam !== false;
  }

  async embedText(text: string, context?: 'query' | 'passage'): Promise<number[]> {
    const url = `${this.baseURL.replace(/\/$/, '')}/embeddings`;
    
    const body: Record<string, any> = {
      input: text,
      model: this.model
    };

    // Only send dimensions if it's not the default or if explicitly enabled
    if (this.sendDimensionsParam && this.dimensions !== 1536 && this.model.includes('text-embedding-3')) {
      body.dimensions = this.dimensions;
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API Error (${response.status}): ${errorText}`);
    }

    const resJson = (await response.json()) as any;
    if (!resJson.data || resJson.data.length === 0 || !resJson.data[0].embedding) {
      throw new Error(`Invalid response format from OpenAI embeddings API: ${JSON.stringify(resJson)}`);
    }

    let embedding = resJson.data[0].embedding as number[];

    // If the returned embedding dimension doesn't match the target, truncate and normalize (Matryoshka style)
    if (embedding.length !== this.dimensions) {
      embedding = embedding.slice(0, this.dimensions);
      embedding = this.normalize(embedding);
    }

    return embedding;
  }

  async embedImage(image: string | Buffer | Uint8Array): Promise<number[]> {
    throw new Error('OpenAIEmbeddingProvider does not support image embeddings natively. Please use LocalCLIPEmbeddingProvider or a custom multimodal provider for image data.');
  }

  private normalize(vector: number[]): number[] {
    let sumSq = 0;
    for (let i = 0; i < vector.length; i++) {
      sumSq += vector[i] * vector[i];
    }
    const magnitude = Math.sqrt(sumSq);
    if (magnitude === 0) return vector;
    return vector.map(x => x / magnitude);
  }
}
