export interface Memory {
  id: string;
  content: string;
  metadata: Record<string, any>;
  createdAt: number;
  embedding?: number[];
}

export interface QueryResult {
  memory: Memory;
  distance: number;
  score?: number; // Combined RRF score for hybrid search
}

export interface EmbeddingProvider {
  readonly dimensions: number;
  embedText(text: string, context?: 'query' | 'passage'): Promise<number[]>;
  embedTexts(texts: string[], context?: 'query' | 'passage'): Promise<number[][]>; // Batch embedding support
  embedImage(image: string | Buffer | Uint8Array): Promise<number[]>;
}

export interface RMemoryConfig {
  dbPath: string;
  collectionName?: string;
  embeddingProvider: EmbeddingProvider;
}

export interface QueryOptions {
  query: string | Buffer | Uint8Array;
  limit?: number;
  filter?: Record<string, any>;
  hybrid?: boolean; // Enable hybrid search (FTS5 + Semantic + RRF)
  decayFactor?: number; // Time decay constant (per hour)
}

export interface DocumentIngestOptions {
  id?: string;
  pathOrBuffer: string | Buffer;
  type: 'pdf' | 'docx' | 'xlsx' | 'txt';
  metadata?: Record<string, any>;
  chunkSize?: number;
  chunkOverlap?: number;
  parentChild?: boolean; // Enable Parent-Child Hierarchical Ingestion
  parentChunkSize?: number;
  parentChunkOverlap?: number;
}

export interface LocalEmbeddingConfig {
  modelName?: string;
  device?: 'cpu' | 'gpu' | 'webgpu';
  dtype?: 'fp32' | 'fp16' | 'q8' | 'q4';
}

export interface ConsolidateOptions {
  filter?: Record<string, any>;
  threshold?: number; // Max cosine distance for clustering (default: 0.15)
}

export interface DatabaseAdapter {
  insert(id: string, content: string, embedding: number[], metadata: Record<string, any>): void;
  query(queryEmbedding: number[], limit: number, filter?: Record<string, any>): QueryResult[];
  queryLexical(queryText: string, limit: number, filter?: Record<string, any>): QueryResult[];
  getById(id: string): Memory | null;
  getAll(filter?: Record<string, any>): Memory[];
  delete(id: string): void;
  clear(): void;
  close(): void;

  // Semantic Cache Methods
  setCache(queryText: string, responseText: string, embedding: number[], ttlSeconds: number): void;
  getCache(queryEmbedding: number[], threshold: number): string | null;
  clearCache(): void;
}
