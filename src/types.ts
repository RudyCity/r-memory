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

export interface DatabaseAdapter {
  insert(id: string, content: string, embedding: number[], metadata: Record<string, any>): void;
  query(queryEmbedding: number[], limit: number, filter?: Record<string, any>): QueryResult[];
  queryLexical(queryText: string, limit: number, filter?: Record<string, any>): QueryResult[];
  getById(id: string): Memory | null;
  delete(id: string): void;
  clear(): void;
  close(): void;
}
