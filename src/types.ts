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
}

export interface DatabaseAdapter {
  insert(id: string, content: string, embedding: number[], metadata: Record<string, any>): void;
  query(queryEmbedding: number[], limit: number, filter?: Record<string, any>): QueryResult[];
  delete(id: string): void;
  clear(): void;
  close(): void;
}
