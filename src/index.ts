import { randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { SQLiteAdapter } from './database/sqlite.js';
import { 
  Memory, 
  QueryResult, 
  EmbeddingProvider, 
  RMemoryConfig, 
  QueryOptions 
} from './types.js';
import { 
  extractTextFromPDF, 
  extractTextFromDocx, 
  extractTextFromXlsx, 
  chunkText 
} from './utils/document.js';

export { 
  Memory, 
  QueryResult, 
  EmbeddingProvider, 
  RMemoryConfig, 
  QueryOptions 
};

export { 
  LocalTextEmbeddingProvider, 
  LocalCLIPEmbeddingProvider 
} from './embeddings/local.js';

export { 
  OpenAIEmbeddingProvider, 
  OpenAIEmbeddingConfig 
} from './embeddings/openai.js';

export {
  extractTextFromPDF,
  extractTextFromDocx,
  extractTextFromXlsx,
  chunkText
};

export class RMemory {
  private db: SQLiteAdapter;
  private provider: EmbeddingProvider;
  private collectionName: string;

  constructor(config: RMemoryConfig) {
    this.provider = config.embeddingProvider;
    this.collectionName = config.collectionName || 'memories';
    
    // Initialize DB adapter with dynamic dimensions
    this.db = new SQLiteAdapter(
      config.dbPath, 
      this.collectionName, 
      this.provider.dimensions
    );
  }

  /**
   * Adds a memory to the agent's database.
   * If an image is provided, its embedding is generated. Otherwise, the text embedding of content is used.
   * 
   * @param options Memory insertion options
   * @returns The memory ID (generated if not provided)
   */
  async addMemory(options: {
    id?: string;
    content: string;
    image?: string | Buffer | Uint8Array;
    metadata?: Record<string, any>;
  }): Promise<string> {
    const id = options.id || randomUUID();
    const content = options.content;
    const metadata = options.metadata || {};
    
    let embedding: number[];
    if (options.image) {
      embedding = await this.provider.embedImage(options.image);
      // Store flag in metadata so we know this memory was generated from an image
      metadata._hasImage = true;
    } else {
      embedding = await this.provider.embedText(content, 'passage');
    }

    this.db.insert(id, content, embedding, metadata);
    return id;
  }

  /**
   * Queries memories based on semantic similarity.
   * Supporting both text-based queries (string) and image-based queries (Buffer / Uint8Array).
   * 
   * @param options Query configuration options
   * @returns Array of query results containing memory details and similarity distance
   */
  async query(options: QueryOptions): Promise<QueryResult[]> {
    const limit = options.limit ?? 5;
    const filter = options.filter;

    let queryEmbedding: number[];
    if (typeof options.query === 'string') {
      queryEmbedding = await this.provider.embedText(options.query, 'query');
    } else {
      queryEmbedding = await this.provider.embedImage(options.query);
    }

    return this.db.query(queryEmbedding, limit, filter);
  }

  /**
   * Deletes a memory by its ID.
   */
  delete(id: string): void {
    this.db.delete(id);
  }

  /**
   * Clears all memories from the collection.
   */
  clear(): void {
    this.db.clear();
  }

  /**
   * Closes the underlying database connection.
   */
  close(): void {
    this.db.close();
  }

  /**
   * Extracts text, chunks it, and adds a whole document (PDF, DOCX, XLSX, TXT) to the memory collection.
   * 
   * @param options Document ingestion configuration
   * @returns Array of generated memory IDs for each chunk
   */
  async addDocument(options: {
    id?: string;
    pathOrBuffer: string | Buffer;
    type: 'pdf' | 'docx' | 'xlsx' | 'txt';
    metadata?: Record<string, any>;
    chunkSize?: number;
    chunkOverlap?: number;
  }): Promise<string[]> {
    const docId = options.id || randomUUID();
    const customMetadata = options.metadata || {};
    const chunkSize = options.chunkSize ?? 500;
    const chunkOverlap = options.chunkOverlap ?? 100;

    let buffer: Buffer;
    if (typeof options.pathOrBuffer === 'string') {
      buffer = readFileSync(options.pathOrBuffer);
    } else {
      buffer = options.pathOrBuffer;
    }

    let text = '';
    switch (options.type) {
      case 'pdf':
        text = await extractTextFromPDF(buffer);
        break;
      case 'docx':
        text = await extractTextFromDocx(buffer);
        break;
      case 'xlsx':
        text = await extractTextFromXlsx(buffer);
        break;
      case 'txt':
        text = buffer.toString('utf-8');
        break;
      default:
        throw new Error(`Unsupported document type: ${options.type}`);
    }

    const chunks = chunkText(text, chunkSize, chunkOverlap);
    const chunkIds: string[] = [];

    for (let i = 0; i < chunks.length; i++) {
      const chunkId = `${docId}-chunk-${i}`;
      const chunkMetadata = {
        ...customMetadata,
        _documentId: docId,
        _chunkIndex: i,
        _chunkCount: chunks.length
      };

      await this.addMemory({
        id: chunkId,
        content: chunks[i],
        metadata: chunkMetadata
      });
      chunkIds.push(chunkId);
    }

    return chunkIds;
  }

  /**
   * Returns whether the fast native sqlite-vec extension was successfully loaded.
   */
  isVectorExtensionLoaded(): boolean {
    return this.db.getIsVectorExtensionLoaded();
  }
}
