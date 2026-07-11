import { randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { SQLiteAdapter } from './database/sqlite.js';
import { 
  Memory, 
  QueryResult, 
  EmbeddingProvider, 
  RMemoryConfig, 
  QueryOptions,
  DocumentIngestOptions
} from './types.js';
import { 
  extractTextFromPDF, 
  extractTextFromDocx, 
  extractTextFromXlsx, 
  chunkText,
  chunkTextParentChild
} from './utils/document.js';

export { 
  Memory, 
  QueryResult, 
  EmbeddingProvider, 
  RMemoryConfig, 
  QueryOptions,
  DocumentIngestOptions
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
  chunkText,
  chunkTextParentChild
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
    embedding?: number[]; // Bypass embedding generation if pre-computed is supplied
  }): Promise<string> {
    const id = options.id || randomUUID();
    const content = options.content;
    const metadata = options.metadata || {};
    
    let embedding: number[];
    if (options.embedding) {
      embedding = options.embedding;
    } else if (options.image) {
      embedding = await this.provider.embedImage(options.image);
      metadata._hasImage = true;
    } else {
      embedding = await this.provider.embedText(content, 'passage');
    }

    this.db.insert(id, content, embedding, metadata);
    return id;
  }

  /**
   * Queries memories based on semantic similarity, support for lexical query, and hybrid search.
   * Resolves parent-child references dynamically to return parent context during retrieval.
   * 
   * @param options Query configuration options
   * @returns Array of query results containing memory details and similarity distance
   */
  async query(options: QueryOptions): Promise<QueryResult[]> {
    const limit = options.limit ?? 5;
    const filter = options.filter;

    // 1. Ingest query and compute raw search results
    let rawResults: QueryResult[];
    if (options.hybrid && typeof options.query === 'string') {
      rawResults = await this.queryHybrid(options.query, limit, filter);
    } else {
      let queryEmbedding: number[];
      if (typeof options.query === 'string') {
        queryEmbedding = await this.provider.embedText(options.query, 'query');
      } else {
        queryEmbedding = await this.provider.embedImage(options.query);
      }
      rawResults = this.db.query(queryEmbedding, limit, filter);
    }

    // 2. Resolve Parent-Child References
    // Swap child chunk content with parent content if _parentId is present in metadata
    const resolvedResults: QueryResult[] = [];
    for (const res of rawResults) {
      const parentId = res.memory.metadata._parentId;
      if (parentId) {
        const parentMemory = this.db.getById(parentId);
        if (parentMemory) {
          resolvedResults.push({
            ...res,
            memory: {
              ...res.memory,
              content: parentMemory.content, // Return parent rich context
              metadata: {
                ...res.memory.metadata,
                _childContent: res.memory.content // Preserve child text
              }
            }
          });
          continue;
        }
      }
      resolvedResults.push(res);
    }

    return resolvedResults;
  }

  /**
   * Performs hybrid search combining semantic (Vector) and lexical (FTS5) queries,
   * merged using Reciprocal Rank Fusion (RRF).
   */
  private async queryHybrid(queryText: string, limit: number, filter?: Record<string, any>): Promise<QueryResult[]> {
    // Retrieve double the limit for more robust ranking merging
    const candidateLimit = limit * 2;

    // A. Run Semantic Vector Search
    const queryEmbedding = await this.provider.embedText(queryText, 'query');
    const semanticResults = this.db.query(queryEmbedding, candidateLimit, filter);

    // B. Run Lexical Keyword Search
    const lexicalResults = this.db.queryLexical(queryText, candidateLimit, filter);

    // C. Combine using Reciprocal Rank Fusion (RRF)
    const rrfScores: Record<string, { memory: Memory; semanticRank?: number; lexicalRank?: number; distance: number }> = {};

    semanticResults.forEach((res, index) => {
      const id = res.memory.id;
      if (!rrfScores[id]) {
        rrfScores[id] = { memory: res.memory, distance: res.distance };
      }
      rrfScores[id].semanticRank = index + 1;
    });

    lexicalResults.forEach((res, index) => {
      const id = res.memory.id;
      if (!rrfScores[id]) {
        // Fallback distance to 1.0 (opposite/unmatched) if not found in semantic results
        rrfScores[id] = { memory: res.memory, distance: 1.0 };
      }
      rrfScores[id].lexicalRank = index + 1;
    });

    const k = 60; // Standard constants for RRF fusion
    const combined: QueryResult[] = [];

    for (const [id, info] of Object.entries(rrfScores)) {
      let score = 0;
      if (info.semanticRank !== undefined) {
        score += 1.0 / (k + info.semanticRank);
      }
      if (info.lexicalRank !== undefined) {
        score += 1.0 / (k + info.lexicalRank);
      }

      combined.push({
        memory: info.memory,
        distance: info.distance,
        score: score
      });
    }

    // Sort by RRF score descending (higher is better)
    combined.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

    return combined.slice(0, limit);
  }

  /**
   * Extracts text, chunks it, and adds a whole document (PDF, DOCX, XLSX, TXT) to the memory collection.
   * Supports both sliding window chunking and Parent-Child hierarchical ingestion.
   * 
   * @param options Document ingestion configuration
   * @returns Array of generated memory IDs for each chunk
   */
  async addDocument(options: DocumentIngestOptions): Promise<string[]> {
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

    const chunkIds: string[] = [];

    if (options.parentChild) {
      const parentChunkSize = options.parentChunkSize ?? 1000;
      const parentChunkOverlap = options.parentChunkOverlap ?? 200;

      const parentChildData = chunkTextParentChild(
        text, 
        parentChunkSize, 
        parentChunkOverlap, 
        chunkSize, 
        chunkOverlap
      );

      for (let pIdx = 0; pIdx < parentChildData.length; pIdx++) {
        const parentId = `${docId}-parent-${pIdx}`;
        const parentText = parentChildData[pIdx].parentText;

        // Store parent chunk without embedding (save resource: fill vector table with dummy zeros)
        const dummyEmbedding = new Array(this.provider.dimensions).fill(0);
        await this.addMemory({
          id: parentId,
          content: parentText,
          metadata: { _isParent: true, _documentId: docId },
          embedding: dummyEmbedding
        });

        // Store child chunks with their generated embeddings
        const childTexts = parentChildData[pIdx].childChunks;
        for (let cIdx = 0; cIdx < childTexts.length; cIdx++) {
          const childId = `${docId}-child-${pIdx}-${cIdx}`;
          const childMetadata = {
            ...customMetadata,
            _documentId: docId,
            _parentId: parentId
          };

          await this.addMemory({
            id: childId,
            content: childTexts[cIdx],
            metadata: childMetadata
          });
          chunkIds.push(childId);
        }
      }
    } else {
      // Standard Flat Chunking Ingestion
      const chunks = chunkText(text, chunkSize, chunkOverlap);
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
    }

    return chunkIds;
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
   * Returns whether the fast native sqlite-vec extension was successfully loaded.
   */
  isVectorExtensionLoaded(): boolean {
    return this.db.getIsVectorExtensionLoaded();
  }
}
