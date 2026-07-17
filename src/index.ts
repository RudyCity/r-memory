import { randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { SQLiteAdapter } from './database/sqlite.js';
import type { 
  Memory, 
  QueryResult, 
  EmbeddingProvider, 
  RMemoryConfig, 
  QueryOptions,
  DocumentIngestOptions,
  ConsolidateOptions
} from './types.js';
import { 
  extractTextFromPDF, 
  extractTextFromDocx, 
  extractTextFromXlsx, 
  chunkText,
  chunkTextParentChild
} from './utils/document.js';

export type { 
  Memory, 
  QueryResult, 
  EmbeddingProvider, 
  RMemoryConfig, 
  QueryOptions,
  DocumentIngestOptions,
  ConsolidateOptions
};

export { 
  LocalTextEmbeddingProvider, 
  LocalCLIPEmbeddingProvider 
} from './embeddings/local.js';

import { OpenAIEmbeddingProvider } from './embeddings/openai.js';
import type { OpenAIEmbeddingConfig } from './embeddings/openai.js';

export { OpenAIEmbeddingProvider };
export type { OpenAIEmbeddingConfig };

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
   * Resolves parent-child references dynamically and applies optional time-decay recency weights.
   * 
   * @param options Query configuration options
   * @returns Array of query results containing memory details and similarity score
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

    // 2. Apply optional Time Decay (Recency Weighting)
    if (options.decayFactor !== undefined && options.decayFactor > 0) {
      const now = Date.now();
      rawResults.forEach(res => {
        // Compute age in hours
        const ageHours = (now - res.memory.createdAt) / (1000 * 60 * 60);
        const decayMultiplier = Math.exp(-options.decayFactor! * ageHours);
        
        // Decay similarity score (where similarity is 1.0 - distance)
        const similarity = 1.0 - res.distance;
        const decayedSimilarity = similarity * decayMultiplier;
        
        // Convert back to cosine distance
        res.distance = 1.0 - decayedSimilarity;
      });

      // Sort by decayed distance ascending (closer matches first)
      rawResults.sort((a, b) => a.distance - b.distance);
    }

    // 3. Resolve Parent-Child References
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

    return resolvedResults.slice(0, limit);
  }

  /**
   * Performs hybrid search combining semantic (Vector) and lexical (FTS5) queries,
   * merged using Reciprocal Rank Fusion (RRF).
   */
  private async queryHybrid(queryText: string, limit: number, filter?: Record<string, any>): Promise<QueryResult[]> {
    const candidateLimit = limit * 2;

    const queryEmbedding = await this.provider.embedText(queryText, 'query');
    const semanticResults = this.db.query(queryEmbedding, candidateLimit, filter);

    const lexicalResults = this.db.queryLexical(queryText, candidateLimit, filter);

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
        rrfScores[id] = { memory: res.memory, distance: 1.0 };
      }
      rrfScores[id].lexicalRank = index + 1;
    });

    const k = 60; 
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

    combined.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

    return combined;
  }

  /**
   * Extracts text, chunks it, and adds a whole document to the database.
   * Performs parallel batch embedding generation to accelerate ingestion speed.
   * 
   * @param options Document ingestion configuration
   * @returns Array of generated memory IDs for each chunk
   */
  async addDocument(options: DocumentIngestOptions): Promise<string[]> {
    const docId = options.id || randomUUID();
    const customMetadata = options.metadata || {};
    const chunkSize = options.chunkSize ?? 500;
    const chunkOverlap = options.chunkOverlap ?? Math.min(100, Math.floor(chunkSize * 0.2));

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
      const parentChunkOverlap = options.parentChunkOverlap ?? Math.min(200, Math.floor(parentChunkSize * 0.2));

      const parentChildData = chunkTextParentChild(
        text, 
        parentChunkSize, 
        parentChunkOverlap, 
        chunkSize, 
        chunkOverlap
      );

      // Collect all child texts across parents to generate batch embeddings in one call
      const allChildTexts: string[] = [];
      parentChildData.forEach(p => allChildTexts.push(...p.childChunks));
      
      const allChildEmbeddings = await this.provider.embedTexts(allChildTexts, 'passage');

      let childEmbedIndex = 0;
      for (let pIdx = 0; pIdx < parentChildData.length; pIdx++) {
        const parentId = `${docId}-parent-${pIdx}`;
        const parentText = parentChildData[pIdx].parentText;

        // Store parent chunk without embedding (saves space)
        const dummyEmbedding = new Array(this.provider.dimensions).fill(0);
        await this.addMemory({
          id: parentId,
          content: parentText,
          metadata: { _isParent: true, _documentId: docId },
          embedding: dummyEmbedding
        });

        // Store child chunks with pre-computed batch embeddings
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
            metadata: childMetadata,
            embedding: allChildEmbeddings[childEmbedIndex++]
          });
          chunkIds.push(childId);
        }
      }
    } else {
      // Flat Ingestion with Batch Embedding
      const chunks = chunkText(text, chunkSize, chunkOverlap);
      const embeddings = await this.provider.embedTexts(chunks, 'passage');

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
          metadata: chunkMetadata,
          embedding: embeddings[i]
        });
        chunkIds.push(chunkId);
      }
    }

    return chunkIds;
  }

  /**
   * Retrieves a cached LLM response for a query based on semantic similarity.
   */
  async getSemanticCache(queryText: string, threshold = 0.05): Promise<string | null> {
    const queryEmbedding = await this.provider.embedText(queryText, 'query');
    return this.db.getCache(queryEmbedding, threshold);
  }

  /**
   * Caches an LLM response for a query.
   */
  async setSemanticCache(queryText: string, responseText: string, ttlSeconds = 3600): Promise<void> {
    const queryEmbedding = await this.provider.embedText(queryText, 'query');
    this.db.setCache(queryText, responseText, queryEmbedding, ttlSeconds);
  }

  /**
   * Clears the semantic cache.
   */
  clearCache(): void {
    this.db.clearCache();
  }

  /**
   * Consolidates older, highly similar memories by clustering them and merging
   * using a developer-provided summarizer LLM callback.
   */
  async consolidate(
    summarizer: (texts: string[]) => Promise<string>,
    options: ConsolidateOptions = {}
  ): Promise<void> {
    const threshold = options.threshold ?? 0.15;
    const filter = options.filter;

    // Retrieve all memories in the collection (without vector query)
    const allMemories = this.db.getAll(filter);

    if (allMemories.length <= 1) return;

    const visited = new Set<string>();

    for (let i = 0; i < allMemories.length; i++) {
      const current = allMemories[i];
      if (visited.has(current.id)) continue;
      if (current.metadata._isParent) continue;

      // Load full memory record containing its embedding float array
      const fullCurrent = this.db.getById(current.id);
      if (!fullCurrent || !fullCurrent.embedding) continue;

      // Query database for nearest neighbors of current memory
      const matches = this.db.query(fullCurrent.embedding, 30, filter);
      const cluster: Memory[] = [];

      for (const match of matches) {
        if (visited.has(match.memory.id)) continue;
        if (match.memory.metadata._isParent) continue;

        if (match.distance <= threshold) {
          cluster.push(match.memory);
        }
      }

      // Summarize and consolidate if there's overlap (> 1 similar memory)
      if (cluster.length > 1) {
        const clusterTexts = cluster.map(m => m.content);
        
        try {
          const summary = await summarizer(clusterTexts);
          
          // Delete merged child memories
          const clusterIds = cluster.map(m => m.id);
          clusterIds.forEach(id => {
            visited.add(id);
            this.db.delete(id);
          });

          // Insert summarized memory back
          const consolidatedMetadata = {
            _isConsolidated: true,
            _consolidatedCount: cluster.length,
            _sourceIds: clusterIds
          };

          await this.addMemory({
            content: summary,
            metadata: consolidatedMetadata
          });
        } catch (e) {
          console.error('Error consolidating cluster:', e);
        }
      }
    }
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
