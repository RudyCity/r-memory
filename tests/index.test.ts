import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { RMemory } from '../src/index.js';
import type { EmbeddingProvider } from '../src/types.js';
import { existsSync, unlinkSync } from 'node:fs';

// A simple mock embedding provider with 3 dimensions
class MockEmbeddingProvider implements EmbeddingProvider {
  readonly dimensions = 3;

  async embedText(text: string, context?: 'query' | 'passage'): Promise<number[]> {
    if (text.includes('Budi')) {
      return [1, 0, 0]; // Vector A
    } else if (text.includes('makanan')) {
      return [0, 1, 0]; // Vector B
    }
    return [0, 0, 1]; // Default
  }

  async embedTexts(texts: string[], context?: 'query' | 'passage'): Promise<number[][]> {
    return Promise.all(texts.map(t => this.embedText(t, context)));
  }

  async embedImage(image: string | Buffer | Uint8Array): Promise<number[]> {
    return [0.707, 0.707, 0]; // Mix Vector
  }
}

describe('RMemory Core Functionality', () => {
  let memory: RMemory;
  const dbPath = 'test_memory_db.sqlite';

  beforeEach(() => {
    // Clean up old DB file
    if (existsSync(dbPath)) {
      try {
        unlinkSync(dbPath);
      } catch (e) {}
    }

    memory = new RMemory({
      dbPath: dbPath,
      collectionName: 'test_collection',
      embeddingProvider: new MockEmbeddingProvider()
    });
  });

  afterEach(() => {
    memory.close();
    if (existsSync(dbPath)) {
      try {
        unlinkSync(dbPath);
      } catch (e) {}
    }
  });

  it('should initialize and report database vector capabilities', () => {
    const isLoaded = memory.isVectorExtensionLoaded();
    console.log(`Test environment vector extension loaded: ${isLoaded}`);
    expect(typeof isLoaded).toBe('boolean');
  });

  it('should support batch text embedding', async () => {
    const texts = ['Halo Budi', 'makanan anjing', 'teks biasa'];
    const embeddings = await memory['provider'].embedTexts(texts);
    expect(embeddings.length).toBe(3);
    expect(embeddings[0]).toEqual([1, 0, 0]);
    expect(embeddings[1]).toEqual([0, 1, 0]);
    expect(embeddings[2]).toEqual([0, 0, 1]);
  });

  it('should get and set semantic cache', async () => {
    const query = 'Siapa pencipta r-memory?';
    const response = 'Penciptanya adalah tim pengembang.';
    
    // Get cache (should be null initially)
    let cached = await memory.getSemanticCache(query);
    expect(cached).toBeNull();

    // Set cache
    await memory.setSemanticCache(query, response, 10); // TTL 10 seconds

    // Get cache (should return the response)
    cached = await memory.getSemanticCache(query);
    expect(cached).toBe(response);

    // Get cache with query having slight keyword difference
    cached = await memory.getSemanticCache('Ada pencipta r-memory?');
    expect(cached).toBe(response);
  });

  it('should apply time decay recency weighting', async () => {
    memory.clear();

    // Insert two memories with same content/vector, but different creation times
    const id1 = await memory.addMemory({ id: 'dec-1', content: 'Info Budi lama' });
    const id2 = await memory.addMemory({ id: 'dec-2', content: 'Info Budi baru' });

    // Manually modify createdAt of dec-1 in SQLite to be 24 hours ago
    const ageMs = 24 * 60 * 60 * 1000;
    const db = memory['db']['db'];
    db.prepare(`UPDATE ${memory['db']['tableName']} SET created_at = ? WHERE id = ?`).run(Date.now() - ageMs, 'dec-1');

    // With decayFactor: 0.1 (per hour decay), the older memory 'dec-1' should rank second!
    const results = await memory.query({
      query: 'Budi',
      decayFactor: 0.1,
      limit: 2
    });

    expect(results.length).toBe(2);
    expect(results[0].memory.id).toBe('dec-2'); // Newest memory should rank first!
    expect(results[1].memory.id).toBe('dec-1'); // Oldest decayed memory should rank second
  });

  it('should consolidate overlapping memories using a summarizer callback', async () => {
    memory.clear();

    // Add two overlapping memories about Budi
    await memory.addMemory({ content: 'Budi suka makan rendang.' });
    await memory.addMemory({ content: 'Budi suka makan nasi padang.' });

    let summarizerCalled = false;
    const summarizer = async (texts: string[]) => {
      summarizerCalled = true;
      expect(texts.length).toBe(2);
      return 'Budi menyukai masakan khas Padang.';
    };

    await memory.consolidate(summarizer, { threshold: 0.05 });

    expect(summarizerCalled).toBe(true);

    // Search for consolidated memory
    const results = await memory.query({
      query: 'Budi',
      limit: 1
    });

    expect(results.length).toBe(1);
    expect(results[0].memory.content).toBe('Budi menyukai masakan khas Padang.');
    expect(results[0].memory.metadata._isConsolidated).toBe(true);
  });

  it('should add memory and query it semantically', async () => {
    const id = await memory.addMemory({
      content: 'Kucing adalah hewan peliharaan yang lucu.',
      metadata: { category: 'animal' }
    });

    expect(id).toBeDefined();

    const results = await memory.query({
      query: 'makanan', // Matches Vector B [0, 1, 0] (no match, distance will be higher)
      limit: 1
    });

    expect(results.length).toBe(1);
    expect(results[0].memory.id).toBe(id);
    expect(results[0].distance).toBeGreaterThanOrEqual(0);
  });

  it('should filter query results by metadata', async () => {
    await memory.addMemory({
      id: 'mem-1',
      content: 'Budi bermain sepak bola.',
      metadata: { sport: 'football', active: true }
    });

    await memory.addMemory({
      id: 'mem-2',
      content: 'Andi bermain bulu tangkis.',
      metadata: { sport: 'badminton', active: true }
    });

    const results = await memory.query({
      query: 'Budi',
      limit: 5,
      filter: { sport: 'badminton' }
    });

    expect(results.length).toBe(1);
    expect(results[0].memory.id).toBe('mem-2');
  });

  it('should delete memories by ID', async () => {
    const id = await memory.addMemory({
      content: 'Budi makan sate kambing.'
    });

    let results = await memory.query({ query: 'Budi', limit: 1 });
    expect(results.length).toBe(1);

    memory.delete(id);

    results = await memory.query({ query: 'Budi', limit: 1 });
    expect(results.length).toBe(0);
  });

  it('should clear all memories', async () => {
    await memory.addMemory({ content: 'Budi suka minum kopi.' });
    await memory.addMemory({ content: 'Andi suka minum teh.' });

    let results = await memory.query({ query: 'Budi', limit: 5 });
    expect(results.length).toBe(2);

    memory.clear();

    results = await memory.query({ query: 'Budi', limit: 5 });
    expect(results.length).toBe(0);
  });

  it('should ingest documents and split them into flat chunks', async () => {
    const txtContent = 'Ini adalah baris pertama dari dokumen uji. Ini adalah baris kedua yang menjelaskan detail. Dan ini baris ketiga untuk melengkapi teks.';
    const buffer = Buffer.from(txtContent, 'utf-8');

    const chunkIds = await memory.addDocument({
      pathOrBuffer: buffer,
      type: 'txt',
      chunkSize: 50,
      chunkOverlap: 10,
      metadata: { docSource: 'test-doc' }
    });

    expect(chunkIds.length).toBeGreaterThan(1);
    expect(chunkIds[0]).toContain('-chunk-0');

    // Query a stored chunk
    const results = await memory.query({
      query: 'dokumen uji',
      limit: 20,
      filter: { docSource: 'test-doc' }
    });

    expect(results.length).toBe(chunkIds.length);
    expect(results[0].memory.metadata._documentId).toBeDefined();
    const hasChunkZero = results.some(r => r.memory.metadata._chunkIndex === 0);
    expect(hasChunkZero).toBe(true);
  });

  it('should run hybrid search (FTS5 + Vector + RRF)', async () => {
    // Add memories with distinct keywords
    await memory.addMemory({ id: 'doc-hy-1', content: 'Kunci pertama tentang sistem pertahanan.' });
    await memory.addMemory({ id: 'doc-hy-2', content: 'Makanan kesukaan kucing adalah ikan segar.' });

    // Run hybrid search
    const results = await memory.query({
      query: 'sistem pertahanan',
      hybrid: true,
      limit: 1
    });

    expect(results.length).toBe(1);
    expect(results[0].memory.content).toContain('sistem pertahanan');
    expect(results[0].score).toBeDefined(); // RRF score should be set
  });

  it('should ingest documents using Parent-Child hierarchy and resolve parent context on query', async () => {
    const parentText = 'Ini adalah paragraf panjang tentang Budi yang bertindak sebagai Parent.';
    const buffer = Buffer.from(parentText, 'utf-8');

    const chunkIds = await memory.addDocument({
      pathOrBuffer: buffer,
      type: 'txt',
      parentChild: true,
      parentChunkSize: 60,
      parentChunkOverlap: 10,
      chunkSize: 20,
      chunkOverlap: 5,
      metadata: { hierarchy: 'parent-child-test' }
    });

    // It should generate multiple child chunks
    expect(chunkIds.length).toBeGreaterThan(1);
    expect(chunkIds[0]).toContain('-child-');

    // Query for a child chunk using 'Budi' (mapped to mock vector [1, 0, 0])
    const results = await memory.query({
      query: 'Budi',
      limit: 1,
      filter: { hierarchy: 'parent-child-test' }
    });

    expect(results.length).toBe(1);
    // The returned content should be the PARENT chunk text containing Budi
    expect(results[0].memory.content).toContain('paragraf panjang');
    expect(results[0].memory.metadata._parentId).toBeDefined();
    expect(results[0].memory.metadata._childContent).toBeDefined();
  });
});
