import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { RMemory } from '../src/index.js';
import { EmbeddingProvider } from '../src/types.js';
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
    expect(memory).toBeDefined();
    // It will be true or false depending on whether sqlite-vec extension is compiled/loaded
    const isVecLoaded = memory.isVectorExtensionLoaded();
    console.log(`Test environment vector extension loaded: ${isVecLoaded}`);
  });

  it('should store and retrieve a text memory', async () => {
    const id = await memory.addMemory({
      content: 'Nama saya Budi',
      metadata: { key: 'val1' }
    });

    expect(id).toBeDefined();
    expect(typeof id).toBe('string');

    const results = await memory.query({
      query: 'Siapa nama Budi?',
      limit: 1
    });

    expect(results.length).toBe(1);
    expect(results[0].memory.content).toBe('Nama saya Budi');
    expect(results[0].memory.metadata.key).toBe('val1');
    expect(results[0].distance).toBeLessThan(0.1); // Cosine distance should be near 0 (perfect match)
  });

  it('should filter memories by metadata', async () => {
    await memory.addMemory({
      id: 'id-1',
      content: 'Budi suka rendang',
      metadata: { session: 'session-A' }
    });

    await memory.addMemory({
      id: 'id-2',
      content: 'Budi suka sate',
      metadata: { session: 'session-B' }
    });

    // Query without filter - should get both
    const allResults = await memory.query({
      query: 'Budi suka apa?',
      limit: 10
    });
    expect(allResults.length).toBe(2);

    // Query with filter
    const filteredResults = await memory.query({
      query: 'Budi suka apa?',
      limit: 10,
      filter: { session: 'session-B' }
    });

    expect(filteredResults.length).toBe(1);
    expect(filteredResults[0].memory.id).toBe('id-2');
    expect(filteredResults[0].memory.content).toBe('Budi suka sate');
  });

  it('should delete a memory', async () => {
    await memory.addMemory({
      id: 'delete-me',
      content: 'Memori tentang Budi yang akan dihapus'
    });

    let results = await memory.query({ query: 'Budi', limit: 1 });
    expect(results.length).toBe(1);

    memory.delete('delete-me');

    results = await memory.query({ query: 'Budi', limit: 1 });
    expect(results.length).toBe(0);
  });

  it('should clear all memories', async () => {
    await memory.addMemory({ content: 'Memori Budi 1' });
    await memory.addMemory({ content: 'Memori Budi 2' });

    let results = await memory.query({ query: 'Budi', limit: 10 });
    expect(results.length).toBe(2);

    memory.clear();

    results = await memory.query({ query: 'Budi', limit: 10 });
    expect(results.length).toBe(0);
  });

  it('should parse, chunk, and store a document (TXT)', async () => {
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
});
