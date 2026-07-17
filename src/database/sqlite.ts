declare const Bun: any;
let Database: any;
if (typeof Bun !== 'undefined') {
  // @ts-ignore
  const { Database: BunDatabase } = await import("bun:sqlite");
  Database = class BunDatabaseWrapper {
    private db: any;
    constructor(dbPath: string) {
      this.db = new BunDatabase(dbPath, { create: true });
    }
    exec(sql: string) {
      this.db.run(sql);
    }
    pragma(sql: string) {
      this.db.run(`PRAGMA ${sql}`);
    }
    prepare(sql: string) {
      const query = this.db.query(sql);
      return {
        get(...args: any[]) {
          const res = query.get(...args);
          return res === null ? undefined : res;
        },
        run(...args: any[]) {
          return query.run(...args);
        },
        all(...args: any[]) {
          return query.all(...args);
        }
      };
    }
  };
} else {
  const { default: BetterSqlite3 } = await import("better-sqlite3");
  Database = BetterSqlite3;
}
import * as sqliteVec from 'sqlite-vec';
import { Memory, QueryResult, DatabaseAdapter } from '../types.js';

export class SQLiteAdapter implements DatabaseAdapter {
  private db: any;
  private isVectorExtensionLoaded = false;
  private tableName: string;
  private vecTableName: string;
  private ftsTableName: string;
  private cacheTableName: string;
  private vecCacheTableName: string;
  private dimensions: number;

  constructor(dbPath: string, collectionName = 'memories', dimensions: number) {
    this.tableName = `memories_${collectionName}`;
    this.vecTableName = `vec_memories_${collectionName}`;
    this.ftsTableName = `fts_${collectionName}`;
    this.cacheTableName = `cache_memories_${collectionName}`;
    this.vecCacheTableName = `vec_cache_memories_${collectionName}`;
    this.dimensions = dimensions;
    
    // Initialize database connection
    this.db = new Database(dbPath);
    
    // Set journal mode to WAL for better performance
    this.db.pragma('journal_mode = WAL');
    
    this.initialize();
  }

  private initialize(): void {
    // Attempt to load sqlite-vec extension
    try {
      sqliteVec.load(this.db);
      // Verify it is working
      const version = this.db.prepare('SELECT vec_version()').get() as any;
      if (version) {
        this.isVectorExtensionLoaded = true;
      }
    } catch (e) {
      // Fallback mode will be used
      this.isVectorExtensionLoaded = false;
    }

    if (this.isVectorExtensionLoaded) {
      // Vector Mode schema
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS ${this.tableName} (
          rowid INTEGER PRIMARY KEY AUTOINCREMENT,
          id TEXT UNIQUE NOT NULL,
          content TEXT NOT NULL,
          metadata TEXT NOT NULL, -- JSON stringified metadata
          created_at INTEGER NOT NULL
        );
        
        CREATE UNIQUE INDEX IF NOT EXISTS idx_${this.tableName}_id ON ${this.tableName}(id);
      `);

      // sqlite-vec virtual table
      this.db.exec(`
        CREATE VIRTUAL TABLE IF NOT EXISTS ${this.vecTableName} USING vec0(
          rowid INTEGER PRIMARY KEY, -- Maps directly to rowid in the metadata table
          embedding float[${this.dimensions}]
        );
      `);

      // Semantic Cache Tables (Vector Mode)
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS ${this.cacheTableName} (
          rowid INTEGER PRIMARY KEY AUTOINCREMENT,
          query_text TEXT UNIQUE NOT NULL,
          response_text TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          ttl_seconds INTEGER NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS ${this.vecCacheTableName} USING vec0(
          rowid INTEGER PRIMARY KEY,
          embedding float[${this.dimensions}]
        );
      `);
    } else {
      // Fallback Mode schema
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS ${this.tableName} (
          rowid INTEGER PRIMARY KEY AUTOINCREMENT,
          id TEXT UNIQUE NOT NULL,
          content TEXT NOT NULL,
          metadata TEXT NOT NULL, -- JSON stringified metadata
          embedding BLOB NOT NULL, -- Serialized Float32Array
          created_at INTEGER NOT NULL
        );
        
        CREATE UNIQUE INDEX IF NOT EXISTS idx_${this.tableName}_id ON ${this.tableName}(id);
      `);

      // Semantic Cache Table (Fallback Mode)
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS ${this.cacheTableName} (
          rowid INTEGER PRIMARY KEY AUTOINCREMENT,
          query_text TEXT UNIQUE NOT NULL,
          response_text TEXT NOT NULL,
          embedding BLOB NOT NULL, -- Serialized Float32Array
          created_at INTEGER NOT NULL,
          ttl_seconds INTEGER NOT NULL
        );
      `);

      // Register the cosine similarity function
      this.db.function('vec_distance_cosine', (a: unknown, b: unknown) => {
        if (!Buffer.isBuffer(a) || !Buffer.isBuffer(b)) {
          return 1.0;
        }
        
        // Map buffer to Float32Array
        const arrA = new Float32Array(a.buffer, a.byteOffset, a.byteLength / 4);
        const arrB = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
        
        let dotProduct = 0;
        const len = Math.min(arrA.length, arrB.length);
        for (let i = 0; i < len; i++) {
          dotProduct += arrA[i] * arrB[i];
        }
        
        return 1.0 - dotProduct;
      });
    }

    // Initialize FTS5 Virtual Table for Lexical Search
    this.db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS ${this.ftsTableName} USING fts5(
        content,
        content='${this.tableName}',
        content_rowid='rowid'
      );
    `);

    // Create database triggers to automatically synchronize FTS5 index with memories content
    this.db.exec(`
      CREATE TRIGGER IF NOT EXISTS trg_${this.tableName}_ai AFTER INSERT ON ${this.tableName} BEGIN
        INSERT INTO ${this.ftsTableName}(rowid, content) VALUES (new.rowid, new.content);
      END;

      CREATE TRIGGER IF NOT EXISTS trg_${this.tableName}_ad AFTER DELETE ON ${this.tableName} BEGIN
        INSERT INTO ${this.ftsTableName}(${this.ftsTableName}, rowid, content) VALUES('delete', old.rowid, old.content);
      END;

      CREATE TRIGGER IF NOT EXISTS trg_${this.tableName}_au AFTER UPDATE ON ${this.tableName} BEGIN
        INSERT INTO ${this.ftsTableName}(${this.ftsTableName}, rowid, content) VALUES('delete', old.rowid, old.content);
        INSERT INTO ${this.ftsTableName}(rowid, content) VALUES (new.rowid, new.content);
      END;
    `);

    // Auto-create index on metadata fields
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_${this.tableName}_meta_docid ON ${this.tableName}(json_extract(metadata, '$._documentId'));
      CREATE INDEX IF NOT EXISTS idx_${this.tableName}_meta_session ON ${this.tableName}(json_extract(metadata, '$.session'));
    `);
  }

  insert(id: string, content: string, embedding: number[], metadata: Record<string, any>): void {
    const floatArray = new Float32Array(embedding);
    const buffer = Buffer.from(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
    const metadataStr = JSON.stringify(metadata);
    const createdAt = Date.now();

    if (this.isVectorExtensionLoaded) {
      const insertMeta = this.db.prepare(`
        INSERT INTO ${this.tableName} (id, content, metadata, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          content = excluded.content,
          metadata = excluded.metadata,
          created_at = excluded.created_at
      `);
      
      const insertVec = this.db.prepare(`
        INSERT INTO ${this.vecTableName} (rowid, embedding)
        VALUES (?, ?)
      `);

      const runTx = this.db.transaction(() => {
        const existing = this.db.prepare(`SELECT rowid FROM ${this.tableName} WHERE id = ?`).get(id) as any;
        if (existing) {
          this.db.prepare(`DELETE FROM ${this.vecTableName} WHERE rowid = ?`).run(BigInt(existing.rowid));
        }

        const info = insertMeta.run(id, content, metadataStr, createdAt);
        const rowid = info.lastInsertRowid;
        
        insertVec.run(BigInt(rowid), buffer);
      });

      runTx();
    } else {
      const stmt = this.db.prepare(`
        INSERT INTO ${this.tableName} (id, content, metadata, embedding, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          content = excluded.content,
          metadata = excluded.metadata,
          embedding = excluded.embedding,
          created_at = excluded.created_at
      `);
      stmt.run(id, content, metadataStr, buffer, createdAt);
    }
  }

  query(queryEmbedding: number[], limit = 5, filter?: Record<string, any>): QueryResult[] {
    const floatArray = new Float32Array(queryEmbedding);
    const buffer = Buffer.from(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
    
    const whereClauses: string[] = [];
    const params: any[] = [];

    if (filter) {
      for (const [key, value] of Object.entries(filter)) {
        whereClauses.push(`json_extract(m.metadata, '$.${key}') = ?`);
        params.push(value);
      }
    }

    if (this.isVectorExtensionLoaded) {
      let sql = `
        SELECT 
          m.id, 
          m.content, 
          m.metadata, 
          m.created_at,
          v.distance
        FROM ${this.vecTableName} v
        JOIN ${this.tableName} m ON m.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
      `;
      
      const queryParams: any[] = [buffer, limit];
      
      if (whereClauses.length > 0) {
        sql += ` AND ${whereClauses.map(c => c.replace('m.', 'm.')).join(' AND ')}`;
        queryParams.push(...params);
      }
      
      const stmt = this.db.prepare(sql);
      const rows = stmt.all(...queryParams) as any[];

      return rows.map(row => ({
        memory: {
          id: row.id,
          content: row.content,
          metadata: JSON.parse(row.metadata),
          createdAt: row.created_at
        },
        distance: row.distance
      }));
    } else {
      let sql = `
        SELECT 
          id, 
          content, 
          metadata, 
          created_at,
          vec_distance_cosine(embedding, ?) as distance
        FROM ${this.tableName} m
        WHERE 1=1
      `;

      const queryParams: any[] = [buffer];

      if (whereClauses.length > 0) {
        sql += ` AND ${whereClauses.join(' AND ')}`;
        queryParams.push(...params);
      }

      sql += ` ORDER BY distance ASC LIMIT ?`;
      queryParams.push(limit);

      const stmt = this.db.prepare(sql);
      const rows = stmt.all(...queryParams) as any[];

      return rows.map(row => ({
        memory: {
          id: row.id,
          content: row.content,
          metadata: JSON.parse(row.metadata),
          createdAt: row.created_at
        },
        distance: row.distance
      }));
    }
  }

  queryLexical(queryText: string, limit = 5, filter?: Record<string, any>): QueryResult[] {
    const whereClauses: string[] = [];
    const params: any[] = [];

    if (filter) {
      for (const [key, value] of Object.entries(filter)) {
        whereClauses.push(`json_extract(m.metadata, '$.${key}') = ?`);
        params.push(value);
      }
    }

    let sql = `
      SELECT 
        m.id, 
        m.content, 
        m.metadata, 
        m.created_at,
        bm25(${this.ftsTableName}) as score
      FROM ${this.ftsTableName} f
      JOIN ${this.tableName} m ON m.rowid = f.rowid
      WHERE f.content MATCH ?
    `;

    const queryParams: any[] = [queryText];

    if (whereClauses.length > 0) {
      sql += ` AND ${whereClauses.map(c => c.replace('m.', 'm.')).join(' AND ')}`;
      queryParams.push(...params);
    }

    sql += ` ORDER BY score ASC LIMIT ?`;
    queryParams.push(limit);

    try {
      const stmt = this.db.prepare(sql);
      const rows = stmt.all(...queryParams) as any[];

      return rows.map(row => ({
        memory: {
          id: row.id,
          content: row.content,
          metadata: JSON.parse(row.metadata),
          createdAt: row.created_at
        },
        distance: row.score
      }));
    } catch (e: any) {
      return [];
    }
  }

  getById(id: string): Memory | null {
    if (this.isVectorExtensionLoaded) {
      const stmt = this.db.prepare(`SELECT rowid, id, content, metadata, created_at FROM ${this.tableName} WHERE id = ?`);
      const row = stmt.get(id) as any;
      if (!row) return null;
      
      let embedding: number[] | undefined;
      try {
        const vecRow = this.db.prepare(`SELECT embedding FROM ${this.vecTableName} WHERE rowid = ?`).get(BigInt(row.rowid)) as any;
        if (vecRow && vecRow.embedding) {
          const arr = new Float32Array(vecRow.embedding.buffer, vecRow.embedding.byteOffset, vecRow.embedding.byteLength / 4);
          embedding = Array.from(arr);
        }
      } catch (e) {}

      return {
        id: row.id,
        content: row.content,
        metadata: JSON.parse(row.metadata),
        createdAt: row.created_at,
        embedding
      };
    } else {
      const stmt = this.db.prepare(`SELECT id, content, metadata, embedding, created_at FROM ${this.tableName} WHERE id = ?`);
      const row = stmt.get(id) as any;
      if (!row) return null;
      
      let embedding: number[] | undefined;
      if (row.embedding) {
        const arr = new Float32Array(row.embedding.buffer, row.embedding.byteOffset, row.embedding.byteLength / 4);
        embedding = Array.from(arr);
      }

      return {
        id: row.id,
        content: row.content,
        metadata: JSON.parse(row.metadata),
        createdAt: row.created_at,
        embedding
      };
    }
  }

  getAll(filter?: Record<string, any>): Memory[] {
    const whereClauses: string[] = [];
    const params: any[] = [];

    if (filter) {
      for (const [key, value] of Object.entries(filter)) {
        whereClauses.push(`json_extract(metadata, '$.${key}') = ?`);
        params.push(value);
      }
    }

    let sql = `SELECT id, content, metadata, created_at FROM ${this.tableName}`;
    if (whereClauses.length > 0) {
      sql += ` WHERE ${whereClauses.join(' AND ')}`;
    }

    const stmt = this.db.prepare(sql);
    const rows = stmt.all(...params) as any[];

    return rows.map(row => ({
      id: row.id,
      content: row.content,
      metadata: JSON.parse(row.metadata),
      createdAt: row.created_at
    }));
  }

  delete(id: string): void {
    if (this.isVectorExtensionLoaded) {
      const getRowid = this.db.prepare(`SELECT rowid FROM ${this.tableName} WHERE id = ?`);
      const deleteMeta = this.db.prepare(`DELETE FROM ${this.tableName} WHERE id = ?`);
      const deleteVec = this.db.prepare(`DELETE FROM ${this.vecTableName} WHERE rowid = ?`);
      
      const runTx = this.db.transaction(() => {
        const row = getRowid.get(id) as any;
        if (row) {
          deleteVec.run(BigInt(row.rowid));
          deleteMeta.run(id);
        }
      });
      runTx();
    } else {
      const stmt = this.db.prepare(`DELETE FROM ${this.tableName} WHERE id = ?`);
      stmt.run(id);
    }
  }

  // Semantic Cache Implementation
  setCache(queryText: string, responseText: string, embedding: number[], ttlSeconds: number): void {
    const floatArray = new Float32Array(embedding);
    const buffer = Buffer.from(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
    const createdAt = Math.floor(Date.now() / 1000);

    if (this.isVectorExtensionLoaded) {
      const insertMeta = this.db.prepare(`
        INSERT INTO ${this.cacheTableName} (query_text, response_text, created_at, ttl_seconds)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(query_text) DO UPDATE SET
          response_text = excluded.response_text,
          created_at = excluded.created_at,
          ttl_seconds = excluded.ttl_seconds
      `);
      
      const insertVec = this.db.prepare(`
        INSERT OR REPLACE INTO ${this.vecCacheTableName} (rowid, embedding)
        VALUES (?, ?)
      `);

      const runTx = this.db.transaction(() => {
        const existing = this.db.prepare(`SELECT rowid FROM ${this.cacheTableName} WHERE query_text = ?`).get(queryText) as any;
        if (existing) {
          this.db.prepare(`DELETE FROM ${this.vecCacheTableName} WHERE rowid = ?`).run(BigInt(existing.rowid));
        }

        const info = insertMeta.run(queryText, responseText, createdAt, ttlSeconds);
        insertVec.run(BigInt(info.lastInsertRowid), buffer);
      });
      runTx();
    } else {
      const stmt = this.db.prepare(`
        INSERT INTO ${this.cacheTableName} (query_text, response_text, embedding, created_at, ttl_seconds)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(query_text) DO UPDATE SET
          response_text = excluded.response_text,
          embedding = excluded.embedding,
          created_at = excluded.created_at,
          ttl_seconds = excluded.ttl_seconds
      `);
      stmt.run(queryText, responseText, buffer, createdAt, ttlSeconds);
    }
  }

  getCache(queryEmbedding: number[], threshold: number): string | null {
    const floatArray = new Float32Array(queryEmbedding);
    const buffer = Buffer.from(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
    const now = Math.floor(Date.now() / 1000);

    if (this.isVectorExtensionLoaded) {
      const stmt = this.db.prepare(`
        SELECT 
          c.response_text,
          c.created_at,
          c.ttl_seconds,
          v.distance
        FROM ${this.vecCacheTableName} v
        JOIN ${this.cacheTableName} c ON c.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = 1
      `);
      
      try {
        const row = stmt.get(buffer) as any;
        if (row && row.distance <= threshold && (row.created_at + row.ttl_seconds > now)) {
          return row.response_text;
        }
      } catch (e) {}
      return null;
    } else {
      const stmt = this.db.prepare(`
        SELECT 
          response_text,
          created_at,
          ttl_seconds,
          vec_distance_cosine(embedding, ?) as distance
        FROM ${this.cacheTableName}
        WHERE (created_at + ttl_seconds > ?)
        ORDER BY distance ASC
        LIMIT 1
      `);
      
      try {
        const row = stmt.get(buffer, now) as any;
        if (row && row.distance <= threshold) {
          return row.response_text;
        }
      } catch (e) {}
      return null;
    }
  }

  clearCache(): void {
    const runTx = this.db.transaction(() => {
      if (this.isVectorExtensionLoaded) {
        this.db.exec(`DELETE FROM ${this.vecCacheTableName}`);
      }
      this.db.exec(`DELETE FROM ${this.cacheTableName}`);
    });
    runTx();
  }

  clear(): void {
    const runTx = this.db.transaction(() => {
      // 1. Delete memories (triggers FTS sync)
      this.db.exec(`DELETE FROM ${this.tableName}`);
      
      // 2. Clear vectors
      if (this.isVectorExtensionLoaded) {
        this.db.exec(`DELETE FROM ${this.vecTableName}`);
      }
      
      // 3. Clear cache
      this.clearCache();
      
      // 4. Reset sequences
      this.db.exec(`DELETE FROM sqlite_sequence WHERE name='${this.tableName}'`);
      this.db.exec(`DELETE FROM sqlite_sequence WHERE name='${this.cacheTableName}'`);
    });
    runTx();
  }

  close(): void {
    this.db.close();
  }

  getIsVectorExtensionLoaded(): boolean {
    return this.isVectorExtensionLoaded;
  }
}
