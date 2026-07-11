import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { Memory, QueryResult, DatabaseAdapter } from '../types.js';

export class SQLiteAdapter implements DatabaseAdapter {
  private db: Database.Database;
  private isVectorExtensionLoaded = false;
  private tableName: string;
  private vecTableName: string;
  private ftsTableName: string;
  private dimensions: number;

  constructor(dbPath: string, collectionName = 'memories', dimensions: number) {
    this.tableName = `memories_${collectionName}`;
    this.vecTableName = `vec_memories_${collectionName}`;
    this.ftsTableName = `fts_${collectionName}`;
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
      // Note: we use float[dimensions] as expected by sqlite-vec
      this.db.exec(`
        CREATE VIRTUAL TABLE IF NOT EXISTS ${this.vecTableName} USING vec0(
          rowid INTEGER PRIMARY KEY, -- Maps directly to rowid in the metadata table
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
        
        // Since embeddings are pre-normalized, cosine distance is 1.0 - dot product
        return 1.0 - dotProduct;
      });
    }

    // Initialize FTS5 Virtual Table for Lexical Search (in BOTH vector and fallback modes)
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

    // Auto-create index on metadata fields if common fields are present
    // This accelerates json_extract queries
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_${this.tableName}_meta_docid ON ${this.tableName}(json_extract(metadata, '$._documentId'));
      CREATE INDEX IF NOT EXISTS idx_${this.tableName}_meta_session ON ${this.tableName}(json_extract(metadata, '$.session'));
    `);
  }

  insert(id: string, content: string, embedding: number[], metadata: Record<string, any>): void {
    // Convert embedding to Float32Array buffer for binary BLOB storage
    const floatArray = new Float32Array(embedding);
    const buffer = Buffer.from(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
    const metadataStr = JSON.stringify(metadata);
    const createdAt = Date.now();

    if (this.isVectorExtensionLoaded) {
      // Use SQLite transactions for safe compound inserts
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
        // First delete if conflicts exist in virtual table
        // As INSERT OR REPLACE is buggy in sqlite-vec, we fetch rowid first
        const existing = this.db.prepare(`SELECT rowid FROM ${this.tableName} WHERE id = ?`).get(id) as any;
        if (existing) {
          this.db.prepare(`DELETE FROM ${this.vecTableName} WHERE rowid = ?`).run(BigInt(existing.rowid));
          // Note: FTS triggers handle synchronization for updates/deletes in memories table
        }

        const info = insertMeta.run(id, content, metadataStr, createdAt);
        const rowid = info.lastInsertRowid;
        
        insertVec.run(BigInt(rowid), buffer);
      });

      runTx();
    } else {
      // Fallback Mode insert
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
    
    // Build query where clause from filter metadata
    const whereClauses: string[] = [];
    const params: any[] = [];

    if (filter) {
      for (const [key, value] of Object.entries(filter)) {
        whereClauses.push(`json_extract(m.metadata, '$.${key}') = ?`);
        params.push(value);
      }
    }

    if (this.isVectorExtensionLoaded) {
      // Vector Mode Query
      // We MATCH query embedding first, then join with metadata table
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
      // Fallback Mode Query
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

    // Lexical Search using FTS5 MATCH
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

    // bm25 score (smaller/more negative is better)
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
        // Normalize: BM25 score could be anything, but distance is lower is better
        // For RRF rank fusion, only ordering matters, so we return the raw score
        distance: row.score
      }));
    } catch (e: any) {
      // If FTS query fails (e.g. syntax issue in MATCH), return empty
      return [];
    }
  }

  getById(id: string): Memory | null {
    const stmt = this.db.prepare(`SELECT id, content, metadata, created_at FROM ${this.tableName} WHERE id = ?`);
    const row = stmt.get(id) as any;
    if (!row) return null;
    return {
      id: row.id,
      content: row.content,
      metadata: JSON.parse(row.metadata),
      createdAt: row.created_at
    };
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

  clear(): void {
    const runTx = this.db.transaction(() => {
      // 1. Delete from memories table (triggers will automatically delete from FTS5!)
      this.db.exec(`DELETE FROM ${this.tableName}`);
      
      // 2. If vector mode, delete from vector table
      if (this.isVectorExtensionLoaded) {
        this.db.exec(`DELETE FROM ${this.vecTableName}`);
      }
      
      // 3. Reset autoincrement sequence
      this.db.exec(`DELETE FROM sqlite_sequence WHERE name='${this.tableName}'`);
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
