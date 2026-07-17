# R-Memory 🧠

A high-performance, light-weight TypeScript memory agent library. It equips AI agents with long-term and short-term memory capabilities using a vector-enabled SQLite database. R-Memory supports both local offline models and OpenAI-compatible APIs for text and multimodal (image) embeddings, alongside built-in document ingestion and semantic chunking.

---

## ✨ Features

*   **⚡ Dual-Mode SQLite Vector Database**:
    *   *Native Mode*: Natively loads the `sqlite-vec` extension and utilizes high-performance virtual `vec0` tables.
    *   *Portable Fallback Mode*: Automatically registers a custom JavaScript-based `vec_distance_cosine` function inside SQLite if the extension is not available. 100% portable across Windows, macOS, and Linux out-of-the-box.
*   **🌐 Flexible Embeddings (Local & OpenAI-Compatible)**:
    *   *Local Text*: Runs the state-of-the-art `Xenova/multilingual-e5-small` model on ONNX Runtime (384 dimensions) for excellent Bahasa Indonesia & English support. Supports **8-bit integer quantization (`dtype: 'q8'`)** and **WebGPU hardware acceleration**.
    *   *Local Multimodal (CLIP)*: Runs the `Xenova/clip-vit-base-patch32` model (512 dimensions) to index images and text in the same vector space, enabling cross-modal searches. Supports **JPEG, PNG, WebP** formats.
    *   *OpenAI-Compatible*: Connects to any OpenAI-compatible API (standard OpenAI, Ollama, LM Studio) for text embeddings.
*   **📐 Matryoshka & Dynamic Dimensions**:
    *   Auto-detects model dimensions and dynamically adjusts SQLite schemas.
    *   Supports Matryoshka Representation Learning (dimension truncation + L2 normalization) for models like `text-embedding-3-small` (e.g. cutting 1536 down to 256 or 512 dimensions).
*   **⚡ Parallel Batch Ingestion**:
    *   Generates embeddings in parallel batch operations using `embedTexts()` to speed up document loader ingestion by up to **4x** on both local models and OpenAI endpoints.
*   **🔎 Hybrid Search (FTS5 + Vector + RRF)**:
    *   Integrates SQLite FTS5 Full-Text Search inside the database synchronized in real-time via automatic SQL triggers.
    *   Uses **Reciprocal Rank Fusion (RRF)** to merge lexical keyword matches and semantic vector matches into a unified ranking list for high-precision retrieval.
*   **⏳ Time-Decay (Recency Weighting)**:
    *   Factor memory age (`decayFactor` per hour) into search results to prioritize fresh facts and context over older information.
*   **📑 Document & Hierarchical Parent-Child Ingestion**:
    *   Direct ingestion of **PDF, Microsoft Word (`.docx`), Excel (`.xlsx`, `.xls`, `.csv`), and plain text (`.txt`)** files.
    *   Preserves spreadsheet table structures by parsing worksheets into CSV strings.
    *   Slides a window parser over extracted text to produce semantic chunks with smart **word boundary alignment** (never cuts words in half).
    *   **Parent-Child RAG Support**: Split documents into large Parent chunks (broad context) and small Child chunks (highly focused embeddings). Searches query the small Child chunks but automatically return the larger Parent chunk to give the LLM full context.
*   **💾 Semantic Cache**:
    *   Store query and response pairs in SQLite vector cache tables with custom similarity threshold matching and TTL expirations to bypass expensive LLM latency and API costs.
*   **🤝 Memory Consolidation (Clustering)**:
    *   Cluster overlapping thoughts using cosine similarity metrics, merge them using a developer-provided summarizer LLM callback, and delete source redundant entries to prevent database bloating.
*   **🔍 Rich Metadata Filtering**:
    *   Store arbitrary metadata alongside memories and query using SQLite's native JSON operations (`json_extract`). Generates indices on metadata fields (`_documentId`, etc.) automatically to accelerate queries.

---

## 📦 Installation

Install the package and its peer dependencies using Bun:

```bash
bun add r-memory better-sqlite3 sqlite-vec @huggingface/transformers sharp pdf-parse mammoth xlsx
```

> **Note**: `bun:sqlite` is automatically used as the database engine under the Bun runtime. Native vector capabilities are enabled by loading the `sqlite-vec` extension via `loadExtension()`. Under Node.js, `better-sqlite3` is used instead. `sharp` is required for local image decoding. `pdf-parse`, `mammoth`, and `xlsx` are required only if you use the document ingestion feature.

---

## 🚀 Quick Start

### 1. Local Text Memory (Quantized & Accelerated)

```typescript
import { RMemory, LocalTextEmbeddingProvider } from 'r-memory';

const memory = new RMemory({
  dbPath: 'memories.db',
  collectionName: 'agent_facts',
  embeddingProvider: new LocalTextEmbeddingProvider({
    dtype: 'q8',       // 8-bit quantization (saves up to 75% RAM!)
    device: 'webgpu'   // Use 'webgpu' for GPU acceleration or 'cpu' (default)
  })
});
```

---

### 2. Time-Decay Recency Weighting

```typescript
// Query budget data, penalizing older memories by 0.05 per hour age
const results = await memory.query({
  query: 'Q4 marketing budget allocation',
  decayFactor: 0.05, // Exp decay per hour
  limit: 1
});
```

---

### 3. Semantic Cache

Store LLM responses and retrieve them semantically for similar queries within a matching threshold.

```typescript
const query = 'What is the operational code for Project Nebula?';

// 1. Try to read from cache (threshold 0.15 matches slight paraphrasing)
let response = await memory.getSemanticCache(query, 0.15);

if (!response) {
  // 2. Call LLM if cache miss
  response = await callYourLLM(query);
  
  // 3. Store in cache (TTL 3600 seconds)
  await memory.setSemanticCache(query, response, 3600);
}
```

---

### 4. Memory Consolidation (Clustering)

Merge overlapping memories periodically using a custom summarizer callback.

```typescript
const summarizerCallback = async (texts: string[]) => {
  // Call your LLM to merge multiple overlapping notes
  return await callLLMSummarize(`Merge these agent observations into a single note: ${texts.join('\n')}`);
};

// Cluster memories with distance <= 0.25, summarize, and clean up source records
await memory.consolidate(summarizerCallback, { threshold: 0.25 });
```

---

### 5. Parent-Child Hierarchical Ingestion

```typescript
// Ingest a PDF file with Parent-Child structure
const chunkIds = await memory.addDocument({
  pathOrBuffer: './nebula_report.pdf',
  type: 'pdf',
  parentChild: true,
  parentChunkSize: 1000,    // Broad parent context
  chunkSize: 200            // Small child chunk for semantic matching
});

// Searching child chunks automatically resolves to the parent chunk!
const results = await memory.query({
  query: 'kepala tim proyek',
  limit: 1
});

console.log(results[0].memory.content); // Returns the 1000-character parent chunk!
console.log(results[0].memory.metadata._childContent); // Contains the specific matched 200-char child chunk
```

---

### 6. Local Multimodal (CLIP - Image & Text)

Index images and search them using either text descriptions or matching images. Supports JPEG, PNG, and **WebP** formats.

```typescript
import { RMemory, LocalCLIPEmbeddingProvider } from 'r-memory';
import { readFileSync } from 'fs';

const memory = new RMemory({
  dbPath: 'multimodal.db',
  collectionName: 'assets',
  embeddingProvider: new LocalCLIPEmbeddingProvider()
});

// Add a WebP image
await memory.addMemory({
  id: 'img-cat',
  content: 'Photo of a sleeping cat on a pillow',
  image: readFileSync('cat.webp')
});

// Search image using text
const results = await memory.query({
  query: 'kucing tidur',
  limit: 1
});
```

---

## 🛠️ API Reference

### `RMemory`

*   `new RMemory(config: RMemoryConfig)`:
    *   `dbPath`: Path to the SQLite database file.
    *   `collectionName`: Name of the SQLite tables namespace.
    *   `embeddingProvider`: An instance of `EmbeddingProvider`.
*   `addMemory(options)`: Inserts a single memory.
    *   `id`: Optional custom string ID.
    *   `content`: The text content.
    *   `image`: Optional image path string, Buffer, or Uint8Array.
    *   `metadata`: Optional JSON metadata object.
    *   `embedding`: Optional pre-computed number array (bypasses embedding model).
*   `addDocument(options)`: Parses a document, chunks it, and indexes it using parallel batch embedding.
    *   `pathOrBuffer`: Path to file or raw file Buffer.
    *   `type`: `'pdf' | 'docx' | 'xlsx' | 'txt'`.
    *   `chunkSize`: Max child chunk size in characters (default: 500).
    *   `chunkOverlap`: Overlap between child chunks (default: scales to 20%).
    *   `parentChild`: Set `true` to enable Parent-Child ingestion.
    *   `parentChunkSize`: Max parent chunk size in characters (default: 1000).
    *   `parentChunkOverlap`: Overlap between parent chunks (default: scales to 20%).
*   `query(options)`: Queries memories by similarity.
    *   `query`: Search query (string for text, Buffer/Uint8Array for image).
    *   `limit`: Max results to return (default: 5).
    *   `filter`: Optional metadata key-value filters.
    *   `hybrid`: Set `true` to enable Hybrid search (lexical + semantic + RRF).
    *   `decayFactor`: Set time-decay multiplier per hour.
*   `getSemanticCache(queryText, threshold)`: Resolves cached LLM responses.
*   `setSemanticCache(queryText, responseText, ttlSeconds)`: Writes LLM responses to cache.
*   `clearCache()`: Truncates cache tables.
*   `consolidate(summarizer, options)`: Merges overlapping records.
*   `delete(id: string)`: Deletes a memory by ID.
*   `clear()`: Clears all memories and cache in the collection.
*   `close()`: Closes the SQLite connection.
*   `isVectorExtensionLoaded()`: Returns `true` if native `sqlite-vec` was loaded.

---

## 📝 License

ISC
