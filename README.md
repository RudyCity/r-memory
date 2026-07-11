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
*   **🔎 Hybrid Search (FTS5 + Vector + RRF)**:
    *   Integrates SQLite FTS5 Full-Text Search inside the database synchronized in real-time via automatic SQL triggers.
    *   Uses **Reciprocal Rank Fusion (RRF)** to merge lexical keyword matches and semantic vector matches into a unified ranking list for high-precision retrieval.
*   **📑 Document & Hierarchical Parent-Child Ingestion**:
    *   Direct ingestion of **PDF, Microsoft Word (`.docx`), Excel (`.xlsx`, `.xls`, `.csv`), and plain text (`.txt`)** files.
    *   Preserves spreadsheet table structures by parsing worksheets into CSV strings.
    *   Slides a window parser over extracted text to produce semantic chunks with smart **word boundary alignment** (never cuts words in half).
    *   **Parent-Child RAG Support**: Split documents into large Parent chunks (broad context) and small Child chunks (highly focused embeddings). Searches query the small Child chunks but automatically return the larger Parent chunk to give the LLM full context.
*   **🔍 Rich Metadata Filtering**:
    *   Store arbitrary metadata alongside memories and query using SQLite's native JSON operations (`json_extract`). Generates indices on metadata fields (`_documentId`, etc.) automatically to accelerate queries.

---

## 📦 Installation

Install the package and its peer dependencies:

```bash
npm install r-memory better-sqlite3 sqlite-vec @huggingface/transformers sharp pdf-parse mammoth xlsx
```

> **Note**: `sharp` is required for local image decoding. `pdf-parse`, `mammoth`, and `xlsx` are required only if you use the document ingestion feature.

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

// Add a memory passage
await memory.addMemory({
  id: 'fact-1',
  content: 'Nama saya Budi, saya tinggal di Jakarta dan menyukai rendang pedas.',
  metadata: { author: 'Budi', session: 'session-123' }
});

// Semantic query
const results = await memory.query({
  query: 'di mana budi tinggal?',
  limit: 1
});
console.log(results[0].memory.content); // "Nama saya Budi, saya tinggal di Jakarta..."
```

---

### 2. Hybrid Search (Lexical + Semantic + RRF)

Combine exact keyword matching (for codes, IDs, or specific names) and semantic search.

```typescript
// Query using hybrid search
const results = await memory.query({
  query: 'kode sandi proyek NEBULA-99',
  hybrid: true, // Enables FTS5 + Vector + RRF
  limit: 1
});

console.log(results[0].memory.content);
console.log(results[0].score); // Combined RRF rank score
```

---

### 3. Parent-Child Hierarchical Ingestion

```typescript
import { RMemory, LocalTextEmbeddingProvider } from 'r-memory';

const memory = new RMemory({
  dbPath: 'memories.db',
  collectionName: 'docs',
  embeddingProvider: new LocalTextEmbeddingProvider()
});

// Ingest a PDF file with Parent-Child structure
const chunkIds = await memory.addDocument({
  pathOrBuffer: './nebula_report.pdf',
  type: 'pdf',
  parentChild: true,
  parentChunkSize: 1000,    // Broad parent context
  parentChunkOverlap: 200,
  chunkSize: 200,           // Small child chunk for semantic matching
  chunkOverlap: 50,
  metadata: { category: 'project-nebula' }
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

### 4. Local Multimodal (CLIP - Image & Text)

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

// Search image using text (Text -> Image)
const results = await memory.query({
  query: 'kucing tidur',
  limit: 1
});
console.log(results[0].memory.content); // "Photo of a sleeping cat..."
```

---

### 5. OpenAI-Compatible Embeddings (with Matryoshka support)

```typescript
import { RMemory, OpenAIEmbeddingProvider } from 'r-memory';

const memory = new RMemory({
  dbPath: 'openai_memories.db',
  collectionName: 'remotes',
  embeddingProvider: new OpenAIEmbeddingProvider({
    baseURL: 'https://api.openai.com/v1', // or local server
    apiKey: 'your-api-key',
    model: 'text-embedding-3-small',
    dimensions: 256, // Truncate dimensions (Matryoshka) - auto L2-normalizes
  })
});
```

---

## 🛠️ API Reference

### `RMemory`

*   `new RMemory(config: RMemoryConfig)`:
    *   `dbPath`: Path to the SQLite database file.
    *   `collectionName`: Name of the SQLite tables namespace (defaults to `'memories'`).
    *   `embeddingProvider`: An instance of `EmbeddingProvider`.
*   `addMemory(options)`: Inserts a single memory.
    *   `id`: Optional custom string ID (defaults to UUID v4).
    *   `content`: The text content.
    *   `image`: Optional image path string, Buffer, or Uint8Array.
    *   `metadata`: Optional JSON metadata object.
    *   `embedding`: Optional pre-computed number array (bypasses embedding model).
*   `addDocument(options)`: Parses a document, chunks it, and indexes it.
    *   `pathOrBuffer`: Path to file or raw file Buffer.
    *   `type`: `'pdf' | 'docx' | 'xlsx' | 'txt'`.
    *   `chunkSize`: Max child chunk size in characters (default: 500).
    *   `chunkOverlap`: Overlap between child chunks (default: 100).
    *   `parentChild`: Set `true` to enable Parent-Child ingestion.
    *   `parentChunkSize`: Max parent chunk size in characters (default: 1000).
    *   `parentChunkOverlap`: Overlap between parent chunks (default: 200).
*   `query(options)`: Queries memories by similarity.
    *   `query`: Search query (string for text, Buffer/Uint8Array for image).
    *   `limit`: Max results to return (default: 5).
    *   `filter`: Optional metadata key-value filters.
    *   `hybrid`: Set `true` to enable Hybrid search (lexical + semantic + RRF).
*   `delete(id: string)`: Deletes a memory by ID.
*   `clear()`: Clears all memories in the collection.
*   `close()`: Closes the SQLite connection.
*   `isVectorExtensionLoaded()`: Returns `true` if native `sqlite-vec` extension was loaded.

---

## 📝 License

ISC
