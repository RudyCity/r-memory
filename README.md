# R-Memory 🧠

A high-performance, light-weight TypeScript memory agent library. It equips AI agents with long-term and short-term memory capabilities using a vector-enabled SQLite database. R-Memory supports both local offline models and OpenAI-compatible APIs for text and multimodal (image) embeddings, alongside built-in document ingestion and semantic chunking.

---

## ✨ Features

*   **⚡ Dual-Mode SQLite Vector Database**:
    *   *Native Mode*: Natively loads the `sqlite-vec` extension and utilizes high-performance virtual `vec0` tables.
    *   *Portable Fallback Mode*: Automatically registers a custom JavaScript-based `vec_distance_cosine` function inside SQLite if the extension is not available. 100% portable across Windows, macOS, and Linux out-of-the-box.
*   **🌐 Flexible Embeddings (Local & OpenAI-Compatible)**:
    *   *Local Text*: Runs the state-of-the-art `Xenova/multilingual-e5-small` model on ONNX Runtime (384 dimensions) for excellent Bahasa Indonesia & English support.
    *   *Local Multimodal (CLIP)*: Runs the `Xenova/clip-vit-base-patch32` model (512 dimensions) to index images and text in the same vector space, enabling cross-modal searches. Supports **JPEG, PNG, WebP** formats.
    *   *OpenAI-Compatible*: Connects to any OpenAI-compatible API (standard OpenAI, Ollama, LM Studio) for text embeddings.
*   **📐 Matryoshka & Dynamic Dimensions**:
    *   Auto-detects model dimensions and dynamically adjusts SQLite schemas.
    *   Supports Matryoshka Representation Learning (dimension truncation + L2 normalization) for models like `text-embedding-3-small` (e.g. cutting 1536 down to 256 or 512 dimensions).
*   **📑 Document & Spreadsheet Ingestion**:
    *   Direct ingestion of **PDF, Microsoft Word (`.docx`), Excel (`.xlsx`, `.xls`, `.csv`), and plain text (`.txt`)** files.
    *   Preserves spreadsheet table structures by parsing worksheets into CSV strings.
    *   Slides a window parser over extracted text to produce semantic chunks (defaults: 500 chars size, 100 chars overlap) with smart **word boundary alignment** (never cuts words in half).
*   **🔍 Rich Metadata Filtering**:
    *   Store arbitrary metadata alongside memories and query using SQLite's native JSON operations (`json_extract`).

---

## 📦 Installation

Install the package and its peer dependencies:

```bash
npm install r-memory better-sqlite3 sqlite-vec @huggingface/transformers sharp pdf-parse mammoth xlsx
```

> **Note**: `sharp` is required for local image decoding. `pdf-parse`, `mammoth`, and `xlsx` are required only if you use the document ingestion feature.

---

## 🚀 Quick Start

### 1. Local Text Memory (Multilingual: Bahasa Indonesia & English)

```typescript
import { RMemory, LocalTextEmbeddingProvider } from 'r-memory';

const memory = new RMemory({
  dbPath: 'memories.db',
  collectionName: 'agent_facts',
  embeddingProvider: new LocalTextEmbeddingProvider() // multilingual-e5-small
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
console.log(results[0].distance);       // Cosine distance score
```

---

### 2. Document & Spreadsheet Ingestion (RAG)

Automatically extract text, chunk it with overlaps, generate embeddings, and index it into SQLite.

```typescript
import { RMemory, LocalTextEmbeddingProvider } from 'r-memory';

const memory = new RMemory({
  dbPath: 'memories.db',
  collectionName: 'docs',
  embeddingProvider: new LocalTextEmbeddingProvider()
});

// Ingest a PDF file or Excel spreadsheet
const chunkIds = await memory.addDocument({
  pathOrBuffer: './financial_report.xlsx',
  type: 'xlsx', // Supported: 'pdf', 'docx', 'xlsx', 'txt'
  chunkSize: 500,
  chunkOverlap: 100,
  metadata: { category: 'financial', year: 2026 }
});

console.log(`Ingested ${chunkIds.length} spreadsheet chunks!`);

// Query spreadsheet semantically
const results = await memory.query({
  query: 'q4 marketing budget distribution',
  limit: 2,
  filter: { category: 'financial' } // Metadata filtering
});
console.log(results[0].memory.content);
```

---

### 3. Local Multimodal (CLIP - Image & Text)

Index images and search them using either text descriptions (cross-modal) or matching images (image-to-image).

```typescript
import { RMemory, LocalCLIPEmbeddingProvider } from 'r-memory';
import { readFileSync } from 'fs';

const memory = new RMemory({
  dbPath: 'multimodal.db',
  collectionName: 'assets',
  embeddingProvider: new LocalCLIPEmbeddingProvider() // clip-vit-base-patch32
});

// Add an image (Buffer/Uint8Array or file path string)
await memory.addMemory({
  id: 'img-cat',
  content: 'Photo of a sleeping cat on a pillow', // optional description
  image: readFileSync('cat.webp'), // Supports JPEG, PNG, WebP
  metadata: { tag: 'animal' }
});

// Query 1: Search image using text (Text -> Image)
const textQueryResults = await memory.query({
  query: 'kucing tidur',
  limit: 1
});
console.log(textQueryResults[0].memory.content); // "Photo of a sleeping cat..."

// Query 2: Search using another image (Image -> Image)
const imageQueryResults = await memory.query({
  query: readFileSync('cat_query.png'),
  limit: 1
});
console.log(imageQueryResults[0].memory.id); // "img-cat"
```

---

### 4. OpenAI-Compatible Embeddings

Use services like standard OpenAI, Ollama, or LM Studio. Includes Matryoshka support for truncating dimensions.

```typescript
import { RMemory, OpenAIEmbeddingProvider } from 'r-memory';

const memory = new RMemory({
  dbPath: 'openai_memories.db',
  collectionName: 'remotes',
  embeddingProvider: new OpenAIEmbeddingProvider({
    baseURL: 'https://api.openai.com/v1', // or http://localhost:11434/v1 for Ollama
    apiKey: 'your-api-key', // optional for local providers
    model: 'text-embedding-3-small',
    dimensions: 256, // Truncate dimensions (Matryoshka) - auto L2-normalizes
  })
});
```

---

## 🛠️ API Reference

### `RMemory`

*   `new RMemory(config: RMemoryConfig)`:
    *   `dbPath`: Path to the SQLite database file (e.g. `'memory.db'` or `':memory:'`).
    *   `collectionName`: Name of the SQLite tables namespace (defaults to `'memories'`).
    *   `embeddingProvider`: An instance of `EmbeddingProvider`.
*   `addMemory(options)`: Inserts a single text or image memory.
    *   `id`: Optional custom string ID (defaults to UUID v4).
    *   `content`: The text content.
    *   `image`: Optional image path string, Buffer, or Uint8Array.
    *   `metadata`: Optional JSON metadata object.
*   `addDocument(options)`: Parses a document, chunks it, embeds it, and stores the chunks.
    *   `pathOrBuffer`: Path to file or raw file Buffer.
    *   `type`: `'pdf' | 'docx' | 'xlsx' | 'txt'`.
    *   `chunkSize`: Max chunk size in characters (default: 500).
    *   `chunkOverlap`: Overlap between consecutive chunks (default: 100).
*   `query(options)`: Queries memories by semantic similarity.
    *   `query`: Search query (string for text query, Buffer/Uint8Array for image query).
    *   `limit`: Max results to return (default: 5).
    *   `filter`: Optional metadata key-value filters.
*   `delete(id: string)`: Deletes a memory by ID.
*   `clear()`: Clears all memories in the collection.
*   `close()`: Closes the SQLite connection.
*   `isVectorExtensionLoaded()`: Returns `true` if fast native `sqlite-vec` was loaded.

---

## 📝 License

ISC
