# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-07-11

### Added
*   **Model Quantization & Accelerator Options**:
    *   Added `dtype` ('q8', 'fp16', 'fp32') and `device` ('cpu', 'gpu', 'webgpu') parameters to local embedding providers (`LocalTextEmbeddingProvider` & `LocalCLIPEmbeddingProvider`).
    *   Allows loading 8-bit quantized ONNX models locally to reduce RAM footprint by up to 75%.
*   **Hybrid Search (Lexical + Semantic + RRF)**:
    *   Integrated SQLite FTS5 Full-Text Search inside `SQLiteAdapter` synchronized via database triggers.
    *   Implemented Reciprocal Rank Fusion (RRF) to merge vector similarity results and keyword match rankings into a single sorted score list.
*   **Parent-Child Hierarchical RAG**:
    *   Added support for splitting documents into large Parent chunks (broad context) and small Child chunks (high-precision vector lookup).
    *   Auto-resolves child memory results to return their parent's content during retrieval, preserving matched snippet details in metadata.

## [1.0.0] - 2026-07-11

### Added
*   **Core Architecture**:
    *   Created orchestrator class `RMemory` handling insertion, semantic queries, deletion, and collections.
    *   Defined strong TS interfaces (`Memory`, `QueryResult`, `EmbeddingProvider`, `DatabaseAdapter`).
*   **Dual-mode SQLite Storage (`sqlite-vec` + Fallback JS)**:
    *   Fast native vector mode utilizing the `sqlite-vec` extension and virtual `vec0` tables.
    *   Zero-dependency JavaScript fallback mode that automatically compiles a custom cosine distance function on top of standard SQLite.
    *   Compound inserts with transaction handling for robust virtual table inserts.
    *   Strict binding mapping using `BigInt` casts for `rowid` inputs to comply with `sqlite-vec` C-extension requirements.
    *   Dynamic metadata filtering using SQLite's built-in JSON operator (`json_extract`).
*   **Local ONNX Embedding Providers**:
    *   `LocalTextEmbeddingProvider`: Standard E5 model (`Xenova/multilingual-e5-small`, 384 dimensions) with auto-prefixing support (`query: `/`passage: `) for high-performance Indonesian and English text retrieval.
    *   `LocalCLIPEmbeddingProvider`: Multimodal model (`Xenova/clip-vit-base-patch32`, 512 dimensions) mapping text and images into a shared vector space, facilitating cross-modal searches.
*   **Image Buffering support**:
    *   Integrated `sharp` for image parsing.
    *   Input formats support (JPEG, PNG, WebP) natively wrapped into global `Blob` objects for seamless decoding inside Transformers.js.
*   **OpenAI-Compatible provider**:
    *   OpenAI-compatible text embedding provider (`OpenAIEmbeddingProvider`).
    *   Support for Matryoshka dimension truncation (e.g. 1536 down to 256/512 dimensions) coupled with automatic L2-normalization.
*   **Document Ingestion System**:
    *   Integrated loaders for PDF (`pdf-parse`), Microsoft Word (`mammoth`), Excel spreadsheets (`xlsx`), and plain text (`.txt`).
    *   Tabular layout preservation for Excel sheets by parsing worksheets directly to CSV formatting.
    *   Sliding window character-based text chunker with overlap support and smart word boundary alignment.
*   **Testing and Examples**:
    *   Comprehensive unit tests (`tests/index.test.ts`) validating insertion, deletion, query scoring, metadata filtering, and document ingestion.
    *   Multimodal example script (`examples/test-memory.ts`) using E5 text search and CLIP image similarity searches.
    *   Document example script (`examples/test-document.ts`) validating txt/xlsx parser text extraction, automatic chunking, and semantic table searches.
