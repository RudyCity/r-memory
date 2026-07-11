import { writeFileSync, existsSync, unlinkSync } from 'node:fs';
import { RMemory, LocalTextEmbeddingProvider } from '../dist/index.js';

const docPath = 'test-advanced-doc.txt';

function createSampleDocument() {
  console.log('Membuat berkas dokumen uji...');
  const text = `
PARAGRAF 1 - TENTANG PROYEK NEBULA:
Proyek Nebula adalah inisiatif pertahanan siber rahasia untuk melindungi infrastruktur satelit nasional. 
Tim pengembang utama berlokasi di Bandung dan dipimpin oleh Dr. Adrian. Kode sandi operasional adalah NEBULA-99.

PARAGRAF 2 - TIM OPERASIONAL KELAYAKAN:
Setiap agen yang tergabung dalam Proyek Nebula wajib melewati sertifikasi keamanan tingkat 5. 
Anggaran proyek ini didanai oleh kementerian pertahanan dan diaudit setiap kuartal secara ketat.

PARAGRAF 3 - PENGEMBANGAN MODEL R-MEMORY:
R-Memory digunakan sebagai memori jangka panjang untuk agen otonom yang memantau anomali jaringan satelit.
Sistem ini menggunakan pencarian hibrida (hybrid search) untuk menggabungkan pencarian kata kunci eksak dan kemiripan semantik.
  `.trim();
  writeFileSync(docPath, text, 'utf-8');
  console.log(`- Berkas dokumen dibuat di: ${docPath}`);
}

async function runAdvancedDemo() {
  createSampleDocument();

  console.log('\nMenginisialisasi RMemory dengan Kuantisasi 8-bit (q8)...');
  // Load model multilingual-e5-small using 8-bit integer quantization to save RAM/CPU load
  const provider = new LocalTextEmbeddingProvider({
    dtype: 'q8', // 8-bit quantization
    device: 'cpu'
  });

  const memory = new RMemory({
    dbPath: 'advanced_memories.db',
    collectionName: 'adv_chunks',
    embeddingProvider: provider
  });

  memory.clear();

  // Ingest Document using Parent-Child Hierarchical Ingestion
  console.log('\nMengimpor dokumen menggunakan metode Parent-Child Hierarchical Ingestion...');
  const chunkIds = await memory.addDocument({
    pathOrBuffer: docPath,
    type: 'txt',
    parentChild: true,
    parentChunkSize: 200,      // Parent chunks get larger context
    parentChunkOverlap: 40,
    chunkSize: 60,             // Child chunks are small for high-precision semantic lookup
    chunkOverlap: 10,
    metadata: { source: 'nebula_doc' }
  });
  console.log(`Dokumen berhasil diimpor menjadi ${chunkIds.length} Child chunks.`);

  console.log('\n--- 🔎 PENCARIAN 1: PENCARIAN VEKTOR STANDAR (SEMANTIC ONLY) ---');
  const q1 = 'Siapa pemimpin tim di Bandung?';
  console.log(`Query: "${q1}"`);
  
  let results = await memory.query({
    query: q1,
    limit: 1
  });

  results.forEach((res, i) => {
    console.log(`  [${i+1}] Jarak: ${res.distance.toFixed(4)}`);
    console.log(`      Child Match : "${res.memory.metadata._childContent}"`);
    console.log(`      Parent Text : "${res.memory.content}"`);
  });

  console.log('\n--- 🔎 PENCARIAN 2: PENCARIAN HIBRIDA (HYBRID SEARCH: VECTOR + FTS5 + RRF) ---');
  // We use a query containing an exact keyword: "NEBULA-99"
  const q2 = 'Apa kode sandi operasional proyek NEBULA-99?';
  console.log(`Query: "${q2}"`);

  results = await memory.query({
    query: q2,
    hybrid: true, // Enable Lexical + Semantic + Reciprocal Rank Fusion
    limit: 1
  });

  results.forEach((res, i) => {
    console.log(`  [${i+1}] Skor RRF: ${(res.score ?? 0).toFixed(6)} | Jarak: ${res.distance.toFixed(4)}`);
    console.log(`      Child Match : "${res.memory.metadata._childContent}"`);
    console.log(`      Parent Text : "${res.memory.content}"`);
  });

  // Clean up
  memory.close();
  try {
    if (existsSync(docPath)) unlinkSync(docPath);
    console.log('\nBerkas dokumen uji sementara berhasil dihapus.');
  } catch (e) {}
}

runAdvancedDemo().catch(console.error);
