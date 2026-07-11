import { writeFileSync, existsSync, unlinkSync } from 'node:fs';
import * as XLSX from 'xlsx';
import { RMemory, LocalTextEmbeddingProvider } from '../dist/index.js';

// Setup file paths
const txtPath = 'test-doc.txt';
const xlsxPath = 'test-sheet.xlsx';

function createSampleFiles() {
  console.log('Membuat file uji dokumen...');

  // 1. Create a TXT file
  const txtContent = `
R-Memory adalah library TypeScript untuk fungsi memori agen AI.
Library ini menggunakan database vector SQLite untuk menyimpan data secara efisien.
Fitur utamanya meliputi pencarian semantik teks, pencarian gambar (multimodal),
dan dukungan ekstraksi dokumen (PDF, Word, TXT, Excel).
  `.trim();
  writeFileSync(txtPath, txtContent, 'utf-8');
  console.log(`- File TXT dibuat di: ${txtPath}`);

  // 2. Create an Excel (.xlsx) file using SheetJS
  const wsData = [
    ['Nama Agen', 'Keahlian', 'Status Operasional', 'Lokasi'],
    ['Nexus-7', 'Analisis Data & Prediksi', 'Aktif', 'Jakarta'],
    ['Vanguard-1', 'Keamanan Siber & Enkripsi', 'Aktif', 'Surabaya'],
    ['Echo-5', 'Pemrosesan Bahasa Alami', 'Standby', 'Bandung'],
    ['Atlas-9', 'Navigasi & Logistik Otomatis', 'Non-Aktif', 'Medan']
  ];
  const ws = XLSX.utils.aoa_to_sheet(wsData);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Daftar Agen');
  XLSX.writeFile(wb, xlsxPath);
  console.log(`- File Excel dibuat di: ${xlsxPath}`);
}

async function runDemo() {
  createSampleFiles();

  console.log('\nMenginisialisasi RMemory dengan model multilingual-e5-small...');
  const provider = new LocalTextEmbeddingProvider();
  const memory = new RMemory({
    dbPath: 'document_memories.db',
    collectionName: 'doc_chunks',
    embeddingProvider: provider
  });

  memory.clear();

  // 1. Ingest TXT Document
  console.log('\nMengimpor dokumen TXT ke memory...');
  const txtChunks = await memory.addDocument({
    pathOrBuffer: txtPath,
    type: 'txt',
    chunkSize: 150,
    chunkOverlap: 30,
    metadata: { source: 'doc_txt', category: 'manual' }
  });
  console.log(`Dokumen TXT berhasil diimpor menjadi ${txtChunks.length} chunks.`);

  // 2. Ingest Excel Document
  console.log('\nMengimpor dokumen Excel (.xlsx) ke memory...');
  const xlsxChunks = await memory.addDocument({
    pathOrBuffer: xlsxPath,
    type: 'xlsx',
    chunkSize: 200,
    chunkOverlap: 40,
    metadata: { source: 'doc_xlsx', category: 'operational' }
  });
  console.log(`Dokumen Excel berhasil diimpor menjadi ${xlsxChunks.length} chunks.`);

  // Verify by Querying
  console.log('\n--- 🔎 MENJALANKAN PENCARIAN SEMANTIK ---');

  // Query 1: Tanya tentang fitur R-Memory (dari TXT)
  const q1 = 'Fitur utama r-memory apa saja?';
  console.log(`\nQuery: "${q1}"`);
  let results = await memory.query({ query: q1, limit: 1 });
  results.forEach((res, i) => {
    console.log(`  [${i+1}] Jarak: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}" | Source: ${res.memory.metadata.source}`);
  });

  // Query 2: Tanya tentang Agen di Surabaya (dari Excel)
  const q2 = 'Agen yang di Surabaya tugasnya apa?';
  console.log(`\nQuery: "${q2}"`);
  results = await memory.query({ query: q2, limit: 1 });
  results.forEach((res, i) => {
    console.log(`  [${i+1}] Jarak: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}" | Source: ${res.memory.metadata.source}`);
  });

  // Query 3: Tanya tentang Echo-5 (dari Excel)
  const q3 = 'Tampilkan status Echo-5';
  console.log(`\nQuery: "${q3}"`);
  results = await memory.query({ query: q3, limit: 1 });
  results.forEach((res, i) => {
    console.log(`  [${i+1}] Jarak: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}" | Source: ${res.memory.metadata.source}`);
  });

  // Clean up
  memory.close();
  try {
    if (existsSync(txtPath)) unlinkSync(txtPath);
    if (existsSync(xlsxPath)) unlinkSync(xlsxPath);
    console.log('\nFile uji dokumen sementara berhasil dihapus.');
  } catch (e) {}
}

runDemo().catch(console.error);
