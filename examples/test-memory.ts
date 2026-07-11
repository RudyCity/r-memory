import { existsSync, readFileSync } from 'node:fs';
import { RMemory, LocalTextEmbeddingProvider, LocalCLIPEmbeddingProvider } from '../dist/index.js';

// Absolute path to the generated test image
const testImagePath = 'C:\\Users\\USER\\.gemini\\antigravity\\brain\\7cec3376-7042-453d-8698-c89f42468bcf\\cat_sleeping_1783775572959.png';

async function runTextMemoryDemo() {
  console.log('\n--- 1. DEMO MEMORI TEKS LOKAL (multilingual-e5-small) ---');
  
  const provider = new LocalTextEmbeddingProvider();
  console.log(`Menginisialisasi model teks. Dimensi embedding: ${provider.dimensions}`);

  const memory = new RMemory({
    dbPath: 'text_memories.db',
    collectionName: 'agent_facts',
    embeddingProvider: provider
  });

  console.log(`Menggunakan SQLite Vector Extension: ${memory.isVectorExtensionLoaded() ? 'YA' : 'TIDAK (Fallback JS Cosine Similarity Aktif)'}`);

  // Hapus memori lama agar bersih
  memory.clear();

  console.log('Menyimpan memori ke database...');
  await memory.addMemory({
    id: 'fact-1',
    content: 'Nama saya Budi, saya tinggal di Jakarta.',
    metadata: { author: 'Budi', type: 'profile', session: '123' }
  });

  await memory.addMemory({
    id: 'fact-2',
    content: 'Saya sangat menyukai makanan pedas terutama rendang daging sapi.',
    metadata: { author: 'Budi', type: 'preference', session: '123' }
  });

  await memory.addMemory({
    id: 'fact-3',
    content: 'Hobi saya adalah bermain sepak bola dan membaca buku sejarah di akhir pekan.',
    metadata: { author: 'Budi', type: 'hobby', session: '456' }
  });

  console.log('Memori berhasil disimpan!\n');

  // Query 1: Makanan kesukaan
  const query1 = 'Apakah Budi suka rendang?';
  console.log(`Querying: "${query1}"`);
  let results = await memory.query({ query: query1, limit: 2 });
  results.forEach((res, i) => {
    console.log(`  [${i+1}] Jarak/Distance: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}"`);
  });

  // Query 2: Lokasi tinggal
  const query2 = 'Di mana kota asal Budi?';
  console.log(`\nQuerying: "${query2}"`);
  results = await memory.query({ query: query2, limit: 1 });
  results.forEach((res, i) => {
    console.log(`  [${i+1}] Jarak/Distance: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}"`);
  });

  // Query 3: Hobi + Metadata Filter (Hanya session '123')
  const query3 = 'Apa olahraga favorit Budi?';
  console.log(`\nQuerying: "${query3}" dengan filter metadata { session: "123" }`);
  results = await memory.query({ 
    query: query3, 
    limit: 2,
    filter: { session: '123' }
  });
  results.forEach((res, i) => {
    console.log(`  [${i+1}] Jarak/Distance: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}" | Metadata: ${JSON.stringify(res.memory.metadata)}`);
  });

  memory.close();
}

async function runMultimodalDemo() {
  console.log('\n--- 2. DEMO MEMORI MULTIMODAL LOKAL (clip-vit-base-patch32) ---');

  if (!existsSync(testImagePath)) {
    console.log(`[WARNING] File gambar uji tidak ditemukan di: ${testImagePath}. Demo multimodal dilewati.`);
    return;
  }

  const provider = new LocalCLIPEmbeddingProvider();
  console.log(`Menginisialisasi model CLIP. Dimensi embedding: ${provider.dimensions}`);

  const memory = new RMemory({
    dbPath: 'multimodal_memories.db',
    collectionName: 'agent_images',
    embeddingProvider: provider
  });

  memory.clear();

  console.log('Membaca file gambar kucing tidur...');
  const imageBuffer = readFileSync(testImagePath);

  console.log('Menyimpan memori gambar ke database...');
  await memory.addMemory({
    id: 'image-cat',
    content: 'Foto anak kucing yang sedang tidur pulas di atas bantal.',
    image: imageBuffer,
    metadata: { category: 'cat', type: 'image' }
  });

  await memory.addMemory({
    id: 'text-dog',
    content: 'Anjing husky berlari kencang di salju yang dingin.',
    metadata: { category: 'dog', type: 'text' }
  });

  console.log('Memori multimodal berhasil disimpan!\n');

  // Query 1: Mencari gambar menggunakan teks (Cross-Modal Search)
  const query1 = 'kucing lucu yang sedang tertidur';
  console.log(`Querying (Teks -> Gambar/Teks): "${query1}"`);
  let results = await memory.query({ query: query1, limit: 2 });
  results.forEach((res, i) => {
    console.log(`  [${i+1}] Jarak/Distance: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}" | Metadata: ${JSON.stringify(res.memory.metadata)}`);
  });

  // Query 2: Mencari memori menggunakan gambar (Image -> Text/Image Search)
  console.log(`\nQuerying (Gambar Kucing -> Database)`);
  // Kita kirim gambar kucing yang sama untuk mencari memori terdekat
  results = await memory.query({ query: imageBuffer, limit: 2 });
  results.forEach((res, i) => {
    console.log(`  [${i+1}] Jarak/Distance: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}" | Metadata: ${JSON.stringify(res.memory.metadata)}`);
  });

  memory.close();
}

async function main() {
  try {
    await runTextMemoryDemo();
    await runMultimodalDemo();
  } catch (error) {
    console.error('Terjadi kesalahan saat menjalankan demo:', error);
  }
}

main();
