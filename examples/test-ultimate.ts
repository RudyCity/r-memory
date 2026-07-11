import { RMemory, LocalTextEmbeddingProvider } from '../dist/index.js';

async function runUltimateDemo() {
  console.log('Menginisialisasi RMemory untuk Ultimate Demo (v1.2.0)...');
  const provider = new LocalTextEmbeddingProvider({ dtype: 'q8' }); // 8-bit model
  
  const memory = new RMemory({
    dbPath: 'ultimate_memories.db',
    collectionName: 'ult_chunks',
    embeddingProvider: provider
  });

  memory.clear();

  // 1. BATCH EMBEDDING INGESTION
  console.log('\n--- ⚡ 1. VERIFIKASI BATCH EMBEDDING INGESTION ---');
  console.time('Waktu Ingestion (Batch)');
  const parentText = `
R-Memory versi 1.2.0 dirilis dengan optimasi performa tingkat tinggi.
Fitur baru meliputi Batch Embedding untuk mempercepat pengolahan dokumen secara paralel.
Selain itu, terdapat fitur Time-Decay (recency weighting) untuk memprioritaskan informasi terbaru.
Fitur Semantic Cache digunakan untuk merespons kueri berulang secara instan menggunakan SQLite.
Dan fitur Memory Consolidation mengelompokkan memori serupa untuk dirangkum oleh LLM.
  `.trim();
  
  const chunkIds = await memory.addDocument({
    pathOrBuffer: Buffer.from(parentText, 'utf-8'),
    type: 'txt',
    parentChild: true,
    parentChunkSize: 150,
    chunkSize: 60,
    metadata: { doc: 'ultimate-release' }
  });
  console.timeEnd('Waktu Ingestion (Batch)');
  console.log(`Berhasil mengimpor ${chunkIds.length} chunks.`);

  // 2. SEMANTIC CACHE WITH DISTANCE PRINTING
  console.log('\n--- 💾 2. VERIFIKASI SEMANTIC CACHE ---');
  const qCache = 'Fitur apa saja yang ada di versi 1.2.0?';
  const mockLLMResponse = 'Versi 1.2.0 mendukung Batch Embedding, Time-Decay, Semantic Cache, dan Consolidation.';

  console.log('Menyimpan respons LLM ke Semantic Cache...');
  await memory.setSemanticCache(qCache, mockLLMResponse, 60);

  // Let's check the actual distance to different queries
  const queriesToTest = [
    'Fitur apa saja yang ada di versi 1.2.0?', // Same
    'Fitur baru apa saja di versi 1.2.0?',     // Paraphrase 1
    'Apa yang baru di r-memory v1.2.0?'       // Paraphrase 2
  ];

  const db = memory['db']['db'];
  const embed = memory['db']['isVectorExtensionLoaded'];

  for (const q of queriesToTest) {
    const qEmbed = await provider.embedText(q, 'query');
    const floatArray = new Float32Array(qEmbed);
    const buffer = Buffer.from(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);

    let distance = 1.0;
    if (embed) {
      const stmt = db.prepare(`
        SELECT v.distance FROM vec_cache_memories_ult_chunks v 
        JOIN cache_memories_ult_chunks c ON c.rowid = v.rowid 
        WHERE v.embedding MATCH ? AND k = 1
      `);
      const row = stmt.get(buffer) as any;
      if (row) distance = row.distance;
    } else {
      const stmt = db.prepare(`
        SELECT vec_distance_cosine(embedding, ?) as distance FROM cache_memories_ult_chunks
      `);
      const row = stmt.get(buffer) as any;
      if (row) distance = row.distance;
    }

    const hit = await memory.getSemanticCache(q, 0.35); // Allow large threshold for testing
    console.log(`- Query: "${q}" | Distance ke Cache: ${distance.toFixed(4)} | Hit (threshold 0.35): ${hit ? 'YA 🎉' : 'TIDAK'}`);
  }

  // 3. TIME-DECAY (RECENCY WEIGHTING)
  console.log('\n--- ⏳ 3. VERIFIKASI TIME-DECAY (RECENCY WEIGHTING) ---');
  memory.clear();
  const idOld = await memory.addMemory({ content: 'Anggaran pemasaran Q4 adalah 500 Juta Rupiah (Lama)' });
  const idNew = await memory.addMemory({ content: 'Anggaran pemasaran Q4 direvisi menjadi 750 Juta Rupiah (Baru)' });

  const ageMs = 48 * 60 * 60 * 1000;
  db.prepare(`UPDATE ${memory['db']['tableName']} SET created_at = ? WHERE id = ?`).run(Date.now() - ageMs, idOld);

  console.log('Pencarian anggaran TANPA time-decay:');
  let searchResults = await memory.query({ query: 'Berapa anggaran pemasaran Q4?', limit: 2 });
  searchResults.forEach((res, i) => console.log(`  [${i+1}] Jarak: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}"`));

  console.log('\nPencarian anggaran DENGAN time-decay (decayFactor: 0.05):');
  searchResults = await memory.query({ query: 'Berapa anggaran pemasaran Q4?', decayFactor: 0.05, limit: 2 });
  searchResults.forEach((res, i) => console.log(`  [${i+1}] Jarak Decayed: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}"`));

  // 4. MEMORY CONSOLIDATION (CLUSTERING) WITH DISTANCE PRINTING
  console.log('\n--- 🤝 4. VERIFIKASI MEMORY CONSOLIDATION ---');
  memory.clear();
  await memory.addMemory({ content: 'Adrian menyukai pemrograman TypeScript.' });
  await memory.addMemory({ content: 'Adrian sangat suka menulis kode dalam bahasa TypeScript.' });
  await memory.addMemory({ content: 'Hari ini hujan deras di Jakarta.' });

  // Let's print the actual distance between Adrian memories
  const m1 = memory['db'].getById(memory['db'].getAll()[0].id);
  const m2 = memory['db'].getById(memory['db'].getAll()[1].id);
  
  if (m1 && m2 && m1.embedding && m2.embedding) {
    const qRes = memory['db'].query(m1.embedding, 5);
    const dist = qRes.find(r => r.memory.id === m2.id)?.distance ?? 1.0;
    console.log(`Jarak cosine antara dua memori Adrian: ${dist.toFixed(4)}`);
    
    console.log('\nMemulai konsolidasi memori dengan threshold otomatis...');
    const mockSummarizer = async (texts: string[]) => {
      console.log(`- Menerima ${texts.length} memori yang mirip untuk diringkas:`);
      texts.forEach(t => console.log(`    * "${t}"`));
      return 'Adrian adalah programmer TypeScript.';
    };

    // Set threshold slightly larger than actual distance to guarantee clustering
    await memory.consolidate(mockSummarizer, { threshold: dist + 0.02 });

    console.log('\nHasil setelah konsolidasi:');
    const allResults = await memory.query({ query: 'Adrian', limit: 5 });
    allResults.forEach((res, i) => {
      console.log(`  [${i+1}] Jarak: ${res.distance.toFixed(4)} | Isi: "${res.memory.content}" | Consolidated: ${res.memory.metadata._isConsolidated ? 'YA' : 'TIDAK'}`);
    });
  }

  memory.close();
}

runUltimateDemo().catch(console.error);
