import { EmbeddingProvider, LocalEmbeddingConfig } from '../types.js';

// Lazy load transformers.js to avoid loading it if only OpenAI-compatible is used
let transformersModule: any = null;

async function getTransformers() {
  if (!transformersModule) {
    // Dynamically import @huggingface/transformers to keep it ESM compatible
    transformersModule = await import('@huggingface/transformers');
  }
  return transformersModule;
}

export class LocalTextEmbeddingProvider implements EmbeddingProvider {
  readonly dimensions = 384;
  private modelName: string;
  private device: string;
  private dtype: string;
  private extractor: any = null;

  constructor(config: LocalEmbeddingConfig = {}) {
    this.modelName = config.modelName || 'Xenova/multilingual-e5-small';
    this.device = config.device || 'cpu';
    this.dtype = config.dtype || 'fp32';
  }

  private async getExtractor() {
    if (!this.extractor) {
      const { pipeline } = await getTransformers();
      this.extractor = await pipeline('feature-extraction', this.modelName, {
        device: this.device,
        dtype: this.dtype
      });
    }
    return this.extractor;
  }

  async embedText(text: string, context: 'query' | 'passage' = 'passage'): Promise<number[]> {
    const extractor = await this.getExtractor();
    
    // E5 models require a prefix
    let processedText = text;
    if (this.modelName.includes('e5-small')) {
      processedText = `${context}: ${text}`;
    }

    const output = await extractor(processedText, {
      pooling: 'mean',
      normalize: true
    });
    
    return Array.from(output.data);
  }

  async embedTexts(texts: string[], context: 'query' | 'passage' = 'passage'): Promise<number[][]> {
    if (texts.length === 0) return [];
    const extractor = await this.getExtractor();

    // E5 models require a prefix
    const processedTexts = this.modelName.includes('e5-small')
      ? texts.map(t => `${context}: ${t}`)
      : texts;

    const output = await extractor(processedTexts, {
      pooling: 'mean',
      normalize: true
    });

    const dims = this.dimensions;
    const data = output.data; // Float32Array containing all embeddings in a flat array
    const results: number[][] = [];
    
    for (let i = 0; i < texts.length; i++) {
      const start = i * dims;
      const end = start + dims;
      results.push(Array.from(data.subarray(start, end)));
    }
    
    return results;
  }

  async embedImage(image: string | Buffer | Uint8Array): Promise<number[]> {
    throw new Error('LocalTextEmbeddingProvider does not support image embeddings. Use LocalCLIPEmbeddingProvider for multimodal memories.');
  }
}

export class LocalCLIPEmbeddingProvider implements EmbeddingProvider {
  readonly dimensions = 512;
  private modelId: string;
  private device: string;
  private dtype: string;
  
  private tokenizer: any = null;
  private textModel: any = null;
  private processor: any = null;
  private visionModel: any = null;

  constructor(config: LocalEmbeddingConfig = {}) {
    this.modelId = config.modelName || 'Xenova/clip-vit-base-patch32';
    this.device = config.device || 'cpu';
    this.dtype = config.dtype || 'fp32';
  }

  private async initTextModel() {
    if (!this.tokenizer || !this.textModel) {
      const { AutoTokenizer, CLIPTextModelWithProjection } = await getTransformers();
      this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);
      this.textModel = await CLIPTextModelWithProjection.from_pretrained(this.modelId, {
        device: this.device,
        dtype: this.dtype
      });
    }
  }

  private async initVisionModel() {
    if (!this.processor || !this.visionModel) {
      const { AutoProcessor, CLIPVisionModelWithProjection } = await getTransformers();
      this.processor = await AutoProcessor.from_pretrained(this.modelId);
      this.visionModel = await CLIPVisionModelWithProjection.from_pretrained(this.modelId, {
        device: this.device,
        dtype: this.dtype
      });
    }
  }

  async embedText(text: string, context?: 'query' | 'passage'): Promise<number[]> {
    await this.initTextModel();
    const textInputs = this.tokenizer(text, { padding: true, truncation: true });
    const { text_embeds } = await this.textModel(textInputs);
    
    // Normalize CLIP embedding to unit length
    const rawData = Array.from(text_embeds.data) as number[];
    return this.normalize(rawData);
  }

  async embedTexts(texts: string[], context?: 'query' | 'passage'): Promise<number[][]> {
    if (texts.length === 0) return [];
    await this.initTextModel();
    
    const textInputs = this.tokenizer(texts, { padding: true, truncation: true });
    const { text_embeds } = await this.textModel(textInputs);
    
    const dims = this.dimensions;
    const data = text_embeds.data; // Float32Array containing all embeddings
    const results: number[][] = [];
    
    for (let i = 0; i < texts.length; i++) {
      const start = i * dims;
      const end = start + dims;
      const rawVector = Array.from(data.subarray(start, end)) as number[];
      results.push(this.normalize(rawVector));
    }
    
    return results;
  }

  async embedImage(imageInput: string | Buffer | Uint8Array): Promise<number[]> {
    await this.initVisionModel();
    const { RawImage } = await getTransformers();
    
    let rawImage;
    if (typeof imageInput === 'string') {
      rawImage = await RawImage.read(imageInput);
    } else {
      // Convert buffer or typed array to global Blob for RawImage compatibility
      const blob = new Blob([imageInput]);
      rawImage = await RawImage.read(blob as any);
    }

    const imageInputs = await this.processor(rawImage);
    const { image_embeds } = await this.visionModel(imageInputs);
    
    const rawData = Array.from(image_embeds.data) as number[];
    return this.normalize(rawData);
  }

  private normalize(vector: number[]): number[] {
    let sumSq = 0;
    for (let i = 0; i < vector.length; i++) {
      sumSq += vector[i] * vector[i];
    }
    const magnitude = Math.sqrt(sumSq);
    if (magnitude === 0) return vector;
    return vector.map(x => x / magnitude);
  }
}
