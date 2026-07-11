import { readFileSync } from 'node:fs';
import { RawImage } from '@huggingface/transformers';

const testImagePath = 'C:\\Users\\USER\\.gemini\\antigravity\\brain\\7cec3376-7042-453d-8698-c89f42468bcf\\cat_sleeping_1783775572959.png';

async function test() {
  const buffer = readFileSync(testImagePath);
  const dataUrl = `data:image/png;base64,${buffer.toString('base64')}`;
  
  try {
    console.log('Attempting to read data URL...');
    const image = await RawImage.read(dataUrl);
    console.log('Success! Dimensions:', image.width, 'x', image.height);
  } catch (e) {
    console.error('Failed with data URL:', e.message);
  }
}

test();
