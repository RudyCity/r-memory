import pdf from 'pdf-parse';
import mammoth from 'mammoth';
import * as XLSX from 'xlsx';

/**
 * Extracts raw text from a PDF document buffer.
 */
export async function extractTextFromPDF(buffer: Buffer): Promise<string> {
  // pdf-parse library exports a default function
  const data = await pdf(buffer);
  return data.text || '';
}

/**
 * Extracts raw text from a Microsoft Word (.docx) document buffer.
 */
export async function extractTextFromDocx(buffer: Buffer): Promise<string> {
  const result = await mammoth.extractRawText({ buffer });
  return result.value || '';
}

/**
 * Extracts comma-separated spreadsheet data from an Excel (.xlsx, .xls, .csv) document buffer.
 */
export async function extractTextFromXlsx(buffer: Buffer): Promise<string> {
  const workbook = XLSX.read(buffer, { type: 'buffer' });
  let text = '';
  for (const sheetName of workbook.SheetNames) {
    const sheet = workbook.Sheets[sheetName];
    const csv = XLSX.utils.sheet_to_csv(sheet);
    if (csv.trim()) {
      text += `Sheet: ${sheetName}\n${csv}\n`;
    }
  }
  return text;
}

/**
 * Splitting text into smaller semantic passages using a sliding window approach with word alignment.
 * 
 * @param text The full text to split
 * @param chunkSize The maximum size of each chunk (characters)
 * @param chunkOverlap The overlap size between consecutive chunks (characters)
 */
export function chunkText(text: string, chunkSize = 500, chunkOverlap = 100): string[] {
  if (chunkSize <= 0) throw new Error('chunkSize must be greater than 0');
  if (chunkOverlap < 0 || chunkOverlap >= chunkSize) {
    throw new Error('chunkOverlap must be non-negative and less than chunkSize');
  }

  const chunks: string[] = [];
  const normalizedText = text.replace(/\r\n/g, '\n').replace(/\s+/g, ' ');
  let index = 0;

  while (index < normalizedText.length) {
    let end = index + chunkSize;
    let chunk = normalizedText.substring(index, end);

    // Align with word boundary if not at the very end of the text
    if (end < normalizedText.length) {
      const lastSpace = chunk.lastIndexOf(' ');
      // Only align if space is found in the last 20% of the chunk
      if (lastSpace > chunkSize * 0.8) {
        chunk = chunk.substring(0, lastSpace);
      }
    }

    const trimmed = chunk.trim();
    if (trimmed) {
      chunks.push(trimmed);
    }

    const step = chunk.length - chunkOverlap;
    index += step > 0 ? step : 1; // Safely advance by at least 1 character
  }

  return chunks;
}
