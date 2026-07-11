declare module 'pdf-parse' {
  interface PDFParseResult {
    numpages: number;
    numrender: number;
    info: any;
    metadata: any;
    text: string;
    version: string;
  }
  
  function pdf(dataBuffer: Buffer, options?: any): Promise<PDFParseResult>;
  export default pdf;
}

declare module 'mammoth' {
  interface MammothResult {
    value: string;
    messages: any[];
  }
  
  export function extractRawText(options: { buffer: Buffer }): Promise<MammothResult>;
}
