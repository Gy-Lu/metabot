/**
 * Reads Feishu documents and converts them to Markdown.
 * Reverse of markdown-to-blocks.ts: Feishu Block → Markdown.
 */
import * as lark from '@larksuiteoapi/node-sdk';
import type { Logger } from '../utils/logger.js';

// Reverse map: Feishu language code → language name for code fences
const REVERSE_LANGUAGE_MAP: Record<number, string> = {
  1: 'plaintext', 2: 'abap', 3: 'ada', 4: 'apache', 5: 'apex',
  6: 'assembly', 7: 'bash', 8: 'csharp', 9: 'cpp', 10: 'c',
  11: 'cobol', 12: 'css', 13: 'coffeescript', 14: 'd', 15: 'dart',
  16: 'delphi', 17: 'django', 18: 'dockerfile', 19: 'erlang', 20: 'fortran',
  21: 'foxpro', 22: 'go', 23: 'groovy', 24: 'html', 25: 'htmlbars',
  26: 'http', 27: 'haskell', 28: 'json', 29: 'java', 30: 'javascript',
  31: 'julia', 32: 'kotlin', 33: 'latex', 34: 'lisp', 36: 'lua',
  38: 'matlab', 39: 'makefile', 40: 'markdown', 41: 'nginx',
  42: 'objective-c', 43: 'openedgeabl', 44: 'perl', 45: 'php',
  47: 'powershell', 48: 'prolog', 49: 'protobuf', 50: 'python', 51: 'r',
  52: 'rpg', 53: 'ruby', 54: 'rust', 55: 'sas', 56: 'scss', 57: 'sql',
  58: 'scala', 59: 'scheme', 60: 'smalltalk', 61: 'swift', 62: 'thrift',
  63: 'typescript', 64: 'vbscript', 65: 'vbnet', 66: 'xml', 67: 'yaml',
  68: 'cmake', 69: 'diff', 70: 'gams', 72: 'less', 73: 'pascal',
  76: 'stata', 80: 'toml',
};

// Block type constants (same as markdown-to-blocks.ts)
const BLOCK_TYPE = {
  PAGE: 1, TEXT: 2,
  HEADING1: 3, HEADING2: 4, HEADING3: 5, HEADING4: 6, HEADING5: 7, HEADING6: 8,
  HEADING7: 9, HEADING8: 10, HEADING9: 11,
  BULLET: 12, ORDERED: 13, CODE: 14, QUOTE: 15, TODO: 17, DIVIDER: 22,
  TABLE: 31, TABLE_CELL: 32,
} as const;

export interface DocReadResult {
  title: string;
  markdown: string;
  documentId: string;
  wordCount: number;
}

export interface ParsedFeishuUrl {
  type: 'docx' | 'wiki' | 'unknown';
  id: string;
}

/** Parse a Feishu document URL to extract type and ID. */
export function parseFeishuUrl(url: string): ParsedFeishuUrl {
  // https://xxx.feishu.cn/docx/ABC123...
  const docxMatch = url.match(/\/docx\/([A-Za-z0-9]+)/);
  if (docxMatch) return { type: 'docx', id: docxMatch[1] };

  // https://xxx.feishu.cn/wiki/ABC123...
  const wikiMatch = url.match(/\/wiki\/([A-Za-z0-9]+)/);
  if (wikiMatch) return { type: 'wiki', id: wikiMatch[1] };

  return { type: 'unknown', id: '' };
}

export class FeishuDocReader {
  constructor(
    private client: lark.Client,
    private logger: Logger,
  ) {}

  /** Read a document by URL (auto-detects docx vs wiki). */
  async readByUrl(url: string): Promise<DocReadResult | null> {
    const parsed = parseFeishuUrl(url);
    if (parsed.type === 'unknown' || !parsed.id) {
      this.logger.warn({ url }, 'Unrecognized Feishu URL format');
      return null;
    }
    if (parsed.type === 'wiki') {
      return this.readWikiNode(parsed.id);
    }
    return this.readDocument(parsed.id);
  }

  /** Read a standalone docx document. */
  async readDocument(documentId: string): Promise<DocReadResult | null> {
    try {
      const blocks = await this.fetchAllBlocks(documentId);
      const title = this.extractTitle(blocks);
      const markdown = this.blocksToMarkdown(blocks);
      return {
        title,
        markdown,
        documentId,
        wordCount: markdown.split(/\s+/).filter(Boolean).length,
      };
    } catch (err: any) {
      this.logger.error({ err: err.msg || err.message, documentId }, 'Failed to read document');
      return null;
    }
  }

  /** Read a wiki page by node token. */
  async readWikiNode(nodeToken: string): Promise<DocReadResult | null> {
    try {
      // Resolve node token to document ID
      const nodeResp = await this.client.wiki.v2.space.getNode({
        params: { token: nodeToken },
      });
      const node = (nodeResp.data as any)?.node;
      if (!node) {
        this.logger.warn({ nodeToken }, 'Wiki node not found');
        return null;
      }

      const objType = node.obj_type;
      const objToken = node.obj_token;

      if (objType !== 'docx') {
        this.logger.warn({ nodeToken, objType }, 'Wiki node is not a docx document');
        return null;
      }

      return this.readDocument(objToken);
    } catch (err: any) {
      this.logger.error({ err: err.msg || err.message, nodeToken }, 'Failed to read wiki node');
      return null;
    }
  }

  /** Fetch all blocks from a document with pagination. */
  private async fetchAllBlocks(documentId: string): Promise<any[]> {
    const allBlocks: any[] = [];
    let pageToken: string | undefined;

    do {
      const resp = await this.client.docx.v1.documentBlock.list({
        path: { document_id: documentId },
        params: {
          page_size: 500,
          ...(pageToken ? { page_token: pageToken } : {}),
        },
      });

      const data = resp.data as any;
      const items = data?.items || [];
      allBlocks.push(...items);
      pageToken = data?.page_token || undefined;
    } while (pageToken);

    return allBlocks;
  }

  /** Extract title from the page block. */
  private extractTitle(blocks: any[]): string {
    const pageBlock = blocks.find((b) => b.block_type === BLOCK_TYPE.PAGE);
    if (pageBlock?.page?.elements) {
      return this.elementsToText(pageBlock.page.elements);
    }
    return '';
  }

  /** Convert blocks to Markdown. */
  private blocksToMarkdown(blocks: any[]): string {
    const lines: string[] = [];
    let orderedIndex = 1;
    let lastWasOrdered = false;

    for (const block of blocks) {
      // Skip page block (it's the root container)
      if (block.block_type === BLOCK_TYPE.PAGE) continue;

      // Reset ordered list counter when leaving ordered blocks
      if (block.block_type !== BLOCK_TYPE.ORDERED && lastWasOrdered) {
        orderedIndex = 1;
        lastWasOrdered = false;
      }

      const line = this.blockToMarkdown(block, orderedIndex);
      if (line !== null) {
        lines.push(line);
        if (block.block_type === BLOCK_TYPE.ORDERED) {
          orderedIndex++;
          lastWasOrdered = true;
        }
      }
    }

    return lines.join('\n');
  }

  /** Convert a single block to Markdown. */
  private blockToMarkdown(block: any, orderedIndex: number): string | null {
    const type = block.block_type;

    // Headings
    if (type >= BLOCK_TYPE.HEADING1 && type <= BLOCK_TYPE.HEADING9) {
      const level = Math.min(type - BLOCK_TYPE.HEADING1 + 1, 6);
      const prefix = '#'.repeat(level);
      const key = `heading${type - BLOCK_TYPE.HEADING1 + 1}`;
      const elements = block[key]?.elements;
      return `${prefix} ${this.elementsToMarkdown(elements)}`;
    }

    switch (type) {
      case BLOCK_TYPE.TEXT: {
        const elements = block.text?.elements;
        const text = this.elementsToMarkdown(elements);
        return text || '';
      }

      case BLOCK_TYPE.BULLET: {
        const elements = block.bullet?.elements;
        return `- ${this.elementsToMarkdown(elements)}`;
      }

      case BLOCK_TYPE.ORDERED: {
        const elements = block.ordered?.elements;
        return `${orderedIndex}. ${this.elementsToMarkdown(elements)}`;
      }

      case BLOCK_TYPE.TODO: {
        const elements = block.todo?.elements;
        const done = block.todo?.style?.done ? 'x' : ' ';
        return `- [${done}] ${this.elementsToMarkdown(elements)}`;
      }

      case BLOCK_TYPE.CODE: {
        const elements = block.code?.elements;
        const langCode = block.code?.language || 1;
        const lang = REVERSE_LANGUAGE_MAP[langCode] || '';
        const langLabel = lang === 'plaintext' ? '' : lang;
        const content = this.elementsToText(elements);
        return `\`\`\`${langLabel}\n${content}\n\`\`\``;
      }

      case BLOCK_TYPE.QUOTE: {
        const elements = block.quote?.elements;
        const text = this.elementsToMarkdown(elements);
        return text.split('\n').map((l) => `> ${l}`).join('\n');
      }

      case BLOCK_TYPE.DIVIDER:
        return '---';

      default:
        return null;
    }
  }

  /** Convert elements array to Markdown with inline formatting. */
  private elementsToMarkdown(elements: any[] | undefined): string {
    if (!elements || elements.length === 0) return '';

    return elements.map((el) => {
      if (el.text_run) {
        const { content, text_element_style: style } = el.text_run;
        if (!content) return '';

        let text = content;

        // Apply inline formatting (order matters: code first, then bold/italic)
        if (style?.inline_code) return `\`${text}\``;
        if (style?.bold && style?.italic) text = `***${text}***`;
        else if (style?.bold) text = `**${text}**`;
        else if (style?.italic) text = `*${text}*`;
        if (style?.strikethrough) text = `~~${text}~~`;
        if (style?.link?.url) {
          const url = decodeURIComponent(style.link.url);
          text = `[${style.bold || style.italic ? content : text}](${url})`;
        }

        return text;
      }
      return '';
    }).join('');
  }

  /** Convert elements array to plain text (no formatting). */
  private elementsToText(elements: any[] | undefined): string {
    if (!elements || elements.length === 0) return '';
    return elements.map((el) => el.text_run?.content || '').join('');
  }
}
