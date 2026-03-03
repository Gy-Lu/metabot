---
name: feishu-doc
description: Read Feishu/Lark documents. Use when the user shares a Feishu document URL or asks you to read content from a Feishu doc or wiki page.
---

## Feishu Document Reader

Read Feishu documents and wiki pages, converting them to Markdown for analysis.

### Quick Commands (fd shortcut)

The `fd` CLI is pre-installed and handles auth automatically.

```bash
fd read <url>              # Read document by Feishu URL (auto-detects docx/wiki)
fd read-id <docId>         # Read document by document ID
fd info <url>              # Get document metadata only (no content)
```

### When to Use

- User shares a Feishu document URL (e.g., `https://xxx.feishu.cn/docx/...` or `https://xxx.feishu.cn/wiki/...`)
- User asks to read, analyze, or reference content from a Feishu doc or wiki page
- User wants to sync or import Feishu content into MetaMemory

### Supported URL Formats

- `https://xxx.feishu.cn/docx/ABC123` — Standalone document
- `https://xxx.feishu.cn/wiki/ABC123` — Wiki page

### Guidelines

- The returned content is Markdown — you can analyze, summarize, or save it to MetaMemory
- Use `fd read` for the full document content, `fd info` for just metadata
- If the user provides a URL, always use `fd read <url>` first
- For wiki pages, the bot needs `wiki:wiki:readonly` permission
- For standalone docs, the bot needs `docx:document:readonly` permission
