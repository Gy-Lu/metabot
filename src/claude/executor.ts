import { execSync, spawn } from 'node:child_process';
import type { BotConfigBase } from '../config.js';
import type { Logger } from '../utils/logger.js';
import { AsyncQueue } from '../utils/async-queue.js';

const isWindows = process.platform === 'win32';

/** Resolve the Codex CLI binary path at module load time. */
function resolveCodexPath(): string {
  if (process.env.CODEX_EXECUTABLE_PATH) return process.env.CODEX_EXECUTABLE_PATH;
  if (process.env.CLAUDE_EXECUTABLE_PATH) return process.env.CLAUDE_EXECUTABLE_PATH; // backward compat
  try {
    const cmd = isWindows ? 'where codex' : 'which codex';
    return execSync(cmd, { encoding: 'utf-8' }).trim().split(/\r?\n/)[0];
  } catch {
    return 'codex';
  }
}

const CODEX_EXECUTABLE = resolveCodexPath();

export interface ApiContext {
  botName: string;
  chatId: string;
}

export interface ExecutorOptions {
  prompt: string;
  cwd: string;
  sessionId?: string;
  abortController: AbortController;
  outputsDir?: string;
  apiContext?: ApiContext;
}

export type SDKMessage = {
  type: string;
  subtype?: string;
  uuid?: string;
  session_id?: string;
  message?: {
    content?: Array<{
      type: string;
      text?: string;
      name?: string;
      id?: string;
      input?: unknown;
    }>;
  };
  // Result fields
  duration_ms?: number;
  duration_api_ms?: number;
  total_cost_usd?: number;
  result?: string;
  is_error?: boolean;
  num_turns?: number;
  errors?: string[];
  // Stream event fields
  event?: {
    type: string;
    index?: number;
    delta?: {
      type: string;
      text?: string;
    };
    content_block?: {
      type: string;
      text?: string;
      name?: string;
      id?: string;
    };
  };
  parent_tool_use_id?: string | null;
};

export interface ExecutionHandle {
  stream: AsyncGenerator<SDKMessage>;
  sendAnswer(toolUseId: string, sessionId: string, answerText: string): void;
  finish(): void;
}

type CodexEvent = {
  type: string;
  thread_id?: string;
  item?: {
    type?: string;
    text?: string;
    command?: string;
    status?: string;
    aggregated_output?: string;
    exit_code?: number | null;
  };
};

export class ClaudeExecutor {
  constructor(
    private config: BotConfigBase,
    private logger: Logger,
  ) {}

  private buildPrompt(prompt: string, outputsDir?: string, apiContext?: ApiContext): string {
    const appendSections: string[] = [];

    if (outputsDir) {
      appendSections.push(`## Output Files\nWhen producing output files for the user (images, PDFs, documents, archives, code files, etc.), copy them to: ${outputsDir}\nUse \`cp\` via the Bash tool. The bridge will automatically send files placed there to the user.`);
    }

    if (apiContext) {
      // botName and chatId are per-session — inject into system prompt to avoid
      // race conditions when multiple chats run concurrently.
      // Port and secret are already set as METABOT_* env vars in config.ts.
      appendSections.push(
        `## MetaBot API\nYou are running as bot "${apiContext.botName}" in chat "${apiContext.chatId}".\nUse the /metabot skill for full API documentation (agent bus, scheduling, bot management).`
      );
    }

    if (appendSections.length === 0) {
      return prompt;
    }

    return `${prompt}\n\n---\n\n${appendSections.join('\n\n')}`;
  }

  private buildCodexArgs(prompt: string, sessionId?: string): string[] {
    const args: string[] = ['exec'];

    if (sessionId) {
      args.push('resume', sessionId);
    }

    args.push('--json');
    args.push('--skip-git-repo-check');
    args.push('--sandbox', 'danger-full-access');

    if (this.config.claude.model) {
      args.push('--model', this.config.claude.model);
    }

    args.push(prompt);
    return args;
  }

  startExecution(options: ExecutorOptions): ExecutionHandle {
    const { prompt, cwd, sessionId, abortController, outputsDir, apiContext } = options;
    const startTime = Date.now();
    const queue = new AsyncQueue<SDKMessage>();
    let finalSent = false;
    let responseText = '';
    let currentSessionId = sessionId;
    let stdoutBuffer = '';
    let stderrBuffer = '';

    const composedPrompt = this.buildPrompt(prompt, outputsDir, apiContext);
    const args = this.buildCodexArgs(composedPrompt, sessionId);

    this.logger.info({ cwd, hasSession: !!sessionId, outputsDir, args }, 'Starting Codex execution');

    const child = spawn(CODEX_EXECUTABLE, args, {
      cwd,
      env: { ...process.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    const pushResult = (subtype: 'success' | 'error', errors?: string[]) => {
      if (finalSent) return;
      finalSent = true;
      queue.enqueue({
        type: 'result',
        subtype,
        session_id: currentSessionId,
        result: responseText,
        is_error: subtype !== 'success',
        duration_ms: Date.now() - startTime,
        errors,
      });
      queue.finish();
    };

    const onCodexEvent = (event: CodexEvent) => {
      if (event.type === 'thread.started' && event.thread_id) {
        currentSessionId = event.thread_id;
        queue.enqueue({
          type: 'system',
          session_id: event.thread_id,
        });
        return;
      }

      if (event.type === 'item.started' && event.item?.type === 'command_execution') {
        queue.enqueue({
          type: 'assistant',
          session_id: currentSessionId,
          parent_tool_use_id: null,
          message: {
            content: [{
              type: 'tool_use',
              name: 'Bash',
              input: { command: event.item.command || '' },
            }],
          },
        });
        return;
      }

      if (event.type === 'item.completed' && event.item?.type === 'agent_message') {
        if (event.item.text) {
          responseText = responseText ? `${responseText}\n\n${event.item.text}` : event.item.text;
          queue.enqueue({
            type: 'assistant',
            session_id: currentSessionId,
            parent_tool_use_id: null,
            message: {
              content: [{ type: 'text', text: responseText }],
            },
          });
        }
        return;
      }

      if (event.type === 'item.completed' && event.item?.type === 'command_execution') {
        queue.enqueue({
          type: 'assistant',
          session_id: currentSessionId,
          parent_tool_use_id: null,
          message: {
            content: [{ type: 'tool_result' }],
          },
        });
      }
    };

    const consumeStdoutLines = (chunk: string) => {
      stdoutBuffer += chunk;
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop() || '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const event = JSON.parse(trimmed) as CodexEvent;
          onCodexEvent(event);
        } catch {
          this.logger.warn({ line: trimmed }, 'Failed to parse Codex JSONL line');
        }
      }
    };

    child.stdout.on('data', (buf: Buffer) => {
      consumeStdoutLines(buf.toString('utf-8'));
    });

    child.stderr.on('data', (buf: Buffer) => {
      stderrBuffer += buf.toString('utf-8');
      if (stderrBuffer.length > 8000) {
        stderrBuffer = stderrBuffer.slice(-8000);
      }
    });

    child.on('error', (err) => {
      this.logger.error({ err }, 'Codex process error');
      pushResult('error', [err.message || 'Codex process failed']);
    });

    child.on('close', (code) => {
      if (stdoutBuffer.trim()) {
        consumeStdoutLines('\n');
      }
      if (code === 0) {
        pushResult('success');
        return;
      }
      const errMsg = stderrBuffer.trim() || `Codex process exited with code ${code ?? 'unknown'}`;
      pushResult('error', [errMsg]);
    });

    const abortChild = () => {
      if (child.killed) return;
      child.kill('SIGTERM');
      setTimeout(() => {
        if (!child.killed) child.kill('SIGKILL');
      }, 1500).unref();
    };
    abortController.signal.addEventListener('abort', abortChild, { once: true });

    async function* wrapStream(streamQueue: AsyncQueue<SDKMessage>, logger: Logger): AsyncGenerator<SDKMessage> {
      try {
        for await (const message of streamQueue) {
          yield message;
        }
      } catch (err: any) {
        if (err.name === 'AbortError' || abortController.signal.aborted) {
          logger.info('Codex execution aborted');
          return;
        }
        throw err;
      }
    }

    return {
      stream: wrapStream(queue, this.logger),
      sendAnswer: (_toolUseId: string, _sid: string, _answerText: string) => {
        this.logger.warn('Codex executor does not support tool_result answers; ignoring sendAnswer');
      },
      finish: () => {
        abortChild();
      },
    };
  }

  async *execute(options: ExecutorOptions): AsyncGenerator<SDKMessage> {
    const handle = this.startExecution(options);
    for await (const message of handle.stream) {
      yield message;
    }
  }
}
