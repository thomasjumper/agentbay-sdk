import type {
  AgentBayConfig,
  StoreOptions,
  RecallOptions,
  RecallResult,
  MemoryEntry,
  HealthStats,
  Message,
} from './types';
import { AgentBayError } from './errors';

const DEFAULT_BASE_URL = 'https://www.aiagentsbay.com';
const SDK_VERSION = '0.1.0';

export class AgentBay {
  private apiKey: string;
  private baseUrl: string;
  private projectId: string | undefined;
  private timeout: number;

  /**
   * Create a new AgentBay client.
   *
   * @example
   * // Simple — just an API key
   * const brain = new AgentBay('ab_live_your_key');
   *
   * @example
   * // Full config
   * const brain = new AgentBay({
   *   apiKey: 'ab_live_your_key',
   *   projectId: 'your-project-id',
   *   timeout: 60_000,
   * });
   */
  constructor(config: string | AgentBayConfig) {
    if (typeof config === 'string') {
      this.apiKey = config;
      this.baseUrl = DEFAULT_BASE_URL;
      this.projectId = undefined;
      this.timeout = 30_000;
    } else {
      if (!config.apiKey) {
        throw new AgentBayError('apiKey is required. Get one at https://www.aiagentsbay.com/dashboard', 0);
      }
      this.apiKey = config.apiKey;
      this.baseUrl = (config.baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
      this.projectId = config.projectId;
      this.timeout = config.timeout ?? 30_000;
    }
  }

  // ------------------------------------------------------------------
  // Memory — Core operations
  // ------------------------------------------------------------------

  /**
   * Store a memory entry in your Knowledge Brain.
   *
   * @param content - The knowledge content to store.
   * @param opts - Optional store options (title, type, tier, tags, etc.).
   * @returns The created entry with its `id`.
   *
   * @example
   * await brain.store('Always use connection pooling', { title: 'DB pattern', type: 'PATTERN' });
   */
  async store(content: string, opts?: Partial<StoreOptions>): Promise<{ id: string; deduplicated?: boolean }> {
    const pid = this.resolveProject();
    const body: Record<string, unknown> = {
      content,
      title: opts?.title || content.slice(0, 80),
      type: opts?.type || 'PATTERN',
      tier: opts?.tier || 'semantic',
    };
    if (opts?.tags) body.tags = opts.tags;
    if (opts?.filePaths) body.filePaths = opts.filePaths;
    if (opts?.confidence !== undefined) body.confidence = opts.confidence;
    if (opts?.aliases) body.aliases = opts.aliases;

    return this.post<{ id: string; deduplicated?: boolean }>(`/api/v1/projects/${pid}/memory`, body);
  }

  /**
   * Recall memories by semantic search.
   *
   * @param query - Natural-language search query.
   * @param opts - Optional recall options (limit, tier, type, tags, etc.).
   * @returns Matching entries with scores and metadata.
   *
   * @example
   * const results = await brain.recall('database connection patterns');
   */
  async recall(query: string, opts?: RecallOptions): Promise<RecallResult> {
    const pid = this.resolveProject();
    const params = new URLSearchParams({ q: query });

    if (opts?.limit) params.set('limit', String(opts.limit));
    if (opts?.tier) params.set('tier', opts.tier);
    if (opts?.type) params.set('type', opts.type);
    if (opts?.tags) params.set('tags', opts.tags.join(','));
    if (opts?.tokenBudget) params.set('tokenBudget', String(opts.tokenBudget));
    if (opts?.fast) params.set('fast', 'true');
    if (opts?.resolution) params.set('resolution', opts.resolution);

    return this.get<RecallResult>(`/api/v1/projects/${pid}/memory?${params.toString()}`);
  }

  /**
   * Verify a memory entry, resetting its confidence decay timer.
   */
  async verify(knowledgeId: string): Promise<void> {
    const pid = this.resolveProject();
    await this.patch(`/api/v1/projects/${pid}/memory`, {
      knowledgeId,
      action: 'verify',
    });
  }

  /**
   * Archive (soft-delete) a memory entry.
   */
  async forget(knowledgeId: string): Promise<void> {
    const pid = this.resolveProject();
    await this.delete(`/api/v1/projects/${pid}/memory`, {
      knowledgeId,
    });
  }

  /**
   * Get memory health statistics for the current project.
   */
  async health(): Promise<HealthStats> {
    const pid = this.resolveProject();
    return this.get<HealthStats>(`/api/v1/projects/${pid}/memory?action=health`);
  }

  // ------------------------------------------------------------------
  // Bulk operations
  // ------------------------------------------------------------------

  /**
   * Store multiple memory entries at once.
   *
   * @param entries - Array of store options.
   * @returns Count of stored and errored entries.
   */
  async bulkStore(entries: StoreOptions[]): Promise<{ stored: number; errors: number }> {
    let stored = 0;
    let errors = 0;

    // The API doesn't have a native bulk endpoint, so we batch serially
    // to respect rate limits. Future versions may use a dedicated endpoint.
    for (const entry of entries) {
      try {
        await this.store(entry.content, entry);
        stored++;
      } catch {
        errors++;
      }
    }

    return { stored, errors };
  }

  /**
   * Run multiple recall queries at once.
   *
   * @param queries - Array of search queries.
   * @param opts - Optional shared options for all queries.
   * @returns Array of recall results, one per query.
   */
  async bulkRecall(queries: string[], opts?: { limit?: number }): Promise<RecallResult[]> {
    return Promise.all(queries.map(q => this.recall(q, opts)));
  }

  // ------------------------------------------------------------------
  // Brain management
  // ------------------------------------------------------------------

  /**
   * Create a new Knowledge Brain (project) for your agent.
   *
   * @param name - Human-readable name for the brain.
   * @param opts - Optional description and framework.
   * @returns The created project ID.
   *
   * @example
   * const { projectId } = await brain.setupBrain('My Agent Brain');
   */
  async setupBrain(
    name: string,
    opts?: { description?: string; framework?: string },
  ): Promise<{ projectId: string }> {
    const body: Record<string, unknown> = { name };
    if (opts?.description) body.description = opts.description;
    if (opts?.framework) body.framework = opts.framework;

    const result = await this.post<{ projectId?: string; project?: { id: string } }>('/api/v1/brain/setup', body);

    const projectId = result.projectId || result.project?.id;
    if (projectId && !this.projectId) {
      this.projectId = projectId;
    }

    return { projectId: projectId! };
  }

  /**
   * Create a snapshot (backup) of the current brain state.
   * Uses the MCP endpoint.
   */
  async snapshot(name: string, reason?: string): Promise<{ id: string }> {
    return this.callMcp('agentbay_brain_snapshot', {
      projectId: this.resolveProject(),
      name,
      reason,
    });
  }

  /**
   * Restore a brain from a previous snapshot.
   * Uses the MCP endpoint.
   */
  async restore(snapshotId: string): Promise<{ entriesRestored: number }> {
    return this.callMcp('agentbay_brain_restore', {
      projectId: this.resolveProject(),
      snapshotId,
    });
  }

  // ------------------------------------------------------------------
  // Messaging
  // ------------------------------------------------------------------

  /**
   * Send a message to another agent or broadcast.
   */
  async send(
    content: string,
    opts?: { recipientId?: string; channel?: string; broadcast?: boolean },
  ): Promise<{ id: string }> {
    return this.callMcp('agentbay_session_handoff', {
      projectId: this.resolveProject(),
      content,
      recipientId: opts?.recipientId,
      channel: opts?.channel || 'sdk',
      broadcast: opts?.broadcast ?? false,
    });
  }

  /**
   * Retrieve messages for the current agent.
   */
  async messages(opts?: { unreadOnly?: boolean; limit?: number }): Promise<Message[]> {
    const result = await this.callMcp('agentbay_session_resume', {
      projectId: this.resolveProject(),
      unreadOnly: opts?.unreadOnly ?? true,
      limit: opts?.limit ?? 20,
    });
    return result.messages || [];
  }

  /**
   * Wake another agent with a reason.
   */
  async wake(agentId: string, reason: string): Promise<{ delivered: boolean }> {
    return this.callMcp('agentbay_session_handoff', {
      projectId: this.resolveProject(),
      recipientId: agentId,
      content: reason,
      channel: 'wake',
      broadcast: false,
    });
  }

  // ------------------------------------------------------------------
  // Projects
  // ------------------------------------------------------------------

  /**
   * List all projects the authenticated user has access to.
   */
  async projects(): Promise<Array<{ id: string; name: string }>> {
    const result = await this.get<{ projects: Array<{ id: string; name: string }> } | Array<{ id: string; name: string }>>('/api/v1/projects');
    if (Array.isArray(result)) return result;
    return result.projects || [];
  }

  /**
   * Set the active project for all subsequent operations.
   */
  setProject(projectId: string): void {
    this.projectId = projectId;
  }

  // ------------------------------------------------------------------
  // Graph (memory relationships)
  // ------------------------------------------------------------------

  /**
   * Create a relationship link between two knowledge entries.
   */
  async link(sourceId: string, targetId: string, type: string): Promise<{ id: string }> {
    return this.callMcp('agentbay_knowledge_manage', {
      projectId: this.resolveProject(),
      action: 'link',
      knowledgeId: sourceId,
      targetId,
      relationType: type,
    });
  }

  /**
   * Get related knowledge entries for a given entry.
   */
  async related(knowledgeId: string): Promise<Array<{ title: string; type: string; relationType: string }>> {
    const result = await this.callMcp('agentbay_project_graph_query', {
      projectId: this.resolveProject(),
      knowledgeId,
    });
    return result.related || [];
  }

  // ------------------------------------------------------------------
  // Timeline
  // ------------------------------------------------------------------

  /**
   * Get the activity timeline for the current project.
   */
  async timeline(limit?: number): Promise<Array<{ title: string; action: string; timestamp: string }>> {
    const result = await this.callMcp('agentbay_activity_query', {
      projectId: this.resolveProject(),
      limit: limit ?? 20,
    });
    return result.events || result.activities || [];
  }

  /**
   * Get a diff of changes since a given timestamp.
   */
  async diff(since: string): Promise<{ added: number; updated: number; deprecated: number }> {
    const result = await this.callMcp('agentbay_activity_query', {
      projectId: this.resolveProject(),
      since,
      type: 'diff',
    });
    return {
      added: result.added ?? 0,
      updated: result.updated ?? 0,
      deprecated: result.deprecated ?? 0,
    };
  }

  // ------------------------------------------------------------------
  // Chat — Auto-memory LLM wrapper
  // ------------------------------------------------------------------

  /**
   * Send messages to an LLM with automatic memory recall and storage.
   * Requires `@anthropic-ai/sdk` or `openai` to be installed separately.
   *
   * @example
   * const response = await brain.chat([
   *   { role: 'user', content: 'How should I handle DB connections?' }
   * ]);
   */
  async chat(
    messages: Array<{ role: string; content: string }>,
    opts?: {
      model?: string;
      provider?: 'anthropic' | 'openai';
      autoRecall?: boolean;
      autoStore?: boolean;
      recallLimit?: number;
      [key: string]: any;
    },
  ): Promise<any> {
    const provider = opts?.provider ?? 'anthropic';
    const model = opts?.model ?? (provider === 'anthropic' ? 'claude-sonnet-4-20250514' : 'gpt-4o');
    const autoRecall = opts?.autoRecall ?? true;
    const autoStore = opts?.autoStore ?? true;
    const recallLimit = opts?.recallLimit ?? 3;

    // Extract pass-through options (everything except our known keys)
    const knownKeys = new Set(['model', 'provider', 'autoRecall', 'autoStore', 'recallLimit']);
    const passThrough: Record<string, any> = {};
    if (opts) {
      for (const [k, v] of Object.entries(opts)) {
        if (!knownKeys.has(k)) passThrough[k] = v;
      }
    }

    // 1. Auto-recall: find relevant memories for the last user message
    let contextPrefix = '';
    if (autoRecall) {
      const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
      if (lastUserMsg) {
        try {
          const recalled = await this.recall(lastUserMsg.content, { limit: recallLimit });
          if (recalled.entries && recalled.entries.length > 0) {
            const memoryLines = recalled.entries.map(
              e => `- [${e.type}] ${e.title}: ${e.content}`,
            );
            contextPrefix =
              'Relevant memories from your Knowledge Brain:\n' +
              memoryLines.join('\n') +
              '\n\n';
          }
        } catch {
          // Recall failure is non-fatal
        }
      }
    }

    // 2. Call LLM
    let response: any;

    if (provider === 'anthropic') {
      let Anthropic: any;
      try {
        // eslint-disable-next-line @typescript-eslint/no-implied-eval
        const modName = '@anthropic-ai/sdk';
        const mod = await (Function('m', 'return import(m)')(modName) as Promise<any>);
        Anthropic = mod.default || mod.Anthropic || mod;
      } catch {
        throw new AgentBayError(
          'Anthropic SDK not found. Install it: npm install @anthropic-ai/sdk',
          0,
          'MISSING_DEPENDENCY',
        );
      }

      const client = new Anthropic();

      // Separate system messages from the rest
      const systemMsgs = messages.filter(m => m.role === 'system');
      const nonSystemMsgs = messages.filter(m => m.role !== 'system');

      let systemText = systemMsgs.map(m => m.content).join('\n\n');
      if (contextPrefix) {
        systemText = contextPrefix + (systemText ? '\n\n' + systemText : '');
      }

      response = await client.messages.create({
        model,
        max_tokens: passThrough.max_tokens ?? 4096,
        ...(systemText ? { system: systemText } : {}),
        messages: nonSystemMsgs.map((m: { role: string; content: string }) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content,
        })),
        ...passThrough,
      });
    } else {
      // OpenAI
      let OpenAI: any;
      try {
        // eslint-disable-next-line @typescript-eslint/no-implied-eval
        const modName = 'openai';
        const mod = await (Function('m', 'return import(m)')(modName) as Promise<any>);
        OpenAI = mod.default || mod.OpenAI || mod;
      } catch {
        throw new AgentBayError(
          'OpenAI SDK not found. Install it: npm install openai',
          0,
          'MISSING_DEPENDENCY',
        );
      }

      const client = new OpenAI();

      // Prepend context as a system message
      const augmentedMessages = contextPrefix
        ? [{ role: 'system', content: contextPrefix }, ...messages]
        : [...messages];

      response = await client.chat.completions.create({
        model,
        messages: augmentedMessages,
        ...passThrough,
      });
    }

    // 3. Auto-store: check for learnable content in the response
    if (autoStore) {
      try {
        let responseText = '';
        if (provider === 'anthropic') {
          responseText = response?.content
            ?.filter((b: any) => b.type === 'text')
            .map((b: any) => b.text)
            .join('') ?? '';
        } else {
          responseText = response?.choices?.[0]?.message?.content ?? '';
        }

        const learnableKeywords = /\b(fix|bug|decided|pattern|issue was|the problem|pitfall|mistake|always|never)\b/i;
        if (responseText && learnableKeywords.test(responseText)) {
          const snippet = responseText.slice(0, 500);
          // Fire and forget — don't block the response
          this.store(snippet, {
            title: snippet.slice(0, 80),
            type: 'PATTERN',
          }).catch(() => {});
        }
      } catch {
        // Auto-store failure is non-fatal
      }
    }

    return response;
  }

  // ------------------------------------------------------------------
  // Mem0-compatible aliases
  // ------------------------------------------------------------------

  /**
   * Store a memory entry with auto-detected title and type.
   * Compatible with Mem0's `add()` API.
   *
   * @param data - The content to store.
   * @param opts - Optional userId and metadata.
   * @returns The created entry with its `id`.
   *
   * @example
   * await brain.add('The bug was caused by a missing null check');
   */
  async add(
    data: string,
    opts?: { userId?: string; metadata?: Record<string, any> },
  ): Promise<{ id: string }> {
    // Auto-detect title: first sentence or first 80 chars
    const firstSentenceMatch = data.match(/^[^.!?\n]+[.!?]?/);
    const title = firstSentenceMatch
      ? firstSentenceMatch[0].slice(0, 80)
      : data.slice(0, 80);

    // Auto-detect type
    let type: 'PITFALL' | 'DECISION' | 'PATTERN' = 'PATTERN';
    const lower = data.toLowerCase();
    if (/\b(bug|error|fix|crash|broke|broken|issue|problem)\b/.test(lower)) {
      type = 'PITFALL';
    } else if (/\b(decided|chose|chosen|decision|went with|picked)\b/.test(lower)) {
      type = 'DECISION';
    }

    const storeOpts: Partial<StoreOptions> = {
      title,
      type,
      tags: opts?.metadata?.tags,
    };

    const result = await this.store(data, storeOpts);
    return { id: result.id };
  }

  /**
   * Search memories by semantic query.
   * Compatible with Mem0's `search()` API.
   *
   * @param query - Natural-language search query.
   * @param opts - Optional limit.
   * @returns Matching entries with scores.
   *
   * @example
   * const results = await brain.search('database patterns');
   */
  async search(query: string, opts?: { limit?: number }): Promise<RecallResult> {
    return this.recall(query, { limit: opts?.limit });
  }

  // ------------------------------------------------------------------
  // Utility
  // ------------------------------------------------------------------

  /**
   * Get the current authenticated user's info.
   */
  async whoami(): Promise<{ name: string; email: string; credits: number }> {
    return this.callMcp('agentbay_whoami', {});
  }

  // ------------------------------------------------------------------
  // Internal: REST helpers
  // ------------------------------------------------------------------

  private resolveProject(projectId?: string): string {
    const pid = projectId || this.projectId;
    if (!pid) {
      throw new AgentBayError(
        'No projectId set. Either pass it in the constructor, call setProject(), or call setupBrain() first.',
        0,
      );
    }
    return pid;
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const url = `${this.baseUrl}${path}`;

    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
      'User-Agent': `agentbay-ts/${SDK_VERSION}`,
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        await this.handleErrorResponse(response);
      }

      if (response.status === 204) {
        return {} as T;
      }

      const text = await response.text();
      if (!text) return {} as T;

      return JSON.parse(text) as T;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof AgentBayError) throw error;

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new AgentBayError(`Request timed out after ${this.timeout}ms`, 0, 'TIMEOUT');
        }
        throw new AgentBayError(`Network error: ${error.message}`, 0, 'NETWORK_ERROR');
      }
      throw new AgentBayError('Unknown error', 0);
    }
  }

  private async handleErrorResponse(response: Response): Promise<never> {
    let detail: string;
    try {
      const body = await response.json() as Record<string, unknown>;
      detail = String(body.error || body.message || JSON.stringify(body));
    } catch {
      detail = await response.text().catch(() => 'Unknown error');
    }

    const status = response.status;

    switch (status) {
      case 401:
        throw new AgentBayError(
          'Invalid API key. Check your key at https://www.aiagentsbay.com/dashboard',
          401,
          'UNAUTHORIZED',
        );
      case 403:
        throw new AgentBayError(`Forbidden: ${detail}`, 403, 'FORBIDDEN');
      case 404:
        throw new AgentBayError(`Not found: ${detail}`, 404, 'NOT_FOUND');
      case 429:
        throw new AgentBayError(
          'Rate limit exceeded. Please slow down or upgrade your plan.',
          429,
          'RATE_LIMITED',
        );
      default:
        throw new AgentBayError(`API error ${status}: ${detail}`, status);
    }
  }

  private get<T>(path: string): Promise<T> {
    return this.request<T>('GET', path);
  }

  private post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>('POST', path, body);
  }

  private patch<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>('PATCH', path, body);
  }

  private delete<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>('DELETE', path, body);
  }

  // ------------------------------------------------------------------
  // Internal: MCP JSON-RPC helper
  // ------------------------------------------------------------------

  /**
   * Call an MCP tool via the HTTP MCP endpoint using JSON-RPC.
   * The MCP endpoint returns SSE (Server-Sent Events) with JSON-RPC responses.
   */
  private async callMcp(tool: string, args: Record<string, unknown>): Promise<any> {
    const url = `${this.baseUrl}/api/mcp`;

    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
      'User-Agent': `agentbay-ts/${SDK_VERSION}`,
    };

    const jsonRpcPayload = {
      jsonrpc: '2.0',
      method: 'tools/call',
      id: 1,
      params: {
        name: tool,
        arguments: args,
      },
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(jsonRpcPayload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        await this.handleErrorResponse(response);
      }

      const contentType = response.headers.get('content-type') || '';

      // Handle SSE responses
      if (contentType.includes('text/event-stream')) {
        return this.parseSseResponse(response);
      }

      // Handle direct JSON responses
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const body = await response.json() as any;

      if (body.error) {
        throw new AgentBayError(
          `MCP error: ${body.error.message || JSON.stringify(body.error)}`,
          0,
          'MCP_ERROR',
        );
      }

      // JSON-RPC result
      if (body.result) {
        // MCP tools return result.content[0].text
        const content = body.result.content;
        if (Array.isArray(content) && content.length > 0 && content[0].text) {
          try {
            return JSON.parse(content[0].text);
          } catch {
            return content[0].text;
          }
        }
        return body.result;
      }

      return body;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof AgentBayError) throw error;

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new AgentBayError(`MCP request timed out after ${this.timeout}ms`, 0, 'TIMEOUT');
        }
        throw new AgentBayError(`MCP network error: ${error.message}`, 0, 'NETWORK_ERROR');
      }
      throw new AgentBayError('Unknown MCP error', 0);
    }
  }

  /**
   * Parse an SSE (Server-Sent Events) response from the MCP endpoint.
   * Collects all `data:` lines and extracts the JSON-RPC result.
   */
  private async parseSseResponse(response: Response): Promise<any> {
    const text = await response.text();
    const lines = text.split('\n');
    let lastData = '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        lastData = line.slice(6);
      }
    }

    if (!lastData) {
      throw new AgentBayError('Empty SSE response from MCP endpoint', 0, 'MCP_EMPTY');
    }

    try {
      const parsed = JSON.parse(lastData);

      if (parsed.error) {
        throw new AgentBayError(
          `MCP error: ${parsed.error.message || JSON.stringify(parsed.error)}`,
          0,
          'MCP_ERROR',
        );
      }

      // Extract result content
      const content = parsed.result?.content;
      if (Array.isArray(content) && content.length > 0 && content[0].text) {
        try {
          return JSON.parse(content[0].text);
        } catch {
          return content[0].text;
        }
      }

      return parsed.result || parsed;
    } catch (error) {
      if (error instanceof AgentBayError) throw error;
      throw new AgentBayError(`Failed to parse MCP response: ${lastData}`, 0, 'MCP_PARSE_ERROR');
    }
  }

  toString(): string {
    const masked = this.apiKey.length > 12
      ? `${this.apiKey.slice(0, 8)}...${this.apiKey.slice(-4)}`
      : '***';
    return `AgentBay(apiKey='${masked}', projectId=${this.projectId ?? 'none'})`;
  }
}
