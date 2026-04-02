# @agentbay/sdk

TypeScript SDK for AgentBay — persistent memory for AI agents.

Zero dependencies. Uses Node.js built-in `fetch`.

## Install

```bash
npm install @agentbay/sdk
```

## Quick Start

```typescript
import { AgentBay } from '@agentbay/sdk';

const brain = new AgentBay({
  apiKey: 'ab_live_your_key',
  projectId: 'your-project-id',
});

// Chat with automatic memory — recalls relevant knowledge before the LLM call,
// stores learnable insights from the response automatically.
const response = await brain.chat([
  { role: 'user', content: 'How should I handle database connections?' }
]);
// Requires @anthropic-ai/sdk (default) or openai installed separately:
//   npm install @anthropic-ai/sdk
```

### Without chat — manual store & recall

```typescript
const brain = new AgentBay('ab_live_your_key');

// Create a brain (project) for your agent
const { projectId } = await brain.setupBrain('My Agent');

// Store knowledge
await brain.store('Always use connection pooling for PostgreSQL', {
  title: 'DB pattern',
  type: 'PATTERN',
  tags: ['database', 'performance'],
});

// Recall by semantic search
const results = await brain.recall('database connection');
console.log(results.entries);
```

## With an existing project

```typescript
const brain = new AgentBay({
  apiKey: 'ab_live_...',
  projectId: 'your-project-id',
});

const results = await brain.recall('deployment steps');
```

## API Reference

### Chat (Auto-Memory LLM Wrapper)

```typescript
// Default: Anthropic Claude with auto-recall + auto-store
const response = await brain.chat([
  { role: 'user', content: 'How do I optimize this query?' }
]);

// OpenAI provider
const response = await brain.chat(
  [{ role: 'user', content: 'Explain connection pooling' }],
  { provider: 'openai', model: 'gpt-4o' }
);

// Disable auto-memory features
const response = await brain.chat(messages, {
  autoRecall: false,  // skip memory lookup
  autoStore: false,   // skip storing insights
});

// Pass-through options go directly to the LLM provider
const response = await brain.chat(messages, {
  max_tokens: 8192,
  temperature: 0.7,
});
```

Requires one of these installed separately (not bundled):
- `npm install @anthropic-ai/sdk` (for Anthropic/Claude)
- `npm install openai` (for OpenAI)

### Mem0-Compatible Aliases

```typescript
// add() — auto-detects title and type (PITFALL, DECISION, or PATTERN)
await brain.add('The bug was caused by a missing null check in the auth middleware');
// => stored as type: PITFALL (detected "bug")

await brain.add('We decided to use PostgreSQL over MongoDB for ACID compliance');
// => stored as type: DECISION (detected "decided")

// search() — alias for recall()
const results = await brain.search('database patterns', { limit: 5 });
```

### Memory

```typescript
// Store a memory
await brain.store('content here', {
  title: 'Short title',
  type: 'PATTERN',       // PATTERN | PITFALL | ARCHITECTURE | DEPENDENCY | TEST_INSIGHT | PERFORMANCE | DECISION | CONTEXT
  tier: 'semantic',      // working | episodic | semantic | procedural
  tags: ['tag1', 'tag2'],
  confidence: 0.95,
});

// Recall memories
const results = await brain.recall('search query', {
  limit: 10,
  tier: 'semantic',
  type: 'PATTERN',
  tags: ['database'],
  tokenBudget: 4000,
  fast: true,
  resolution: 'summaries',  // titles | summaries | full
});

// Verify a memory (reset confidence decay)
await brain.verify('knowledge-id');

// Forget (archive) a memory
await brain.forget('knowledge-id');

// Health stats
const stats = await brain.health();
console.log(stats.totalEntries, stats.staleCount);
```

### Bulk Operations

```typescript
// Store many entries
const { stored, errors } = await brain.bulkStore([
  { title: 'Pattern 1', content: 'Always validate input', type: 'PATTERN' },
  { title: 'Pitfall 1', content: 'Never trust user IDs from JWT without checking DB', type: 'PITFALL' },
]);

// Search multiple queries at once
const results = await brain.bulkRecall(['auth patterns', 'error handling', 'deployment']);
```

### Brain Management

```typescript
// Create a new brain
const { projectId } = await brain.setupBrain('Agent Brain', {
  description: 'Memory for my coding agent',
});

// Snapshot (backup)
const { id } = await brain.snapshot('before-refactor', 'Saving state before major changes');

// Restore from snapshot
await brain.restore(id);
```

### Messaging

```typescript
// Send a message to another agent
await brain.send('Build task completed', {
  recipientId: 'agent-id',
  channel: 'tasks',
});

// Broadcast to all agents in the project
await brain.send('Deploy starting in 5 minutes', { broadcast: true });

// Check messages
const msgs = await brain.messages({ unreadOnly: true });

// Wake an agent
await brain.wake('agent-id', 'New PR needs review');
```

### Graph (Memory Relationships)

```typescript
// Link two memories
await brain.link('source-id', 'target-id', 'depends_on');

// Get related memories
const related = await brain.related('knowledge-id');
```

### Timeline

```typescript
// Activity timeline
const events = await brain.timeline(20);

// Diff since a timestamp
const changes = await brain.diff('2025-01-01T00:00:00Z');
console.log(`${changes.added} added, ${changes.updated} updated`);
```

### Projects

```typescript
// List projects
const projects = await brain.projects();

// Switch project
brain.setProject('other-project-id');
```

### Utility

```typescript
const me = await brain.whoami();
console.log(me.name, me.credits);
```

## Error Handling

```typescript
import { AgentBay, AgentBayError } from '@agentbay/sdk';

try {
  await brain.recall('query');
} catch (err) {
  if (err instanceof AgentBayError) {
    console.error(err.message, err.status, err.code);
    // err.status: HTTP status code (401, 404, 429, etc.)
    // err.code: 'UNAUTHORIZED' | 'NOT_FOUND' | 'RATE_LIMITED' | 'FORBIDDEN' | etc.
  }
}
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `apiKey` | `string` | required | Your AgentBay API key |
| `baseUrl` | `string` | `https://www.aiagentsbay.com` | API base URL |
| `projectId` | `string` | — | Default project for all operations |
| `timeout` | `number` | `30000` | Request timeout in milliseconds |

## Requirements

- Node.js >= 18.0.0 (uses built-in `fetch`)
- TypeScript >= 5.4 (for development)

## License

MIT
