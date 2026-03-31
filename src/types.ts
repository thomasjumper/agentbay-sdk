export interface StoreOptions {
  title: string;
  content: string;
  type?: 'PATTERN' | 'PITFALL' | 'ARCHITECTURE' | 'DEPENDENCY' | 'TEST_INSIGHT' | 'PERFORMANCE' | 'DECISION' | 'CONTEXT';
  tier?: 'working' | 'episodic' | 'semantic' | 'procedural';
  tags?: string[];
  filePaths?: string[];
  confidence?: number;
  aliases?: string[];
}

export interface RecallOptions {
  limit?: number;
  tier?: string;
  type?: string;
  tags?: string[];
  tokenBudget?: number;
  fast?: boolean;
  resolution?: 'titles' | 'summaries' | 'full';
}

export interface MemoryEntry {
  id: string;
  title: string;
  content: string;
  summary: string | null;
  type: string;
  tier: string;
  tags: string[];
  confidence: number;
  score: number;
  tokenCount: number;
}

export interface RecallResult {
  entries: MemoryEntry[];
  totalTokens: number;
  searchMode: string;
  strategies: string[];
}

export interface HealthStats {
  totalEntries: number;
  byTier: Record<string, number>;
  byType: Record<string, number>;
  totalTokens: number;
  staleCount: number;
  lowConfidenceCount: number;
}

export interface Message {
  id: string;
  from: string;
  content: string;
  channel: string;
  isBroadcast: boolean;
  createdAt: string;
}

export interface AgentBayConfig {
  apiKey: string;
  baseUrl?: string;
  projectId?: string;
  timeout?: number;
}
