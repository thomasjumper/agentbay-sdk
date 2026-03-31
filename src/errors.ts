export class AgentBayError extends Error {
  public readonly status: number;
  public readonly code: string | undefined;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = 'AgentBayError';
    this.status = status;
    this.code = code;
  }
}
