/**
 * SSE client for POST /stream (Phase 4, Step 2).
 *
 * The browser's EventSource only issues GET requests and cannot send a
 * JSON body, so the wire format is consumed by hand: fetch the response as
 * a stream and parse the `event:` / `data:` frames as they arrive.
 *
 * Frames are split on a blank line, and a partial frame is carried over
 * between reads. A network chunk boundary falls wherever TCP puts it, not
 * on frame boundaries, so parsing each chunk independently would silently
 * drop the token straddling the split - producing an answer missing a word
 * somewhere in the middle, which is far worse than a visible failure.
 *
 * Both CRLF and LF line endings are accepted. sse-starlette, which serves
 * /stream, emits CRLF, so frames are separated by a CR LF CR LF sequence.
 * Splitting on a bare LF LF matches nothing against that, and the failure
 * is silent: a valid 200 response, every frame delivered, and not one of
 * them parsed. Verified against the real endpoint rather than assumed.
 */

export type Citation = {
  chunk_id: string;
  municipality_name: string;
  bylaw_name: string;
  section_number: string;
  section_title: string | null;
  page_number: number | null;
  source_url: string;
  language: string;
};

export type StreamHandlers = {
  onCitations?: (citations: Citation[]) => void;
  onToken?: (token: string) => void;
  onError?: (message: string) => void;
};

export type StreamRequest = {
  query: string;
  municipality_id: string;
  province_code: string;
  language: string;
  session_id?: string;
};

/** Thrown for HTTP-level failures, before any SSE frame arrives. */
export class StreamHttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }

  /**
   * Rate limiting is an expected condition on a free-tier deployment, not
   * a fault. The UI says so in plain terms rather than showing an error.
   */
  get isRateLimited(): boolean {
    return this.status === 429;
  }
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const SESSION_HEADER = "X-Session-Id";
const SESSION_STORAGE_KEY = "bylaw-assistant-session";

/**
 * Opaque per-tab id used only for rate limiting.
 *
 * Kept in sessionStorage so one tab keeps one budget, and held per tab
 * rather than per browser so a second tab is not throttled by the first.
 * It identifies nobody: the backend combines it with the caller's IP to
 * form a limiter key and never stores it in the audit log.
 */
function sessionId(): string {
  if (typeof window === "undefined") return "server";
  try {
    const existing = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existing) return existing;
    const fresh = crypto.randomUUID();
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, fresh);
    return fresh;
  } catch {
    // Private browsing can refuse storage; a per-call id still gives the
    // IP half of the key something to combine with.
    return crypto.randomUUID();
  }
}

async function messageFor(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    // Non-JSON error body; fall through to the generic message.
  }
  return `Request failed (HTTP ${response.status}).`;
}

function dispatch(frame: string, handlers: StreamHandlers): void {
  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return;

  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return;
  }

  switch (eventName) {
    case "citations":
      if (Array.isArray(payload)) handlers.onCitations?.(payload as Citation[]);
      break;
    case "token":
      if (typeof payload === "string") handlers.onToken?.(payload);
      break;
    case "error":
      handlers.onError?.(
        typeof payload === "string" ? payload : "The assistant failed.",
      );
      break;
    default:
      break;
  }
}

export async function streamAnswer(
  request: StreamRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      [SESSION_HEADER]: sessionId(),
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new StreamHttpError(response.status, await messageFor(response));
  }
  if (!response.body) {
    throw new StreamHttpError(response.status, "The response had no body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      // `stream: true` keeps a multi-byte character split across chunks
      // intact - French bylaw text is full of accented characters.
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split(/\r?\n\r?\n/);
      // The trailing element is an incomplete frame; hold it for the next
      // read rather than parsing a truncated token. This also covers a
      // chunk that ends part-way through a CR LF CR LF separator.
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        if (frame.trim()) dispatch(frame, handlers);
      }
    }

    if (buffer.trim()) dispatch(buffer, handlers);
  } finally {
    reader.releaseLock();
  }
}
