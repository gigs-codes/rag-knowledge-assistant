// Thin API client — the only file that knows the backend's URL shape.
// Components call these functions and get typed data back; if the backend
// contract changes, this is the one file to update.

export interface DocumentOut {
  id: string;
  filename: string;
  num_chunks: number;
  uploaded_at: string;
}

export interface Citation {
  document_id: string;
  filename: string;
  chunk_index: number;
  text: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  latency_ms: number;
}

export interface UserOut {
  username: string;
  role: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: UserOut;
}

const BASE = "/api";
const TOKEN_KEY = "eka_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

// Every request goes through here so the Authorization header only needs
// to be built in one place. `init.headers` is kept as a plain object
// (not merged via the Headers class) since every call site already only
// ever passes a plain object, and this stays consistent with that.
async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const token = getToken();
  const headers = {
    ...(init.headers as Record<string, string> | undefined),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  return fetch(`${BASE}${path}`, { ...init, headers });
}

export async function register(username: string, password: string): Promise<TokenResponse> {
  return handle(
    await apiFetch("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    })
  );
}

export async function login(username: string, password: string): Promise<TokenResponse> {
  return handle(
    await apiFetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    })
  );
}

export async function getMe(): Promise<UserOut> {
  return handle(await apiFetch("/auth/me"));
}

export async function listDocuments(): Promise<DocumentOut[]> {
  return handle(await apiFetch("/documents"));
}

export async function uploadDocument(file: File): Promise<DocumentOut> {
  const form = new FormData();
  form.append("file", file);
  const res = await apiFetch("/documents/upload", { method: "POST", body: form });
  const data = await handle<{ document: DocumentOut }>(res);
  return data.document;
}

export async function deleteDocument(id: string): Promise<void> {
  await handle(await apiFetch(`/documents/${id}`, { method: "DELETE" }));
}

export async function askQuestion(question: string, documentId?: string): Promise<QueryResponse> {
  return handle(
    await apiFetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, document_id: documentId ?? null }),
    })
  );
}

export async function submitFeedback(
  question: string,
  answer: string,
  rating: "up" | "down"
): Promise<void> {
  await handle(
    await apiFetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, answer, rating, source: "query" }),
    })
  );
}

export type StreamEvent =
  | { type: "citations"; citations: Citation[] }
  | { type: "token"; text: string }
  | { type: "done"; latency_ms: number };

// Why fetch + manual stream reading instead of the browser's EventSource
// API: EventSource only supports GET requests with no request body, and
// the question needs to go up as JSON — reading the response body as a
// stream ourselves works with POST just fine. SSE's wire format
// (`data: {...}\n\n`) is simple enough to parse by hand rather than
// pulling in a library for it.
export async function askQuestionStream(
  question: string,
  documentId: string | undefined,
  onEvent: (event: StreamEvent) => void
): Promise<void> {
  const res = await apiFetch("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_id: documentId ?? null }),
  });
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line; a chunk from the network
    // may contain zero, one, or several complete events, so we split on
    // that separator and keep any trailing partial event in `buffer` for
    // the next read.
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      onEvent(JSON.parse(line.slice("data:".length).trim()));
    }
  }
}
