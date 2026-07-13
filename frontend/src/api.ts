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

const BASE = "/api";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

export async function listDocuments(): Promise<DocumentOut[]> {
  return handle(await fetch(`${BASE}/documents`));
}

export async function uploadDocument(file: File): Promise<DocumentOut> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents/upload`, { method: "POST", body: form });
  const data = await handle<{ document: DocumentOut }>(res);
  return data.document;
}

export async function deleteDocument(id: string): Promise<void> {
  await handle(await fetch(`${BASE}/documents/${id}`, { method: "DELETE" }));
}

export async function askQuestion(question: string, documentId?: string): Promise<QueryResponse> {
  return handle(
    await fetch(`${BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, document_id: documentId ?? null }),
    })
  );
}
