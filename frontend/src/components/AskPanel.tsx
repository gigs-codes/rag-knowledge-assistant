import { useEffect, useState } from "react";
import { askQuestionStream, type Citation } from "../api";

interface Props {
  documentId: string | null;
}

interface Exchange {
  question: string;
  answer?: string; // grows incrementally as stream tokens arrive
  citations?: Citation[];
  latencyMs?: number;
  streaming?: boolean;
  error?: string;
}

const HISTORY_KEY = "eka_history";

function loadHistory(): Exchange[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    // Persisted history never has `streaming: true` mid-stream (a page
    // refresh mid-answer just leaves it incomplete) — normalize on load
    // so an interrupted answer doesn't render a permanently-stuck spinner.
    const parsed: Exchange[] = raw ? JSON.parse(raw) : [];
    return parsed.map((ex) => ({ ...ex, streaming: false }));
  } catch {
    return [];
  }
}

export function AskPanel({ documentId }: Props) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<Exchange[]>(loadHistory);

  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  }, [history]);

  const updateLast = (updater: (ex: Exchange) => Exchange) => {
    setHistory((h) => h.map((ex, i) => (i === h.length - 1 ? updater(ex) : ex)));
  };

  const handleAsk = async () => {
    if (!question.trim() || loading) return;
    const q = question.trim();
    setQuestion("");
    setLoading(true);
    setHistory((h) => [...h, { question: q, answer: "", streaming: true }]);

    try {
      await askQuestionStream(q, documentId ?? undefined, (event) => {
        if (event.type === "citations") {
          updateLast((ex) => ({ ...ex, citations: event.citations }));
        } else if (event.type === "token") {
          updateLast((ex) => ({ ...ex, answer: (ex.answer ?? "") + event.text }));
        } else if (event.type === "done") {
          updateLast((ex) => ({ ...ex, latencyMs: event.latency_ms, streaming: false }));
        }
      });
    } catch (e) {
      updateLast((ex) => ({ ...ex, error: (e as Error).message, streaming: false }));
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="flex flex-col h-full">
      {history.length > 0 && (
        <div className="flex justify-end pb-2">
          <button onClick={clearHistory} className="text-xs text-neutral-400 hover:text-neutral-700">
            Clear conversation
          </button>
        </div>
      )}
      <div className="flex-1 overflow-y-auto flex flex-col gap-6 pb-4">
        {history.length === 0 && (
          <p className="text-neutral-500 text-sm">
            Ask a question about your uploaded documents. Answers are grounded only in retrieved
            content and include citations back to the source chunk.
          </p>
        )}
        {history.map((ex, i) => (
          <div key={i} className="flex flex-col gap-2">
            <div className="self-end bg-neutral-900 text-white text-sm px-4 py-2 rounded-2xl rounded-br-sm max-w-[80%] shadow-sm">
              {ex.question}
            </div>
            {ex.error && (
              <div className="text-sm text-red-700 max-w-[80%]">Error: {ex.error}</div>
            )}
            {!ex.error && (ex.answer || ex.streaming) && (
              <div className="flex flex-col gap-2 max-w-[85%]">
                {ex.answer ? (
                  <div className="bg-neutral-100 text-neutral-900 text-sm px-4 py-3 rounded-2xl rounded-bl-sm whitespace-pre-wrap border border-neutral-200 shadow-sm">
                    {ex.answer}
                    {ex.streaming && <span className="animate-pulse">▍</span>}
                  </div>
                ) : (
                  <div className="text-sm text-neutral-500 animate-pulse">Thinking…</div>
                )}
                {ex.latencyMs !== undefined && (
                  <div className="text-xs text-neutral-400">{ex.latencyMs} ms</div>
                )}
                {ex.citations && ex.citations.length > 0 && (
                  <div className="flex flex-col gap-1">
                    {ex.citations.map((c, ci) => (
                      <details
                        key={ci}
                        className="text-xs bg-white border border-neutral-200 rounded-md px-3 py-2"
                      >
                        <summary className="cursor-pointer text-neutral-500">
                          [{ci + 1}] {c.filename} &middot; chunk {c.chunk_index} &middot; score{" "}
                          {c.score}
                        </summary>
                        <p className="mt-2 text-neutral-600 whitespace-pre-wrap">{c.text}</p>
                      </details>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="flex gap-2 border-t border-neutral-200 pt-4">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          placeholder="Ask a question about your documents..."
          className="flex-1 bg-white text-neutral-900 text-sm rounded-lg px-4 py-3 outline-none border border-neutral-300 focus:ring-2 focus:ring-neutral-400 placeholder:text-neutral-400"
        />
        <button
          onClick={handleAsk}
          disabled={loading || !question.trim()}
          className="bg-neutral-900 hover:bg-neutral-700 disabled:opacity-40 disabled:hover:bg-neutral-900 text-white text-sm font-medium px-5 rounded-lg transition-colors"
        >
          Ask
        </button>
      </div>
    </div>
  );
}
