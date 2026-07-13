import { useState } from "react";
import { askQuestion, type QueryResponse } from "../api";

interface Props {
  documentId: string | null;
}

interface Exchange {
  question: string;
  response?: QueryResponse;
  error?: string;
}

export function AskPanel({ documentId }: Props) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<Exchange[]>([]);

  const handleAsk = async () => {
    if (!question.trim() || loading) return;
    const q = question.trim();
    setQuestion("");
    setLoading(true);
    setHistory((h) => [...h, { question: q }]);
    try {
      const response = await askQuestion(q, documentId ?? undefined);
      setHistory((h) => h.map((ex, i) => (i === h.length - 1 ? { ...ex, response } : ex)));
    } catch (e) {
      setHistory((h) =>
        h.map((ex, i) => (i === h.length - 1 ? { ...ex, error: (e as Error).message } : ex))
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto flex flex-col gap-6 pb-4">
        {history.length === 0 && (
          <p className="text-[#8b6f52] text-sm">
            Ask a question about your uploaded documents. Answers are grounded only in retrieved
            content and include citations back to the source chunk.
          </p>
        )}
        {history.map((ex, i) => (
          <div key={i} className="flex flex-col gap-2">
            <div className="self-end bg-[#c47a00] text-white text-sm px-4 py-2 rounded-2xl rounded-br-sm max-w-[80%] shadow-sm">
              {ex.question}
            </div>
            {ex.error && (
              <div className="text-sm text-red-700 max-w-[80%]">Error: {ex.error}</div>
            )}
            {ex.response && (
              <div className="flex flex-col gap-2 max-w-[85%]">
                <div className="bg-white text-[#2c1b12] text-sm px-4 py-3 rounded-2xl rounded-bl-sm whitespace-pre-wrap border border-[#e4d3bb] shadow-sm">
                  {ex.response.answer}
                </div>
                <div className="text-xs text-[#a8916f]">{ex.response.latency_ms} ms</div>
                {ex.response.citations.length > 0 && (
                  <div className="flex flex-col gap-1">
                    {ex.response.citations.map((c, ci) => (
                      <details
                        key={ci}
                        className="text-xs bg-white/70 border border-[#e4d3bb] rounded-md px-3 py-2"
                      >
                        <summary className="cursor-pointer text-[#8b6f52]">
                          [{ci + 1}] {c.filename} &middot; chunk {c.chunk_index} &middot; score{" "}
                          {c.score}
                        </summary>
                        <p className="mt-2 text-[#7a5c40] whitespace-pre-wrap">{c.text}</p>
                      </details>
                    ))}
                  </div>
                )}
              </div>
            )}
            {!ex.response && !ex.error && (
              <div className="text-sm text-[#8b6f52] max-w-[80%] animate-pulse">Thinking…</div>
            )}
          </div>
        ))}
      </div>

      <div className="flex gap-2 border-t border-[#e4d3bb] pt-4">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          placeholder="Ask a question about your documents..."
          className="flex-1 bg-white text-[#2c1b12] text-sm rounded-lg px-4 py-3 outline-none border border-[#e4d3bb] focus:ring-2 focus:ring-[#e8a74a] placeholder:text-[#a8916f]"
        />
        <button
          onClick={handleAsk}
          disabled={loading || !question.trim()}
          className="bg-[#c47a00] hover:bg-[#e8a74a] disabled:opacity-40 disabled:hover:bg-[#c47a00] text-white text-sm font-medium px-5 rounded-lg transition-colors"
        >
          Ask
        </button>
      </div>
    </div>
  );
}
