import { useState } from "react";
import { DocumentPanel } from "./components/DocumentPanel";
import { AskPanel } from "./components/AskPanel";

function App() {
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  return (
    <div className="h-screen coffee-grid-bg text-[#2c1b12] flex flex-col">
      <header className="border-b border-[#e4d3bb] bg-[#faf3ea]/80 backdrop-blur-sm px-6 py-4">
        <h1 className="text-lg font-semibold">Enterprise Knowledge Assistant</h1>
        <p className="text-xs text-[#8b6f52]">
          Local RAG · Ollama (phi3:mini) · bge-small-en-v1.5 embeddings · ChromaDB
        </p>
      </header>
      <div className="flex flex-1 overflow-hidden">
        <aside className="w-72 border-r border-[#e4d3bb] bg-[#faf3ea]/60 backdrop-blur-sm p-4 overflow-y-auto">
          <DocumentPanel selectedId={selectedDocId} onSelect={setSelectedDocId} />
        </aside>
        <main className="flex-1 p-6 overflow-hidden">
          <AskPanel documentId={selectedDocId} />
        </main>
      </div>
    </div>
  );
}

export default App;
