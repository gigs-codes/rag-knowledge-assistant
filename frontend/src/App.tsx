import { useState } from "react";
import { DocumentPanel } from "./components/DocumentPanel";
import { AskPanel } from "./components/AskPanel";
import { LoginForm } from "./components/LoginForm";
import { useAuth } from "./auth";

function App() {
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const { user, loading, logout } = useAuth();

  if (loading) return null;
  if (!user) return <LoginForm />;

  return (
    <div className="h-screen bg-white text-neutral-900 flex flex-col">
      <header className="border-b border-neutral-200 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold">Enterprise Knowledge Assistant</h1>
          <p className="text-xs text-neutral-500">
            Local RAG · Ollama (phi3:mini) · bge-small-en-v1.5 embeddings · ChromaDB
          </p>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <span className="text-neutral-600">
            {user.username} <span className="text-neutral-400">({user.role})</span>
          </span>
          <button onClick={logout} className="text-xs text-neutral-500 hover:text-neutral-900">
            Sign out
          </button>
        </div>
      </header>
      <div className="flex flex-1 overflow-hidden">
        <aside className="w-72 border-r border-neutral-200 bg-neutral-50 p-4 overflow-y-auto">
          <DocumentPanel selectedId={selectedDocId} onSelect={setSelectedDocId} role={user.role} />
        </aside>
        <main className="flex-1 p-6 overflow-hidden">
          <AskPanel documentId={selectedDocId} />
        </main>
      </div>
    </div>
  );
}

export default App;
