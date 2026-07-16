import { useEffect, useRef, useState } from "react";
import { deleteDocument, listDocuments, uploadDocument, type DocumentOut } from "../api";

interface Props {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

export function DocumentPanel({ selectedId, onSelect }: Props) {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = () => listDocuments().then(setDocuments).catch((e) => setError(e.message));

  useEffect(() => {
    refresh();
  }, []);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      await uploadDocument(file);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const handleDelete = async (id: string, filename: string) => {
    // A confirmation dialog is the cheapest possible safeguard against a
    // misclick — deletion is irreversible (the vectors and the source PDF
    // are both gone), and there's no undo anywhere in this app.
    if (!window.confirm(`Delete "${filename}"? This cannot be undone.`)) return;
    await deleteDocument(id);
    if (selectedId === id) onSelect(null);
    await refresh();
  };

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-sm font-semibold text-neutral-500 uppercase tracking-wide mb-2">
          Documents
        </h2>
        <label className="flex flex-col items-center justify-center border border-dashed border-neutral-300 rounded-lg h-24 cursor-pointer hover:border-neutral-500 hover:bg-neutral-100 transition-colors text-sm text-neutral-500">
          <span>{uploading ? "Uploading & embedding..." : "Click to upload a document"}</span>
          {!uploading && <span className="text-xs text-neutral-400 mt-1">PDF, DOCX, TXT, or MD</span>}
          <input
            ref={fileInput}
            type="file"
            accept=".pdf,.docx,.txt,.md"
            className="hidden"
            disabled={uploading}
            onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
          />
        </label>
        {error && <p className="text-red-700 text-xs mt-2">{error}</p>}
      </div>

      <div className="flex flex-col gap-1">
        <button
          onClick={() => onSelect(null)}
          className={`text-left text-sm px-3 py-2 rounded-md transition-colors ${
            selectedId === null
              ? "bg-neutral-900 text-white"
              : "text-neutral-600 hover:bg-neutral-200"
          }`}
        >
          All documents
        </button>
        {documents.map((doc) => (
          <div
            key={doc.id}
            className={`group flex items-center justify-between text-sm px-3 py-2 rounded-md cursor-pointer transition-colors ${
              selectedId === doc.id
                ? "bg-neutral-900 text-white"
                : "text-neutral-800 hover:bg-neutral-200"
            }`}
            onClick={() => onSelect(doc.id)}
          >
            <div className="truncate">
              <div className="truncate">{doc.filename}</div>
              <div className="text-xs text-neutral-400">{doc.num_chunks} chunks</div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDelete(doc.id, doc.filename);
              }}
              className="opacity-0 group-hover:opacity-100 text-neutral-400 hover:text-red-700 text-xs px-2"
            >
              delete
            </button>
          </div>
        ))}
        {documents.length === 0 && (
          <p className="text-xs text-neutral-400 px-3">No documents uploaded yet.</p>
        )}
      </div>
    </div>
  );
}
