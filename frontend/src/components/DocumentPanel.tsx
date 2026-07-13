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

  const handleDelete = async (id: string) => {
    await deleteDocument(id);
    if (selectedId === id) onSelect(null);
    await refresh();
  };

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-sm font-semibold text-[#5a4636] uppercase tracking-wide mb-2">
          Documents
        </h2>
        <label className="flex items-center justify-center border border-dashed border-[#c9a97e] rounded-lg h-24 cursor-pointer hover:border-[#c47a00] hover:bg-[#f3ddb2]/40 transition-colors text-sm text-[#8b6f52]">
          {uploading ? "Uploading & embedding..." : "Click to upload a PDF"}
          <input
            ref={fileInput}
            type="file"
            accept="application/pdf"
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
              ? "bg-[#f3ddb2]/60 text-[#8b4e00]"
              : "text-[#7a5c40] hover:bg-[#f0e4d3]"
          }`}
        >
          All documents
        </button>
        {documents.map((doc) => (
          <div
            key={doc.id}
            className={`group flex items-center justify-between text-sm px-3 py-2 rounded-md cursor-pointer transition-colors ${
              selectedId === doc.id
                ? "bg-[#f3ddb2]/60 text-[#8b4e00]"
                : "text-[#4a3626] hover:bg-[#f0e4d3]"
            }`}
            onClick={() => onSelect(doc.id)}
          >
            <div className="truncate">
              <div className="truncate">{doc.filename}</div>
              <div className="text-xs text-[#8b6f52]">{doc.num_chunks} chunks</div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDelete(doc.id);
              }}
              className="opacity-0 group-hover:opacity-100 text-[#8b6f52] hover:text-red-700 text-xs px-2"
            >
              delete
            </button>
          </div>
        ))}
        {documents.length === 0 && (
          <p className="text-xs text-[#a8916f] px-3">No documents uploaded yet.</p>
        )}
      </div>
    </div>
  );
}
