"""
Document management endpoints: upload, list, delete.

Why routes stay thin: this file only does HTTP concerns (accept upload,
validate content-type, translate service exceptions to HTTP status codes).
All real logic — extraction, chunking, embedding, storage — lives in
IngestionService, which means it's testable without spinning up FastAPI
at all.
"""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.deps import get_ingestion_service, get_registry
from app.api.security import get_current_user, require_role
from app.models.schemas import DocumentOut, UploadResponse
from app.services.document_registry_base import DocumentRegistryBase
from app.services.ingestion_service import SUPPORTED_EXTENSIONS, IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(require_role("admin"))])
async def upload_document(
    file: UploadFile,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    # Validated by extension, not content-type: browsers send inconsistent
    # (or empty) MIME types for .txt/.md/.docx depending on OS and how the
    # file was created, whereas the extension is what the user actually
    # sees and chose. This is also a cheap upfront rejection before
    # reading a potentially large file into memory.
    extension = Path(file.filename or "").suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    file_bytes = await file.read()
    try:
        record = ingestion_service.ingest_document(file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return UploadResponse(document=DocumentOut(**record), message="Document ingested successfully.")


@router.get("", response_model=list[DocumentOut], dependencies=[Depends(get_current_user)])
def list_documents(registry: DocumentRegistryBase = Depends(get_registry)):
    return registry.list()


@router.delete("/{document_id}", dependencies=[Depends(require_role("admin"))])
def delete_document(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    registry: DocumentRegistryBase = Depends(get_registry),
):
    if registry.get(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    ingestion_service.delete_document(document_id)
    return {"message": "Document deleted."}
