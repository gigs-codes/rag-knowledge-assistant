"""
Document management endpoints: upload, list, delete.

Why routes stay thin: this file only does HTTP concerns (accept upload,
validate content-type, translate service exceptions to HTTP status codes).
All real logic — extraction, chunking, embedding, storage — lives in
IngestionService, which means it's testable without spinning up FastAPI
at all.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.deps import get_ingestion_service, get_registry
from app.models.schemas import DocumentOut, UploadResponse
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    try:
        record = ingestion_service.ingest_pdf(file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return UploadResponse(document=DocumentOut(**record), message="Document ingested successfully.")


@router.get("", response_model=list[DocumentOut])
def list_documents(registry: DocumentRegistry = Depends(get_registry)):
    return registry.list()


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    registry: DocumentRegistry = Depends(get_registry),
):
    if registry.get(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    ingestion_service.delete_document(document_id)
    return {"message": "Document deleted."}
