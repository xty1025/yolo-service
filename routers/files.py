from fastapi import APIRouter, File,UploadFile
router = APIRouter(prefix="/files",tags=["文件"])

@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    data = file.file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(data),
    }