"""HTTP surface for detector, reader, and NLP outputs."""

from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from visual_nlp.api.settings import load_settings
from visual_nlp.pipeline.document import DocumentPipeline
from visual_nlp.pipeline.image import ImagePipeline, build_pipeline

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PDF_SUFFIXES = {".pdf"}
ALLOWED_SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES
CONTENT_TYPE_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "application/pdf": ".pdf",
}


def _load_pipeline() -> ImagePipeline:
    settings = load_settings()
    return build_pipeline(
        reader_config=settings.reader_config,
        reader_checkpoint=settings.reader_checkpoint,
        detector_config=settings.detector_config,
        detector_checkpoint=settings.detector_checkpoint,
        nlp_config=settings.nlp_config,
        nlp_checkpoint=settings.nlp_checkpoint,
    )


def _suffix_for_upload(upload: UploadFile) -> str:
    filename = Path(upload.filename or "")
    suffix = filename.suffix.lower()
    if suffix in ALLOWED_SUFFIXES:
        return suffix
    mapped = CONTENT_TYPE_SUFFIX.get((upload.content_type or "").split(";")[0].strip().lower())
    if mapped:
        return mapped
    raise HTTPException(status_code=400, detail="Unsupported file type. Upload an image or PDF.")


def _require_models(pipeline: ImagePipeline, mode: str, use_nlp: bool) -> None:
    if mode == "line" and pipeline.reader is None:
        raise HTTPException(status_code=503, detail="Reader is not loaded.")
    if mode == "full" and (pipeline.detector is None or pipeline.reader is None):
        raise HTTPException(status_code=503, detail="Full-image mode requires detector and reader.")
    if use_nlp and pipeline.nlp is None:
        raise HTTPException(status_code=503, detail="NLP is not loaded.")


def _write_upload(upload: UploadFile, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        shutil.copyfileobj(upload.file, handle)
        return Path(handle.name)


def _validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc


def create_app(pipeline: ImagePipeline | None = None) -> FastAPI:
    injected = pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = injected if injected is not None else _load_pipeline()
        app.state.document_pipeline = DocumentPipeline(app.state.pipeline)
        app.state.lock = threading.Lock()
        yield

    app = FastAPI(
        title="Visual NLP",
        version="0.1.0",
        description="HTTP access to detector, reader, and NLP document outputs.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict:
        loaded: ImagePipeline = request.app.state.pipeline
        return {
            "status": "ok",
            "device": str(loaded.device),
            "models": {
                "reader": "loaded" if loaded.reader is not None else "absent",
                "detector": "loaded" if loaded.detector is not None else "absent",
                "nlp": "loaded" if loaded.nlp is not None else "absent",
            },
        }

    @app.post("/predict")
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        mode: str | None = Form(None),
        use_nlp: bool = Form(False),
    ) -> dict:
        suffix = _suffix_for_upload(file)
        resolved_mode = mode or ("full" if suffix in PDF_SUFFIXES else "line")
        if resolved_mode not in {"line", "full"}:
            raise HTTPException(status_code=400, detail="mode must be 'line' or 'full'.")
        pipeline_obj: ImagePipeline = request.app.state.pipeline
        _require_models(pipeline_obj, resolved_mode, use_nlp)
        path = _write_upload(file, suffix)
        try:
            if suffix in IMAGE_SUFFIXES:
                _validate_image(path)
            document_id = Path(file.filename or path.name).stem
            with request.app.state.lock:
                if suffix in PDF_SUFFIXES:
                    result = request.app.state.document_pipeline.run(
                        path,
                        mode=resolved_mode,
                        use_nlp=use_nlp,
                        document_id=document_id,
                    )
                else:
                    result = pipeline_obj.run(
                        path,
                        mode=resolved_mode,
                        use_nlp=use_nlp,
                        document_id=document_id,
                    )
        except HTTPException:
            raise
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not process file: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)
        return result.to_dict()

    return app


app = create_app()
