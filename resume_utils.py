"""
Resume text extraction — turns an uploaded resume file (sent as base64 over the
chat endpoint) into plain text the LLM can read, for PDF, DOCX, and TXT.

Kept as plain, non-LLM Python: extracting text from a known file format is a
mechanical task, not something that benefits from a model call.
"""

import base64
import io

MAX_RESUME_BYTES = 5 * 1024 * 1024  # 5 MB — generous for a resume, cheap to enforce


class ResumeExtractionError(Exception):
    """Raised when a resume file can't be decoded or read (bad base64, corrupt
    file, unsupported extension). Callers should treat this the same as any
    other "please clarify" case — not a crash."""


def extract_resume_text(filename: str, base64_content: str) -> str:
    ext = (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""

    try:
        raw = base64.b64decode(base64_content, validate=True)
    except Exception as exc:
        raise ResumeExtractionError(f"Couldn't decode the uploaded file ({exc}).") from exc

    if len(raw) > MAX_RESUME_BYTES:
        raise ResumeExtractionError(
            f"That file is too large ({len(raw) / 1024 / 1024:.1f} MB) — please upload a resume under 5 MB."
        )

    if ext == "pdf":
        return _extract_pdf(raw)
    if ext == "docx":
        return _extract_docx(raw)
    if ext in ("txt", "md", ""):
        try:
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception as exc:
            raise ResumeExtractionError(f"Couldn't read the file as text ({exc}).") from exc

    raise ResumeExtractionError(
        f"Unsupported resume file type '.{ext}' — please upload a .pdf, .docx, or .txt file."
    )


def _extract_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ResumeExtractionError(
            "PDF support isn't installed on the server (missing `pypdf`). Run `pip install pypdf`."
        ) from exc

    reader = PdfReader(io.BytesIO(raw))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = text.strip()
    if not text:
        raise ResumeExtractionError(
            "Couldn't find any text in that PDF — it may be a scanned image without a text layer."
        )
    return text


def _extract_docx(raw: bytes) -> str:
    try:
        import docx
    except ImportError as exc:
        raise ResumeExtractionError(
            "DOCX support isn't installed on the server (missing `python-docx`). Run `pip install python-docx`."
        ) from exc

    document = docx.Document(io.BytesIO(raw))
    text = "\n".join(p.text for p in document.paragraphs).strip()
    if not text:
        raise ResumeExtractionError("Couldn't find any text in that DOCX file.")
    return text
