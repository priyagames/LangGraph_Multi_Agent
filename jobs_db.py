import os
import random
import re
from datetime import date
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

DB_NAME = "jd_intake_db"
COLLECTION_NAME = "jobs"

_client = None


class JobsDatabaseError(RuntimeError):
    """Raised when job lookup/publish cannot connect to or query MongoDB."""


def _get_collection():
    """Return the `jobs` collection, connecting to MongoDB (via MONGO_URI) on first use."""
    global _client
    if _client is None:
        try:
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri:
                raise JobsDatabaseError(
                    "MONGO_URI environment variable is not set. "
                    "Add your MongoDB Atlas connection string to .env (see .env.example)."
                )
            _client = MongoClient(mongo_uri)
            _client.admin.command("ping")
        except JobsDatabaseError:
            raise
        except ServerSelectionTimeoutError as exc:
            raise JobsDatabaseError(
                "MongoDB Atlas is not reachable from this machine. This usually means your current IP is not allowed in Atlas Network Access, or the cluster is offline. Add your IP in Atlas > Network Access and retry."
            ) from exc
        except Exception as exc:
            raise JobsDatabaseError(
                "The job database is unavailable right now. Please try again in a moment."
            ) from exc
    return _client[DB_NAME][COLLECTION_NAME]


def _db_call(function_name: str, action):
    try:
        return action()
    except JobsDatabaseError:
        raise
    except Exception as exc:
        raise JobsDatabaseError(
            f"The job database is unavailable while {function_name}. Please try again in a moment."
        ) from exc


def find_duplicate_job(title: str, company: str):
    """Return the existing job doc with the same title+company (case/whitespace-insensitive), or None.

    Compares against precomputed lowercase fields rather than a regex match — a raw keyword
    dropped into MongoDB's $regex can crash on special characters (seen firsthand in the
    Day 1 job search tool), and it's slower than an indexable equality check anyway.
    """
    return _db_call(
        "checking for duplicates",
        lambda: _get_collection().find_one({
            "title_lower": title.strip().lower(),
            "company_lower": company.strip().lower(),
        }),
    )


def _slug_job_id(company: str) -> str:
    """Create a recruiter-friendly, URL-safe slug from a company name."""
    slug = re.sub(r"[^A-Z0-9]+", "-", (company or "job").strip().upper()).strip("-")
    return slug[:20] or "JOB"


def _generate_job_id(collection, company: str) -> str:
    """Create a readable job code like 'ACME-1048' and keep it unique."""
    prefix = _slug_job_id(company)
    pattern = rf"^{re.escape(prefix)}-(\d+)$"
    highest = 0

    if collection is not None and hasattr(collection, "find"):
        for doc in collection.find({"job_id": {"$regex": pattern}}, {"job_id": 1, "_id": 0}):
            match = re.search(r"-(\d+)$", doc.get("job_id", ""))
            if match:
                highest = max(highest, int(match.group(1)))

    next_number = highest + 1 if highest else random.randint(1000, 9999)
    candidate = f"{prefix}-{next_number}"

    if collection is not None and hasattr(collection, "find_one"):
        if collection.find_one({"job_id": candidate}):
            return _generate_job_id(collection, company)

    return candidate


def _ensure_job_id(collection, doc: dict) -> dict:
    """Backfill readable IDs for legacy docs created before job_id was introduced."""
    if doc.get("job_id"):
        return doc

    company = (doc.get("company") or doc.get("title") or "JOB").strip() or "JOB"
    job_id = _generate_job_id(collection, company)
    collection.update_one({"_id": doc["_id"]}, {"$set": {"job_id": job_id}})
    doc["job_id"] = job_id
    return doc


def delete_job_by_id(job_id: str) -> bool:
    """Delete exactly one job using its human-readable job_id."""
    job_id = (job_id or "").strip()
    if not job_id:
        raise ValueError("A valid job ID is required to delete a job.")

    def action():
        collection = _get_collection()
        result = collection.delete_one({"job_id": job_id})
        return result.deleted_count > 0

    return _db_call("deleting a job", action)


def update_job_by_id(job_id: str, updates: dict) -> dict | None:
    """Update a job by exact job_id, rejecting duplicate title/company combinations."""
    job_id = (job_id or "").strip()
    if not job_id:
        raise ValueError("A valid job ID is required to update a job.")
    if not isinstance(updates, dict) or not updates:
        raise ValueError("At least one field update is required.")

    def action():
        collection = _get_collection()
        existing = collection.find_one({"job_id": job_id})
        if not existing:
            return None

        normalized_updates = {}
        new_title = existing.get("title")
        new_company = existing.get("company")

        for key, value in updates.items():
            if value is None or value == "":
                continue
            if key in {"title", "company", "location", "exp", "work_mode", "description"}:
                normalized_updates[key] = str(value)
            elif key == "quality_score":
                normalized_updates[key] = int(value)

        if "title" in normalized_updates:
            new_title = normalized_updates["title"]
            normalized_updates["title_lower"] = new_title.strip().lower()
        if "company" in normalized_updates:
            new_company = normalized_updates["company"]
            normalized_updates["company_lower"] = new_company.strip().lower()

        if new_title and new_company:
            duplicate = collection.find_one({
                "title_lower": new_title.strip().lower(),
                "company_lower": new_company.strip().lower(),
                "job_id": {"$ne": job_id},
            })
            if duplicate:
                raise ValueError(f"A job with the same title and company already exists: {duplicate.get('job_id')}")

        result = collection.find_one_and_update(
            {"job_id": job_id},
            {"$set": normalized_updates},
            return_document=True,
        )
        return result

    return _db_call("updating a job", action)


def publish_job_in_db(parsed: dict, description: str, quality_score: int) -> str:
    """Insert a published job posting. Returns a recruiter-friendly job code."""
    def action():
        collection = _get_collection()
        company = (parsed.get("company") or "Company").strip()
        job_id = _generate_job_id(collection, company)
        doc = {
            **parsed,
            "job_id": job_id,
            "title_lower": parsed["title"].strip().lower(),
            "company_lower": company.lower(),
            "description": description,
            "quality_score": quality_score,
            "date_posted": date.today().isoformat(),
        }
        result = collection.insert_one(doc)
        return job_id

    return _db_call("publishing a job", action)


def get_recent_jobs(limit: int = 5, fields: Optional[list] = None) -> list:
    """Return the `limit` most recently published jobs, newest first.

    `fields` optionally restricts which parsed fields come back (title, company,
    location, exp, work_mode) — the caller (the chat agent) picks this based on
    what the user actually asked for, e.g. "just the job titles". `_id` is always
    converted to a string and always included so callers can reference a specific
    job unambiguously; `description` and `quality_score` are always included too
    since they're commonly useful and cheap to return.

    Sorting is by `_id` (a MongoDB ObjectId, monotonically increasing with insertion
    time) rather than `date_posted`, since `date_posted` is only day-granularity and
    ties would leave ordering among same-day postings undefined.
    """

    def action():
        collection = _get_collection()
        limit_value = max(1, int(limit))

        projection = None
        if fields:
            allowed = {"title", "company", "location", "exp", "work_mode"}
            projection = {f: 1 for f in fields if f in allowed}
            projection["_id"] = 1
            projection["job_id"] = 1
        else:
            projection = {"job_id": 1, "_id": 1}

        cursor = collection.find({}, projection).sort("_id", -1).limit(limit_value)
        jobs = []
        for doc in cursor:
            doc = _ensure_job_id(collection, doc)
            doc["_id"] = str(doc["_id"])
            jobs.append(doc)
        return jobs

    return _db_call("retrieving recent jobs", action)


def find_job_for_resume_match(reference: str):
    """Resolve a human-given reference — a job_id or a title (full or partial) —
    to a single published job doc, or None if nothing matches.

    Tries an exact job_id lookup first (cheap, unambiguous), then falls back to
    a case-insensitive substring match on title. If the substring match is
    ambiguous (multiple jobs), returns None rather than guessing — the caller
    should ask the human to disambiguate rather than silently picking one.
    """

    def action():
        collection = _get_collection()
        reference_text = reference.strip()
        if not reference_text:
            return None

        try:
            by_id = collection.find_one({"_id": ObjectId(reference_text)})
            if by_id:
                by_id = _ensure_job_id(collection, by_id)
                by_id["_id"] = str(by_id["_id"])
                return by_id
        except (InvalidId, TypeError):
            pass

        by_job_id = collection.find_one({"job_id": reference_text})
        if by_job_id:
            by_job_id = _ensure_job_id(collection, by_job_id)
            by_job_id["_id"] = str(by_job_id["_id"])
            return by_job_id

        matches = list(collection.find({"title_lower": {"$regex": _escape_regex(reference_text.lower())}}))
        if len(matches) == 1:
            matches[0]["_id"] = str(matches[0]["_id"])
            return matches[0]
        return None

    return _db_call("matching a job reference", action)


def _escape_regex(text: str) -> str:
    """Escape regex special characters before dropping user text into $regex —
    an unescaped title (e.g. containing '(', '+', '.') can otherwise throw."""
    import re
    return re.escape(text)
