import sqlite3
import uuid

from flask import Flask, jsonify, render_template, request
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from jd_intake_pipeline import build_graph
from jobs_db import JobsDatabaseError, delete_job_by_id, update_job_by_id
from resume_utils import ResumeExtractionError, extract_resume_text

app = Flask(__name__)

# check_same_thread=False: Flask's dev server can serve requests on a different
# thread than the one that opened this connection.
_conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(_conn)
graph = build_graph(checkpointer)


def _format_result(result: dict) -> str:
    """Turn the final graph state into a polished, recruiter-friendly summary."""
    status = result["status"]

    if status == "answered":
        jobs = result.get("query_result", [])
        if not jobs:
            return "No recent jobs found."

        lines = ["Recent jobs:"]
        for index, job in enumerate(jobs, start=1):
            title = job.get("title") or "Untitled role"
            company = job.get("company") or "Unknown company"
            location = job.get("location") or "Location not specified"
            exp = job.get("exp") or "Experience not specified"
            work_mode = job.get("work_mode") or "Mode not specified"
            job_id = job.get("job_id") or job.get("_id") or "Unknown ID"
            lines.append(f"\n{index}. {title}")
            lines.append(f"   • Job ID: {job_id}")
            lines.append(f"   • Company: {company}")
            lines.append(f"   • Location: {location}")
            lines.append(f"   • Experience: {exp}")
            lines.append(f"   • Work mode: {work_mode}")
        return "\n".join(lines)

    if status == "published":
        return (
            "✅ Job posting created\n"
            f"Job ID: {result['job_id']}\n"
            "The listing is now live in the database."
        )

    if status == "rejected":
        return (
            "ℹ️ Duplicate posting detected\n"
            "This job already exists, so I did not create a new listing."
        )

    if status == "needs_input":
        return (
            "⚠️ Manual review required\n"
            "I couldn't complete this posting automatically, so it has been flagged for review."
        )

    if status == "qualified":
        job = result.get("target_job") or {}
        job_label = job.get("title", "that job")
        score = result.get("qualify_score")
        feedback = result.get("qualify_feedback") or "No additional feedback available."
        return (
            "Resume uploaded and processed successfully\n"
            f"Role: {job_label}\n"
            f"Score: {score}/100\n"
            f"Feedback: {feedback}"
        )

    return f"Status: {status}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    message = data.get("message", "")
    msg_lower = (message or "").strip().lower()
    thread_id = data.get("thread_id")

    if "thank you" in msg_lower or "thanks" in msg_lower:
        return jsonify({"reply": "You're welcome! Happy to help.", "thread_id": None, "done": True})

    if "delete" in msg_lower:
        match = __import__("re").search(r"\b[A-Z]{2,}-\d+\b", message, flags=__import__("re").I)
        job_id = match.group(0).upper() if match else None
        if not job_id:
            return jsonify({
                "reply": "To delete a job, send the exact job ID. Example: delete ACME-1048",
                "thread_id": None,
                "done": True,
            })
        try:
            deleted = delete_job_by_id(job_id)
            if deleted:
                return jsonify({"reply": f"Job {job_id} deleted successfully.", "thread_id": None, "done": True})
            return jsonify({"reply": f"No job found with ID {job_id}.", "thread_id": None, "done": True})
        except Exception as exc:
            return jsonify({"reply": str(exc), "thread_id": None, "done": True})

    if "update" in msg_lower:
        match = __import__("re").search(r"\b[A-Z]{2,}-\d+\b", message, flags=__import__("re").I)
        job_id = match.group(0).upper() if match else None
        if not job_id:
            return jsonify({"reply": "To update a job, send the exact job ID and the new details. Example: update ACME-1048 company to Acme Analytics", "thread_id": None, "done": True})
        updates = {}
        text = message.strip()
        lower_text = text.lower()
        if "company" in lower_text:
            updates["company"] = text.split("company", 1)[1].split(" to ", 1)[1].strip() if " to " in lower_text else text.split("company", 1)[1].strip()
        if "title" in lower_text:
            updates["title"] = text.split("title", 1)[1].split(" to ", 1)[1].strip() if " to " in lower_text else text.split("title", 1)[1].strip()
        if "location" in lower_text:
            updates["location"] = text.split("location", 1)[1].split(" to ", 1)[1].strip() if " to " in lower_text else text.split("location", 1)[1].strip()
        if "exp" in lower_text:
            updates["exp"] = text.split("exp", 1)[1].split(" to ", 1)[1].strip() if " to " in lower_text else text.split("exp", 1)[1].strip()
        if "work mode" in lower_text:
            updates["work_mode"] = text.split("work mode", 1)[1].split(" to ", 1)[1].strip() if " to " in lower_text else text.split("work mode", 1)[1].strip()
        if not updates:
            return jsonify({"reply": "I can update fields like title, company, location, exp, and work mode for the matching job ID.", "thread_id": None, "done": True})
        try:
            updated = update_job_by_id(job_id, updates)
            if updated:
                return jsonify({"reply": f"Job {job_id} updated successfully.", "thread_id": None, "done": True})
            return jsonify({"reply": f"No job found with ID {job_id}.", "thread_id": None, "done": True})
        except Exception as exc:
            return jsonify({"reply": str(exc), "thread_id": None, "done": True})

    # A resume upload rides along on the same /chat call as an ordinary message
    # (per the "same chat interface" requirement) — the client base64-encodes
    # the file and sends it as `resume_base64` + `resume_filename` alongside
    # whatever text names the job to score it against, e.g. "score this for
    # the Backend Engineer role".
    resume_base64 = data.get("resume_base64")
    resume_filename = data.get("resume_filename", "")

    try:
        if thread_id:
            # An open thread means the pipeline is mid-pause waiting on a missing
            # field, or on which job to score a resume against - this message is
            # the answer, not a new submission.
            config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke(Command(resume=message), config=config)
        elif resume_base64:
            try:
                resume_text = extract_resume_text(resume_filename, resume_base64)
            except ResumeExtractionError as exc:
                # Not a graph run at all - the file itself couldn't be read, so
                # there's nothing to invoke. Report it the same conversational way
                # as everything else and let the user retry with a fixed upload.
                return jsonify({"reply": str(exc), "thread_id": None, "done": True})

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke(
                {"raw_jd": message, "resume_text": resume_text, "target_job_reference": message},
                config=config,
            )
        else:
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke({"raw_jd": message}, config=config)
    except JobsDatabaseError as exc:
        return jsonify({"reply": str(exc), "thread_id": None, "done": True})
    except Exception as exc:
        app.logger.exception("Unexpected error while processing /chat request")
        return jsonify(
            {"reply": "The job service is temporarily unavailable. Please try again in a moment.", "thread_id": None, "done": True}
        )

    # The node-by-node trace (which node ran, what it decided) is dev-facing
    # debug info, not something a recruiter should see in the chat — print it
    # to the server console so `python app.py` still shows exactly what the
    # agent is doing under the hood, without leaking that into the reply text.
    for line in result.get("log", []):
        print(f"[trace] {line}")

    if "__interrupt__" in result:
        question = result["__interrupt__"][0].value["question"]
        return jsonify({"reply": question, "thread_id": thread_id, "done": False})

    return jsonify({"reply": _format_result(result), "thread_id": None, "done": True})


if __name__ == "__main__":
    app.run(debug=True)
