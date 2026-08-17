"""
JD Intake Pipeline — a LangGraph workflow, not a multi-agent system.

Every node below is a deterministic step: some are plain Python (duplicate check,
routing decisions), some make a single LLM call for one specific job (intent
classification, extraction, field validation, scoring, rewriting). None of them
reason over their own tools in a loop the way Day 1's `create_agent()` agent did —
that's the point of this example. LangGraph here is used to orchestrate an explicit,
typed, resumable *workflow*, with loops that have circuit breakers and a
human-in-the-loop pause for missing/invalid information.

Chat-facing behavior added on top of the original "parse -> score -> publish" flow:

1. Every fresh message goes through `intake`, a single LLM call that decides
   whether it's an "add a job" submission or a "get me the N recent JDs" style
   lookup, and (for submissions) extracts *and* validates fields in the same
   pass — the graph branches from there. This is one round-trip instead of the
   three separate calls an earlier version made, to keep token usage down.
2. Field validation checks not just whether a value is *present* (that was
   already true) but whether it's *plausible* for that field — "Cat" for years
   of experience or "Tennis" for a location is caught here, even though both
   are non-empty strings a presence-only check would have accepted.
3. `ask_field` is the single node that pauses the graph for a human answer, for
   both "this field is missing" and "this field's value doesn't make sense" cases.
   Every pause is logged and reported back to the caller with which node raised it,
   so the chat transcript shows exactly what the agent is doing and why, not just a
   bare question.
"""

import json
import os
import re
import sys
import uuid
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from jobs_db import find_duplicate_job, find_job_for_resume_match, get_recent_jobs, publish_job_in_db

# Gemini's replies can contain arbitrary Unicode (e.g. accented characters); on
# Windows consoles the default stdout encoding isn't UTF-8, so those characters
# print as mojibake without this.
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(override=True)

# LangSmith tracing hangs indefinitely on this stack - confirmed via thread-dump,
# it's NOT a network issue (smith.langchain.com is reachable fine). The tracer's
# on_chat_model_start() calls langchain_core.env.get_runtime_environment(), which
# calls platform.win32_ver(); on Python 3.14 + this Windows setup that makes a WMI
# query that never returns. Any traced LLM call hangs forever, right there in the
# main thread, before a single byte reaches LangSmith. See README Troubleshooting
# for how to check whether this affects you and how to re-enable tracing.
# LANGSMITH_TRACING=false in .env isn't enough on its own - some tracing paths key
# off LANGSMITH_API_KEY just being *present*. Force both off here so
# ChatGoogleGenerativeAI.invoke() never takes that path.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGSMITH_API_KEY", None)
os.environ.pop("LANGCHAIN_API_KEY", None)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
# Structured-JSON calls (intake, single-field validation, scoring) never need more
# than a few hundred tokens back — capping output keeps a runaway/rambling reply
# from burning tokens for no benefit. rewrite_jd produces a full JD, so it uses
# the uncapped `llm` above instead of this bound instance.
llm_short = llm.bind(max_output_tokens=400)

REQUIRED_FIELDS = ["title", "company", "location", "exp", "work_mode"]
QUALITY_THRESHOLD = 75
MAX_REVISIONS = 2
MAX_MISSING_INFO_ATTEMPTS = 6  # total question cap across BOTH missing-field asks and invalid-field clarifications
DEFAULT_RECENT_JOBS_LIMIT = 5
MAX_JOB_REFERENCE_ATTEMPTS = 3  # how many times we'll ask "which job?" before giving up


class PendingField(TypedDict):
    field: str
    reason: Optional[str]  # None => field is missing entirely; str => field is present but invalid, this is why


class JDState(TypedDict):
    raw_jd: str
    intent: str  # "add_job" | "query_jobs" | "qualify_resume"
    query_limit: int
    query_fields: list
    query_result: list
    parsed: dict
    pending_fields: list  # list[PendingField] — the queue ask_field works through
    missing_info_attempts: int
    quality_score: int
    quality_feedback: str
    rewritten_jd: str
    revision_count: int
    duplicate_of: Optional[str]
    status: str
    job_id: Optional[str]
    log: list
    # Resume-qualification path (only populated when a resume was uploaded — see app.py)
    resume_text: Optional[str]
    target_job_reference: str  # the job title/id the human gave, or "" if not yet given
    target_job: Optional[dict]
    qualify_attempts: int
    qualify_score: int
    qualify_feedback: str


def extract_text(content) -> str:
    """Gemini returns message content as a list of blocks instead of a plain string."""
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
    return content


def _strip_code_fence(text: str) -> str:
    """Gemini often wraps JSON replies in ```json ... ``` fences even when told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _log(state: JDState, note: str) -> list:
    """Every log line is prefixed with the node name, so the chat surface can show
    exactly which step of the graph produced it instead of an opaque status string."""
    return state.get("log", []) + [note]


# ─── Nodes ────────────────────────────────────────────────────────────────────

def intake(state: JDState) -> dict:
    """Single LLM call that replaces three separate calls the first version of
    this pipeline made (classify intent, extract fields, validate fields).
    Folding them into one prompt cuts token spend roughly 3x on the common path,
    since each separate call would have re-sent the whole message as context
    anyway — one round-trip amortizes that instead of paying for it three times.

    This only ever runs on a fresh message (see app.py — resumed answers to an
    open interrupt skip straight back into the graph via Command(resume=...) and
    never re-enter at START), so `state["raw_jd"]` here is always the user's
    original chat message, not a follow-up answer to a field question.
    """
    prompt = (
        "A recruiter sent this chat message. Decide intent, then act on it.\n\n"
        "intent=\"query_jobs\" if they're asking to look up recent job postings "
        "(e.g. \"get me the 3 recent JDs\", \"titles of the last 5 jobs\"). "
        "Then also give limit (int, default "
        f"{DEFAULT_RECENT_JOBS_LIMIT}) and fields (subset of {REQUIRED_FIELDS}, "
        "empty list = full JD).\n\n"
        "intent=\"add_job\" otherwise (they're submitting a job description). "
        f"Then extract fields {REQUIRED_FIELDS} as strings (\"\" if not mentioned), "
        "and for every non-empty field judge whether its value is plausible for "
        "that field (not just non-empty). For `exp`, accept realistic durations such "
        "as '2 years', '6 months', '2+ years', 'internship', or 'Fresher'; do not "
        "reject these just because they are not in years-only form. For `location`, "
        "examples like 'Tennis' are invalid. List invalid ones in `invalid` as "
        "{field, reason}.\n\n"
        "Reply as strict JSON only, no commentary, no markdown fences, exactly:\n"
        '{"intent": "add_job"|"query_jobs", "limit": <int>, "fields": [...], '
        '"parsed": {<field>: "<value>", ...}, "invalid": [{"field": "<f>", "reason": "<why>"}]}\n'
        "Omit keys that don't apply to the chosen intent.\n\n"
        f"Message:\n{state['raw_jd']}"
    )
    response = llm_short.invoke(prompt)
    text = _strip_code_fence(extract_text(response.content))
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {}

    intent = result.get("intent") if result.get("intent") in ("add_job", "query_jobs") else "add_job"

    if intent == "query_jobs":
        limit = int(result.get("limit") or DEFAULT_RECENT_JOBS_LIMIT)
        fields = [f for f in (result.get("fields") or []) if f in REQUIRED_FIELDS]
        return {
            "intent": intent,
            "query_limit": limit,
            "query_fields": fields,
            "log": _log(state, f"intake: intent=query_jobs, limit={limit}, fields={fields}"),
        }

    raw_parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
    parsed = {field: str(raw_parsed.get(field, "")).strip() for field in REQUIRED_FIELDS}
    invalid_by_field = {
        entry.get("field"): str(entry.get("reason", "")) or "that value doesn't look right"
        for entry in (result.get("invalid") or [])
        if isinstance(entry, dict) and entry.get("field") in REQUIRED_FIELDS
    }

    pending = []
    for field in REQUIRED_FIELDS:
        if not parsed[field]:
            pending.append({"field": field, "reason": None})
        elif field in invalid_by_field:
            pending.append({"field": field, "reason": invalid_by_field[field]})

    note = f"intake: intent=add_job, parsed={parsed}"
    if invalid_by_field:
        note += f", invalid={invalid_by_field}"
    return {
        "intent": intent,
        "parsed": parsed,
        "pending_fields": pending,
        "log": _log(state, note),
    }


def route_after_intake(state: JDState) -> str:
    if state["intent"] == "query_jobs":
        return "query_jobs"
    return "ask_field" if state["pending_fields"] else "check_duplicate"


def query_jobs(state: JDState) -> dict:
    """Terminal: fetch the N most recent published jobs from the DB, optionally
    projected down to just the fields the user asked for (e.g. just titles)."""
    limit = state.get("query_limit") or DEFAULT_RECENT_JOBS_LIMIT
    fields = state.get("query_fields") or []
    jobs = get_recent_jobs(limit=limit, fields=fields or None)
    return {
        "query_result": jobs,
        "status": "answered",
        "log": _log(state, f"query_jobs: fetched {len(jobs)} recent job(s)" + (f", fields={fields}" if fields else "")),
    }


# ─── Resume qualification path ─────────────────────────────────────────────────
#
# Entered directly from START (see route_start / build_graph) whenever the caller
# already extracted resume text from an uploaded file — app.py does that before
# invoking the graph, so `intake`'s classification call is skipped entirely for
# this path (no point asking an LLM "is this a resume?" when we already know).

def route_start(state: JDState) -> str:
    return "identify_target_job" if state.get("resume_text") else "intake"


def identify_target_job(state: JDState) -> dict:
    """Plain lookup — no LLM. `target_job_reference` is either the chat message
    itself (first attempt, on the assumption a short message naming the job was
    sent alongside the resume) or the human's answer to `ask_job_reference`.
    """
    reference = (state.get("target_job_reference") or state.get("raw_jd") or "").strip()
    job = find_job_for_resume_match(reference) if reference else None
    return {
        "target_job_reference": reference,
        "target_job": job,
        "log": _log(state, f"identify_target_job: reference='{reference}', found={bool(job)}"),
    }


def route_after_identify_job(state: JDState) -> str:
    if state["target_job"]:
        return "qualify_resume"
    if state.get("qualify_attempts", 0) >= MAX_JOB_REFERENCE_ATTEMPTS:
        return "escalate"
    return "ask_job_reference"


def ask_job_reference(state: JDState) -> dict:
    """Pauses the graph to ask which job to score the resume against — either
    because none was given, or the given title/id didn't match exactly one
    posting (ambiguous or not found)."""
    reference = interrupt({
        "field": "target_job",
        "reason": None,
        "question": "Which job should I score this resume against? You can give the job title or the job ID.",
    })
    return {
        "target_job_reference": str(reference).strip(),
        "qualify_attempts": state.get("qualify_attempts", 0) + 1,
        "log": _log(state, f"ask_job_reference: got '{str(reference).strip()}'"),
    }


def qualify_resume(state: JDState) -> dict:
    """LLM call: how well does this resume match the target job's requirements?
    Terminal node — this is the "qualify score" the recruiter asked for.
    """
    job = state["target_job"]
    job_text = job.get("description") or ", ".join(f"{k}: {v}" for k, v in job.items() if k in REQUIRED_FIELDS)
    prompt = (
        "Score how well this resume matches the job below, from 0-100, based on "
        "relevant skills, years of experience, and role fit. Reply as strict "
        'JSON, no markdown fences: {"score": <int 0-100>, "feedback": '
        '"<2-3 sentence explanation covering strengths and gaps>"}.\n\n'
        f"Job ({job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}):\n{job_text}\n\n"
        f"Resume:\n{state['resume_text']}"
    )
    response = llm_short.invoke(prompt)
    text = _strip_code_fence(extract_text(response.content))
    try:
        result = json.loads(text)
        score = int(result.get("score", 0))
        feedback = str(result.get("feedback", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        score, feedback = 0, "Could not parse a qualification score from the model's reply."

    return {
        "qualify_score": score,
        "qualify_feedback": feedback,
        "status": "qualified",
        "log": _log(state, f"qualify_resume: {score}/100 against '{job.get('title')}' - {feedback}"),
    }


def _is_plausible_exp_value(value: str) -> bool:
    """Keep exp validation deterministic: months, years, internships, and fresher entries are all plausible."""
    text = (value or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in ["fresher", "fresh graduate", "internship", "intern", "trainee"]):
        return True
    if re.search(r"\b\d+\s*(?:\+\s*)?(?:month|months|mo|m|year|years|yr|yrs)\b", text):
        return True
    if re.search(r"\b\d+\s*(?:to|-)\s*\d+\s*(?:month|months|mo|m|year|years|yr|yrs)\b", text):
        return True
    return False


def _validate_single_field(field: str, value: str) -> dict:
    """LLM call for exactly one field/value pair — used only inside `ask_field`,
    to re-check a human's answer to a single question. Kept deliberately tiny
    (one field, short prompt, capped output) since this is the one call that can
    repeat multiple times in a single conversation if answers keep failing.
    Returns {"valid": bool, "reason": str}.
    """
    if field == "exp":
        valid = _is_plausible_exp_value(value)
        return {"valid": valid, "reason": "" if valid else "Use a duration like '2 years', '6 months', 'Internship', or 'Fresher'."}

    prompt = (
        f"Is \"{value}\" a plausible value for the job-posting field \"{field}\"? "
        "(e.g. 'Cat' is not valid for years of experience, 'Tennis' is not valid "
        "for a location). Reply as strict JSON only, no commentary, no markdown "
        'fences: {"valid": <bool>, "reason": "<short reason, only if invalid>"}.'
    )
    response = llm_short.invoke(prompt)
    text = _strip_code_fence(extract_text(response.content))
    try:
        result = json.loads(text)
        return {"valid": bool(result.get("valid", True)), "reason": str(result.get("reason", ""))}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"valid": True, "reason": ""}


def ask_field(state: JDState) -> dict:
    """Pauses the graph (interrupt) and asks a human about ONE field at a time —
    either because it's missing, or because its value failed validation.

    The reply is taken verbatim as that field's candidate value, then immediately
    re-validated (a single-field LLM check) before being accepted, so a second bad
    answer ("Puppy" after "Cat") gets caught right away instead of slipping through
    to publish_job. No side effects happen before `interrupt()`: on resume, the
    node re-runs from the top, so anything before the interrupt call would
    otherwise execute twice.
    """
    pending = state["pending_fields"]
    current = pending[0]
    field, reason = current["field"], current["reason"]

    if reason:
        question = (
            f"The {field} you gave — \"{state['parsed'].get(field, '')}\" — doesn't look valid "
            f"({reason}). Could you give a valid {field}?"
        )
    else:
        question = f"What is the {field}?"

    reply = str(interrupt({"field": field, "reason": reason, "question": question})).strip()

    attempts = state.get("missing_info_attempts", 0) + 1
    result = _validate_single_field(field, reply)

    parsed = {**state["parsed"], field: reply}
    rest = pending[1:]

    if not result["valid"]:
        rest = rest + [{"field": field, "reason": result["reason"] or "that value doesn't look right"}]
        log_note = f"ask_field: got '{reply}' for {field}, still invalid ({result['reason']})"
    else:
        log_note = f"ask_field: got '{reply}' for {field}, accepted"

    return {
        "parsed": parsed,
        "pending_fields": rest,
        "missing_info_attempts": attempts,
        "log": _log(state, log_note),
    }


def route_after_ask_field(state: JDState) -> str:
    if state.get("missing_info_attempts", 0) >= MAX_MISSING_INFO_ATTEMPTS and state["pending_fields"]:
        return "escalate"
    return "ask_field" if state["pending_fields"] else "check_duplicate"


def check_duplicate(state: JDState) -> dict:
    """Plain database lookup — no LLM. A duplicate is a fact, not a judgment call."""
    existing = find_duplicate_job(state["parsed"]["title"], state["parsed"]["company"])
    duplicate_of = str(existing["_id"]) if existing else None
    return {
        "duplicate_of": duplicate_of,
        "log": _log(state, f"check_duplicate: duplicate_of={duplicate_of}"),
    }


def route_after_duplicate_check(state: JDState) -> str:
    return "reject" if state["duplicate_of"] else "score_quality"


def score_quality(state: JDState) -> dict:
    """LLM scores clarity, specificity, and bias-free language of the current draft."""
    jd_text = state.get("rewritten_jd") or state["raw_jd"]
    prompt = (
        "Score this job description from 0-100 on clarity, specificity, and "
        "bias-free language. Reply as strict JSON, no markdown fences: "
        '{"score": <int 0-100>, "feedback": "<1-2 sentence critique>"}.\n\n'
        f"Job description:\n{jd_text}"
    )
    response = llm_short.invoke(prompt)
    text = _strip_code_fence(extract_text(response.content))
    try:
        result = json.loads(text)
        score = int(result.get("score", 0))
        feedback = str(result.get("feedback", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        score, feedback = 0, "Could not parse a quality score from the model's reply."

    return {
        "quality_score": score,
        "quality_feedback": feedback,
        "log": _log(state, f"score_quality: {score}/100 - {feedback}"),
    }


def route_after_score(state: JDState) -> str:
    """Plain Python: the threshold and revision cap are fixed rules, not LLM discretion."""
    if state["quality_score"] >= QUALITY_THRESHOLD:
        return "publish_job"
    if state.get("revision_count", 0) < MAX_REVISIONS:
        return "rewrite_jd"
    return "escalate"


def rewrite_jd(state: JDState) -> dict:
    """LLM rewrites the current draft using the quality feedback."""
    jd_text = state.get("rewritten_jd") or state["raw_jd"]
    prompt = (
        f"Rewrite this job description to address the following feedback: "
        f"\"{state['quality_feedback']}\". Reply with only the rewritten job "
        "description, no commentary, no markdown fences.\n\n"
        f"Job description:\n{jd_text}"
    )
    response = llm.invoke(prompt)
    rewritten = extract_text(response.content).strip()
    revision_count = state.get("revision_count", 0) + 1
    return {
        "rewritten_jd": rewritten,
        "revision_count": revision_count,
        "log": _log(state, f"rewrite_jd: revision #{revision_count}"),
    }


def publish_job(state: JDState) -> dict:
    """Terminal: writes the final job posting to MongoDB."""
    description = state.get("rewritten_jd") or state["raw_jd"]
    job_id = publish_job_in_db(state["parsed"], description, state["quality_score"])
    return {
        "job_id": job_id,
        "status": "published",
        "log": _log(state, f"publish_job: job_id={job_id}"),
    }


def reject(state: JDState) -> dict:
    """Terminal: duplicate postings are auto-rejected, no human judgment needed."""
    return {
        "status": "rejected",
        "log": _log(state, "reject: duplicate of an existing job posting"),
    }


def escalate(state: JDState) -> dict:
    """Terminal: too many missing/invalid-field questions asked, or too many
    failed quality revisions. Either way this pipeline gives up and hands off to a
    human outside the graph — it does NOT publish a JD built on an unresolved
    invalid field (e.g. it will never create a posting with exp="Cat")."""
    return {
        "status": "needs_input",
        "log": _log(state, "escalate: needs manual review"),
    }


# ─── Graph assembly ───────────────────────────────────────────────────────────

def build_graph(checkpointer):
    graph = StateGraph(JDState)

    graph.add_node("intake", intake)
    graph.add_node("query_jobs", query_jobs)
    graph.add_node("ask_field", ask_field)
    graph.add_node("check_duplicate", check_duplicate)
    graph.add_node("score_quality", score_quality)
    graph.add_node("rewrite_jd", rewrite_jd)
    graph.add_node("publish_job", publish_job)
    graph.add_node("reject", reject)
    graph.add_node("escalate", escalate)
    graph.add_node("identify_target_job", identify_target_job)
    graph.add_node("ask_job_reference", ask_job_reference)
    graph.add_node("qualify_resume", qualify_resume)

    graph.add_conditional_edges(START, route_start, {
        "intake": "intake",
        "identify_target_job": "identify_target_job",
    })
    graph.add_conditional_edges("intake", route_after_intake, {
        "query_jobs": "query_jobs",
        "ask_field": "ask_field",
        "check_duplicate": "check_duplicate",
    })
    graph.add_edge("query_jobs", END)

    graph.add_conditional_edges("identify_target_job", route_after_identify_job, {
        "qualify_resume": "qualify_resume",
        "ask_job_reference": "ask_job_reference",
        "escalate": "escalate",
    })
    graph.add_edge("ask_job_reference", "identify_target_job")
    graph.add_edge("qualify_resume", END)

    graph.add_conditional_edges("ask_field", route_after_ask_field, {
        "ask_field": "ask_field",
        "escalate": "escalate",
        "check_duplicate": "check_duplicate",
    })
    graph.add_conditional_edges("check_duplicate", route_after_duplicate_check, {
        "reject": "reject",
        "score_quality": "score_quality",
    })
    graph.add_conditional_edges("score_quality", route_after_score, {
        "publish_job": "publish_job",
        "rewrite_jd": "rewrite_jd",
        "escalate": "escalate",
    })
    graph.add_edge("rewrite_jd", "score_quality")
    graph.add_edge("publish_job", END)
    graph.add_edge("reject", END)
    graph.add_edge("escalate", END)

    return graph.compile(checkpointer=checkpointer)


# ─── CLI runner ────────────────────────────────────────────────────────────────
#
# Default mode is interactive: pauses just prompt for input right there in the
# same process, like any normal CLI tool. `--resume` is kept as a separate,
# explicit mode for the one demo that actually needs two process invocations:
# proving the pipeline survives being killed and resumed later, purely from
# checkpoints.sqlite plus a thread_id. Day-to-day use should never need it.

def report_final(result: dict) -> None:
    status = result.get("status")
    if status == "answered":
        print("\nRecent jobs:")
        for job in result.get("query_result", []):
            print(f"  - {job}")
    elif status == "qualified":
        job = result.get("target_job") or {}
        print(f"\nQualification score: {result.get('qualify_score')}/100 (against '{job.get('title')}')")
        print(f"  {result.get('qualify_feedback')}")
    else:
        print(f"\nStatus: {status}")
    for line in result["log"]:
        print(f"  - {line}")
    if status == "published":
        print(f"\nPublished job_id: {result['job_id']}")


def run_interactive(app, raw_jd: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"[thread_id: {thread_id}]")
    result = app.invoke({"raw_jd": raw_jd}, config=config)

    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        question = interrupt_value["question"]
        reply = input(f"\n{question}\n> ")
        result = app.invoke(Command(resume=reply), config=config)

    report_final(result)


if __name__ == "__main__":
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        app = build_graph(checkpointer)

        if len(sys.argv) >= 2 and sys.argv[1] == "--resume":
            if len(sys.argv) < 4:
                print('Usage: python jd_intake_pipeline.py --resume <thread_id> "<your answer>"')
                sys.exit(1)
            thread_id = sys.argv[2]
            reply = " ".join(sys.argv[3:])
            config = {"configurable": {"thread_id": thread_id}}
            result = app.invoke(Command(resume=reply), config=config)

            while "__interrupt__" in result:
                interrupt_value = result["__interrupt__"][0].value
                question = interrupt_value["question"]
                reply = input(f"\n{question}\n> ")
                result = app.invoke(Command(resume=reply), config=config)

            report_final(result)

        elif len(sys.argv) >= 2:
            thread_id = str(uuid.uuid4())
            raw_jd_input = " ".join(sys.argv[1:])
            run_interactive(app, raw_jd_input, thread_id)

        else:
            print('Usage: python jd_intake_pipeline.py "<raw job description text>"')
            print('       python jd_intake_pipeline.py --resume <thread_id> "<your answer>"  (after killing a paused run)')
            sys.exit(1)
