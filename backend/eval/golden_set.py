"""
Golden evaluation set: a fixture document plus hand-written question/answer
pairs whose correctness a human already verified.

Why a fixture document instead of evaluating against whatever's currently
uploaded: reproducibility. Anyone running `python -m eval.run_eval` gets
the same document, the same questions, and therefore comparable scores
run-over-run and machine-over-machine — evaluating against a real user's
uploaded PDF would make results impossible to compare across runs.

Each entry has an `expect_answerable` flag: most questions ARE answerable
from the fixture text, but one deliberately is not — that one exists
specifically to check the "say you don't know" grounding behavior
(chat_service.py's SYSTEM_PROMPT), which is a different failure mode than
faithfulness and needs its own check.
"""

FIXTURE_DOCUMENT_TEXT = """Acme Corp Remote Work Policy

Employees may work remotely up to 3 days per week.
Remote work requests must be approved by a direct manager
at least 48 hours in advance.
All remote employees must be reachable via Slack during core
hours, which are 10am to 4pm in the employee's local timezone.
Equipment stipend: employees working remotely receive a one-time
500 USD stipend for home office equipment, claimable via the
Finance portal within the first 90 days of a remote arrangement."""

FIXTURE_FILENAME = "eval_fixture_remote_work_policy.pdf"

GOLDEN_SET = [
    {
        "question": "How many days per week can employees work remotely?",
        "expect_answerable": True,
        "reference_answer": "Up to 3 days per week.",
    },
    {
        "question": "How far in advance must remote work requests be approved?",
        "expect_answerable": True,
        "reference_answer": "At least 48 hours in advance.",
    },
    {
        "question": "What are the core hours employees must be reachable during?",
        "expect_answerable": True,
        "reference_answer": "10am to 4pm in the employee's local timezone.",
    },
    {
        "question": "What is the equipment stipend amount for remote workers?",
        "expect_answerable": True,
        "reference_answer": "A one-time 500 USD stipend.",
    },
    {
        "question": "How long do employees have to claim the equipment stipend?",
        "expect_answerable": True,
        "reference_answer": "Within the first 90 days of a remote arrangement.",
    },
    {
        "question": "Who must approve remote work requests?",
        "expect_answerable": True,
        "reference_answer": "A direct manager.",
    },
    {
        "question": "What is the company's parental leave policy?",
        "expect_answerable": False,
        "reference_answer": None,
    },
    {
        "question": "Can employees expense a new laptop under this policy?",
        "expect_answerable": False,
        "reference_answer": None,
    },
]
