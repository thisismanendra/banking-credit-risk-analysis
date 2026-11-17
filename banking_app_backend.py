
import json
import re
import time
import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GRADE_FILE = BASE_DIR / "reports" / "grade_statistics.json"

spec = importlib.util.spec_from_file_location(
    "banking_base_runtime",
    BASE_DIR / "banking_assistant_runtime.py"
)

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def _load_grade_result():

    with open(GRADE_FILE, "r", encoding="utf-8") as file:
        grades = json.load(file)

    grades = sorted(
        grades,
        key=lambda x: x["grade"]
    )

    if [g["grade"] for g in grades] != list("ABCDEFG"):
        raise RuntimeError(
            "Saved grade statistics are incomplete."
        )

    total_loans = sum(g["total_loans"] for g in grades)
    charged_off = sum(g["charged_off"] for g in grades)

    if total_loans != 1190174:
        raise RuntimeError(
            f"Unexpected labelled loan count: {total_loans}"
        )

    overall_rate = round(
        charged_off / total_loans * 100,
        2
    )

    lines = [
        "Historical Charged Off rates across all LendingClub grades:"
    ]

    for g in grades:

        lines.append(
            f"Grade {g['grade']}: "
            f"{g['charged_off_rate']:.2f}% "
            f"({g['charged_off']:,} Charged Off out of "
            f"{g['total_loans']:,} loans)."
        )

    lines.append(
        f"Overall: {overall_rate:.2f}% "
        f"({charged_off:,} Charged Off out of "
        f"{total_loans:,} loans)."
    )

    lines.append(
        "Grade G has the highest historical Charged Off rate, "
        "while Grade A has the lowest."
    )

    return {
        "answer": "\n".join(lines),
        "source": "LendingClub accepted loans",
        "grounded": True,
        "dataset_records": total_loans,
        "filters": ["All loan grades A-G"],
        "evidence": {
            "grades": grades,
            "overall": {
                "total_loans": total_loans,
                "charged_off": charged_off,
                "default_rate_percent": overall_rate
            }
        }
    }


_GRADE_RESULT = _load_grade_result()


def is_grade_comparison(question):

    q = str(question).lower()

    grade = "grade" in q or "grades" in q

    rate = any(
        word in q
        for word in [
            "default", "charged off", "charge-off"
        ]
    )

    comparison = any(
        word in q
        for word in [
            "compare", "comparison", "across",
            "all", "each", "grade-wise", "grade wise"
        ]
    )

    return grade and rate and comparison


def ask(question):

    if is_grade_comparison(question):
        return dict(_GRADE_RESULT)

    return base.ask(question)


def clear_cache():
    base.clear_cache()


def cache_size():
    return base.cache_size()
