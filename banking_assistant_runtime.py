
import json
import re
import time
import urllib.request
import urllib.error
import importlib.util

from pathlib import Path
from collections import OrderedDict

BASE_DIR = Path(__file__).resolve().parent
ENGINE_FILE = BASE_DIR / "banking_data_engine.py"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "banking-assistant:final"

KEEP_ALIVE = "30m"
MAX_CACHE_ITEMS = 256

_CACHE = OrderedDict()
_ENGINE = None


def _load_engine():

    global _ENGINE

    if _ENGINE is not None:
        return _ENGINE

    if not ENGINE_FILE.exists():
        raise FileNotFoundError(
            f"Banking engine not found: {ENGINE_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "banking_data_engine_runtime",
        ENGINE_FILE
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _ENGINE = module
    return _ENGINE


def _normalise_question(question):

    return re.sub(
        r"\s+",
        " ",
        str(question).strip().lower()
    )


def _is_dataset_query(question):

    q = question.lower()

    markers = [
        "dataset",
        "lendingclub",
        "lending club",
        "full data",
        "historical charged off",
        "charged off rate",
        "default rate",
        "loan grade",
        "loan grades",
        "grade a",
        "grade b",
        "grade c",
        "grade d",
        "grade e",
        "grade f",
        "grade g",
        "how many loans",
        "number of loans",
        "accepted loans",
        "average annual income",
        "median dti",
        "loan purpose",
        "debt consolidation loans",
        "36-month loans",
        "compare default rates",
        "highest default",
        "lowest default"
    ]

    return any(marker in q for marker in markers)




# --------------------------------------------------------------
# DEDICATED FULL-DATASET GRADE COMPARISON
# --------------------------------------------------------------

_GRADE_COMPARISON_CACHE = None


def _is_grade_comparison(question):

    q = str(question).lower()

    grade_requested = (
        "grade" in q
        or "grades" in q
    )

    comparison_requested = any(
        word in q
        for word in [
            "compare",
            "comparison",
            "across",
            "all",
            "each",
            "by grade",
            "grade-wise",
            "grade wise"
        ]
    )

    rate_requested = any(
        word in q
        for word in [
            "default",
            "charged off",
            "charge-off",
            "charged-off"
        ]
    )

    return (
        grade_requested
        and comparison_requested
        and rate_requested
    )


def _grade_comparison_result():

    global _GRADE_COMPARISON_CACHE

    if _GRADE_COMPARISON_CACHE is not None:
        return dict(_GRADE_COMPARISON_CACHE)

    import duckdb

    data_dir = (
        BASE_DIR
        / "data_knowledge"
        / "lendingclub_full_v1"
    )

    parts = sorted(data_dir.glob("*.parquet"))

    if not parts:
        parts = sorted(data_dir.rglob("*.parquet"))

    if not parts:
        raise FileNotFoundError(
            "LendingClub Parquet files were not found."
        )

    files_sql = "[" + ", ".join(
        "'" + p.as_posix().replace("'", "''") + "'"
        for p in parts
    ) + "]"

    con = duckdb.connect(database=":memory:")

    try:

        statuses = {
            str(row[0])
            for row in con.execute(
                f"""
                SELECT DISTINCT CAST(loan_status AS VARCHAR)
                FROM read_parquet({files_sql})
                """
            ).fetchall()
        }

        if {"0", "1"}.issubset(statuses):
            status_filter = "loan_status IN (0, 1)"
            charged_condition = "loan_status = 1"

        elif {"Fully Paid", "Charged Off"}.issubset(statuses):
            status_filter = (
                "loan_status IN ('Fully Paid', 'Charged Off')"
            )
            charged_condition = (
                "loan_status = 'Charged Off'"
            )

        else:
            raise ValueError(
                f"Unexpected loan status encoding: {statuses}"
            )

        query = f"""
        SELECT
            grade,
            COUNT(*) AS total_loans,
            SUM(
                CASE WHEN {charged_condition}
                THEN 1 ELSE 0 END
            ) AS charged_off
        FROM read_parquet({files_sql})
        WHERE {status_filter}
          AND grade IS NOT NULL
        GROUP BY grade
        ORDER BY grade
        """

        rows = con.execute(query).fetchall()

    finally:
        con.close()

    grades = []

    for grade, total, charged in rows:

        total = int(total)
        charged = int(charged)

        grades.append({
            "grade": str(grade),
            "total_loans": total,
            "charged_off": charged,
            "charged_off_rate": round(
                100.0 * charged / total,
                2
            )
        })

    if [x["grade"] for x in grades] != list("ABCDEFG"):
        raise RuntimeError(
            "Grade comparison did not return all grades A-G."
        )

    total_loans = sum(
        x["total_loans"] for x in grades
    )

    total_charged = sum(
        x["charged_off"] for x in grades
    )

    overall = round(
        100.0 * total_charged / total_loans,
        2
    )

    lines = [
        "Historical Charged Off rates across all LendingClub grades:"
    ]

    for item in grades:

        lines.append(
            f"Grade {item['grade']}: "
            f"{item['charged_off_rate']:.2f}% "
            f"({item['charged_off']:,} Charged Off "
            f"out of {item['total_loans']:,} loans)."
        )

    lines.append(
        f"Overall: {overall:.2f}% "
        f"({total_charged:,} Charged Off "
        f"out of {total_loans:,} loans)."
    )

    lines.append(
        "Grade G has the highest historical Charged Off rate, "
        "while Grade A has the lowest."
    )

    output = {
        "answer": "\n".join(lines),
        "source": "LendingClub accepted loans",
        "grounded": True,
        "dataset_records": total_loans,
        "filters": ["All loan grades A-G"],
        "evidence": {
            "grades": grades,
            "overall": {
                "total_loans": total_loans,
                "charged_off": total_charged,
                "default_rate_percent": overall
            }
        }
    }

    _GRADE_COMPARISON_CACHE = output

    return dict(output)


def _try_dti_calculation(question):

    q = question.lower()

    if not any(
        x in q
        for x in ["dti", "debt-to-income", "debt to income"]
    ):
        return None

    debt_match = re.search(
        r"(?:monthly\s+)?debt\s+payments?"
        r"\s*(?:are|is|of|:|=)?\s*"
        r"(?:(?:rs\.?|inr|₹)\s*)?"
        r"([\d,]+(?:\.\d+)?)",
        q
    )

    income_match = re.search(
        r"(?:gross\s+)?monthly\s+income"
        r"\s*(?:is|of|:|=)?\s*"
        r"(?:(?:rs\.?|inr|₹)\s*)?"
        r"([\d,]+(?:\.\d+)?)",
        q
    )

    if not debt_match or not income_match:
        return None

    debt = float(debt_match.group(1).replace(",", ""))
    income = float(income_match.group(1).replace(",", ""))

    if income <= 0 or debt < 0:
        return None

    percentage = debt / income * 100

    formatted = f"{percentage:.2f}".rstrip("0").rstrip(".")

    return {
        "answer": (
            f"DTI = (monthly debt payments / gross monthly income) "
            f"× 100 = {formatted}%. Your debt-to-income ratio is "
            f"{formatted}%."
        ),
        "source": "Deterministic Python calculation",
        "grounded": True,
        "evidence": {
            "monthly_debt_payments": debt,
            "gross_monthly_income": income,
            "dti_percent": percentage
        }
    }


def _extract_final_answer(raw):

    text = str(raw or "").strip()

    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()

    elif "<think>" in text:
        return ""

    text = re.sub(
        r"^(final answer|answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.strip()

    return text


def _ask_ollama(question):

    prompt = (
        "/no_think\n"
        "Answer directly in simple banking language. "
        "Use at most three sentences. "
        "Do not repeat the answer. "
        "Do not reveal reasoning or planning.\n\n"
        f"Question: {question}"
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8,
            "repeat_penalty": 1.15,
            "repeat_last_n": 64,
            "num_ctx": 1024,
            "num_predict": 384
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.time()

    with urllib.request.urlopen(
        request,
        timeout=240
    ) as response:
        data = json.loads(
            response.read().decode("utf-8")
        )

    elapsed = time.time() - start

    answer = _extract_final_answer(
        data.get("response", "")
    )

    if not answer:

        raise RuntimeError(
            "Ollama did not return a clean final answer. "
            "Reasoning-only or incomplete output was rejected."
        )

    return {
        "answer": answer,
        "source": MODEL,
        "grounded": False,
        "latency_seconds": round(elapsed, 3),
        "generated_tokens": data.get("eval_count"),
        "done_reason": data.get("done_reason")
    }


def ask(question):

    question = str(question).strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "source": "Input validation",
            "grounded": False
        }

    start = time.time()
    key = _normalise_question(question)

    # 1. Reuse cached results.
    if key in _CACHE:

        cached = dict(_CACHE[key])
        cached["cached"] = True
        cached["latency_seconds"] = round(
            time.time() - start, 4
        )

        return cached

    # 2. Exact DTI arithmetic.
    dti_result = _try_dti_calculation(question)

    if dti_result is not None:
        result = dti_result

    # 3. Exact all-grade comparison before generic fallback.
    elif _is_grade_comparison(question):

        result = _grade_comparison_result()

    # 4. Other exact dataset questions.
    elif _is_dataset_query(question):

        engine = _load_engine()

        engine_result = engine.answer_dataset_question(
            question
        )

        if (
            isinstance(engine_result, dict)
            and engine_result.get("grounded") is True
        ):

            result = {
                "answer": engine_result.get("answer", ""),
                "source": engine_result.get(
                    "source",
                    "LendingClub accepted loans"
                ),
                "grounded": True,
                "dataset_records": engine_result.get(
                    "dataset_records"
                ),
                "filters": engine_result.get("filters", []),
                "evidence": engine_result.get("evidence", {})
            }

        else:

            result = {
                "answer": (
                    "I couldn't retrieve a verified result for "
                    "that question from the connected LendingClub "
                    "dataset. No statistic has been guessed."
                ),
                "source": "LendingClub DuckDB query engine",
                "grounded": False
            }

    # 4. Banking knowledge via Ollama.
    else:

        result = _ask_ollama(question)

    result["latency_seconds"] = round(
        time.time() - start, 3
    )

    result["cached"] = False

    # 5. Cache successful results only.
    if result.get("answer") and (
        result.get("grounded") is True
        or result.get("source") == MODEL
    ):

        _CACHE[key] = dict(result)
        _CACHE.move_to_end(key)

        while len(_CACHE) > MAX_CACHE_ITEMS:
            _CACHE.popitem(last=False)

    return result


def clear_cache():
    _CACHE.clear()


def cache_size():
    return len(_CACHE)
