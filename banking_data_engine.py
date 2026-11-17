
import json
import re
import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "data_knowledge" / "lendingclub_full_v1"
MANIFEST_PATH = STORE_DIR / "manifest.json"
SUMMARY_PATH = BASE_DIR / "data_knowledge" / "banking_dataset_knowledge.json"

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    MANIFEST = json.load(f)

TOTAL_RECORDS = int(MANIFEST["labelled_records"])

FIELDS = {
    "loan amount": "loan_amnt",
    "loan_amnt": "loan_amnt",
    "amount borrowed": "loan_amnt",
    "interest rate": "int_rate",
    "int_rate": "int_rate",
    "monthly installment": "installment",
    "installment": "installment",
    "annual income": "annual_inc",
    "income": "annual_inc",
    "annual_inc": "annual_inc",
    "debt-to-income ratio": "dti",
    "debt to income ratio": "dti",
    "dti": "dti",
    "fico score": "fico_range_high",
    "credit score": "fico_range_high",
    "fico": "fico_range_high",
    "revolving balance": "revol_bal",
    "revol_bal": "revol_bal",
    "revolving utilization": "revol_util",
    "credit utilization": "revol_util",
    "revol_util": "revol_util",
    "open accounts": "open_acc",
    "open_acc": "open_acc",
    "total accounts": "total_acc",
    "total_acc": "total_acc",
    "delinquencies": "delinq_2yrs",
    "late payments": "delinq_2yrs",
    "delinq_2yrs": "delinq_2yrs",
    "mortgage accounts": "mort_acc",
    "mort_acc": "mort_acc"
}

NUMERIC_FIELDS = set(FIELDS.values())

_CONN = None
_PURPOSES = None


def clean(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def get_connection():
    global _CONN

    if _CONN is None:
        _CONN = duckdb.connect(database=":memory:")

        parquet_glob = (
            STORE_DIR / "part-*.parquet"
        ).as_posix()

        escaped = parquet_glob.replace("'", "''")

        _CONN.execute(
            f"CREATE VIEW loans AS "
            f"SELECT * FROM read_parquet('{escaped}')"
        )

    return _CONN


def query(sql, parameters=None):
    con = get_connection()
    return con.execute(sql, parameters or []).fetchall()


def known_purposes():
    global _PURPOSES

    if _PURPOSES is None:
        rows = query(
            "SELECT DISTINCT purpose FROM loans "
            "WHERE purpose IS NOT NULL"
        )

        _PURPOSES = [
            str(row[0]).strip()
            for row in rows
        ]

    return _PURPOSES


def detect_filters(q):
    conditions = []
    parameters = []
    description = []

    # Grade filters A-G
    grade_match = re.search(
        r"\bgrade\s+([a-g])\b", q
    )

    if grade_match:
        grade = grade_match.group(1).upper()
        conditions.append(
            "UPPER(TRIM(CAST(grade AS VARCHAR))) = ?"
        )
        parameters.append(grade)
        description.append(f"grade {grade}")

    # Loan term filters
    term_match = re.search(
        r"\b(36|60)\s*months?\b", q
    )

    if term_match:
        months = term_match.group(1)

        conditions.append(
            "TRIM(CAST(term AS VARCHAR)) LIKE ?"
        )
        parameters.append(f"%{months}%")
        description.append(f"{months}-month term")

    # Purpose filter
    normalized_q = q.replace("_", " ")

    for purpose in sorted(
        known_purposes(),
        key=lambda item: len(item),
        reverse=True
    ):
        normalized = purpose.lower().replace("_", " ").strip()

        if normalized and normalized in normalized_q:
            conditions.append(
                "LOWER(REPLACE(TRIM(CAST(purpose AS VARCHAR)), "
                "'_', ' ')) = ?"
            )
            parameters.append(normalized)
            description.append(f"purpose {normalized}")
            break

    return conditions, parameters, description


def result(answer, evidence, filters=None):
    return {
        "answer": answer,
        "source": "LendingClub accepted loans",
        "dataset_records": TOTAL_RECORDS,
        "filters": filters or [],
        "evidence": evidence,
        "grounded": True
    }


def answer_dataset_question(question):
    q = clean(question)

    conditions, params, filter_description = detect_filters(q)

    where = (
        " WHERE " + " AND ".join(conditions)
        if conditions else ""
    )

    # ----------------------------------------------------------
    # Specific grade ranking queries
    # ----------------------------------------------------------

    if (
        "highest default" in q
        or "highest charged off" in q
        or "highest default rate" in q
    ) and "grade" in q:

        rows = query("""
            SELECT grade,
                   COUNT(*) AS loan_count,
                   SUM(CASE WHEN loan_status=1 THEN 1 ELSE 0 END)
                       AS charged_off,
                   AVG(CASE WHEN loan_status=1 THEN 1.0 ELSE 0.0 END)
                       * 100 AS default_rate
            FROM loans
            GROUP BY grade
            ORDER BY default_rate DESC
            LIMIT 1
        """)

        grade, count, charged, rate = rows[0]

        return result(
            f"Grade {grade} has the highest historical "
            f"Charged Off rate at {rate:.2f}%, based on "
            f"{count:,} loans in the full {TOTAL_RECORDS:,}-record dataset.",
            {
                "grade": grade,
                "loan_count": count,
                "charged_off": charged,
                "default_rate_percent": rate
            }
        )

    if (
        "lowest default" in q
        or "lowest charged off" in q
        or "lowest default rate" in q
    ) and "grade" in q:

        rows = query("""
            SELECT grade,
                   COUNT(*) AS loan_count,
                   AVG(CASE WHEN loan_status=1 THEN 1.0 ELSE 0.0 END)
                       * 100 AS default_rate
            FROM loans
            GROUP BY grade
            ORDER BY default_rate ASC
            LIMIT 1
        """)

        grade, count, rate = rows[0]

        return result(
            f"Grade {grade} has the lowest historical "
            f"Charged Off rate at {rate:.2f}%, based on "
            f"{count:,} loans.",
            {
                "grade": grade,
                "loan_count": count,
                "default_rate_percent": rate
            }
        )

    # ----------------------------------------------------------
    # Count queries
    # ----------------------------------------------------------

    if any(x in q for x in [
        "how many", "number of", "count of", "count "
    ]) and not any(x in q for x in [
        "average", "median", "default rate"
    ]):

        if "fully paid" in q:
            extra = " AND loan_status=0" if conditions else \
                    " WHERE loan_status=0"

            rows = query(
                f"SELECT COUNT(*) FROM loans{where}{extra}",
                params
            )
            count = rows[0][0]

            return result(
                f"There are {count:,} Fully Paid loans"
                + (f" matching {', '.join(filter_description)}."
                   if filter_description else " in the full dataset."),
                {"fully_paid_count": count},
                filter_description
            )

        if "charged off" in q:
            extra = " AND loan_status=1" if conditions else \
                    " WHERE loan_status=1"

            rows = query(
                f"SELECT COUNT(*) FROM loans{where}{extra}",
                params
            )
            count = rows[0][0]

            return result(
                f"There are {count:,} Charged Off loans"
                + (f" matching {', '.join(filter_description)}."
                   if filter_description else " in the full dataset."),
                {"charged_off_count": count},
                filter_description
            )

        if "grade" in q or "term" in q or "purpose" in q:
            rows = query(
                f"SELECT COUNT(*) FROM loans{where}",
                params
            )
            count = rows[0][0]

            return result(
                f"There are {count:,} loans"
                + (
                    f" matching {', '.join(filter_description)}."
                    if filter_description
                    else " in the full dataset."
                ),
                {"loan_count": count},
                filter_description
            )

    # ----------------------------------------------------------
    # Grouped default rates
    # ----------------------------------------------------------

    if "by grade" in q or "per grade" in q:
        rows = query("""
            SELECT grade, COUNT(*) AS loans,
                   AVG(CASE WHEN loan_status=1 THEN 1.0 ELSE 0.0 END)
                   * 100 AS default_rate
            FROM loans
            GROUP BY grade
            ORDER BY grade
        """)

        table = [
            {
                "grade": row[0],
                "loans": row[1],
                "default_rate_percent": round(row[2], 3)
            }
            for row in rows
        ]

        answer = "; ".join(
            f"Grade {r['grade']}: {r['default_rate_percent']:.2f}% "
            f"({r['loans']:,} loans)"
            for r in table
        )

        return result(
            "Historical Charged Off rates by grade: " + answer,
            {"grade_comparison": table}
        )

    # ----------------------------------------------------------
    # Default rate queries
    # ----------------------------------------------------------

    if any(x in q for x in [
        "default rate", "charged off rate",
        "default percentage", "risk rate",
        "percentage default"
    ]):

        rows = query(
            f"""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN loan_status=1 THEN 1 ELSE 0 END)
                       AS charged_off,
                   AVG(CASE WHEN loan_status=1 THEN 1.0 ELSE 0.0 END)
                       * 100 AS rate
            FROM loans{where}
            """,
            params
        )

        count, charged, rate = rows[0]

        label = (
            f" for {', '.join(filter_description)}"
            if filter_description else ""
        )

        return result(
            f"The historical Charged Off rate{label} is "
            f"{rate:.2f}%, based on {count:,} loans "
            f"({charged:,} Charged Off).",
            {
                "count": count,
                "charged_off": charged,
                "default_rate_percent": rate
            },
            filter_description
        )

    # ----------------------------------------------------------
    # Numeric average / median / min / max
    # ----------------------------------------------------------

    selected_field = None

    for alias, field in sorted(
        FIELDS.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
        if alias in q:
            selected_field = field
            break

    if selected_field:

        if "median" in q:
            function = "MEDIAN"
            label = "median"
        elif "maximum" in q or "highest" in q or "maximum value" in q:
            function = "MAX"
            label = "maximum"
        elif "minimum" in q or "lowest value" in q:
            function = "MIN"
            label = "minimum"
        else:
            function = "AVG"
            label = "average"

        rows = query(
            f"""
            SELECT COUNT({selected_field}),
                   {function}({selected_field})
            FROM loans{where}
            """,
            params
        )

        count, value = rows[0]

        if value is None:
            return result(
                "No non-missing values match this query.",
                {"count": count},
                filter_description
            )

        alias_name = selected_field.replace("_", " ")

        return result(
            f"The {label} {alias_name} is {value:,.3f}, "
            f"calculated from {count:,} non-missing records"
            + (
                f" matching {', '.join(filter_description)}."
                if filter_description else "."
            ),
            {
                "field": selected_field,
                "statistic": label,
                "value": value,
                "non_missing_count": count
            },
            filter_description
        )

    # ----------------------------------------------------------
    # Overall dataset questions
    # ----------------------------------------------------------

    if any(x in q for x in [
        "how many loans", "how many records",
        "dataset size", "total records",
        "total loans"
    ]):

        return result(
            f"The full LendingClub dataset contains "
            f"{TOTAL_RECORDS:,} labelled accepted loans: "
            f"{MANIFEST['fully_paid']:,} Fully Paid and "
            f"{MANIFEST['charged_off']:,} Charged Off.",
            {
                "total": TOTAL_RECORDS,
                "fully_paid": MANIFEST["fully_paid"],
                "charged_off": MANIFEST["charged_off"]
            }
        )

    # ----------------------------------------------------------
    # Data-grounded fallback
    # ----------------------------------------------------------

    return {
        "answer": (
            "The question was not matched to a supported dataset "
            "operation. Available operations include record counts, "
            "default rates, grade and term comparisons, loan purposes, "
            "and numeric averages, medians, minimums and maximums. "
            "I will not invent a dataset statistic."
        ),
        "source": "LendingClub accepted loans",
        "dataset_records": TOTAL_RECORDS,
        "filters": [],
        "evidence": {
            "available_fields": list(FIELDS.values()) + [
                "grade", "sub_grade", "term", "purpose",
                "home_ownership", "loan_status"
            ]
        },
        "grounded": False
    }


def ask(question):
    return answer_dataset_question(question)["answer"]
