
import json
import re
import urllib.request
import urllib.error

from src.deterministic_verifier import (
    verify_evidence_deterministically,
    normalize_terms,
    STOP_WORDS,
)


MODEL_NAME = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ============================================================
# 1. QUESTION CONCEPTS
# ============================================================

CONCEPT_PATTERNS = {
    "meal_expense": [
        r"\bmeals?\b",
        r"\bmeal expense\b",
        r"\bmeal reimbursement\b",
        r"\bfood expense\b",
    ],
    "domestic_travel": [
        r"\bdomestic travel\b",
        r"\bbusiness travel\b",
        r"\btravel expense\b",
        r"\btravel reimbursement\b",
    ],
    "internet_reimbursement": [
        r"\binternet\b",
        r"\bbroadband\b",
        r"\binternet reimbursement\b",
    ],
    "annual_bonus": [
        r"\bbonus\b",
        r"\bincentive\b",
    ],
    "retirement_fund": [
        r"\bretirement fund\b",
        r"\bretirement contribution\b",
        r"\bprovident fund\b",
    ],
    "professional_course": [
        r"\bprofessional courses?\b",
        r"\bcertifications?\b",
        r"\bcourse reimbursements?\b",
        r"\bprofessional development\b",
    ],
    "health_insurance": [
        r"\bhealth insurance\b",
        r"\bmedical insurance\b",
    ],

    # Avoid matching base salary/basic salary as salary payment.
    "salary_payment": [
        r"\bsalary payments?\b",
        r"\bsalaries\b",
        r"(?<!base )(?<!basic )\bsalary\b",
        r"\bpayday\b",
    ],

    "password": [
        r"\bpasswords?\b",
    ],

    # FIX: Support both singular and plural forms.
    "security_incident": [
        r"\bsecurity incidents?\b",
        r"\bcybersecurity incidents?\b",
        r"\bsecurity breaches?\b",
    ],

    "security_update": [
        r"\bsecurity updates?\b",
        r"\bsoftware updates?\b",
        r"\bsecurity patches?\b",
    ],
    "data_access_scope": [
        r"\bdata access\b",
        r"\bconfidential information\b",
        r"\bbusiness authorization\b",
        r"\baccess authorization\b",
    ],
    "remote_work": [
        r"\bremote work\b",
        r"\bwork from home\b",
        r"\bwfh\b",
    ],
    "probation_period": [
        r"\bprobation\b",
    ],
    "annual_leave": [
        r"\bannual leave\b",
        r"\bpaid leave\b",
        r"\bvacation leave\b",
    ],
    "expense_submission": [
        r"\bexpense submission\b",
        r"\bsubmit(?:ting|ted)?\s+expenses?\b",
        r"\bexpenses?\s+(?:must be\s+)?submitted\b",
    ],
    "receipt": [
        r"\breceipts?\b",
        r"\bdigital invoices?\b",
        r"\boriginal invoices?\b",
    ],
    "late_arrival": [
        r"\blate\b",
        r"\blate arrival\b",
        r"\btardy\b",
    ],
    "least_privilege": [
        r"\bleast privilege\b",
        r"\bminimum necessary access\b",
    ],
    "mfa": [
        r"\bmfa\b",
        r"\bmulti.factor authentication\b",
    ],
    "full_disk_encryption": [
        r"\bfull.disk encryption\b",
        r"\bdisk encryption\b",
    ],
    "screen_lock": [
        r"\bscreen lock\b",
        r"\bauto.lock\b",
        r"\bautomatic screen lock\b",
    ],
}


def get_question_concepts(text):

    normalized = str(text).lower()
    concepts = set()

    for concept, patterns in CONCEPT_PATTERNS.items():

        for pattern in patterns:

            if re.search(pattern, normalized):
                concepts.add(concept)
                break

    if "retirement_fund" in concepts:
        concepts.discard("salary_payment")

    return concepts


# ============================================================
# 2. OLLAMA CONNECTION
# ============================================================

def call_ollama(prompt):

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=300
    ) as response:

        result = json.loads(
            response.read().decode("utf-8")
        )

    return json.loads(result["response"])


# ============================================================
# 3. CHUNK METADATA HELPERS
# ============================================================

def get_chunk_metadata(chunk):

    metadata = chunk.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    return metadata


def get_chunk_source(chunk):

    metadata = get_chunk_metadata(chunk)

    return chunk.get(
        "source",
        metadata.get("source", "Unknown"),
    )


def get_chunk_page(chunk):

    metadata = get_chunk_metadata(chunk)

    return chunk.get(
        "page",
        metadata.get("page", "Unknown"),
    )


# ============================================================
# 4. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    return " ".join(
        str(text).lower().split()
    )


# ============================================================
# 5. EVIDENCE DEDUPLICATION
# ============================================================

def evidence_fingerprint(item):

    statement = item.get("statement", "")

    statement = re.sub(
        r"^(?:retirement benefits|annual performance bonus|"
        r"professional development|health benefits|salary payment)\s+",
        "",
        statement,
        flags=re.IGNORECASE,
    )

    statement = re.sub(
        r"(?:[.!?]\s*)?salary payments?\s+are\s+processed\s*$",
        "",
        statement,
        flags=re.IGNORECASE,
    )

    statement = re.sub(
        r"\bmeal reimbursement\b",
        "meal expense",
        statement,
        flags=re.IGNORECASE,
    )

    terms = normalize_terms(statement) - STOP_WORDS

    if "meal_expense" in terms:

        terms.discard("meal")
        terms.discard("reimbursement")
        terms.discard("reimbursements")
        terms.discard("reimbursable")

    numbers = tuple(
        re.findall(
            r"(?<!\w)\d+(?:,\d{3})*(?:\.\d+)?(?!\w)",
            statement,
        )
    )

    return (
        get_chunk_source(item),
        get_chunk_page(item),
        frozenset(terms),
        numbers,
    )


# ============================================================
# 6. CONTEXT BUILDING
# ============================================================

def build_context(retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
Document {i}
Source: {get_chunk_source(chunk)}
Page: {get_chunk_page(chunk)}
Content: {chunk.get('text', '')}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# 7. LLM EVIDENCE VALIDATION
# ============================================================

def validate_llm_evidence(
    evidence,
    retrieved_chunks
):

    validated = []
    seen = set()

    if not isinstance(evidence, list):
        return validated

    for item in evidence:

        if not isinstance(item, dict):
            continue

        statement = item.get("statement")

        if (
            not isinstance(statement, str)
            or not statement.strip()
        ):
            continue

        normalized_statement = normalize_text(
            statement
        )

        for chunk in retrieved_chunks:

            chunk_text = normalize_text(
                chunk.get("text", "")
            )

            if normalized_statement in chunk_text:

                evidence_item = {
                    "statement": statement.strip(),
                    "source": get_chunk_source(chunk),
                    "page": get_chunk_page(chunk),
                }

                key = evidence_fingerprint(
                    evidence_item
                )

                if key not in seen:

                    validated.append(
                        evidence_item
                    )

                    seen.add(key)

                break

    return validated


# ============================================================
# 8. LLM EVIDENCE EXTRACTION
# ============================================================

def extract_evidence(
    question,
    context,
    retrieved_chunks
):

    prompt = f"""
You are an enterprise document evidence extraction system.

Extract only exact evidence relevant to the user's question.

Rules:
1. Extract only information explicitly present in the documents.
2. Preserve numerical values and units.
3. Preserve conditions and restrictions.
4. Do not invent facts.
5. Do not include unrelated policies.
6. Do not combine unrelated statements.
7. Never generate statements describing missing information.
8. Extract every relevant statement.
9. Return an empty list if no relevant evidence exists.
10. Copy statements accurately from the supplied documents.
11. For questions asking WHEN or WHAT DATE, extract only the
    statement that explicitly establishes the relevant date or schedule.
12. Do not include general processing-method statements unless asked.

Return ONLY valid JSON in this format:

{{
    "evidence": [
        {{
            "statement": "Exact statement from the document"
        }}
    ]
}}

If no relevant evidence exists, return:

{{
    "evidence": []
}}

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}
"""

    response = call_ollama(prompt)

    if not isinstance(response, dict):
        return []

    evidence = response.get("evidence", [])

    return validate_llm_evidence(
        evidence,
        retrieved_chunks,
    )


# ============================================================
# 9. STRICT QUESTION-TO-EVIDENCE FILTER
# ============================================================

SALARY_TIMING_PATTERNS = [
    r"\blast working day\b",
    r"\blast business day\b",
    r"\blast day of (?:each|every|the) month\b",
    r"\bend of (?:each|every|the) month\b",
    r"\b\d{1,2}(?:st|nd|rd|th)? of (?:each|every|the) month\b",
    r"\bmonthly payroll date\b",
]


def is_salary_timing_question(question):

    return (
        "salary_payment" in get_question_concepts(question)
        and bool(
            re.search(
                r"\b(when|what date|which day|payday|date)\b",
                str(question).lower(),
            )
        )
    )


def contains_salary_payment_schedule(statement):

    return any(
        re.search(pattern, str(statement).lower())
        for pattern in SALARY_TIMING_PATTERNS
    )


# FIX: Recognize both numeric digits and written numbers.
NUMBER_WORD_PATTERN = (
    r"\b(?:"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|"
    r"half|a couple of|a few"
    r")\b"
)

NUMBER_WITH_UNIT_PATTERN = (
    NUMBER_WORD_PATTERN
    + r"\s+(?:"
    r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"percent|percentage|employees?|characters?|"
    r"rupees?|inr"
    r")\b"
)


def contains_numeric_value(statement):

    statement = str(statement).lower()

    # Digits: 12, 15, 1,200, 15.5, etc.
    if re.search(
        r"(?<!\w)\d+(?:,\d{3})*(?:\.\d+)?(?!\w)",
        statement,
    ):
        return True

    # Written numbers: one hour, fifteen days, etc.
    if re.search(
        NUMBER_WITH_UNIT_PATTERN,
        statement,
    ):
        return True

    return False


def filter_evidence_for_question(question, evidence):

    question_concepts = get_question_concepts(question)

    salary_timing_question = is_salary_timing_question(
        question
    )

    question_lower = str(question).lower()

    numeric_question = bool(
        re.search(
            r"\b(maximum|minimum|how much|what amount|"
            r"amount|limit|rate|percentage|percent)\b",
            question_lower,
        )
    )

    filtered = []
    seen = set()

    for item in evidence:

        if not isinstance(item, dict):
            continue

        statement = item.get("statement", "")

        if (
            not isinstance(statement, str)
            or not statement.strip()
        ):
            continue

        evidence_concepts = get_question_concepts(
            statement
        )

        # Salary questions must not accept unrelated bonus evidence.
        if "salary_payment" in question_concepts:

            if "salary_payment" not in evidence_concepts:
                continue

        # Salary payment date questions require an explicit schedule.
        if salary_timing_question:

            if not contains_salary_payment_schedule(statement):
                continue

        # Require relevant policy concept overlap.
        if question_concepts:

            if not question_concepts.intersection(
                evidence_concepts
            ):
                continue

        # FIX: Accept written numbers such as "one hour".
        if numeric_question:

            if not contains_numeric_value(statement):
                continue

        key = evidence_fingerprint(item)

        if key in seen:
            continue

        filtered.append(item)
        seen.add(key)

    return filtered


# ============================================================
# 10. RETRIEVAL-BASED EVIDENCE RECOVERY
# ============================================================

def recover_evidence_from_chunks(
    question,
    retrieved_chunks
):

    question_terms = (
        normalize_terms(question) - STOP_WORDS
    )

    if not question_terms:
        return []

    question_concepts = get_question_concepts(
        question
    )

    recovered = []
    seen = set()

    generic_terms = {
        "maximum",
        "minimum",
        "expense",
        "expenses",
        "reimbursement",
        "reimbursements",
        "employee",
        "employees",
        "per",
        "day",
        "days",
        "amount",
        "limit",
        "cost",
        "value",
        "much",
        "what",
        "how",
        "within",
        "time",
    }

    specific_question_terms = (
        question_terms - generic_terms
    )

    if not specific_question_terms:
        specific_question_terms = question_terms

    indexed_chunks = list(
        enumerate(retrieved_chunks)
    )

    indexed_chunks.sort(
        key=lambda item: (
            get_chunk_source(item[1]),
            get_chunk_page(item[1]),
            item[1].get("chunk_id", item[0]),
        )
    )

    merged_chunks = []
    current_group = None

    for original_position, chunk in indexed_chunks:

        text = chunk.get("text", "")

        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            continue

        source = get_chunk_source(chunk)
        page = get_chunk_page(chunk)

        chunk_id = chunk.get(
            "chunk_id",
            original_position,
        )

        if current_group is None:

            current_group = {
                "text": text.strip(),
                "source": source,
                "page": page,
                "last_chunk_id": chunk_id,
            }

            continue

        same_document = (
            current_group["source"] == source
            and current_group["page"] == page
        )

        try:

            consecutive = (
                int(chunk_id)
                == int(current_group["last_chunk_id"]) + 1
            )

        except (TypeError, ValueError):

            consecutive = False

        if same_document and consecutive:

            current_group["text"] += (
                " " + text.strip()
            )

            current_group["last_chunk_id"] = chunk_id

        else:

            merged_chunks.append(current_group)

            current_group = {
                "text": text.strip(),
                "source": source,
                "page": page,
                "last_chunk_id": chunk_id,
            }

    if current_group is not None:
        merged_chunks.append(current_group)

    for chunk in merged_chunks:

        normalized_document = " ".join(
            chunk["text"].split()
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            normalized_document,
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_terms = (
                normalize_terms(sentence) - STOP_WORDS
            )

            if not sentence_terms:
                continue

            sentence_concepts = get_question_concepts(
                sentence
            )

            if question_concepts:

                if not question_concepts.intersection(
                    sentence_concepts
                ):
                    continue

            else:

                overlap = (
                    specific_question_terms
                    .intersection(sentence_terms)
                )

                if not overlap:
                    continue

                coverage = (
                    len(overlap)
                    / max(len(specific_question_terms), 1)
                )

                if coverage < 0.35:
                    continue

            evidence_item = {
                "statement": sentence,
                "source": chunk["source"],
                "page": chunk["page"],
            }

            key = evidence_fingerprint(
                evidence_item
            )

            if key in seen:
                continue

            recovered.append(evidence_item)
            seen.add(key)

    return recovered


# ============================================================
# 11. COMBINE AND DEDUPLICATE EVIDENCE
# ============================================================

def combine_evidence(
    llm_evidence,
    recovered_evidence
):

    combined = []
    seen = set()

    all_evidence = (
        llm_evidence + recovered_evidence
    )

    for item in all_evidence:

        statement = item.get("statement", "")

        if (
            not isinstance(statement, str)
            or not statement.strip()
        ):
            continue

        key = evidence_fingerprint(item)

        if key in seen:
            continue

        combined.append(item)
        seen.add(key)

    return combined


# ============================================================
# 12. DETERMINISTIC EVIDENCE VERIFICATION
# ============================================================

def verify_evidence(question, evidence):

    if not evidence:

        return {
            "supported": False,
            "reason": (
                "No relevant evidence was extracted "
                "from the retrieved documents."
            ),
        }

    deterministic_result = (
        verify_evidence_deterministically(
            question,
            evidence,
        )
    )

    if deterministic_result["supported"] is not None:
        return deterministic_result

    return {
        "supported": False,
        "reason": deterministic_result["reason"],
    }


# ============================================================
# 13. MAIN ANSWERABILITY PIPELINE
# ============================================================


def check_answerability(
    question,
    retrieved_chunks
):
    """
    Check whether retrieved document chunks contain sufficient
    evidence to answer the question.

    This implementation does not require Ollama or any external
    LLM for evidence extraction, making it cloud-compatible.
    """

    if not retrieved_chunks:
        return {
            "supported": False,
            "reason": "No document evidence was retrieved.",
            "evidence": [],
        }

    # Stage 1: Recover evidence directly from retrieved chunks.
    evidence = recover_evidence_from_chunks(
        question,
        retrieved_chunks,
    )

    # Stage 2: Apply strict question-specific relevance filtering.
    evidence = filter_evidence_for_question(
        question,
        evidence,
    )

    # Stage 3: Perform deterministic evidence verification.
    verification = verify_evidence(
        question,
        evidence,
    )

    return {
        "supported": verification["supported"],
        "reason": verification["reason"],
        "evidence": evidence,
    }
