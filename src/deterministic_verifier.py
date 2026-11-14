
import re
from typing import Any


# ============================================================
# 1. TEXT NORMALIZATION
# ============================================================

def normalize(text: str) -> str:

    text = str(text).lower()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9\s%]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 2. PHRASE ALIASES
# ============================================================

PHRASE_ALIASES = {

    # Leave
    "maternity leaves": "maternity_leave",
    "maternity leave": "maternity_leave",
    "paternity leaves": "paternity_leave",
    "paternity leave": "paternity_leave",
    "paid parental leave": "parental_leave",
    "parental leave": "parental_leave",
    "casual leave": "casual_leave",
    "paid sick leave": "sick_leave",
    "sick leave": "sick_leave",
    "paid annual leave": "annual_leave",
    "annual leave": "annual_leave",
    "paid vacation": "annual_leave",
    "yearly vacation": "annual_leave",
    "annual vacation": "annual_leave",
    "vacation entitlement": "annual_leave",
    "emergency leave": "emergency_leave",
    "carry forward": "leave_carry_forward",
    "carried forward": "leave_carry_forward",
    "carry over": "leave_carry_forward",
    "carried over": "leave_carry_forward",
    "consecutive leave days": "consecutive_leave",
    "consecutive days of leave": "consecutive_leave",

    # Security
    "principle of least privilege":
    "least_privilege access_control_principle",
    "least privilege":
    "least_privilege access_control_principle",
    "minimum necessary permissions":
    "least_privilege access_control_principle",
    "minimum required permissions":
    "least_privilege access_control_principle",
    "access control principle": "access_control_principle",
    "access control model": "access_control_principle",
    "multi factor authentication": "mfa",
    "multifactor authentication": "mfa",
    "two factor authentication": "mfa",
    "two factor": "mfa",
    "biometric authentication": "biometric_authentication",
    "biometric verification": "biometric_authentication",
    "hardware security keys": "hardware_security_key",
    "hardware security key": "hardware_security_key",
    "security keys": "hardware_security_key",
    "security key": "hardware_security_key",
    "full disk encryption": "full_disk_encryption",
    "disk encryption": "full_disk_encryption",
    "entire storage drives": "full_disk_encryption",
    "entire storage drive": "full_disk_encryption",
    "entire hard drive": "full_disk_encryption",
    "automatic screen locking": "screen_lock",
    "automatic screen lock": "screen_lock",
    "screen locking": "screen_lock",
    "screen lock": "screen_lock",
    "locks its screen": "screen_lock",
    "lock its screen": "screen_lock",
    "leaves their screen": "screen_lock",

    # Passwords
    "minimum password length": "password_length",
    "password length": "password_length",
    "password lengths": "password_length",
    "password expiration intervals": "password_expiration",
    "password expiration interval": "password_expiration",
    "password expiration": "password_expiration",
    "password expiry interval": "password_expiration",
    "password expiry": "password_expiration",
    "passwords containing at least": "password_length",
    "passwords": "password",
    "characters": "character",

    # Employment
    "how many employees": "employee_count",
    "number of employees": "employee_count",
    "total employees": "employee_count",
    "employee count": "employee_count",
    "staff count": "employee_count",
    "workforce size": "employee_count",
    "working hours": "work_schedule",
    "work hours": "work_schedule",
    "office hours": "work_schedule",
    "workday": "work_schedule",
    "work day": "work_schedule",
    "start work": "work_schedule",
    "start working": "work_schedule",
    "begin work": "work_schedule",
    "begin working": "work_schedule",
    "finish work": "work_schedule",
    "finish working": "work_schedule",
    "end work": "work_schedule",
    "end working": "work_schedule",
    "start time": "work_schedule",
    "end time": "work_schedule",
    "begin workday": "work_schedule",
    "finish workday": "work_schedule",
    "notice period": "notice_period",
    "probationary period": "probation_period",
    "probation period": "probation_period",
    "probation duration": "probation_period",
    "performance reviews": "performance_review",
    "performance review": "performance_review",
    "review employee performance": "performance_review",
    "management review": "performance_review",
    "five month probation": "probation_period",
    "six month probation": "probation_period",
    "probation lasts": "probation_period",
    "probation is": "probation_period",
    "probation of": "probation_period",

    # Insurance
    "dental insurance": "dental_insurance",
    "health insurance": "health_insurance",
    "medical insurance": "health_insurance",

    # Remote work
    "unlimited remote working": "remote_work",
    "remote working": "remote_work",
    "working remotely": "remote_work",
    "work remotely": "remote_work",
    "remotely": "remote_work",
    "remote work": "remote_work",
    "work from home": "remote_work",

    # Security incidents
    "security incidents": "security_incident",
    "security incident": "security_incident",
    "incident reporting deadline": "security_incident",
    "incident reporting": "security_incident",

    # Data access
    "assigned job responsibilities": "data_access_scope",
    "assigned responsibilities": "data_access_scope",
    "regardless of their role": "data_access_scope",
    "regardless of role": "data_access_scope",
    "all company information": "data_access_scope",
    "information required": "data_access_scope",
    "information unrelated to their job": "data_access_scope",
    "information unrelated to the job": "data_access_scope",
    "unrelated to their job": "data_access_scope",
    "unrelated to the job": "data_access_scope",
    "not related to their job": "data_access_scope",
    "not required for their job": "data_access_scope",
    "outside their job responsibilities": "data_access_scope",
    "outside assigned responsibilities": "data_access_scope",

    # Calendar units
    "working days": "working_day",
    "working day": "working_day",
    "calendar days": "calendar_day",
    "calendar day": "calendar_day",

    # Apex compensation
    "annual performance bonus": "annual_bonus",
    "performance bonus": "annual_bonus",
    "annual bonus": "annual_bonus",
    "bonus payment": "annual_bonus",
    "salary payment": "salary_payment",
    "salary payments": "salary_payment",
    "last working day": "salary_payment",
    "retirement fund contribution": "retirement_fund",
    "retirement fund": "retirement_fund",
    "provident fund contribution": "retirement_fund",
    "professional course reimbursement": "professional_course",
    "course reimbursement": "professional_course",
    "certification reimbursement": "certification_reimbursement",
    "professional development reimbursement": "professional_course",
    "professional course cert reimbursement": "professional_course",
    "professional course certification reimbursement": "professional_course",
    "professional courses certifications reimbursement": "professional_course",
    "professional courses and certifications": "professional_course",
    "course certification reimbursement": "professional_course",
    "course and certification reimbursement": "professional_course",
    "professional course cert": "professional_course",
    "professional course certification": "professional_course",
    "annual professional course reimbursement": "professional_course",

    # Apex security
    "security updates": "security_update",
    "security update": "security_update",
    "software updates": "security_update",
    "software update": "security_update",
    "cybersecurity incidents": "security_incident",
    "cybersecurity incident": "security_incident",
    "business authorization": "business_authorization",

    # Apex workplace
    "reporting manager": "reporting_manager",
    "reporting to the manager": "reporting_manager",
    "late arrival": "late_arrival",
    "arriving late": "late_arrival",
    "late for work": "late_arrival",

    # Apex expenses
    "domestic business travel": "domestic_travel",
    "business travel": "domestic_travel",
    "travel expense limit": "domestic_travel",
    "travel expenses": "domestic_travel",
    "meal expenses": "meal_expense",
    "meal expense": "meal_expense",
    "daily meal allowance": "meal_expense",
    "internet reimbursement": "internet_reimbursement",
    "internet reimbursements": "internet_reimbursement",
    "home internet expenses": "internet_reimbursement",
    "home internet reimbursement": "internet_reimbursement",
    "broadband reimbursement": "internet_reimbursement",
    "internet allowance": "internet_reimbursement",
    "equipment allowance": "equipment_allowance",
    "work from home equipment": "equipment_allowance",
    "work from home equipment allowance": "equipment_allowance",
    "home office equipment": "equipment_allowance",
    "home office equipment allowance": "equipment_allowance",
    "original receipts": "receipt_requirement",
    "valid digital invoices": "receipt_requirement",
    "digital invoices": "receipt_requirement",
    "expense submission deadline": "expense_submission",
    "expense date": "expense_submission",
    "local business transportation": "business_transportation",
    "business transportation": "business_transportation",
}


# ============================================================
# 3. STOP WORDS
# ============================================================

STOP_WORDS = {
    "a", "an", "the",
    "is", "are", "was", "were",
    "be", "been", "being",
    "do", "does", "did",
    "has", "have", "had",
    "can", "could", "would",
    "should", "will", "shall",
    "may", "might", "must",
    "of", "in", "on", "at",
    "to", "for", "from", "by",
    "with", "and", "or", "but",
    "what", "which", "who",
    "whom", "whose", "when",
    "where", "why", "how",
    "many", "much", "often",
    "frequently",
    "company", "companies",
    "employees", "employee",
    "provided", "provide", "provides",
    "available", "required", "require",
    "their", "they", "them",
    "it", "its", "this", "that",
    "these", "those",
    "same", "equivalent", "identical",
    "minimum", "maximum",
    "more", "less", "than",
    "number", "amount", "quantity",
    "period", "duration", "time",
    "days", "weeks", "months", "years",
    "expect", "expected", "allow", "allowed",
    "specified", "specify", "policy",
    "follow", "follows", "model",
    "principle", "happen", "automatically",
    "use", "using", "within", "per", "every",
}


# ============================================================
# 4. TERM NORMALIZATION
# ============================================================

def normalize_terms(text: str) -> set:

    normalized = normalize(text)

    aliases = sorted(
        PHRASE_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for phrase, replacement in aliases:

        pattern = r"\b" + re.escape(phrase) + r"\b"

        normalized = re.sub(
            pattern,
            replacement,
            normalized
        )

    return set(normalized.split())


# ============================================================
# 5. NUMBER EXTRACTION
# ============================================================

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2,
    "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60,
}


def extract_numbers(text: str) -> list:

    normalized = str(text).lower()

    numbers = []

    matches = re.findall(
        r"(?<![a-zA-Z])\d{1,3}(?:,\d{3})+(?:\.\d+)?"
        r"|\b\d+(?:\.\d+)?\b",
        normalized
    )

    for match in matches:
        numbers.append(float(match.replace(",", "")))

    for word, value in NUMBER_WORDS.items():

        if re.search(
            r"\b" + word + r"\b",
            normalized
        ):
            numbers.append(float(value))

    return numbers


# ============================================================
# 6. EVIDENCE TEXT EXTRACTION
# ============================================================

def get_evidence_text(evidence: Any) -> str:

    if isinstance(evidence, str):
        return evidence

    if isinstance(evidence, dict):

        return str(
            evidence.get("statement")
            or evidence.get("text")
            or evidence.get("content")
            or ""
        )

    if isinstance(evidence, list):

        statements = []

        for item in evidence:

            if isinstance(item, str):
                statements.append(item)

            elif isinstance(item, dict):

                statement = (
                    item.get("statement")
                    or item.get("text")
                    or item.get("content")
                    or ""
                )

                if statement:
                    statements.append(str(statement))

        return " ".join(statements)

    return ""


# ============================================================
# 7. QUESTION INTENT GUARDS
# ============================================================

def check_specific_question_intent(
    question: str,
    evidence: Any
) -> bool:

    q = normalize(question)
    e = normalize(get_evidence_text(evidence))

    # External contractor access must be explicitly documented.
    asks_about_contractors = bool(
        re.search(
            r"\b(external contractors?|third party contractors?|outside contractors?)\b",
            q
        )
    )

    evidence_mentions_contractors = bool(
        re.search(
            r"\b(external contractors?|third party contractors?|outside contractors?)\b",
            e
        )
    )

    if asks_about_contractors and not evidence_mentions_contractors:
        return False

    # Salary payment timing must be explicitly documented.
    asks_salary_timing = (
        bool(re.search(r"\b(salary|salaries|payday|pay date)\b", q))
        and bool(
            re.search(
                r"\b(when|what date|which day|receive|received|paid|payday)\b",
                q
            )
        )
    )

    salary_payment_patterns = [
        r"\blast working day\b",
        r"\blast business day\b",
        r"\blast day of (?:each|every|the) month\b",
        r"\bend of (?:each|every|the) month\b",
        r"\b\d{1,2}(?:st|nd|rd|th)? of (?:each|every|the) month\b",
    ]

    if asks_salary_timing:

        return any(
            re.search(pattern, e)
            for pattern in salary_payment_patterns
        )

    strict_phrases = {

        "equipment allowance": [
            "equipment allowance",
            "work from home equipment",
            "home office equipment",
        ],

        "internet reimbursement": [
            "internet reimbursement",
            "internet reimbursements",
            "home internet",
            "broadband reimbursement",
            "internet allowance",
        ],

        "leave carry-forward": [
            "carry forward",
            "carried forward",
            "carry over",
            "carried over",
        ],

        "biometric authentication": [
            "biometric",
            "fingerprint",
            "facial recognition",
            "face recognition",
            "iris scan",
        ],

        "password expiration": [
            "password expiration",
            "password expiry",
            "password expires",
            "password rotation",
            "password must be changed",
            "change passwords every",
        ],

        "hardware security keys": [
            "hardware security key",
            "security key",
            "security keys",
        ],

        "bonus": [
            "bonus",
            "incentive",
        ],

        "salary payment": [
            "salary",
            "salaries",
            "salary payment",
            "salary payments",
            "monthly salary",
            "payday",
        ],

        "retirement fund": [
            "retirement fund",
            "provident fund",
        ],

        "professional course": [
            "professional course",
            "professional courses",
            "course reimbursement",
            "courses and certifications",
            "certification reimbursement",
            "certifications",
            "professional development",
            "professional development reimbursement",
        ],

        "domestic travel": [
            "domestic travel",
            "business travel",
            "travel expense",
        ],

        "meal expense": [
            "meal expense",
            "meal allowance",
        ],

        "receipt requirement": [
            "receipt",
            "digital invoice",
            "invoice",
        ],

        "expense submission": [
            "expense submission",
            "expense date",
        ],

        "confidential information access": [
            "confidential information",
            "access confidential information",
            "confidential data",
            "business authorization",
        ],

        "security update": [
            "security update",
            "software update",
        ],
    }

    for concept, phrases in strict_phrases.items():

        asks_concept = any(
            re.search(r"\b" + re.escape(phrase) + r"\b", q)
            for phrase in phrases
        )

        if not asks_concept:
            continue

        evidence_mentions_concept = any(
            re.search(r"\b" + re.escape(phrase) + r"\b", e)
            for phrase in phrases
        )

        if not evidence_mentions_concept:
            return False

    # Carry-forward needs explicit evidence.
    if re.search(
        r"\b(carry forward|carried forward|carry over|carried over)\b",
        q
    ):
        if not re.search(
            r"\b(carry forward|carried forward|carry over|carried over|roll over|rolled over)\b",
            e
        ):
            return False

    # Biometric authentication needs explicit evidence.
    if re.search(
        r"\b(biometric|fingerprint|facial recognition|face recognition|iris scan)\b",
        q
    ):
        if not re.search(
            r"\b(biometric|fingerprint|facial recognition|face recognition|iris scan)\b",
            e
        ):
            return False

    # Password expiration and password length are different.
    if re.search(
        r"\b(password expiration|password expiry|password expires|password rotation)\b",
        q
    ):
        if not re.search(
            r"\b(password expiration|password expiry|password expires|password rotation|password must be changed|change passwords every)\b",
            e
        ):
            return False

    # Consecutive leave needs explicit evidence.
    if (
        re.search(r"\b(consecutive|continuous)\b", q)
        and re.search(r"\bleave\b", q)
    ):
        if not re.search(r"\b(consecutive|continuous)\b", e):
            return False

    # Employee count must explicitly mention people.
    if re.search(
        r"\b(how many employees|number of employees|employee count|staff count|workforce size)\b",
        q
    ):
        if not re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand)\s+(employees|staff members|people|workers)\b",
            e
        ):
            return False

    # Equivalence questions need explicit equivalence evidence.
    if re.search(
        r"\b(same as|equivalent to|identical to)\b",
        q
    ):
        if not re.search(
            r"\b(same as|equivalent to|identical to)\b",
            e
        ):
            return False

    # Dental insurance.
    if "dental insurance" in q and "dental insurance" not in e:
        return False

    return True


# ============================================================
# 8. SUBJECT MATCHING
# ============================================================

def get_question_and_evidence_terms(question, evidence):

    question_terms = normalize_terms(question) - STOP_WORDS

    evidence_terms = (
        normalize_terms(get_evidence_text(evidence))
        - STOP_WORDS
    )

    return question_terms, evidence_terms


def get_subject_overlap(question, evidence):

    question_terms, evidence_terms = (
        get_question_and_evidence_terms(question, evidence)
    )

    generic_terms = {
        "day", "week", "month", "year",
        "hour", "minute", "character",
        "working_day", "calendar_day",
        "number", "amount", "quantity",
        "percentage", "percent",
    }

    question_specific = question_terms - generic_terms
    evidence_specific = evidence_terms - generic_terms

    return question_specific.intersection(evidence_specific)


# ============================================================
# 9. NUMERIC VERIFICATION
# ============================================================

def verify_numeric_limit(question: str, evidence: Any):

    evidence_text = get_evidence_text(evidence)

    question_normalized = normalize(question)
    evidence_normalized = normalize(evidence_text)

    question_terms, statement_terms = (
        get_question_and_evidence_terms(question, evidence)
    )

    evidence_numbers = extract_numbers(evidence_text)

    if not evidence_numbers:
        return None

    # Employee count must have explicit employee evidence.
    if "employee_count" in question_terms:
        return None

    # Leave categories must match.
    leave_categories = {
        "maternity_leave",
        "paternity_leave",
        "parental_leave",
        "casual_leave",
        "sick_leave",
        "annual_leave",
        "emergency_leave",
    }

    question_leave = question_terms.intersection(leave_categories)
    statement_leave = statement_terms.intersection(leave_categories)

    if question_leave and not question_leave.intersection(statement_leave):
        return None

    # Password expiration and length are different.
    if "password_expiration" in question_terms:
        if "password_expiration" not in statement_terms:
            return None

    # --------------------------------------------------------
    # PASSWORD LENGTH VERIFICATION
    # --------------------------------------------------------

    password_length_match = False

    if "password_length" in question_terms:

        evidence_has_password = (
            "password" in statement_terms
        )

        evidence_has_character_unit = bool(
            re.search(
                r"\bcharacters?\b",
                evidence_normalized
            )
        )

        password_length_match = (
            evidence_has_password
            and evidence_has_character_unit
        )

        if not password_length_match:
            return None

    # Specific policy concepts must match.
    special_concepts = {
        "probation_period",
        "performance_review",
        "remote_work",
        "security_incident",
        "work_schedule",
        "annual_leave",
        "sick_leave",
        "public_holidays",
        "notice_period",
        "internet_reimbursement",
        "equipment_allowance",
        "health_insurance",
        "annual_bonus",
        "retirement_fund",
        "salary_payment",
        "professional_course",
        "certification_reimbursement",
        "security_update",
        "domestic_travel",
        "meal_expense",
        "receipt_requirement",
        "expense_submission",
        "business_transportation",
    }

    # Password length is intentionally checked separately above.
    question_special = question_terms.intersection(special_concepts)
    evidence_special = statement_terms.intersection(special_concepts)

    if (
        question_special
        and not question_special.intersection(evidence_special)
    ):
        return None

    # Bonus and salary increment are different.
    asks_bonus = bool(
        re.search(r"\b(bonus|incentive)\b", question_normalized)
    )

    if asks_bonus and not re.search(
        r"\b(bonus|incentive)\b",
        evidence_normalized
    ):
        return None

    # Reimbursement categories must match.
    reimbursement_concepts = {
        "internet_reimbursement",
        "equipment_allowance",
        "professional_course",
        "certification_reimbursement",
    }

    question_reimbursement = question_terms.intersection(
        reimbursement_concepts
    )

    evidence_reimbursement = statement_terms.intersection(
        reimbursement_concepts
    )

    if (
        question_reimbursement
        and not question_reimbursement.intersection(evidence_reimbursement)
    ):
        return None

    # Require a matching subject unless this is a verified
    # password-length question.
    subject_overlap = get_subject_overlap(question, evidence)

    if not subject_overlap and not password_length_match:
        return None

    # Match requested units, including common adverbs.
    unit_groups = {
        "day": {
            "day", "days", "daily"
        },
        "week": {
            "week", "weeks", "weekly"
        },
        "month": {
            "month", "months", "monthly"
        },
        "year": {
            "year", "years", "yearly", "annually", "annual"
        },
        "hour": {
            "hour", "hours", "hourly"
        },
        "minute": {
            "minute", "minutes"
        },
        "character": {
            "character", "characters"
        },
    }

    question_units = set()
    evidence_units = set()

    for unit, variants in unit_groups.items():

        if any(
            re.search(
                r"\b" + re.escape(variant) + r"\b",
                question_normalized
            )
            for variant in variants
        ):
            question_units.add(unit)

        if any(
            re.search(
                r"\b" + re.escape(variant) + r"\b",
                evidence_normalized
            )
            for variant in variants
        ):
            evidence_units.add(unit)

    if question_units and not question_units.issubset(evidence_units):
        return None

    # Percentage questions must have percentage evidence.
    asks_percentage = bool(
        re.search(
            r"\b(percent|percentage)\b|%",
            question.lower()
        )
    )

    if asks_percentage:
        if not re.search(
            r"%|\bpercent(?:age)?\b",
            evidence_text.lower()
        ):
            return None

    # Currency questions must have currency evidence.
    asks_currency = bool(
        re.search(
            r"\b(inr|rupees?|rs)\b|₹",
            question.lower()
        )
    )

    if asks_currency:
        if not re.search(
            r"\b(inr|rupees?|rs)\b|₹",
            evidence_text.lower()
        ):
            return None

    return {
        "supported": True,
        "reason": (
            "The evidence contains a matching policy concept "
            "and an explicit numerical value."
        )
    }


# ============================================================
# 10. TIME FACT VERIFICATION
# ============================================================

def verify_time_fact(question: str, evidence: Any):

    evidence_text = get_evidence_text(evidence)

    q = normalize(question)
    e = normalize(evidence_text)

    question_terms, evidence_terms = (
        get_question_and_evidence_terms(question, evidence)
    )

    # Salary payment timing.
    asks_salary_timing = (
        bool(
            re.search(
                r"\b(salary|salaries|payday|pay date)\b",
                q
            )
        )
        and bool(
            re.search(
                r"\b(when|what date|which day|receive|received|paid|payday)\b",
                q
            )
        )
    )

    salary_payment_patterns = [
        r"\blast working day\b",
        r"\blast business day\b",
        r"\blast day of (?:each|every|the) month\b",
        r"\bend of (?:each|every|the) month\b",
        r"\b\d{1,2}(?:st|nd|rd|th)? of (?:each|every|the) month\b",
    ]

    if asks_salary_timing:

        if any(
            re.search(pattern, e)
            for pattern in salary_payment_patterns
        ):
            return {
                "supported": True,
                "reason": (
                    "The evidence explicitly specifies "
                    "the salary payment schedule."
                )
            }

        return None

    # Working hours.
    asks_work_schedule = (
        "work_schedule" in question_terms
        or bool(
            re.search(
                r"\b(start work|begin work|finish work|end work|workday|working hours|office hours)\b",
                q
            )
        )
    )

    has_time = bool(
        re.search(
            r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
            e
        )
    )

    if asks_work_schedule and has_time:
        return {
            "supported": True,
            "reason": (
                "The evidence specifies the company's "
                "working-hour information."
            )
        }

    return None


# ============================================================
# 11. DIRECT FACT VERIFICATION
# ============================================================

def verify_direct_fact(question: str, evidence: Any):

    evidence_text = get_evidence_text(evidence)
    q = normalize(question)

    question_terms, statement_terms = (
        get_question_and_evidence_terms(question, evidence)
    )

    if not check_specific_question_intent(question, evidence):
        return None

    # Confidential information access verification.
    if (
        re.search(
            r"\b(confidential information|confidential data)\b",
            q
        )
        and re.search(
            r"\b(who|authorized|authorised|access)\b",
            q
        )
    ):

        if (
            re.search(
                r"\b(confidential information|confidential data)\b",
                evidence_text,
                re.IGNORECASE
            )
            and re.search(
                r"\b(explicit business authorization|business authorization)\b",
                evidence_text,
                re.IGNORECASE
            )
        ):
            return {
                "supported": True,
                "reason": (
                    "The Apex IT Security Policy explicitly states "
                    "that confidential information can only be accessed "
                    "by employees with business authorization."
                )
            }

    # Leave category guard.
    leave_categories = {
        "maternity_leave",
        "paternity_leave",
        "parental_leave",
        "casual_leave",
        "sick_leave",
        "annual_leave",
        "emergency_leave",
    }

    question_leave = question_terms.intersection(leave_categories)
    statement_leave = statement_terms.intersection(leave_categories)

    if question_leave and not question_leave.intersection(statement_leave):
        return None

    # Password expiration guard.
    if "password_expiration" in question_terms:

        if "password_expiration" not in statement_terms:
            return None

    # Password length guard.
    # Do not require the artificial "password_length" alias
    # to appear in the evidence.
    if "password_length" in question_terms:

        evidence_normalized = normalize(evidence_text)

        if (
            "password" not in statement_terms
            or not re.search(
                r"\bcharacters?\b",
                evidence_normalized
            )
        ):
            return None

    # Strict concepts.
    strict_concepts = {
        "biometric_authentication",
        "hardware_security_key",
        "dental_insurance",
        "employee_count",
    }

    question_strict = question_terms.intersection(strict_concepts)

    if (
        question_strict
        and not question_strict.intersection(statement_terms)
    ):
        return None

    # Least privilege / confidential information access.
    access_concepts = {
        "least_privilege",
        "access_control_principle",
        "data_access_scope",
        "business_authorization",
    }

    access_question = question_terms.intersection(access_concepts)
    access_evidence = statement_terms.intersection(access_concepts)

    if access_question:

        if (
            "data_access_scope" in access_question
            or "business_authorization" in access_question
        ):

            if "business_authorization" in statement_terms:
                return {
                    "supported": True,
                    "reason": (
                        "The evidence explicitly states "
                        "that confidential information "
                        "access requires business authorization."
                    )
                }

            return None

        if access_evidence:
            return {
                "supported": True,
                "reason": (
                    "The evidence describes access restrictions "
                    "based on assigned job responsibilities."
                )
            }

    # Screen lock.
    if "screen_lock" in question_terms:

        if "screen_lock" in statement_terms:
            return {
                "supported": True,
                "reason": (
                    "The evidence specifies automatic "
                    "screen-lock protection."
                )
            }

        return None

    # Full-disk encryption.
    if "full_disk_encryption" in question_terms:

        if "full_disk_encryption" in statement_terms:
            return {
                "supported": True,
                "reason": (
                    "The evidence specifies full-disk "
                    "encryption for company laptops."
                )
            }

        return None

    # Multi-factor authentication.
    if "mfa" in question_terms:

        if "mfa" in statement_terms:
            return {
                "supported": True,
                "reason": (
                    "The evidence specifies multi-factor "
                    "authentication requirements."
                )
            }

        return None

    # Apex-specific and Northstar important concepts.
    important_concepts = {
        "work_schedule",
        "remote_work",
        "performance_review",
        "probation_period",
        "notice_period",
        "security_incident",
        "health_insurance",
        "annual_leave",
        "sick_leave",
        "public_holidays",
        "emergency_leave",
        "late_arrival",
        "annual_bonus",
        "retirement_fund",
        "salary_payment",
        "professional_course",
        "certification_reimbursement",
        "security_update",
        "domestic_travel",
        "meal_expense",
        "internet_reimbursement",
        "equipment_allowance",
        "receipt_requirement",
        "expense_submission",
        "business_transportation",
        "business_authorization",
    }

    important_overlap = (
        question_terms.intersection(important_concepts)
        .intersection(statement_terms)
    )

    if important_overlap:
        return {
            "supported": True,
            "reason": (
                "The extracted evidence contains matching "
                "factual concepts relevant to the question."
            )
        }

    # General concept overlap.
    overlap = question_terms.intersection(statement_terms)

    if not overlap:
        return None

    coverage = (
        len(overlap) / len(question_terms)
        if question_terms else 0
    )

    if coverage >= 0.50:
        return {
            "supported": True,
            "reason": (
                "The extracted evidence contains sufficient "
                "matching factual concepts."
            )
        }

    return None


# ============================================================
# 12. MULTI-PART QUESTION VERIFICATION
# ============================================================

def split_multi_part_question(question: str) -> list:

    normalized = normalize(question)

    parts = re.split(r"\s+and\s+", normalized)

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def verify_multi_part_question(question: str, evidence: Any):

    parts = split_multi_part_question(question)

    if len(parts) <= 1:
        return None

    for part in parts:

        result = verify_evidence_deterministically(
            part,
            evidence
        )

        if result.get("supported") is not True:
            return None

    return {
        "supported": True,
        "reason": (
            "All parts of the question are supported "
            "by the extracted evidence."
        )
    }


# ============================================================
# 13. MAIN DETERMINISTIC VERIFIER
# ============================================================

def verify_evidence_deterministically(
    question: str,
    evidence: Any
):

    if not evidence:
        return {
            "supported": None,
            "reason": "No evidence was provided."
        }

    evidence_text = get_evidence_text(evidence)

    if not evidence_text.strip():
        return {
            "supported": None,
            "reason": "Evidence contains no usable text."
        }

    # 1. Strict intent checks.
    if not check_specific_question_intent(question, evidence):
        return {
            "supported": None,
            "reason": (
                "The evidence does not establish the "
                "specific policy feature requested."
            )
        }

    # 2. Multi-part verification.
    multi_result = verify_multi_part_question(
        question,
        evidence
    )

    if multi_result is not None:
        return multi_result

    # 3. Numeric verification.
    numeric_result = verify_numeric_limit(
        question,
        evidence
    )

    if numeric_result is not None:
        return numeric_result

    # Do not allow quantitative questions to bypass
    # numeric verification and reach generic fact matching.
    quantitative_intent = bool(
        re.search(
            r"\b(how much|how many|maximum|minimum|max|min|limit|amount|"
            r"percentage|percent|rate)\b",
            normalize(question)
        )
    )

    if quantitative_intent:
        return {
            "supported": None,
            "reason": (
                "The evidence does not satisfy the "
                "required numerical verification."
            )
        }

    # 4. Time verification.
    time_result = verify_time_fact(
        question,
        evidence
    )

    if time_result is not None:
        return time_result

    # 5. Direct fact verification.
    direct_result = verify_direct_fact(
        question,
        evidence
    )

    if direct_result is not None:
        return direct_result

    # 6. Fail-closed fallback.
    return {
        "supported": None,
        "reason": (
            "No deterministic verification rule matched. "
            "The verifier cannot establish sufficient evidence."
        )
    }





