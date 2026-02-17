"""
Question-type classifier for MedQA validation cases (P1).

Classifies USMLE-style questions by type using heuristic regex patterns
on the question stem. This enables type-aware scoring and stratified reporting.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from validation.base import ValidationCase


class QuestionType(str, Enum):
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    MECHANISM = "mechanism"
    LAB_FINDING = "lab_finding"
    PHARMACOLOGY = "pharmacology"
    EPIDEMIOLOGY = "epidemiology"
    ETHICS = "ethics"
    ANATOMY = "anatomy"
    OTHER = "other"


# Pattern -> QuestionType mapping (checked in order, first match wins)
_STEM_PATTERNS: list[tuple[str, QuestionType]] = [
    # Diagnostic
    (r"most likely diagnosis", QuestionType.DIAGNOSTIC),
    (r"most likely cause", QuestionType.DIAGNOSTIC),
    (r"most likely explanation", QuestionType.DIAGNOSTIC),
    (r"what is the diagnosis", QuestionType.DIAGNOSTIC),
    (r"diagnosis is", QuestionType.DIAGNOSTIC),
    (r"most likely condition", QuestionType.DIAGNOSTIC),
    (r"most likely has", QuestionType.DIAGNOSTIC),
    (r"most likely suffer", QuestionType.DIAGNOSTIC),
    (r"most likely experiencing", QuestionType.DIAGNOSTIC),

    # Mechanism / Pathophysiology
    (r"mechanism of action", QuestionType.MECHANISM),
    (r"pathophysiology", QuestionType.MECHANISM),
    (r"mediator.*(responsible|involved)", QuestionType.MECHANISM),
    (r"(inhibit|block|activate).*receptor", QuestionType.MECHANISM),
    (r"cross[\s-]?link", QuestionType.MECHANISM),
    (r"most likely (due to|caused by|result of|secondary to)", QuestionType.MECHANISM),

    # Pharmacology (before treatment to catch drug-mechanism questions)
    (r"drug.*(target|mechanism|receptor|inhibit)", QuestionType.PHARMACOLOGY),
    (r"(target|act on|bind).*(receptor|enzyme|channel)", QuestionType.PHARMACOLOGY),
    (r"mode of action", QuestionType.PHARMACOLOGY),

    # Lab / Findings
    (r"most likely (finding|result)", QuestionType.LAB_FINDING),
    (r"expected (finding|result|value)", QuestionType.LAB_FINDING),
    (r"characteristic (finding|feature|appearance)", QuestionType.LAB_FINDING),
    (r"(agar|culture|stain|gram|biopsy).*(show|reveal|demonstrate)", QuestionType.LAB_FINDING),
    (r"(laboratory|lab).*(result|finding|value)", QuestionType.LAB_FINDING),
    (r"most likely (show|reveal|demonstrate)", QuestionType.LAB_FINDING),

    # Anatomy
    (r"(structure|nerve|artery|vein|muscle|ligament).*(damaged|injured|affected|involved)", QuestionType.ANATOMY),
    (r"which.*(nerve|artery|vein|muscle|vessel)", QuestionType.ANATOMY),

    # Epidemiology
    (r"(risk factor|prevalence|incidence|odds ratio|relative risk)", QuestionType.EPIDEMIOLOGY),
    (r"most (common|frequent).*(cause|risk|complication)", QuestionType.EPIDEMIOLOGY),

    # Treatment / Management (after mechanism/pharm to avoid misclassification)
    (r"most appropriate (next step|management|treatment|intervention|therapy|pharmacotherapy)", QuestionType.TREATMENT),
    (r"best (next step|initial step|management|treatment)", QuestionType.TREATMENT),
    (r"recommended (treatment|management|therapy)", QuestionType.TREATMENT),
    (r"most appropriate.*(action|course)", QuestionType.TREATMENT),
    (r"next (best )?step in (management|treatment|evaluation)", QuestionType.TREATMENT),
]

# Ethics keywords -- if ANY of these appear AND the question looks like treatment,
# reclassify as ethics
_ETHICS_KEYWORDS = re.compile(
    r"(tell|inform|disclose|report|consent|refuse|autonomy|confidentiality|"
    r"assent|surrogate|advance directive|do not resuscitate|DNR|ethics|ethical|"
    r"duty to warn|breach|malpractice|negligence|capacity|competence)",
    re.IGNORECASE,
)


def classify_question(case: "ValidationCase") -> QuestionType:
    """
    Classify a MedQA question by type using heuristics on the question stem.

    Looks at metadata["question_stem"] first, falls back to
    ground_truth["full_question"], then input_text.

    Returns:
        QuestionType enum value
    """
    stem = case.metadata.get("question_stem", "")
    full_q = case.ground_truth.get("full_question", case.input_text)

    # Classify on stem first (more specific), then full question
    result = QuestionType.OTHER
    for text in [stem, full_q]:
        if not text:
            continue
        text_lower = text.lower()
        for pattern, qtype in _STEM_PATTERNS:
            if re.search(pattern, text_lower):
                result = qtype
                break
        if result != QuestionType.OTHER:
            break

    # Ethics override: if classified as TREATMENT but ethics keywords present,
    # reclassify as ETHICS
    if result == QuestionType.TREATMENT:
        search_text = stem + " " + full_q
        if _ETHICS_KEYWORDS.search(search_text):
            result = QuestionType.ETHICS

    return result


def classify_question_from_text(question_text: str) -> QuestionType:
    """
    Classify a raw question string (no ValidationCase needed).
    Useful for ad-hoc classification.
    """
    text_lower = question_text.lower()
    for pattern, qtype in _STEM_PATTERNS:
        if re.search(pattern, text_lower):
            # Ethics override
            if qtype == QuestionType.TREATMENT and _ETHICS_KEYWORDS.search(question_text):
                return QuestionType.ETHICS
            return qtype
    return QuestionType.OTHER


# Convenience: which types are "pipeline-appropriate"?
DIAGNOSTIC_TYPES = {QuestionType.DIAGNOSTIC}
PIPELINE_APPROPRIATE_TYPES = {
    QuestionType.DIAGNOSTIC,
    QuestionType.TREATMENT,
    QuestionType.LAB_FINDING,
}
