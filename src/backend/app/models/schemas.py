# [Track A: Baseline]
"""
Domain models for the Clinical Decision Support Agent.

These Pydantic models define the structured data flowing through the agent pipeline.
Every tool consumes and produces typed models — no loose dicts or unstructured text.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Confidence(str, Enum):
    VERY_LOW = "very low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very high"
    UNRECOGNIZED = "unrecognized — requires review"


class AgentStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ──────────────────────────────────────────────
# Patient Data Models
# ──────────────────────────────────────────────

class Medication(BaseModel):
    name: str = Field(..., description="Medication name")
    dose: Optional[str] = Field(None, description="Dosage, e.g. '10mg daily'")
    rxcui: Optional[str] = Field(None, description="RxNorm concept ID")


class LabResult(BaseModel):
    test_name: str = Field(..., description="Lab test name")
    value: str = Field(..., description="Result value with units")
    reference_range: Optional[str] = Field(None, description="Normal reference range")
    is_abnormal: Optional[bool] = Field(None, description="Whether result is abnormal")


class VitalSigns(BaseModel):
    blood_pressure: Optional[str] = None
    heart_rate: Optional[str] = None
    temperature: Optional[str] = None
    respiratory_rate: Optional[str] = None
    oxygen_saturation: Optional[str] = None
    weight: Optional[str] = None
    height: Optional[str] = None


class PatientProfile(BaseModel):
    """Structured patient profile — output of the Patient Data Parser tool."""
    age: Optional[int] = None
    gender: Gender = Gender.UNKNOWN
    chief_complaint: str = Field(..., description="Primary reason for visit")
    history_of_present_illness: str = Field("", description="HPI narrative")
    past_medical_history: List[str] = Field(default_factory=list)
    current_medications: List[Medication] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    lab_results: List[LabResult] = Field(default_factory=list)
    vital_signs: Optional[VitalSigns] = None
    social_history: Optional[str] = None
    family_history: Optional[str] = None
    additional_notes: Optional[str] = None


# ──────────────────────────────────────────────
# Clinical Reasoning Models
# ──────────────────────────────────────────────

class DiagnosisCandidate(BaseModel):
    diagnosis: str = Field(..., description="Diagnosis name")
    icd10_code: Optional[str] = Field(None, description="ICD-10 code if known")
    likelihood: Confidence = Field(..., description="Estimated likelihood")
    likelihood_raw: Optional[str] = Field(None, description="Original model output if it didn't match a standard value")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence from patient data")
    reasoning: str = Field("", description="Clinical reasoning chain")

    @field_validator("likelihood", mode="before")
    @classmethod
    def normalize_likelihood(cls, v: str) -> str:
        """Map non-standard likelihood values to the closest Confidence enum member."""
        if isinstance(v, Confidence):
            return v
        raw = str(v).strip().lower()
        # Direct match
        for member in Confidence:
            if raw == member.value:
                return member
        # Fuzzy mapping for common LLM variations
        mapping = {
            "unlikely": "very low",
            "very unlikely": "very low",
            "rare": "very low",
            "possible": "low",
            "less likely": "low",
            "somewhat likely": "moderate",
            "medium": "moderate",
            "intermediate": "moderate",
            "likely": "high",
            "probable": "high",
            "very likely": "very high",
            "highly likely": "very high",
            "almost certain": "very high",
        }
        if raw in mapping:
            return Confidence(mapping[raw])
        # Unknown value — flag for clinical review, raw value preserved in likelihood_raw
        return Confidence.UNRECOGNIZED

    def __init__(self, **data):
        # Capture the raw value before validation normalizes it
        raw_val = data.get("likelihood")
        super().__init__(**data)
        if isinstance(raw_val, str):
            raw_lower = raw_val.strip().lower()
            known = {m.value for m in Confidence}
            # Also check the fuzzy mapping keys
            mapped = {
                "unlikely", "very unlikely", "rare", "possible", "less likely",
                "somewhat likely", "medium", "intermediate", "likely",
                "probable", "very likely", "highly likely", "almost certain",
            }
            if raw_lower not in known and raw_lower not in mapped:
                self.likelihood_raw = raw_val.strip()


class RecommendedAction(BaseModel):
    action: str = Field(..., description="Recommended action (test, referral, treatment)")
    priority: Severity = Field(..., description="Priority level")
    rationale: str = Field("", description="Why this action is recommended")


class ClinicalReasoningResult(BaseModel):
    """Output of the Clinical Reasoning Agent (MedGemma)."""
    differential_diagnosis: List[DiagnosisCandidate] = Field(
        default_factory=list, description="Ranked differential diagnosis"
    )
    risk_assessment: Optional[str] = Field(None, description="Overall risk assessment")
    recommended_workup: List[RecommendedAction] = Field(
        default_factory=list, description="Recommended tests, referrals, treatments"
    )
    reasoning_chain: str = Field("", description="Full chain-of-thought reasoning")


# ──────────────────────────────────────────────
# Drug Interaction Models
# ──────────────────────────────────────────────

class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: Severity
    description: str
    clinical_significance: Optional[str] = None
    source: str = Field("OpenFDA", description="Data source")


class DrugInteractionResult(BaseModel):
    """Output of the Drug Interaction Checker tool."""
    interactions_found: List[DrugInteraction] = Field(default_factory=list)
    medications_checked: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Guideline Retrieval Models
# ──────────────────────────────────────────────

class GuidelineExcerpt(BaseModel):
    title: str = Field(..., description="Guideline or source title")
    excerpt: str = Field(..., description="Relevant excerpt text")
    source: str = Field(..., description="Publication or organization")
    url: Optional[str] = None
    relevance_score: Optional[float] = None


class GuidelineRetrievalResult(BaseModel):
    """Output of the Guideline Retrieval (RAG) tool."""
    query: str = Field(..., description="The query used for retrieval")
    excerpts: List[GuidelineExcerpt] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Conflict Detection Models
# ──────────────────────────────────────────────

class ConflictType(str, Enum):
    OMISSION = "omission"                # Guideline recommends X, patient not receiving X
    CONTRADICTION = "contradiction"      # Patient's current care contradicts guideline
    DOSAGE = "dosage"                    # Dose adjustment criteria apply to this patient
    MONITORING = "monitoring"            # Required monitoring not documented/ordered
    ALLERGY_RISK = "allergy_risk"        # Guideline suggests drug patient is allergic to
    INTERACTION_GAP = "interaction_gap"  # Drug interaction not addressed in current plan


class ClinicalConflict(BaseModel):
    """A single detected conflict between guidelines and patient data."""
    conflict_type: ConflictType = Field(..., description="Category of the conflict")
    severity: Severity = Field(..., description="Potential clinical impact")

    @field_validator("conflict_type", mode="before")
    @classmethod
    def _normalise_conflict_type(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v

    @field_validator("severity", mode="before")
    @classmethod
    def _normalise_severity(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v
    guideline_source: str = Field("", description="Which guideline flagged this")
    guideline_text: str = Field("", description="What the guideline recommends")
    patient_data: str = Field("", description="Relevant patient data that conflicts")
    description: str = Field("", description="Plain-language explanation of the gap")
    suggested_resolution: Optional[str] = Field(
        None, description="Potential resolution for the clinician to consider"
    )


class ConflictDetectionResult(BaseModel):
    """Output of the Conflict Detection tool."""
    conflicts: List[ClinicalConflict] = Field(default_factory=list)
    guidelines_checked: int = Field(0, description="Number of guidelines compared")
    summary: str = Field("", description="Brief summary of conflict analysis")


# ──────────────────────────────────────────────
# Final CDS Report
# ──────────────────────────────────────────────

class CDSReport(BaseModel):
    """
    The final Clinical Decision Support report — synthesized by MedGemma
    from all tool outputs. This is what the clinician sees.
    """
    patient_summary: str = Field(..., description="Concise patient summary")
    differential_diagnosis: List[DiagnosisCandidate] = Field(default_factory=list)
    drug_interaction_warnings: List[DrugInteraction] = Field(default_factory=list)
    guideline_recommendations: List[str] = Field(
        default_factory=list, description="Guideline-concordant recommendations"
    )
    suggested_next_steps: List[RecommendedAction] = Field(default_factory=list)
    caveats: List[str] = Field(
        default_factory=list,
        description="Limitations, uncertainties, and disclaimers"
    )
    conflicts: List[ClinicalConflict] = Field(
        default_factory=list,
        description="Detected conflicts between guidelines and patient care"
    )
    sources_cited: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Agent Orchestration Models
# ──────────────────────────────────────────────

class AgentStep(BaseModel):
    """Represents a single step in the agent pipeline, streamed to the frontend."""
    step_id: str
    step_name: str
    status: AgentStepStatus = AgentStepStatus.PENDING
    tool_name: Optional[str] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class AgentState(BaseModel):
    """Full state of the agent pipeline for a given case."""
    case_id: str
    steps: List[AgentStep] = Field(default_factory=list)
    patient_profile: Optional[PatientProfile] = None
    clinical_reasoning: Optional[ClinicalReasoningResult] = None
    drug_interactions: Optional[DrugInteractionResult] = None
    guideline_retrieval: Optional[GuidelineRetrievalResult] = None
    conflict_detection: Optional[ConflictDetectionResult] = None
    final_report: Optional[CDSReport] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ──────────────────────────────────────────────
# API Request / Response Models
# ──────────────────────────────────────────────

class CaseSubmission(BaseModel):
    """API request to submit a new patient case for analysis."""
    patient_text: str = Field(
        ...,
        description="Free-text patient case description or structured data",
        min_length=10,
    )
    include_drug_check: bool = Field(True, description="Run drug interaction check")
    include_guidelines: bool = Field(True, description="Retrieve relevant guidelines")


class CaseResponse(BaseModel):
    """API response for a submitted case."""
    case_id: str
    status: str
    message: str


class CaseResult(BaseModel):
    """API response with the full case result."""
    case_id: str
    state: AgentState
    report: Optional[CDSReport] = None
