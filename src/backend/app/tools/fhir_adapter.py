"""
FHIR R4 input adapter for the CDS pipeline.

Converts FHIR R4 resources (Patient, Condition, MedicationStatement,
Observation, AllergyIntolerance) into the pipeline's internal
PatientProfile model.

Supports:
  - FHIR Patient resource → demographics
  - FHIR Condition resources → past medical history
  - FHIR MedicationStatement / MedicationRequest → current medications
  - FHIR Observation resources → lab results + vitals
  - FHIR AllergyIntolerance → allergies

Usage:
    from app.tools.fhir_adapter import FHIRAdapter
    adapter = FHIRAdapter()
    profile = adapter.to_patient_profile(fhir_bundle)

Reference: https://www.hl7.org/fhir/R4/
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

from app.models.schemas import (
    Gender,
    LabResult,
    Medication,
    PatientProfile,
    VitalSigns,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# FHIR resource type constants
# ──────────────────────────────────────────────

RESOURCE_PATIENT = "Patient"
RESOURCE_CONDITION = "Condition"
RESOURCE_MEDICATION_STATEMENT = "MedicationStatement"
RESOURCE_MEDICATION_REQUEST = "MedicationRequest"
RESOURCE_OBSERVATION = "Observation"
RESOURCE_ALLERGY = "AllergyIntolerance"
RESOURCE_BUNDLE = "Bundle"

# LOINC codes for vital signs
VITAL_LOINC = {
    "85354-9": "blood_pressure",     # Blood pressure panel
    "8480-6": "systolic_bp",          # Systolic BP
    "8462-4": "diastolic_bp",        # Diastolic BP
    "8867-4": "heart_rate",          # Heart rate
    "8310-5": "temperature",         # Body temperature
    "9279-1": "respiratory_rate",    # Respiratory rate
    "2708-6": "oxygen_saturation",   # SpO2 (older code)
    "59408-5": "oxygen_saturation",  # SpO2 (preferred)
    "29463-7": "weight",             # Body weight
    "8302-2": "height",              # Body height
}


class FHIRAdapter:
    """
    Converts FHIR R4 resources to the CDS pipeline's PatientProfile.

    Accepts either:
      - A FHIR Bundle (resourceType: "Bundle" with entry[])
      - A dict with resource-type keys (e.g., {"Patient": {...}, "Conditions": [...]})
      - Individual resources
    """

    def to_patient_profile(
        self,
        fhir_data: Dict[str, Any],
        chief_complaint: str = "",
        hpi: str = "",
    ) -> PatientProfile:
        """
        Convert FHIR data into a PatientProfile.

        Args:
            fhir_data: FHIR Bundle, or dict of resources, or single resource
            chief_complaint: Chief complaint text (may not be in FHIR data)
            hpi: History of present illness (may not be in FHIR data)

        Returns:
            PatientProfile ready for the CDS pipeline
        """
        resources = self._extract_resources(fhir_data)

        # Demographics from Patient resource
        age, gender, social_hx, family_hx = self._parse_patient(
            resources.get(RESOURCE_PATIENT)
        )

        # Past medical history from Conditions
        pmh = self._parse_conditions(resources.get(RESOURCE_CONDITION, []))

        # Medications
        meds = self._parse_medications(
            resources.get(RESOURCE_MEDICATION_STATEMENT, [])
            + resources.get(RESOURCE_MEDICATION_REQUEST, [])
        )

        # Labs & Vitals from Observations
        observations = resources.get(RESOURCE_OBSERVATION, [])
        labs, vitals = self._parse_observations(observations)

        # Allergies
        allergies = self._parse_allergies(resources.get(RESOURCE_ALLERGY, []))

        # Compose chief complaint from conditions if not provided
        if not chief_complaint and pmh:
            chief_complaint = f"Patient with: {', '.join(pmh[:3])}"

        return PatientProfile(
            age=age,
            gender=gender,
            chief_complaint=chief_complaint or "See FHIR data",
            history_of_present_illness=hpi,
            past_medical_history=pmh,
            current_medications=meds,
            allergies=allergies,
            lab_results=labs,
            vital_signs=vitals if any(
                v is not None for v in [
                    vitals.blood_pressure, vitals.heart_rate, vitals.temperature,
                    vitals.respiratory_rate, vitals.oxygen_saturation,
                    vitals.weight, vitals.height,
                ]
            ) else None,
            social_history=social_hx,
            family_history=family_hx,
            additional_notes="Imported from FHIR R4 resources",
        )

    def _extract_resources(
        self, fhir_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract typed resources from various FHIR input formats.

        Returns dict: {resourceType: resource_or_list}
        """
        result: Dict[str, Any] = {
            RESOURCE_PATIENT: None,
            RESOURCE_CONDITION: [],
            RESOURCE_MEDICATION_STATEMENT: [],
            RESOURCE_MEDICATION_REQUEST: [],
            RESOURCE_OBSERVATION: [],
            RESOURCE_ALLERGY: [],
        }

        resource_type = fhir_data.get("resourceType", "")

        # Case 1: FHIR Bundle
        if resource_type == RESOURCE_BUNDLE:
            entries = fhir_data.get("entry", [])
            for entry in entries:
                resource = entry.get("resource", entry)
                self._classify_resource(resource, result)
            return result

        # Case 2: Single resource
        if resource_type in (
            RESOURCE_PATIENT, RESOURCE_CONDITION,
            RESOURCE_MEDICATION_STATEMENT, RESOURCE_MEDICATION_REQUEST,
            RESOURCE_OBSERVATION, RESOURCE_ALLERGY,
        ):
            self._classify_resource(fhir_data, result)
            return result

        # Case 3: Dict with resource-type keys
        for key, value in fhir_data.items():
            # Normalize key
            norm_key = key.replace(" ", "")
            if norm_key.lower() == "patient" and isinstance(value, dict):
                result[RESOURCE_PATIENT] = value
            elif norm_key.lower() in ("condition", "conditions"):
                items = value if isinstance(value, list) else [value]
                result[RESOURCE_CONDITION].extend(items)
            elif norm_key.lower() in ("medicationstatement", "medicationstatements", "medications"):
                items = value if isinstance(value, list) else [value]
                result[RESOURCE_MEDICATION_STATEMENT].extend(items)
            elif norm_key.lower() in ("medicationrequest", "medicationrequests"):
                items = value if isinstance(value, list) else [value]
                result[RESOURCE_MEDICATION_REQUEST].extend(items)
            elif norm_key.lower() in ("observation", "observations"):
                items = value if isinstance(value, list) else [value]
                result[RESOURCE_OBSERVATION].extend(items)
            elif norm_key.lower() in ("allergyintolerance", "allergies"):
                items = value if isinstance(value, list) else [value]
                result[RESOURCE_ALLERGY].extend(items)

        return result

    def _classify_resource(
        self, resource: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Classify a single resource into the result dict."""
        rt = resource.get("resourceType", "")
        if rt == RESOURCE_PATIENT:
            result[RESOURCE_PATIENT] = resource
        elif rt == RESOURCE_CONDITION:
            result[RESOURCE_CONDITION].append(resource)
        elif rt == RESOURCE_MEDICATION_STATEMENT:
            result[RESOURCE_MEDICATION_STATEMENT].append(resource)
        elif rt == RESOURCE_MEDICATION_REQUEST:
            result[RESOURCE_MEDICATION_REQUEST].append(resource)
        elif rt == RESOURCE_OBSERVATION:
            result[RESOURCE_OBSERVATION].append(resource)
        elif rt == RESOURCE_ALLERGY:
            result[RESOURCE_ALLERGY].append(resource)

    # ──────────────────────────────────────────────
    # Individual resource parsers
    # ──────────────────────────────────────────────

    def _parse_patient(
        self, patient: Optional[Dict[str, Any]]
    ) -> tuple[Optional[int], Gender, Optional[str], Optional[str]]:
        """
        Extract demographics from FHIR Patient resource.

        Returns: (age, gender, social_history, family_history)
        """
        if not patient:
            return None, Gender.UNKNOWN, None, None

        # Gender
        fhir_gender = (patient.get("gender") or "").lower()
        gender_map = {
            "male": Gender.MALE,
            "female": Gender.FEMALE,
            "other": Gender.OTHER,
            "unknown": Gender.UNKNOWN,
        }
        gender = gender_map.get(fhir_gender, Gender.UNKNOWN)

        # Age from birthDate
        age = None
        birth_date_str = patient.get("birthDate")
        if birth_date_str:
            try:
                birth_date = datetime.strptime(birth_date_str[:10], "%Y-%m-%d").date()
                today = date.today()
                age = today.year - birth_date.year
                if (today.month, today.day) < (birth_date.month, birth_date.day):
                    age -= 1
            except (ValueError, TypeError):
                pass

        # Social / family history from extensions (if present)
        social_hx = None
        family_hx = None
        for ext in patient.get("extension", []):
            url = ext.get("url", "")
            if "social-history" in url.lower():
                social_hx = ext.get("valueString", "")
            elif "family-history" in url.lower():
                family_hx = ext.get("valueString", "")

        return age, gender, social_hx, family_hx

    def _parse_conditions(
        self, conditions: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract past medical history from FHIR Condition resources."""
        pmh = []
        for cond in conditions:
            # Get display text from code
            code_obj = cond.get("code", {})
            display = self._get_codeable_display(code_obj)
            if display:
                # Append clinical status if available
                status = cond.get("clinicalStatus", {})
                status_text = self._get_codeable_display(status)
                if status_text and status_text.lower() not in ("active", ""):
                    display = f"{display} ({status_text})"
                pmh.append(display)
        return pmh

    def _parse_medications(
        self, medications: List[Dict[str, Any]]
    ) -> List[Medication]:
        """Extract medications from FHIR MedicationStatement/Request resources."""
        meds = []
        for med_resource in medications:
            name = ""
            dose = None
            rxcui = None

            # Medication name from medicationCodeableConcept
            med_code = med_resource.get("medicationCodeableConcept", {})
            if med_code:
                name = self._get_codeable_display(med_code)
                rxcui = self._get_rxcui(med_code)

            # Or from medicationReference (simplified — just extract display)
            if not name:
                med_ref = med_resource.get("medicationReference", {})
                name = med_ref.get("display", "Unknown medication")

            # Dosage
            dosages = med_resource.get("dosage", med_resource.get("dosageInstruction", []))
            if dosages and isinstance(dosages, list) and len(dosages) > 0:
                d = dosages[0]
                dose_parts = []

                # Dose quantity
                dose_qty = d.get("doseAndRate", [{}])
                if dose_qty and isinstance(dose_qty, list):
                    dr = dose_qty[0]
                    dq = dr.get("doseQuantity", {})
                    if dq:
                        val = dq.get("value", "")
                        unit = dq.get("unit", dq.get("code", ""))
                        if val:
                            dose_parts.append(f"{val}{unit}")

                # Timing
                timing = d.get("timing", {})
                repeat = timing.get("repeat", {})
                freq = repeat.get("frequency")
                period = repeat.get("period")
                period_unit = repeat.get("periodUnit")
                if freq and period and period_unit:
                    unit_map = {"d": "day", "h": "hour", "wk": "week", "mo": "month"}
                    dose_parts.append(f"{freq}x/{unit_map.get(period_unit, period_unit)}")

                # Route
                route = d.get("route", {})
                route_text = self._get_codeable_display(route)
                if route_text:
                    dose_parts.append(route_text)

                # Text fallback
                if not dose_parts:
                    dose_text = d.get("text", "")
                    if dose_text:
                        dose_parts.append(dose_text)

                dose = " ".join(dose_parts) if dose_parts else None

            if name:
                meds.append(Medication(name=name, dose=dose, rxcui=rxcui))

        return meds

    def _parse_observations(
        self, observations: List[Dict[str, Any]]
    ) -> tuple[List[LabResult], VitalSigns]:
        """
        Separate FHIR Observations into lab results and vital signs.

        Uses LOINC codes to identify vital signs.
        """
        labs = []
        vitals = VitalSigns()
        systolic = None
        diastolic = None

        for obs in observations:
            code_obj = obs.get("code", {})
            loinc = self._get_loinc(code_obj)
            display = self._get_codeable_display(code_obj)
            value_str = self._get_observation_value(obs)
            ref_range = self._get_reference_range(obs)

            if not display and not loinc:
                continue

            # Check if this is a vital sign
            if loinc in VITAL_LOINC:
                vital_field = VITAL_LOINC[loinc]
                if vital_field == "systolic_bp":
                    systolic = value_str
                elif vital_field == "diastolic_bp":
                    diastolic = value_str
                elif vital_field == "blood_pressure":
                    # BP panel — check components
                    for comp in obs.get("component", []):
                        comp_code = comp.get("code", {})
                        comp_loinc = self._get_loinc(comp_code)
                        comp_val = self._get_observation_value(comp)
                        if comp_loinc == "8480-6":  # systolic
                            systolic = comp_val
                        elif comp_loinc == "8462-4":  # diastolic
                            diastolic = comp_val
                else:
                    setattr(vitals, vital_field, value_str)
            else:
                # It's a lab result
                is_abnormal = None
                interpretation = obs.get("interpretation", [])
                if interpretation:
                    interp_code = ""
                    if isinstance(interpretation, list) and interpretation:
                        interp_code = self._get_codeable_display(interpretation[0]).lower()
                    if any(w in interp_code for w in ("high", "low", "abnormal", "critical")):
                        is_abnormal = True
                    elif "normal" in interp_code:
                        is_abnormal = False

                labs.append(LabResult(
                    test_name=display or f"LOINC:{loinc}",
                    value=value_str or "No value",
                    reference_range=ref_range,
                    is_abnormal=is_abnormal,
                ))

        # Assemble blood pressure
        if systolic and diastolic:
            vitals.blood_pressure = f"{systolic}/{diastolic}"
        elif systolic:
            vitals.blood_pressure = f"{systolic}/?"
        elif diastolic:
            vitals.blood_pressure = f"?/{diastolic}"

        return labs, vitals

    def _parse_allergies(
        self, allergies: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract allergy names from FHIR AllergyIntolerance resources."""
        result = []
        for allergy in allergies:
            code_obj = allergy.get("code", {})
            display = self._get_codeable_display(code_obj)
            if display:
                # Add reaction info if available
                reactions = allergy.get("reaction", [])
                if reactions:
                    manifestations = []
                    for rxn in reactions:
                        for m in rxn.get("manifestation", []):
                            m_display = self._get_codeable_display(m)
                            if m_display:
                                manifestations.append(m_display)
                    if manifestations:
                        display = f"{display} ({', '.join(manifestations)})"
                result.append(display)
        return result

    # ──────────────────────────────────────────────
    # Utility helpers
    # ──────────────────────────────────────────────

    def _get_codeable_display(self, codeable: Dict[str, Any]) -> str:
        """Extract display text from a FHIR CodeableConcept."""
        if not codeable:
            return ""
        # Try text first
        text = codeable.get("text", "")
        if text:
            return text
        # Try coding display
        codings = codeable.get("coding", [])
        if codings and isinstance(codings, list):
            for coding in codings:
                display = coding.get("display", "")
                if display:
                    return display
            # Fall back to code
            return codings[0].get("code", "")
        return ""

    def _get_loinc(self, codeable: Dict[str, Any]) -> Optional[str]:
        """Extract LOINC code from a CodeableConcept."""
        for coding in codeable.get("coding", []):
            system = coding.get("system", "")
            if "loinc" in system.lower():
                return coding.get("code")
        return None

    def _get_rxcui(self, codeable: Dict[str, Any]) -> Optional[str]:
        """Extract RxCUI from a CodeableConcept."""
        for coding in codeable.get("coding", []):
            system = coding.get("system", "")
            if "rxnorm" in system.lower():
                return coding.get("code")
        return None

    def _get_observation_value(self, obs: Dict[str, Any]) -> Optional[str]:
        """Extract the value from a FHIR Observation."""
        # valueQuantity
        vq = obs.get("valueQuantity")
        if vq:
            val = vq.get("value", "")
            unit = vq.get("unit", vq.get("code", ""))
            return f"{val} {unit}".strip()

        # valueString
        vs = obs.get("valueString")
        if vs:
            return vs

        # valueCodeableConcept
        vcc = obs.get("valueCodeableConcept")
        if vcc:
            return self._get_codeable_display(vcc)

        # valueBoolean / valueInteger
        for key in ("valueBoolean", "valueInteger", "valueDateTime"):
            if key in obs:
                return str(obs[key])

        return None

    def _get_reference_range(self, obs: Dict[str, Any]) -> Optional[str]:
        """Extract reference range from a FHIR Observation."""
        ref_ranges = obs.get("referenceRange", [])
        if not ref_ranges:
            return None

        rr = ref_ranges[0]
        low = rr.get("low", {})
        high = rr.get("high", {})
        text = rr.get("text", "")

        if text:
            return text

        low_val = low.get("value", "")
        high_val = high.get("value", "")
        unit = low.get("unit", high.get("unit", ""))

        if low_val and high_val:
            return f"{low_val}-{high_val} {unit}".strip()
        elif low_val:
            return f">= {low_val} {unit}".strip()
        elif high_val:
            return f"<= {high_val} {unit}".strip()

        return None


# ──────────────────────────────────────────────
# Convenience: convert FHIR Bundle to free text
# ──────────────────────────────────────────────

def fhir_to_text(fhir_data: Dict[str, Any]) -> str:
    """
    Convert FHIR data to a free-text clinical summary.

    This is an alternative to structured parsing — useful when
    passing directly to the existing free-text pipeline.
    """
    adapter = FHIRAdapter()
    profile = adapter.to_patient_profile(fhir_data)

    lines = []

    if profile.age and profile.gender:
        lines.append(f"{profile.age}-year-old {profile.gender.value}.")
    elif profile.age:
        lines.append(f"{profile.age}-year-old patient.")

    if profile.chief_complaint:
        lines.append(f"Chief complaint: {profile.chief_complaint}")

    if profile.history_of_present_illness:
        lines.append(f"HPI: {profile.history_of_present_illness}")

    if profile.past_medical_history:
        lines.append(f"PMH: {', '.join(profile.past_medical_history)}")

    if profile.current_medications:
        med_strs = []
        for m in profile.current_medications:
            s = m.name
            if m.dose:
                s += f" {m.dose}"
            med_strs.append(s)
        lines.append(f"Medications: {', '.join(med_strs)}")

    if profile.allergies:
        lines.append(f"Allergies: {', '.join(profile.allergies)}")

    if profile.vital_signs:
        vs = profile.vital_signs
        vital_parts = []
        if vs.blood_pressure:
            vital_parts.append(f"BP {vs.blood_pressure}")
        if vs.heart_rate:
            vital_parts.append(f"HR {vs.heart_rate}")
        if vs.temperature:
            vital_parts.append(f"Temp {vs.temperature}")
        if vs.respiratory_rate:
            vital_parts.append(f"RR {vs.respiratory_rate}")
        if vs.oxygen_saturation:
            vital_parts.append(f"SpO2 {vs.oxygen_saturation}")
        if vital_parts:
            lines.append(f"Vitals: {', '.join(vital_parts)}")

    if profile.lab_results:
        lab_strs = []
        for lab in profile.lab_results:
            s = f"{lab.test_name}: {lab.value}"
            if lab.reference_range:
                s += f" (ref: {lab.reference_range})"
            if lab.is_abnormal:
                s += " [ABNORMAL]"
            lab_strs.append(s)
        lines.append(f"Labs: {'; '.join(lab_strs)}")

    if profile.social_history:
        lines.append(f"Social history: {profile.social_history}")

    if profile.family_history:
        lines.append(f"Family history: {profile.family_history}")

    return "\n".join(lines)
