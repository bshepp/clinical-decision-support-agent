# [Track D: Arbitrated Parallel]
"""
Track D — Configuration for specialist-arbiter experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SpecialistDef:
    """Definition of a domain-specialist reasoning agent."""
    specialist_id: str
    name: str
    system_prompt_addendum: str
    """
    Appended to the base clinical reasoning system prompt to give this
    specialist its domain-specific lens. For example, a cardiologist
    specialist will emphasise cardiac aetiologies.
    """


@dataclass
class ArbitratedConfig:
    """Configuration for one arbitrated-parallel experiment."""
    config_id: str
    specialists: List[SpecialistDef]
    max_rounds: int = 2
    """
    Number of arbiter-specialist rounds:
      - Round 1: specialists reason independently; arbiter merges
      - Round 2+: arbiter sends tailored re-prompts to specialists
                  that disagree, then re-merges
    """
    arbiter_temperature: float = 0.2
    specialist_temperature: float = 0.3
    max_tokens_specialist: int = 3072
    max_tokens_arbiter: int = 3000
    description: str = ""


# ──────────────────────────────────────────────
# Specialist library
# ──────────────────────────────────────────────

SPECIALIST_CARDIOLOGY = SpecialistDef(
    specialist_id="cardio",
    name="Cardiologist",
    system_prompt_addendum=(
        "You are a board-certified cardiologist. Focus on cardiac aetiologies: "
        "ACS, arrhythmias, valvular disease, heart failure, pericardial disease, "
        "aortic dissection, PE, and cardiac-adjacent conditions. Weight your "
        "differential toward cardiovascular causes when the presentation is consistent."
    ),
)

SPECIALIST_NEUROLOGY = SpecialistDef(
    specialist_id="neuro",
    name="Neurologist",
    system_prompt_addendum=(
        "You are a board-certified neurologist. Focus on neurological aetiologies: "
        "stroke, TIA, seizure, CNS infections, demyelinating disease, neuropathies, "
        "movement disorders, and neuro-oncological conditions. Weight your "
        "differential toward neurological causes when the presentation is consistent."
    ),
)

SPECIALIST_ID = SpecialistDef(
    specialist_id="id",
    name="Infectious Disease Specialist",
    system_prompt_addendum=(
        "You are a board-certified infectious disease specialist. Focus on "
        "infectious aetiologies: bacterial, viral, fungal, parasitic infections, "
        "tropical diseases, HIV-related conditions, sepsis, endocarditis, "
        "osteomyelitis, and opportunistic infections. Consider travel history, "
        "immune status, and epidemiological risk factors."
    ),
)

SPECIALIST_GENERAL_IM = SpecialistDef(
    specialist_id="im",
    name="General Internist",
    system_prompt_addendum=(
        "You are a board-certified internal medicine physician. Take a broad, "
        "systems-based approach. Consider metabolic, endocrine, autoimmune, "
        "oncological, and multisystem aetiologies. Look for Occam's razor — "
        "a single unifying diagnosis. Also consider common conditions that "
        "subspecialists might overlook."
    ),
)

SPECIALIST_EMERGENCY = SpecialistDef(
    specialist_id="em",
    name="Emergency Medicine Physician",
    system_prompt_addendum=(
        "You are a board-certified emergency physician. Prioritise life-threatening "
        "'can't miss' diagnoses first: aortic dissection, PE, MI, tension pneumothorax, "
        "meningitis, ruptured ectopic, necrotising fasciitis. Use a worst-first "
        "approach to the differential. Time-sensitive conditions go to the top."
    ),
)


# ──────────────────────────────────────────────
# Experiment configurations
# ──────────────────────────────────────────────

CONFIGS = [
    ArbitratedConfig(
        config_id="D0_3spec_1round",
        specialists=[SPECIALIST_GENERAL_IM, SPECIALIST_CARDIOLOGY, SPECIALIST_ID],
        max_rounds=1,
        description="3 specialists (IM, Cardio, ID), single arbiter round",
    ),
    ArbitratedConfig(
        config_id="D1_5spec_1round",
        specialists=[
            SPECIALIST_GENERAL_IM, SPECIALIST_CARDIOLOGY, SPECIALIST_NEUROLOGY,
            SPECIALIST_ID, SPECIALIST_EMERGENCY,
        ],
        max_rounds=1,
        description="5 specialists, single arbiter round",
    ),
    ArbitratedConfig(
        config_id="D2_3spec_2rounds",
        specialists=[SPECIALIST_GENERAL_IM, SPECIALIST_CARDIOLOGY, SPECIALIST_ID],
        max_rounds=2,
        description="3 specialists, 2 arbiter rounds with tailored resubmission",
    ),
    ArbitratedConfig(
        config_id="D3_5spec_2rounds",
        specialists=[
            SPECIALIST_GENERAL_IM, SPECIALIST_CARDIOLOGY, SPECIALIST_NEUROLOGY,
            SPECIALIST_ID, SPECIALIST_EMERGENCY,
        ],
        max_rounds=2,
        description="5 specialists, 2 arbiter rounds — full ensemble",
    ),
]
