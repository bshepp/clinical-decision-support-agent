"""
Tool: Drug Interaction Checker

Checks for drug-drug interactions using the OpenFDA API and RxNorm
for medication normalization.

This is a NON-LLM tool — it queries external databases, demonstrating
that MedGemma works alongside traditional tools in the agent pipeline.
"""
from __future__ import annotations

import logging
from typing import List

import httpx

from app.config import settings
from app.models.schemas import (
    DrugInteraction,
    DrugInteractionResult,
    Medication,
    Severity,
)

logger = logging.getLogger(__name__)

OPENFDA_INTERACTION_URL = "https://api.fda.gov/drug/event.json"
RXNORM_INTERACTION_URL = "https://rxnav.nlm.nih.gov/REST/interaction/list.json"


class DrugInteractionTool:
    """Checks drug interactions via OpenFDA and RxNorm APIs."""

    def __init__(self):
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def run(
        self,
        current_medications: List[Medication],
        proposed_medications: List[str] | None = None,
    ) -> DrugInteractionResult:
        """
        Check for drug interactions among current and proposed medications.

        Args:
            current_medications: Patient's current medication list
            proposed_medications: Any new medications being considered

        Returns:
            DrugInteractionResult with found interactions and warnings
        """
        all_med_names = [m.name for m in current_medications]
        if proposed_medications:
            all_med_names.extend(proposed_medications)

        if len(all_med_names) < 2:
            return DrugInteractionResult(
                medications_checked=all_med_names,
                warnings=["Fewer than 2 medications — no interaction check needed"],
            )

        interactions = []
        warnings = []

        # Try RxNorm interaction API (NIH — free, no key needed)
        try:
            rxnorm_interactions = await self._check_rxnorm(all_med_names)
            interactions.extend(rxnorm_interactions)
        except Exception as e:
            logger.warning(f"RxNorm API failed: {e}")
            warnings.append(f"RxNorm API unavailable: {e}")

        # Try OpenFDA as supplementary source
        try:
            fda_interactions = await self._check_openfda(all_med_names)
            interactions.extend(fda_interactions)
        except Exception as e:
            logger.warning(f"OpenFDA API failed: {e}")
            warnings.append(f"OpenFDA API unavailable: {e}")

        # Deduplicate
        interactions = self._deduplicate(interactions)

        return DrugInteractionResult(
            interactions_found=interactions,
            medications_checked=all_med_names,
            warnings=warnings,
        )

    async def _check_rxnorm(self, med_names: List[str]) -> List[DrugInteraction]:
        """Query RxNorm Interaction API."""
        client = await self._get_client()
        interactions = []

        # First, resolve drug names to RxCUIs
        rxcuis = []
        for name in med_names:
            try:
                resp = await client.get(
                    f"{settings.rxnorm_base_url}/rxcui.json",
                    params={"name": name, "search": 1},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    id_group = data.get("idGroup", {})
                    rxnorm_id = id_group.get("rxnormId")
                    if rxnorm_id:
                        # rxnormId can be a list of strings
                        if isinstance(rxnorm_id, list):
                            rxcuis.append(rxnorm_id[0])
                        else:
                            rxcuis.append(str(rxnorm_id))
            except Exception:
                continue

        if len(rxcuis) < 2:
            return interactions

        # Query interaction API with RxCUIs
        try:
            resp = await client.get(
                RXNORM_INTERACTION_URL,
                params={"rxcuis": "+".join(rxcuis)},
            )
            if resp.status_code == 200:
                data = resp.json()
                interaction_groups = data.get("fullInteractionTypeGroup", [])
                for group in interaction_groups:
                    for itype in group.get("fullInteractionType", []):
                        for pair in itype.get("interactionPair", []):
                            desc = pair.get("description", "")
                            severity_str = pair.get("severity", "N/A")
                            names = [
                                concept.get("minConceptItem", {}).get("name", "Unknown")
                                for concept in pair.get("interactionConcept", [])
                            ]
                            interactions.append(
                                DrugInteraction(
                                    drug_a=names[0] if len(names) > 0 else "Unknown",
                                    drug_b=names[1] if len(names) > 1 else "Unknown",
                                    severity=self._map_severity(severity_str),
                                    description=desc,
                                    source="RxNorm/NLM",
                                )
                            )
        except Exception as e:
            logger.warning(f"RxNorm interaction query failed: {e}")

        return interactions

    async def _check_openfda(self, med_names: List[str]) -> List[DrugInteraction]:
        """Query OpenFDA for adverse event reports involving these drugs together."""
        client = await self._get_client()
        interactions = []

        # Check pairs of drugs for co-reported adverse events
        for i, drug_a in enumerate(med_names):
            for drug_b in med_names[i + 1 :]:
                try:
                    search = f'patient.drug.medicinalproduct:"{drug_a}"+AND+patient.drug.medicinalproduct:"{drug_b}"'
                    params = {"search": search, "limit": 1}
                    if settings.openfda_api_key:
                        params["api_key"] = settings.openfda_api_key

                    resp = await client.get(OPENFDA_INTERACTION_URL, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        total = data.get("meta", {}).get("results", {}).get("total", 0)
                        if total > 100:
                            interactions.append(
                                DrugInteraction(
                                    drug_a=drug_a,
                                    drug_b=drug_b,
                                    severity=Severity.MODERATE,
                                    description=f"{total} adverse event reports found involving both {drug_a} and {drug_b}.",
                                    clinical_significance="Review recommended based on adverse event frequency",
                                    source="OpenFDA",
                                )
                            )
                except Exception:
                    continue

        return interactions

    @staticmethod
    def _map_severity(severity_str: str) -> Severity:
        """Map RxNorm severity strings to our Severity enum."""
        s = severity_str.lower()
        if "high" in s or "severe" in s or "serious" in s:
            return Severity.HIGH
        elif "moderate" in s:
            return Severity.MODERATE
        elif "low" in s or "minor" in s:
            return Severity.LOW
        return Severity.MODERATE  # Default

    @staticmethod
    def _deduplicate(interactions: List[DrugInteraction]) -> List[DrugInteraction]:
        """Remove duplicate interactions (same drug pair from different sources)."""
        seen = set()
        unique = []
        for interaction in interactions:
            key = tuple(sorted([interaction.drug_a.lower(), interaction.drug_b.lower()]))
            if key not in seen:
                seen.add(key)
                unique.append(interaction)
        return unique
