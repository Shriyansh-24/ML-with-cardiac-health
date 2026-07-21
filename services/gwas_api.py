"""
services/gwas_api.py — Step 6: GWAS Catalog gene-disease association service.

WHAT THIS MODULE DOES
    Fetches real gene-disease association data from the EMBL-EBI GWAS Catalog
    REST API v2 for cardiac-related genes. Returns a structured summary per
    condition: total associations, top significant hits by p-value, and a
    breakdown of associated traits.

    Results are cached in memory for 1 hour to avoid hammering the EBI API
    on every form submission. If the API is unreachable, the module falls
    back gracefully with an empty summary.

HOW IT WORKS
    1. For each gene in a condition group (e.g. KCNQ1, KCNH2, SCN5A for LQTS),
       query the GWAS Catalog associations endpoint filtered by mapped_gene
    2. Collect and merge results across genes
    3. Sort by p-value ascending (most significant first)
    4. Cache individual gene results for 1 hour

API REFERENCE
    https://www.ebi.ac.uk/gwas/rest/api/v2/docs
    https://www.ebi.ac.uk/gwas/rest/api/v2/associations?mapped_gene=<GENE>
"""

import time
import logging
from typing import Optional
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api/v2"
ASSOCIATIONS_URL = f"{GWAS_API_BASE}/associations"

# How many associations to fetch per gene (API default is 20 per page)
PAGE_SIZE = 20

# How many notable associations to return per condition (most significant)
NOTABLE_LIMIT = 10

# In-memory cache TTL
CACHE_TTL_SECONDS = 3600  # 1 hour

# Minimum delay between API requests (15/sec limit, be polite)
REQUEST_DELAY_SECONDS = 0.3

# Genes grouped by condition
CONDITION_GENES: dict[str, list[str]] = {
    "LQTS": ["KCNQ1", "KCNH2", "SCN5A"],
    "FH": ["LDLR", "APOB"],
}

# ── Cache ───────────────────────────────────────────────────────────────────

_cache: dict[str, tuple[datetime, dict]] = {}


def _get_cached(key: str) -> Optional[dict]:
    """Return cached data if still fresh, else None."""
    entry = _cache.get(key)
    if entry is None:
        return None
    cached_at, data = entry
    if datetime.now() - cached_at > timedelta(seconds=CACHE_TTL_SECONDS):
        del _cache[key]
        return None
    return data


def _set_cache(key: str, data: dict) -> None:
    """Store data in cache with current timestamp."""
    _cache[key] = (datetime.now(), data)


# ── GWAS Catalog API Calls ──────────────────────────────────────────────────


def _fetch_gene_associations(gene_symbol: str) -> list[dict]:
    """
    Fetch GWAS associations for a single gene.

    Args:
        gene_symbol: HGNC gene symbol, e.g. "KCNQ1".

    Returns:
        List of association dicts (raw API response items).

    Raises:
        requests.RequestException: If the API call fails.
    """
    params = {
        "mapped_gene": gene_symbol,
        "size": str(PAGE_SIZE),
    }
    resp = requests.get(ASSOCIATIONS_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # The v2 API returns _embedded -> associations
    embedded = data.get("_embedded", {})
    associations = embedded.get("associations", [])

    logger.info(
        "GWAS Catalog for %s: fetched %s associations",
        gene_symbol,
        len(associations),
    )
    return associations


# ── Parsing (v2 API structure) ─────────────────────────────────────────────

# The GWAS Catalog v2 REST API returns associations with the following
# structure relevant fields:
#   - snp_allele: list of {rs_id, effect_allele} — primary variant identifier
#   - locations: list of strings e.g. "19:11091630"
#   - efo_traits: list of {efo_id, efo_trait}
#   - p_value, beta, or_value, reported_trait, reported_genes
#   - snp_interaction: often None for many associations


def _extract_variant_id(association: dict) -> str:
    """Extract rsID from an association's snp_allele list."""
    snp_alleles = association.get("snp_allele")
    if isinstance(snp_alleles, list):
        for entry in snp_alleles:
            rsid = entry.get("rs_id", "")
            if rsid:
                return rsid
    return "—"


def _extract_pvalue(association: dict) -> Optional[float]:
    """Extract the p-value as a float, or None."""
    raw = association.get("p_value")
    if raw is None or raw == "—":
        return None
    try:
        # Handle both numeric and string formats (e.g. 2e-12, 3.5e-8)
        return float(raw)
    except (ValueError, TypeError):
        return None


def _extract_risk_allele(association: dict) -> str:
    """Extract the effect/risk allele from snp_allele."""
    snp_alleles = association.get("snp_allele")
    if isinstance(snp_alleles, list):
        for entry in snp_alleles:
            allele = entry.get("effect_allele", "")
            if allele:
                return allele
    return "—"


def _extract_trait(association: dict) -> str:
    """Extract the EFO trait name."""
    efo_traits = association.get("efo_traits", [])
    if isinstance(efo_traits, list) and efo_traits:
        return efo_traits[0].get("efo_trait", "Unknown")
    return association.get("reported_trait", "Unknown")


def _extract_effect_size(association: dict) -> Optional[float]:
    """Extract the effect size (beta or odds ratio), or None."""
    # Try beta first (most common for continuous traits)
    beta = association.get("beta")
    if beta is not None:
        try:
            return float(beta)
        except (ValueError, TypeError):
            pass
    # Fall back to or_value
    or_val = association.get("or_value")
    if or_val is not None:
        try:
            return float(or_val)
        except (ValueError, TypeError):
            pass
    return None


def _extract_reported_genes(association: dict) -> list[str]:
    """Extract the list of reported genes."""
    return association.get("reported_genes", [])


def _format_pvalue(pval: Optional[float]) -> str:
    """Format a p-value for display (scientific notation for very small values)."""
    if pval is None:
        return "—"
    if pval < 0.0001:
        return f"{pval:.2e}"
    return f"{pval:.4f}"


def _format_effect(val: Optional[float]) -> str:
    """Format an effect size for display (beta or odds ratio)."""
    if val is None:
        return ""
    if abs(val) < 10:
        return f"{val:.2f}"
    return f"{val:.1f}"


def _parse_association(association: dict) -> Optional[dict]:
    """
    Parse a raw GWAS Catalog association into a simplified dict.

    Returns None if the association lacks a useful p-value.
    """
    pval = _extract_pvalue(association)

    # Skip entries without a meaningful p-value
    if pval is None:
        return None

    return {
        "rs_id": _extract_variant_id(association),
        "trait": _extract_trait(association),
        "p_value": pval,
        "p_value_formatted": _format_pvalue(pval),
        "risk_allele": _extract_risk_allele(association),
        "effect_size": _extract_effect_size(association),
        "effect_size_formatted": _format_effect(_extract_effect_size(association)),
        "reported_genes": _extract_reported_genes(association),
    }


# ── Grouped Fetching ────────────────────────────────────────────────────────


def fetch_condition_associations(condition_code: str) -> dict:
    """
    Fetch and summarise GWAS Catalog associations for a condition.

    Fetches data for all genes associated with the condition and merges
    the results into a single summary.

    Args:
        condition_code: "LQTS" or "FH".

    Returns:
        dict with keys:
            - condition: the condition code
            - genes: list of genes queried
            - total_associations: total count across all genes
            - notable_associations: top hits sorted by p-value (ascending)
            - summary_by_trait: dict mapping trait_name -> count
            - fetched_at: ISO timestamp
            - cached: bool
            - error: None or str

    Example:
        {
            "condition": "LQTS",
            "genes": ["KCNQ1", "KCNH2", "SCN5A"],
            "total_associations": 87,
            "notable_associations": [
                {
                    "rs_id": "rs123456",
                    "trait": "Long QT Syndrome",
                    "p_value": 3.2e-15,
                    "p_value_formatted": "3.20e-15",
                    "risk_allele": "A",
                    "odds_ratio": 2.35,
                    "beta": None,
                    "reported_genes": ["KCNQ1"]
                },
                ...
            ],
            "summary_by_trait": {
                "Long QT Syndrome": 12,
                "Sudden cardiac death": 8,
                ...
            },
            "fetched_at": "2026-07-18T14:30:00",
            "cached": False,
            "error": None
        }
    """
    genes = CONDITION_GENES.get(condition_code)
    if not genes:
        return {
            "condition": condition_code,
            "genes": [],
            "total_associations": 0,
            "notable_associations": [],
            "summary_by_trait": {},
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "cached": False,
            "error": f"Unknown condition code: {condition_code}",
        }

    # Build a composite cache key from sorted gene names
    cache_key = f"gwas_{condition_code}_{'_'.join(sorted(genes))}"

    # Check cache first
    cached = _get_cached(cache_key)
    if cached is not None:
        cached["cached"] = True
        return cached

    result: dict = {
        "condition": condition_code,
        "genes": genes[:],
        "total_associations": 0,
        "notable_associations": [],
        "summary_by_trait": {},
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "cached": False,
        "error": None,
    }

    all_parsed: list[dict] = []
    trait_counts: dict[str, int] = {}
    total_raw = 0

    try:
        for gene in genes:
            try:
                raw_associations = _fetch_gene_associations(gene)
            except requests.RequestException as e:
                logger.warning("GWAS Catalog request failed for %s: %s", gene, e)
                continue

            total_raw += len(raw_associations)

            for assoc in raw_associations:
                parsed = _parse_association(assoc)
                if parsed is None:
                    continue

                # Tag with gene if not already obvious from reported_genes
                if not parsed["reported_genes"]:
                    parsed["reported_genes"] = [gene]

                all_parsed.append(parsed)

                # Tally traits
                trait = parsed["trait"]
                trait_counts[trait] = trait_counts.get(trait, 0) + 1

            # Polite delay between gene requests
            time.sleep(REQUEST_DELAY_SECONDS)

        # Sort by p-value ascending (most significant first)
        all_parsed.sort(key=lambda a: a["p_value"] if a["p_value"] is not None else float("inf"))

        result["total_associations"] = total_raw
        result["notable_associations"] = all_parsed[:NOTABLE_LIMIT]
        result["summary_by_trait"] = dict(
            sorted(trait_counts.items(), key=lambda x: x[1], reverse=True)
        )

    except (KeyError, ValueError, TypeError) as e:
        logger.warning("GWAS Catalog response parsing failed for %s: %s", condition_code, e)
        result["error"] = f"Failed to parse GWAS Catalog response: {e}"

    _set_cache(cache_key, result)
    return result


# ── Public API ──────────────────────────────────────────────────────────────


def fetch_all_condition_associations() -> dict[str, dict]:
    """
    Fetch GWAS data for all supported conditions.

    Returns a dict keyed by condition code ("LQTS", "FH"), each containing
    the result dict from fetch_condition_associations.

    This is the main entry point for app.py to call once per request.
    """
    results: dict[str, dict] = {}
    for condition_code in CONDITION_GENES:
        results[condition_code] = fetch_condition_associations(condition_code)
    return results
