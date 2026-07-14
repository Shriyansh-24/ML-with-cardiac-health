"""
services/clinvar_api.py — Module 2: ClinVar gene variant data service.

WHAT THIS MODULE DOES
    Fetches real variant data from the NCBI ClinVar database for a given
    gene symbol (e.g. MYH7) using the E-utilities API. Returns a structured
    summary: total variant count, breakdown by clinical significance, and a
    list of notable pathogenic/likely-pathogenic variants.

    Results are cached in memory for 1 hour to avoid hammering the NCBI API
    on every form submission. If the API is unreachable or returns an error,
    the module falls back gracefully with an empty summary rather than crashing.

HOW IT WORKS
    1. esearch — find all VariantIDs for a given gene (e.g. MYH7)
    2. esummary — fetch details for the most recent variants (up to 100)
    3. Classify by clinical significance, extract notable variants
    4. Cache the result so the next request within 1 hour is instant

API REFERENCE
    NCBI ClinVar E-utilities:
        https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/
        https://www.ncbi.nlm.nih.gov/home/develop/api/

    Rate limit: 3 requests/second without an API key. We add a small delay
    between sequential requests to stay under this limit.
"""

import time
import logging
from typing import Optional
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# How many recent variants to fetch details for (highest UIDs = most recent)
DETAILS_LIMIT = 50

# How many pathogenic/likely-pathogenic variants to return as "notable"
NOTABLE_LIMIT = 15

# In-memory cache TTL
CACHE_TTL_SECONDS = 3600  # 1 hour

# Minimum delay between NCBI API requests (3/sec max without key)
REQUEST_DELAY_SECONDS = 0.4

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


# ── NCBI API Calls ─────────────────────────────────────────────────────────

def _search_variants(gene_symbol: str, retmax: int = 100) -> tuple[list[str], int]:
    """
    Search ClinVar for all variant IDs associated with a gene.

    Returns both the list of UID strings and the total variant count
    from a single API call — avoids a redundant second request.

    Args:
        gene_symbol: HGNC gene symbol, e.g. "MYH7".
        retmax: Maximum IDs to return (NCBI default is 20).

    Returns:
        (id_list, total_count): list of UID strings and total count.

    Raises:
        requests.RequestException: If the API call fails entirely.
    """
    params = {
        "db": "clinvar",
        "term": f"{gene_symbol}[gene]",
        "retmax": str(retmax),
        "retmode": "json",
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    esearch_result = data.get("esearchresult", {})
    id_list = esearch_result.get("idlist", [])
    total_count = int(esearch_result.get("count", 0))

    logger.info(
        "ClinVar esearch for %s: found %s variants total, fetched %s IDs",
        gene_symbol,
        total_count,
        len(id_list),
    )
    return id_list, total_count


def _fetch_summaries(variant_ids: list[str]) -> list[dict]:
    """
    Fetch esummary data for a list of variant IDs.

    Args:
        variant_ids: List of ClinVar variant UID strings.

    Returns:
        List of parsed variant dicts.

    Raises:
        requests.RequestException: If the API call fails entirely.
    """
    if not variant_ids:
        return []

    params = {
        "db": "clinvar",
        "id": ",".join(variant_ids),
        "retmode": "json",
    }
    resp = requests.get(ESUMMARY_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # The result key contains uid -> record mapping
    records = data.get("result", {})
    # Remove the "uids" list itself, keep only the records
    uids = records.pop("uids", [])
    return [records[uid] for uid in uids if uid in records]


# ── Parsing ─────────────────────────────────────────────────────────────────


def _get_clinical_significance(variant: dict) -> str:
    """Extract the clinical significance description from a variant record."""
    germline = variant.get("germline_classification", {})
    if not germline:
        return "Not classified"
    return germline.get("description", "Not specified")


def _get_condition(variant: dict) -> str:
    """Extract the primary associated condition name."""
    germline = variant.get("germline_classification", {})
    trait_set = germline.get("trait_set", [])
    if trait_set:
        return trait_set[0].get("trait_name", "Unknown")
    return "Unknown"


def _get_variant_type(variant: dict) -> str:
    """Extract the variant type from the first variation in the set."""
    var_set = variant.get("variation_set", [])
    if var_set:
        return var_set[0].get("variant_type", "unknown")
    return "unknown"


def _get_protein_change(variant: dict) -> str:
    """Extract the protein change, if available."""
    return variant.get("protein_change") or "—"


def _classify_significance(sig: str) -> str:
    """Normalise clinical significance into broad buckets."""
    sig_lower = sig.lower()
    if "pathogenic" in sig_lower and "likely" not in sig_lower:
        return "Pathogenic"
    elif "likely pathogenic" in sig_lower:
        return "Likely pathogenic"
    elif "uncertain" in sig_lower or "conflicting" in sig_lower:
        return "Uncertain / Conflicting"
    elif "benign" in sig_lower and "likely" not in sig_lower:
        return "Benign"
    elif "likely benign" in sig_lower:
        return "Likely benign"
    elif "risk factor" in sig_lower:
        return "Risk factor"
    elif "drug response" in sig_lower:
        return "Drug response"
    elif "protective" in sig_lower:
        return "Protective"
    elif "association" in sig_lower:
        return "Association"
    else:
        return "Other"


# ── Main Public Function ────────────────────────────────────────────────────


def fetch_gene_variants(gene_symbol: str = "MYH7") -> dict:
    """
    Fetch and summarise ClinVar variant data for a given gene.

    Results are cached in memory for 1 hour. The cache key is the gene
    symbol, so requesting the same gene twice within the TTL is instant.

    Args:
        gene_symbol: HGNC gene symbol (default "MYH7").

    Returns:
        dict with keys:
            - gene: the gene symbol requested
            - total_variants: total count from esearch
            - summary_by_significance: dict mapping bucket -> count
            - notable_variants: list of pathogenic/likely-pathogenic variants
              (up to NOTABLE_LIMIT)
            - fetched_at: ISO timestamp of when data was fetched
            - cached: bool indicating if response came from cache
            - error: None if successful, str if an error occurred

    Example return:
        {
            "gene": "MYH7",
            "total_variants": 5884,
            "summary_by_significance": {
                "Pathogenic": 342,
                "Likely pathogenic": 156,
                "Uncertain / Conflicting": 3225,
                "Likely benign": 452,
                "Benign": 103,
                "Other": 1606,
            },
            "notable_variants": [
                {
                    "uid": "4856610",
                    "title": "NM_000257.4(MYH7):c.2021T>C (p.Ile674Thr)",
                    "clinical_significance": "Likely pathogenic",
                    "variant_type": "single nucleotide variant",
                    "condition": "Hypertrophic cardiomyopathy 1",
                    "protein_change": "I674T",
                },
                ...
            ],
            "fetched_at": "2026-07-14T12:00:00",
            "cached": False,
            "error": None,
        }
    """
    # Check cache first
    cached = _get_cached(gene_symbol)
    if cached is not None:
        cached["cached"] = True
        return cached

    result: dict = {
        "gene": gene_symbol,
        "total_variants": 0,
        "summary_by_significance": {},
        "notable_variants": [],
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "cached": False,
        "error": None,
    }

    try:
        # Step 1: search for variant IDs (returns total count from same call)
        variant_ids, total_count = _search_variants(gene_symbol, retmax=DETAILS_LIMIT + 50)
        result["total_variants"] = total_count

        time.sleep(REQUEST_DELAY_SECONDS)

        # Step 2: fetch summaries (only the most recent IDs, which are most relevant)
        recent_ids = variant_ids[:DETAILS_LIMIT]
        records = _fetch_summaries(recent_ids)

        if not records:
            result["error"] = f"ClinVar returned no variant details for {gene_symbol}"
            _set_cache(gene_symbol, result)
            return result

        # Step 3: classify and summarise
        by_sig: dict[str, int] = {}
        notable: list[dict] = []

        for rec in records:
            sig = _get_clinical_significance(rec)
            bucket = _classify_significance(sig)
            by_sig[bucket] = by_sig.get(bucket, 0) + 1

            # Collect pathogenic / likely pathogenic variants as "notable"
            if bucket in ("Pathogenic", "Likely pathogenic") and len(notable) < NOTABLE_LIMIT:
                notable.append({
                    "uid": rec.get("uid", ""),
                    "title": rec.get("title", ""),
                    "clinical_significance": sig,
                    "variant_type": _get_variant_type(rec),
                    "condition": _get_condition(rec),
                    "protein_change": _get_protein_change(rec),
                })

        result["summary_by_significance"] = dict(
            sorted(by_sig.items(), key=lambda x: x[1], reverse=True)
        )
        result["notable_variants"] = notable
        result["fetched_at"] = datetime.now().isoformat(timespec="seconds")

    except requests.RequestException as e:
        logger.warning("ClinVar API request failed for %s: %s", gene_symbol, e)
        result["error"] = f"Could not reach ClinVar API: {e}"
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("ClinVar response parsing failed for %s: %s", gene_symbol, e)
        result["error"] = f"Failed to parse ClinVar response: {e}"

    _set_cache(gene_symbol, result)
    return result
