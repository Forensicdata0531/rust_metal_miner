# python_entropy_analyzer/bayes.py
"""
Very lightweight Bayesian scoring of nibble frequencies.

We treat each nibble (0–F) as a category and keep simple
counts. From counts we derive posterior probabilities
with a symmetric Dirichlet prior (Laplace smoothing).
"""

from typing import Dict, List

NIBBLES = [format(i, "x") for i in range(16)]


def count_nibbles(hash_bytes: bytes) -> Dict[str, int]:
    """
    Count nibble (hex digit) frequencies in a single 32-byte hash.
    Returns a dict: {nibble_hex: count}
    """
    counts: Dict[str, int] = {n: 0 for n in NIBBLES}
    for nib in hash_bytes.hex():
        counts[nib] += 1
    return counts


def aggregate_counts(hashes: List[bytes]) -> Dict[str, int]:
    """
    Aggregate nibble counts over a batch of hashes.
    """
    counts: Dict[str, int] = {n: 0 for n in NIBBLES}
    for h in hashes:
        for nib in h.hex():
            counts[nib] += 1
    return counts


def posterior_from_counts(counts: Dict[str, int], alpha: float = 1.0) -> Dict[str, float]:
    """
    Convert counts into posterior probabilities with Dirichlet(alpha) prior.
    alpha=1.0 → Laplace smoothing.
    """
    posterior: Dict[str, float] = {}
    total = sum(counts.values()) + alpha * len(counts)
    for n, c in counts.items():
        posterior[n] = (c + alpha) / total
    return posterior


def bayes_weight(posterior: Dict[str, float]) -> float:
    """
    Compute a very simple "weight" = negative log likelihood
    (roughly an information measure).
    """
    from math import log2
    return -sum(p * log2(p) for p in posterior.values())


# Quick test / example
if __name__ == "__main__":
    import os
    # Make a random hash and compute posterior + weight
    random_hash = os.urandom(32)
    c = count_nibbles(random_hash)
    p = posterior_from_counts(c)
    w = bayes_weight(p)
    print("Random hash weight:", w)
