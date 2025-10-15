# python_entropy_analyzer/entropy.py
"""
Lightweight entropy utilities for nibble-level analysis of 32-byte hashes.

Functions are intentionally minimal-dependency (only stdlib) so this module
can run on low-memory systems.

A "nibble" = 4-bit chunk. A 32-byte hash -> 64 nibbles.
"""

from collections import Counter, deque
import math
from typing import List, Iterable, Tuple, Deque


def shannon_entropy(nibbles: Iterable[int]) -> float:
    """
    Compute Shannon entropy (bits) for a sequence of nibble values (0..15).
    Returns entropy in bits (0..4 for nibble symbols).
    """
    counts = Counter(nibbles)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def entropy_per_nibble(batch_hashes: Iterable[bytes]) -> List[float]:
    """
    Given an iterable of hash byte-strings (each should be 32 bytes),
    compute per-nibble Shannon entropy across the batch.

    Returns a list of 64 floats (entropy for nibble positions 0..63).
    Position 0 corresponds to the high nibble of byte 0.
    """
    # Initialize 64 counters for each nibble position
    counters: List[Counter] = [Counter() for _ in range(64)]
    count_hashes = 0

    for h in batch_hashes:
        if not isinstance(h, (bytes, bytearray)):
            # try to accept hex string too
            raise TypeError("Each hash must be bytes/bytearray of length 32")
        if len(h) != 32:
            # ignore / skip invalid-length items
            continue
        count_hashes += 1
        # For each byte, extract high nibble then low nibble
        for byte_idx in range(32):
            b = h[byte_idx]
            high = (b >> 4) & 0xF
            low = b & 0xF
            nibble_pos = byte_idx * 2
            counters[nibble_pos][high] += 1
            counters[nibble_pos + 1][low] += 1

    if count_hashes == 0:
        # No valid hashes -> return zeros
        return [0.0] * 64

    entropies: List[float] = [0.0] * 64
    for i, c in enumerate(counters):
        total = sum(c.values())
        if total == 0:
            entropies[i] = 0.0
        else:
            e = 0.0
            for v in c.values():
                p = v / total
                if p > 0:
                    e -= p * math.log2(p)
            entropies[i] = e

    return entropies


def batch_entropy_summary(batch_hashes: Iterable[bytes]) -> Tuple[List[float], float, float]:
    """
    Compute per-nibble entropies for a batch and return (entropies, mean_entropy, median_entropy)
    - entropies: list[64]
    - mean_entropy: float average across the 64 positions
    - median_entropy: float median across the 64 positions
    """
    ent = entropy_per_nibble(batch_hashes)
    # mean
    mean = sum(ent) / len(ent) if ent else 0.0
    # median
    sorted_ent = sorted(ent)
    mid = len(sorted_ent) // 2
    if len(sorted_ent) % 2 == 1:
        median = sorted_ent[mid]
    else:
        median = 0.5 * (sorted_ent[mid - 1] + sorted_ent[mid])
    return ent, mean, median


class RollingEntropy:
    """
    Rolling window of per-nibble entropy vectors.

    Keeps last `window` batches (default 64). Offers running average per nibble
    without storing all historical hashes.
    """

    def __init__(self, window: int = 64):
        if window <= 0:
            raise ValueError("window must be > 0")
        self.window: int = window
        self.buffer: Deque[List[float]] = deque(maxlen=window)
        # Maintain running sums for quick average computation
        self.running_sums: List[float] = [0.0] * 64

    def push(self, ent_vector: List[float]):
        """
        Push a new per-nibble entropy vector (length 64).
        """
        if len(ent_vector) != 64:
            raise ValueError("entropy vector must be length 64")
        if len(self.buffer) == self.window:
            # will discard oldest; subtract it from sums
            oldest = self.buffer[0]
            for i in range(64):
                self.running_sums[i] -= oldest[i]
        self.buffer.append(ent_vector)
        for i in range(64):
            self.running_sums[i] += ent_vector[i]

    def mean(self) -> List[float]:
        """
        Return the per-nibble running mean (length 64).
        If no items, returns zeros.
        """
        n = len(self.buffer)
        if n == 0:
            return [0.0] * 64
        return [s / n for s in self.running_sums]

    def global_mean(self) -> float:
        """
        Return the mean across all nibble positions (scalar).
        """
        m = self.mean()
        return sum(m) / len(m) if m else 0.0

    def clear(self):
        self.buffer.clear()
        self.running_sums = [0.0] * 64


# Quick self-test / example usage when run as a script
if __name__ == "__main__":
    import os
    # small test with two example hashes (32 bytes each)
    example_hashes = [
        bytes.fromhex("00060f7054542a4325b750b9384c6c05f8dd624756062d5995f36bff3cbf76b6"),
        bytes.fromhex("0013372065e692e3f8f5c2a831418d6ee2eac8be72eb42a232d825383b2caf7e")
    ]
    ent, mean, med = batch_entropy_summary(example_hashes)
    print("Per-nibble entropy (first 8):", ent[:8])
    print("Mean entropy:", mean)
    print("Median entropy:", med)
