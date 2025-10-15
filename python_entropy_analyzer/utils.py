# python_entropy_analyzer/utils.py
"""
Utility functions to help collect and convert miner output
into a format suitable for the entropy analysis modules.
"""

import re
from typing import List, Iterable


# A regex that matches the 32-byte hash you print in Rust.
# It will find both hex inside brackets or plain hex.
HASH_RE = re.compile(r'(?:0x)?([0-9a-fA-F]{64})')


def extract_hashes_from_line(line: str) -> List[bytes]:
    """
    Extract all 32-byte hashes from a single log line.
    Returns a list of bytes objects.
    """
    hashes: List[bytes] = []
    for match in HASH_RE.finditer(line):
        h_hex = match.group(1)
        try:
            h_bytes = bytes.fromhex(h_hex)
            if len(h_bytes) == 32:
                hashes.append(h_bytes)
        except ValueError:
            continue
    return hashes


def extract_hashes_from_lines(lines: Iterable[str]) -> List[bytes]:
    """
    Extract 32-byte hashes from an iterable of lines.
    Returns a flat list of bytes objects.
    """
    all_hashes: List[bytes] = []
    for line in lines:
        all_hashes.extend(extract_hashes_from_line(line))
    return all_hashes


def load_hashes_from_logfile(path: str, max_lines: int = None) -> List[bytes]:
    """
    Read a log file and extract all 32-byte hashes.
    - path: path to your miner log
    - max_lines: optional limit on how many lines to read
    Returns list of bytes objects.
    """
    hashes: List[bytes] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            hashes.extend(extract_hashes_from_line(line))
    return hashes


def batchify(seq: List[bytes], batch_size: int) -> Iterable[List[bytes]]:
    """
    Yield successive chunks (batches) from a list of hashes.
    Example:
        for batch in batchify(hashes, 256):
            ...
    """
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


# Quick self-test / example
if __name__ == "__main__":
    sample_lines = [
        'Lowest hash this batch: nonce 49975943 → 00060f7054542a4325b750b9384c6c05f8dd624756062d5995f36bff3cbf76b6',
        'Lowest hash this batch: nonce 49980809 → 0013372065e692e3f8f5c2a831418d6ee2eac8be72eb42a232d825383b2caf7e'
    ]
    hashes = extract_hashes_from_lines(sample_lines)
    print(f"Extracted {len(hashes)} hashes. First hash (hex):", hashes[0].hex())
