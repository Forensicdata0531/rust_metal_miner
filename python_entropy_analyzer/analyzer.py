# python_entropy_analyzer/analyzer.py
import time
import binascii
from collections import deque, Counter
import os

# Path to the miner's hash log
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "hashes.log")

# Circular buffer to store recent hashes
BUFFER_SIZE = 512
hash_buffer = deque(maxlen=BUFFER_SIZE)

# Convert bytes to nibbles (4-bit chunks)
def bytes_to_nibbles(byte_array):
    nibbles = []
    for b in byte_array:
        nibbles.append((b >> 4) & 0xF)
        nibbles.append(b & 0xF)
    return nibbles

# Shannon entropy calculation for nibbles
def shannon_entropy(nibbles):
    counts = Counter(nibbles)
    total = len(nibbles)
    entropy = -sum((count / total) * (0 if count == 0 else math.log2(count / total)) for count in counts.values())
    return entropy

# Main loop: read new lines from log
def main():
    print("[INFO] Starting entropy analyzer...")
    with open(LOG_FILE, "r") as f:
        f.seek(0, 2)  # move to the end of the file

        while True:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue

            try:
                # Extract hash bytes from line
                # Expecting lines like: Lowest hash this batch: nonce 12345 → [0, 1, 2, ...]
                parts = line.strip().split("→")
                hash_str = parts[1].strip().replace("[", "").replace("]", "")
                hash_bytes = bytes(int(x, 16) for x in hash_str.split(","))
                
                # Store in circular buffer
                if len(hash_bytes) == 32:
                    hash_buffer.append(hash_bytes)

                    # Convert to nibbles and calculate entropy
                    nibbles = bytes_to_nibbles(hash_bytes)
                    entropy = shannon_entropy(nibbles)
                    print(f"Entropy: {entropy:.3f} bits/nibble, Nonce: {parts[0].split()[-1]}")

            except Exception as e:
                print(f"[WARN] Failed to parse line: {line.strip()} | Error: {e}")
                continue

if __name__ == "__main__":
    import math
    main()
