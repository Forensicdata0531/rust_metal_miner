from qiskit import QuantumCircuit, execute
from qiskit_aer import Aer
from qiskit.execute_function import execute

# -----------------------------
# Simple ancilla pool
# -----------------------------
class AncillaPool:
    def __init__(self, ancilla_indices):
        self.ancilla = ancilla_indices
    def slice(self, start, length):
        return self.ancilla[start:start+length]

# -----------------------------
# 4-bit ripple-carry adder for nibbles
# -----------------------------
def nibble_ripple_carry_adder(qc, a_bits, b_bits, sum_bits, carry_in, carry_out):
    # super simplified toy adder (not full correct carry prop!)
    for i in range(4):
        qc.cx(a_bits[i], sum_bits[i])
        qc.cx(b_bits[i], sum_bits[i])
    qc.cx(carry_in, carry_out)

# -----------------------------
# 1-bit reversible Ch function
# -----------------------------
def reversible_ch(qc, x, y, z, out):
    qc.ccx(x, y, out)
    qc.x(x)
    qc.ccx(x, z, out)
    qc.x(x)

# -----------------------------
# 1-bit reversible Maj function
# -----------------------------
def reversible_maj(qc, x, y, z, out):
    qc.ccx(x, y, out)
    qc.ccx(x, z, out)
    qc.ccx(y, z, out)

# -----------------------------
# 4-bit reversible XOR
# -----------------------------
def reversible_xor(qc, a_bits, b_bits, out_bits):
    for a,b,out in zip(a_bits, b_bits, out_bits):
        qc.cx(a, out)
        qc.cx(b, out)

# -----------------------------
# Helper: rotate nibble list left
# -----------------------------
def rotate_left(bits, shift):
    n = len(bits)
    return [bits[(i - shift) % n] for i in range(n)]

# -----------------------------
# Single reversible SHA-256 round, nibble-based
# -----------------------------
def reversible_sha_round_nibble(qc, a,b,c,d,e,f,g,h,w, ancilla_pool, uncompute=False):
    # process 8 nibbles (32-bit word)
    for nib in range(8):
        a_n = a[nib*4:(nib+1)*4]
        b_n = b[nib*4:(nib+1)*4]
        c_n = c[nib*4:(nib+1)*4]
        d_n = d[nib*4:(nib+1)*4]
        e_n = e[nib*4:(nib+1)*4]
        f_n = f[nib*4:(nib+1)*4]
        g_n = g[nib*4:(nib+1)*4]
        h_n = h[nib*4:(nib+1)*4]
        w_n = w[nib*4:(nib+1)*4]

        temp1 = ancilla_pool.slice(0,4)
        temp2 = ancilla_pool.slice(4,4)
        s1    = ancilla_pool.slice(8,4)
        ch_bits = ancilla_pool.slice(12,4)
        s0    = ancilla_pool.slice(16,4)
        maj_bits=ancilla_pool.slice(20,4)
        carry1 = ancilla_pool.ancilla[24]
        carry2 = ancilla_pool.ancilla[25]

        # S1
        e_rot2 = rotate_left(e_n,2)
        e_rot3 = rotate_left(e_n,3)
        e_rot1 = rotate_left(e_n,1)
        reversible_xor(qc, e_rot2, e_rot3, s1)
        reversible_xor(qc, s1, e_rot1, s1)

        # Ch
        for i in range(4):
            reversible_ch(qc, e_n[i], f_n[i], g_n[i], ch_bits[i])

        # temp1 = h + S1 + Ch + W
        nibble_ripple_carry_adder(qc, h_n, s1, temp1, carry1, carry2)
        nibble_ripple_carry_adder(qc, temp1, ch_bits, temp1, carry1, carry2)
        nibble_ripple_carry_adder(qc, temp1, w_n, temp1, carry1, carry2)

        # S0
        a_rot2 = rotate_left(a_n,2)
        a_rot3 = rotate_left(a_n,3)
        a_rot1 = rotate_left(a_n,1)
        reversible_xor(qc, a_rot2, a_rot3, s0)
        reversible_xor(qc, s0, a_rot1, s0)

        # Maj
        for i in range(4):
            reversible_maj(qc, a_n[i], b_n[i], c_n[i], maj_bits[i])

        # temp2 = S0 + Maj
        nibble_ripple_carry_adder(qc, s0, maj_bits, temp2, carry1, carry2)

        # update state (toy)
        h_n[:] = g_n
        g_n[:] = f_n
        f_n[:] = e_n
        e_n[:] = [(d_n[i]+temp1[i])%2 for i in range(4)]
        d_n[:] = c_n
        c_n[:] = b_n
        b_n[:] = a_n
        a_n[:] = [(temp1[i]+temp2[i])%2 for i in range(4)]

        if uncompute:
            # reverse operations if needed (toy)
            pass

# -----------------------------
# 64-round reversible SHA-256
# -----------------------------
def reversible_sha64_nibble(qc, a,b,c,d,e,f,g,h,W_list, ancilla_pool, uncompute=False):
    for round_idx in range(64):
        reversible_sha_round_nibble(qc, a,b,c,d,e,f,g,h,W_list[round_idx], ancilla_pool, uncompute)

# -----------------------------
# Phase flip if leading nibble == 0000
# -----------------------------
def phase_flip_target(qc, output_bits):
    qc.x(output_bits)  # flip zeros to ones
    qc.h(output_bits[-1])
    qc.mcx(output_bits[:-1], output_bits[-1])
    qc.h(output_bits[-1])
    qc.x(output_bits)

# -----------------------------
# Diffuser for Grover
# -----------------------------
def diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)

# -----------------------------
# Grover oracle with ancilla reuse and target difficulty
# -----------------------------
def grover_sha_oracle_nibble(qc, input_qubits, ancilla_pool, W_list):
    a = input_qubits[0:32]
    b = input_qubits[32:64]
    c = input_qubits[64:96]
    d = input_qubits[96:128]
    e = input_qubits[128:160]
    f = input_qubits[160:192]
    g = input_qubits[192:224]
    h = input_qubits[224:256]

    reversible_sha64_nibble(qc, a,b,c,d,e,f,g,h,W_list, ancilla_pool, uncompute=True)
    # phase flip on first nibble of a
    phase_flip_target(qc, a[0:4])

# -----------------------------
# Example usage with multiple Grover iterations
# -----------------------------
if __name__ == "__main__":
    # 288 qubits for state + 26 ancilla
    ancilla_indices = list(range(288, 288+26))
    ancilla_pool = AncillaPool(ancilla_indices)
    qc = QuantumCircuit(288+26, 4)

    input_qubits = list(range(288))
    # Example 64 message words (toy)
    W_list = [input_qubits[0:32] for _ in range(64)]

    # Initialize in superposition
    for q in input_qubits:
        qc.h(q)

    # Perform a few Grover iterations
    iterations = 2
    for _ in range(iterations):
        grover_sha_oracle_nibble(qc, input_qubits, ancilla_pool, W_list)
        diffuser(qc, input_qubits[0:8])  # diffuse on first few qubits (toy)

    # Measure first 4 input qubits
    qc.measure(input_qubits[0:4], range(4))

    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=512).result()
    counts = result.get_counts()
    print("Grover oracle 64-round nibble counts:", counts)
    print(qc.draw('text'))
