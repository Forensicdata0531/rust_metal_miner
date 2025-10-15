from qiskit import QuantumCircuit, Aer, execute

# -----------------------------
# 4-bit ripple-carry adder for nibbles
# -----------------------------
def nibble_ripple_carry_adder(qc, a_bits, b_bits, sum_bits, carry_in, carry_out):
    carry = [carry_in] + [qc.qregs[0][0]]*3  # intermediate carry qubits
    for i in range(4):
        qc.cx(a_bits[i], sum_bits[i])
        qc.cx(b_bits[i], sum_bits[i])
        qc.cx(carry[i], sum_bits[i])
        qc.ccx(a_bits[i], b_bits[i], carry[i+1])
        qc.ccx(a_bits[i], carry[i], sum_bits[i])
        qc.ccx(b_bits[i], carry[i], sum_bits[i])
    qc.cx(carry[4], carry_out)

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
def reversible_sha_round_nibble(qc, a,b,c,d,e,f,g,h,w, uncompute=False):
    # Process each 32-bit word in 8 nibbles
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

        temp1 = [qc.qregs[0][i] for i in range(4)]
        temp2 = [qc.qregs[0][i+4] for i in range(4)]
        carry1 = qc.qregs[0][8]
        carry2 = qc.qregs[0][9]

        # S1 = e_rot2 XOR e_rot3 XOR e_rot1
        e_rot2 = rotate_left(e_n,2)
        e_rot3 = rotate_left(e_n,3)
        e_rot1 = rotate_left(e_n,1)
        s1 = [qc.qregs[0][i+10] for i in range(4)]
        reversible_xor(qc, e_rot2, e_rot3, s1)
        reversible_xor(qc, s1, e_rot1, s1)

        # Ch(e,f,g)
        ch_bits = [qc.qregs[0][i+14] for i in range(4)]
        for i in range(4):
            reversible_ch(qc, e_n[i], f_n[i], g_n[i], ch_bits[i])

        # temp1 = h + S1 + Ch + w_n
        nibble_ripple_carry_adder(qc, h_n, s1, temp1, carry1, carry2)
        nibble_ripple_carry_adder(qc, temp1, ch_bits, temp1, carry1, carry2)
        nibble_ripple_carry_adder(qc, temp1, w_n, temp1, carry1, carry2)

        # S0 = rotate a nibble
        a_rot2 = rotate_left(a_n,2)
        a_rot3 = rotate_left(a_n,3)
        a_rot1 = rotate_left(a_n,1)
        s0 = [qc.qregs[0][i+18] for i in range(4)]
        reversible_xor(qc, a_rot2, a_rot3, s0)
        reversible_xor(qc, s0, a_rot1, s0)

        # Maj(a,b,c)
        maj_bits = [qc.qregs[0][i+22] for i in range(4)]
        for i in range(4):
            reversible_maj(qc, a_n[i], b_n[i], c_n[i], maj_bits[i])

        # temp2 = S0 + Maj
        nibble_ripple_carry_adder(qc, s0, maj_bits, temp2, carry1, carry2)

        # Update nibble state
        h_n[:] = g_n
        g_n[:] = f_n
        f_n[:] = e_n
        e_n[:] = [(d_n[i]+temp1[i])%2 for i in range(4)]
        d_n[:] = c_n
        c_n[:] = b_n
        b_n[:] = a_n
        a_n[:] = [(temp1[i]+temp2[i])%2 for i in range(4)]

        # Uncompute temporary ancilla
        if uncompute:
            nibble_ripple_carry_adder(qc, s0, maj_bits, temp2, carry1, carry2)
            nibble_ripple_carry_adder(qc, temp1, w_n, temp1, carry1, carry2)
            nibble_ripple_carry_adder(qc, temp1, ch_bits, temp1, carry1, carry2)
            nibble_ripple_carry_adder(qc, h_n, s1, temp1, carry1, carry2)

# -----------------------------
# 64-round reversible SHA-256 (nibble-based)
# -----------------------------
def reversible_sha64_nibble(qc, a,b,c,d,e,f,g,h,W_list, uncompute=False):
    for round_idx in range(64):
        reversible_sha_round_nibble(qc, a,b,c,d,e,f,g,h,W_list[round_idx], uncompute)

# -----------------------------
# Grover oracle wrapper (toy)
# -----------------------------
def grover_sha_oracle_nibble(qc, input_qubits, target_qubits, W_list):
    a = input_qubits[0:32]
    b = input_qubits[32:64]
    c = input_qubits[64:96]
    d = input_qubits[96:128]
    e = input_qubits[128:160]
    f = input_qubits[160:192]
    g = input_qubits[192:224]
    h = input_qubits[224:256]

    reversible_sha64_nibble(qc, a,b,c,d,e,f,g,h,W_list, uncompute=True)

    # Toy phase flip
    for i in range(4):
        qc.cx(a[i], target_qubits[i])
    qc.cz(target_qubits[0], target_qubits[1])
    for i in range(4):
        qc.cx(a[i], target_qubits[i])

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    qc = QuantumCircuit(300,4)  # 288 input + 4 target

    # Input qubits (32*9 words = 288)
    input_qubits = list(range(288))
    target_qubits = list(range(288,292))

    # Example 64 message words (toy, reuse qubits)
    W_list = [input_qubits[0:32] for _ in range(64)]  # toy placeholders

    # Initialize superposition
    for q in input_qubits:
        qc.h(q)

    # Apply Grover oracle
    grover_sha_oracle_nibble(qc, input_qubits, target_qubits, W_list)

    # Measure first 4 input qubits
    qc.measure(input_qubits[0:4], range(4))

    # Simulate
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1024).result()
    counts = result.get_counts()
    print("Grover oracle 64-round nibble example counts:", counts)

    # Draw circuit
    print(qc.draw('text'))
