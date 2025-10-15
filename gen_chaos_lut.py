# gen_chaos_lut.py
for block in range(10):
    start = 0.015625 + block * 0.001
    vals = [f"{start + i*0.00001:.6f}f" for i in range(100)]
    print(f"constant float chaos_lut_{block}[100] = {{")
    for i in range(0, 100, 10):
        print("    " + ", ".join(vals[i:i+10]) + ",")
    print("};\n")
