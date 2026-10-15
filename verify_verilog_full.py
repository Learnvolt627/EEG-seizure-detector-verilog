import math, re, shutil, subprocess
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ---------- 1. Same data split and model as main.py ----------
df = pd.read_csv('BEED_Data.csv')
X = df.drop(columns=['y'])
y = (df['y'] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=10, max_depth=5,
                               class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

Xt = X_test.values.astype(int)
truth = y_test.values
sk_pred = model.predict(X_test)
N = len(Xt)

# Inputs must fit in signed 16 bits, since the hardware ports are 16-bit
assert Xt.min() >= -32768 and Xt.max() <= 32767, "Input value outside 16-bit range!"

# ---------- 2. Check tree_rules.json uses floor thresholds ----------
def json_thr(n):
    if n['type'] == 'leaf':
        return []
    return [n['threshold']] + json_thr(n['left']) + json_thr(n['right'])

def sk_thr(t, node=0):
    if t.children_left[node] == -1:
        return []
    return ([math.floor(t.threshold[node])]
            + sk_thr(t, t.children_left[node]) + sk_thr(t, t.children_right[node]))

with open('tree_rules.json') as f:
    rules = json.load(f)
fixed = all(json_thr(r) == sk_thr(m.tree_) for r, m in zip(rules, model.estimators_))
print("tree_rules.json uses floor thresholds:", fixed)
if not fixed:
    raise SystemExit("Apply the floor fix, re-run extract_rules.py and generate_verilog.py, then retry.")

# ---------- 3. Python emulation of the hardware (floor thresholds, 6-of-10 vote) ----------
def tree_vote(tree, x):
    t = tree.tree_
    node = 0
    while t.children_left[node] != -1:
        thr = math.floor(t.threshold[node])
        node = t.children_right[node] if x[t.feature[node]] > thr else t.children_left[node]
    return int(np.argmax(t.value[node][0]))

emu_pred = np.array([1 if sum(tree_vote(tr, x) for tr in model.estimators_) >= 6 else 0
                     for x in Xt])

# ---------- 4. Write the test vectors and a testbench for all N samples ----------
with open('test_vectors.hex', 'w') as f:
    for row in Xt:
        f.write(' '.join(f"{int(v) & 0xFFFF:04x}" for v in row) + '\n')   # 16-bit two's complement

ports = ', '.join(f".x{i}(x{i})" for i in range(1, 17))
loads = '\n'.join(f"        x{i} = mem[i*16+{i-1}];" for i in range(1, 17))
tb = f"""`timescale 1ns / 1ps
module full_tb;
    reg clk = 0, rst = 1;
    reg signed [15:0] x1, x2, x3, x4, x5, x6, x7, x8;
    reg signed [15:0] x9, x10, x11, x12, x13, x14, x15, x16;
    wire seizure_alert;
    reg [15:0] mem [0:{N*16-1}];
    integer i;

    seizure_detector uut (.clk(clk), .rst(rst), {ports}, .seizure_alert(seizure_alert));

    always #5 clk = ~clk;

    initial begin
        $readmemh("test_vectors.hex", mem);
        repeat (3) @(posedge clk);
        rst = 0;
        for (i = 0; i < {N}; i = i + 1) begin
            @(negedge clk);
{loads}
            @(posedge clk);
            #1;
            $display("RESULT %0d %b", i, seizure_alert);
        end
        $finish;
    end
endmodule
"""
with open('full_tb.v', 'w') as f:
    f.write(tb)

# ---------- 5. Compile and run the real Verilog ----------
for tool in ('iverilog', 'vvp'):
    if shutil.which(tool) is None:
        raise SystemExit(f"{tool} not found on PATH")
subprocess.run(['iverilog', '-o', 'full_sim.out', 'seizure_detector.v', 'full_tb.v'], check=True)
out = subprocess.run(['vvp', 'full_sim.out'], capture_output=True, text=True, check=True).stdout

hw = {}
for m in re.finditer(r'RESULT (\d+) ([01xXzZ])', out):
    hw[int(m.group(1))] = m.group(2)
assert len(hw) == N, f"Expected {N} results, got {len(hw)}"
assert all(v in '01' for v in hw.values()), "Some outputs were X/Z"
hw_pred = np.array([int(hw[i]) for i in range(N)])

# ---------- 6. Report ----------
print(f"\nSamples run through the actual Verilog: {N}")
print("Verilog vs Python emulation mismatches:", int((hw_pred != emu_pred).sum()))
print("Verilog vs sklearn mismatches:         ", int((hw_pred != sk_pred).sum()))
print("\nVerilog vs ground truth:\n", confusion_matrix(truth, hw_pred))
tn, fp, fn, tp = confusion_matrix(truth, hw_pred).ravel()
print(f"\nSeizure recall (sensitivity): {tp/(tp+fn):.4f}")
print(f"Specificity:                  {tn/(tn+fp):.4f}")
print(f"Accuracy:                     {(tp+tn)/N:.4f}")