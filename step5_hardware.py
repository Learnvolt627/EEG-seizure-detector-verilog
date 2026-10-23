"""
step5_hardware.py  --  Step 5: from "cost proxy" to real hardware numbers.

The story: in Step 4 we priced designs with a made-up unit (comparator-bits). Now we do what a
chip designer does. For seven designs we (1) write real Verilog, (2) prove in simulation that the
Verilog gives the same answers as the Python model on all 1,600 test rows, and (3) let a
synthesis tool (Yosys) turn the Verilog into logic and COUNT what it needs:

    generic gates  : AND / OR / MUX ... the kind of count an ASIC flow starts from
    LUT4s (iCE40)  : the kind of count a small FPGA flow reports
    flip-flops     : the one output register

The designs:
    today         forest 10x5, all 16 inputs, 16-bit numbers  (what your Verilog is now)
    free_10bit    the same forest, but 10-bit numbers (the data never needed more)
    tree7_16bit   ONE tree, depth 7, 16-bit numbers  (separates "one tree" from "narrow numbers")
    tree7_10bit   the same tree with 10-bit numbers
    robust        ONE tree, depth 7, all 16 inputs, 6-bit numbers
    aggressive    ONE tree, depth 8, all 16 inputs, 5-bit numbers  (clears the bar by a hair)
    fewer_inputs  ONE tree, depth 10, only 6 inputs, 6-bit numbers

"Dropping low bits" is done by right-shifting, which in hardware is free: the design just
receives the top w bits of the ADC and ignores the rest.

Needs: iverilog + vvp (you have them) and Yosys. If Yosys is missing:  pip install yowasp-yosys
Run from C:\\Biomed_project inside the venv:   python step5_hardware.py
(the first Yosys run can take a minute while it sets itself up)
"""

import json
import math
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

SEED = 42
OUT = "step5_out"                                    # all generated files go here
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("BEED_Data.csv")
names = list(df.columns[:-1])                        # X1 ... X16
X_full = df.drop(columns=["y"]).to_numpy().astype(np.int64)
y = (df["y"] > 0).astype(int).to_numpy()

ALL = list(range(len(names)))
SIX = [names.index(n) for n in ["X16", "X8", "X2", "X7", "X15", "X11"]]    # from Step 3

# label, kind, trees, depth, input columns, low bits dropped, Verilog width (None = just enough)
DESIGNS = [
    ("today",        "forest", 10, 5,  ALL, 0, 16),
    ("free_10bit",   "forest", 10, 5,  ALL, 0, None),
    ("tree7_16bit",  "tree",    1, 7,  ALL, 0, 16),     # same tree as robust, full-width numbers
    ("tree7_10bit",  "tree",    1, 7,  ALL, 0, None),   # same tree, just-enough width
    ("robust",       "tree",    1, 7,  ALL, 4, None),
    ("aggressive",   "tree",    1, 8,  ALL, 5, None),
    ("fewer_inputs", "tree",    1, 10, SIX, 4, None),
]

# The same 80/20 split as main.py, so the 1,600 test rows are the ones you already know.
idx = np.arange(len(y))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)


# ------------------------------------------------------------------ small helpers
def width_needed(values):
    """Signed bits needed to hold these whole numbers."""
    return math.ceil(math.log2(int(np.abs(values).max()) + 1)) + 1


def make_model(kind, trees, depth, n_inputs):
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=SEED)
    return RandomForestClassifier(n_estimators=trees, max_depth=depth, max_features=min(4, n_inputs),
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def tree_list(model, kind):
    return [model] if kind == "tree" else list(model.estimators_)


def emulate(trees, row):
    """Walk every tree the way the Verilog does (threshold = floor, 'x > thr' goes right),
    then apply the vote rule. This is our independent Python model of the hardware."""
    votes = 0
    for tree in trees:
        t = tree.tree_
        node = 0
        while t.children_left[node] != -1:
            thr = math.floor(t.threshold[node])
            node = t.children_right[node] if row[t.feature[node]] > thr else t.children_left[node]
        votes += int(np.argmax(t.value[node][0]))
    return int(votes >= len(trees) // 2 + 1)


def literal(value, width):
    """A signed Verilog constant, e.g. 6'sd5 or -6'sd5."""
    return f"-{width}'sd{abs(value)}" if value < 0 else f"{width}'sd{value}"


def tree_expression(t, node, width, in_names):
    """One tree as a nested 'condition ? right : left' expression."""
    if t.children_left[node] == -1:
        return f"1'b{int(np.argmax(t.value[node][0]))}"
    name = in_names[t.feature[node]]
    thr = math.floor(t.threshold[node])
    right = tree_expression(t, t.children_right[node], width, in_names)
    left = tree_expression(t, t.children_left[node], width, in_names)
    return f"(({name} > {literal(thr, width)}) ? {right} : {left})"


def write_verilog(path, trees, width, in_names):
    n = len(trees)
    ports = ", ".join(in_names)
    text = ["// Auto-generated by step5_hardware.py",
            "module seizure_detector (",
            "    input wire clk,",
            "    input wire rst,",
            f"    input wire signed [{width - 1}:0] {ports},",
            "    output reg seizure_alert",
            ");", ""]
    for i in range(n):
        text.append(f"    wire tree_{i}_vote;")
    for i, tr in enumerate(trees):
        text.append(f"    assign tree_{i}_vote = {tree_expression(tr.tree_, 0, width, in_names)};")
    if n == 1:
        decision = "tree_0_vote"
        text.append("")
    else:
        vote_bits = n.bit_length()
        need = n // 2 + 1
        text += ["", f"    wire [{vote_bits - 1}:0] total_votes;",
                 "    assign total_votes = " + " + ".join(f"tree_{i}_vote" for i in range(n)) + ";", ""]
        decision = f"(total_votes >= {vote_bits}'d{need}) ? 1'b1 : 1'b0"
    text += ["    always @(posedge clk or posedge rst) begin",
             "        if (rst) seizure_alert <= 1'b0;",
             f"        else     seizure_alert <= {decision};",
             "    end", "endmodule", ""]
    with open(path, "w") as f:
        f.write("\n".join(text))


def write_testbench(path, vec_file, width, in_names, n_rows):
    k = len(in_names)
    ports = ", ".join(f".{nm}({nm})" for nm in in_names)
    loads = "\n".join(f"            {nm} = mem[i*{k}+{j}];" for j, nm in enumerate(in_names))
    text = f"""`timescale 1ns / 1ps
module tb;
    reg clk = 0, rst = 1;
    reg signed [{width - 1}:0] {", ".join(in_names)};
    wire seizure_alert;
    reg [15:0] mem [0:{n_rows * k - 1}];
    integer i;
    seizure_detector uut (.clk(clk), .rst(rst), {ports}, .seizure_alert(seizure_alert));
    always #5 clk = ~clk;
    initial begin
        $readmemh("{vec_file}", mem);
        repeat (3) @(posedge clk);
        rst = 0;
        for (i = 0; i < {n_rows}; i = i + 1) begin
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
    with open(path, "w") as f:
        f.write(text)


def simulate(label, vfile, width, in_names, Xq_test):
    """Run the Verilog in Icarus on every test row and return its 0/1 answers."""
    vec = f"{OUT}/{label}_vectors.hex"
    with open(vec, "w") as f:
        for row in Xq_test:
            f.write(" ".join(f"{int(v) & 0xFFFF:04x}" for v in row) + "\n")
    tb = f"{OUT}/{label}_tb.v"
    write_testbench(tb, vec, width, in_names, len(Xq_test))
    sim = f"{OUT}/{label}_sim.out"
    subprocess.run(["iverilog", "-o", sim, vfile, tb], check=True)
    out = subprocess.run(["vvp", sim], capture_output=True, text=True, check=True).stdout
    got = {int(m.group(1)): m.group(2) for m in re.finditer(r"RESULT (\d+) ([01xXzZ])", out)}
    assert len(got) == len(Xq_test), f"expected {len(Xq_test)} results, got {len(got)}"
    assert all(v in "01" for v in got.values()), "some outputs were X/Z"
    return np.array([int(got[i]) for i in range(len(Xq_test))])


def find_yosys():
    for cmd in ("yosys", "yowasp-yosys"):
        path = shutil.which(cmd)
        if path:
            return path
    return None


def synthesize(yosys, vfile, label, flow):
    """Run Yosys and return its cell counts. flow = 'generic' (gates) or 'ice40' (LUT4s)."""
    stats = f"{OUT}/{label}_{flow}.json"
    synth = "synth" if flow == "generic" else "synth_ice40"
    script = f"read_verilog {vfile}; {synth} -top seizure_detector -flatten; tee -q -o {stats} stat -json"
    subprocess.run([yosys, "-q", "-p", script], check=True, capture_output=True, text=True)
    with open(stats) as f:
        data = json.load(f)
    block = data.get("design") or next(iter(data["modules"].values()))
    cells = block.get("num_cells_by_type") or block.get("cells") or {}
    flops = sum(n for c, n in cells.items() if "DFF" in c)
    if flow == "generic":
        return {"gates": sum(cells.values()) - flops, "flops": flops}
    return {"lut4": cells.get("SB_LUT4", 0), "carry": cells.get("SB_CARRY", 0), "ice40_flops": flops}


# ------------------------------------------------------------------ the main loop
yosys = find_yosys()
if yosys is None:
    print("Yosys not found, so synthesis will be skipped. To get the hardware counts run:")
    print("    pip install yowasp-yosys")
    print("and then run this script again.\n")

rows = []
for label, kind, trees, depth, cols, shift, width_override in DESIGNS:
    print(f"--- {label} ---")
    Xq = np.floor_divide(X_full[:, cols], 2 ** shift)          # drop low bits (a right shift)
    width = width_override or width_needed(Xq)
    in_names = [names[c].lower() for c in cols]

    model = make_model(kind, trees, depth, len(cols))
    model.fit(Xq[train_idx], y[train_idx])
    trs = tree_list(model, kind)
    comparators = sum(t.tree_.node_count - t.tree_.n_leaves for t in trs)

    vfile = f"{OUT}/{label}.v"
    write_verilog(vfile, trs, width, in_names)

    Xq_test, y_test = Xq[test_idx], y[test_idx]
    hw = simulate(label, vfile, width, in_names, Xq_test)
    emu = np.array([emulate(trs, r) for r in Xq_test])
    sk = (model.predict(Xq_test) if kind == "tree" else
          (sum(t.predict(Xq_test).astype(int) for t in trs) >= len(trs) // 2 + 1).astype(int))
    tn, fp, fn, tp = confusion_matrix(y_test, hw, labels=[0, 1]).ravel()

    row = {"design": label, "trees": trees, "depth": depth, "inputs": len(cols), "bits": width,
           "comparators": comparators,
           "verilog_vs_emulation": int((hw != emu).sum()), "verilog_vs_sklearn": int((hw != sk).sum()),
           "recall": tp / (tp + fn), "spec": tn / (tn + fp), "acc": (tp + tn) / len(y_test)}
    print(f"  Verilog on {len(y_test)} test rows: recall={row['recall']:.4f} spec={row['spec']:.4f} "
          f"acc={row['acc']:.4f} | mismatches vs Python emulation={row['verilog_vs_emulation']}, "
          f"vs sklearn={row['verilog_vs_sklearn']}")

    if yosys:
        row.update(synthesize(yosys, vfile, label, "generic"))
        row.update(synthesize(yosys, vfile, label, "ice40"))
        print(f"  Yosys: {row['gates']} generic gates + {row['flops']} flop(s); "
              f"{row['lut4']} LUT4s on iCE40")
    rows.append(row)

res = pd.DataFrame(rows)
res.to_csv("results_step5_hardware.csv", index=False)

# ------------------------------------------------------------------ the summary
print("\n=== Summary (test set = the same 1,600 rows as main.py) ===")
cols_show = ["design", "comparators", "inputs", "bits", "recall", "spec", "acc",
             "verilog_vs_emulation", "verilog_vs_sklearn"]
print(res[cols_show].round(4).to_string(index=False))

if yosys:
    base = res[res.design == "today"].iloc[0]
    print("\n=== Real hardware counts from Yosys (smaller is better) ===")
    show = res[["design", "gates", "flops", "lut4"]].copy()
    show["gates_vs_today"] = (base.gates / res.gates).round(1).astype(str) + "x smaller"
    show["lut4_vs_today"] = (base.lut4 / res.lut4).round(1).astype(str) + "x smaller"
    print(show.to_string(index=False))
    print("\n(gates = generic logic gates after Yosys's generic synthesis, flops excluded;")
    print(" lut4 = 4-input lookup tables when targeting a Lattice iCE40 FPGA)")
print("\nSaved: results_step5_hardware.csv and the Verilog files in the step5_out folder")