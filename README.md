# How small can an EEG seizure detector get in hardware?

A hardware-aware machine-learning study on the BEED EEG dataset. A Random Forest seizure detector is
converted to synthesizable Verilog, verified against Python, and synthesized with Yosys. We then ask:
**how small can the detector get (trees, depth, inputs, bit width) before it stops being good enough?**

> Status: proof-of-concept study done in simulation and synthesis. No silicon, power or timing
> measurements. Dataset has no subject IDs, so results do not show generalisation to new patients.

## Headline result

One depth-7 decision tree on 6-bit inputs catches 99.6% of seizures like a 10-tree forest, using about
**12x fewer FPGA logic cells** (49 vs 588 LUT4s on Lattice iCE40, via Yosys). About 4.2x of that comes from
using one tree instead of ten, and a further 2.9x from narrower numbers. The cost is about 1.6 points of
specificity (more false alarms).

| Design                       | Trees | Depth | Bits | LUT4s  | Gates   | Recall | Specificity | Accuracy |
| ---------------------------- | ----- | ----- | ---- | ------ | ------- | ------ | ----------- | -------- |
| Original forest              | 10    | 5     | 16   | 588    | 1363    | 99.58% | 98.00%      | 99.19%   |
| One tree, full width         | 1     | 7     | 16   | 140    | 417     | 99.42% | 96.50%      | 98.69%   |
| **One tree, 6-bit (chosen)** | 1     | 7     | 6    | **49** | **122** | 99.50% | 96.50%      | 98.75%   |
| One tree, 5-bit              | 1     | 8     | 5    | 43     | 99      | 98.58% | 95.50%      | 97.81%   |

(Hardware rows: the 1,600-row test split of `main.py`, seed 42; Verilog simulated in Icarus, 0 mismatches
against the Python model on every row.)

Robustness (20 random 75/25 splits, design fixed in advance): the chosen design had recall 99.61% +/- 0.13,
specificity 96.59% +/- 0.58 and cleared recall >= 0.99 and specificity >= 0.95 in 20 of 20 splits. The 5-bit design
cleared it in only 9 of 20. Software with no hardware limit (SVM, small neural net, 100-tree forest) reaches about
99.7% accuracy and about 99% specificity, which is the ceiling.

## Run order

Needs Python 3, scikit-learn, pandas, numpy, matplotlib, Icarus Verilog (`iverilog`, `vvp`), and Yosys
(`pip install yowasp-yosys`). Seed is 42 throughout.

| Step | Script                                                                        | What it does                                                         |
| ---- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 0    | `main.py`, `extract_rules.py`, `generate_verilog.py`, `generate_testbench.py` | Original forest to Verilog                                           |
| 1    | `step1_baseline.py`                                                           | Dataset checks and baselines with repeated cross-validation          |
| 2    | `step2_size_sweep.py`                                                         | Trees x depth sweep, cost = comparators                              |
| 3    | `step3_inputs_and_checks.py`                                                  | Fewest inputs, plain tree, vote rule                                 |
| 4    | `step4_combine.py`                                                            | Bit width x inputs x model size (84 designs)                         |
| 5    | `step5_hardware.py`                                                           | Real Verilog + Yosys synthesis for 7 designs, verified in simulation |
| 6    | `step6_final_check.py`                                                        | Holdout exam, confidence intervals, 20-seed stability                |
| 7    | `step7_margin_rule.py`                                                        | Same exam with a safety-margin selection rule                        |

## Method notes

- Hardware rule: each tree votes 0/1; alert if votes >= n//2 + 1; output registered (one clock of latency).
- Thresholds are floored (`x > floor(t)`) because inputs are integers. An earlier `round()` bug caused 5 mismatches
  against software and was fixed.
- Lower bit widths drop low bits (floor division by 2^s), like a coarser ADC; in hardware this is a free right shift.
- Cost proxy used during search: comparators x bit width. Synthesis later agreed (predicted about 3.1x for the
  16-to-6-bit change, measured 2.9x).
- Selection rule: "cheapest design that barely clears the bar" picked a fragile design (5-bit, depth 8). Requiring a margin
  on development data (recall >= 0.993, specificity >= 0.96) selected the 6-bit depth-7 tree, which held up.

## Limitations

- BEED has no subject column, so there is no leave-one-subject-out test. Many very different models score near
  perfectly, so BEED looks easy and the results may not transfer to harder data.
- The documentation calls the 16 columns channels, but correlations look like consecutive samples of one signal. We call
  them "inputs", not electrodes.
- Several design choices were first explored on the whole dataset; Steps 6 and 7 re-check them on a locked holdout
  but the 20 splits reuse the same rows, so their spread understates real uncertainty.
- Counts come from generic Yosys synthesis, not place-and-route, power or timing analysis.

## Related work (not claimed as novel)

Hardware random forests for iEEG seizure detection (Razi et al., IECBES 2022); boosted shallow trees for seizure
detection (Shoaran et al., EMBC 2016); tree-to-FPGA tools (hls4ml, conifer); BEED has been used with SVM/KNN/ANN in software
(Goksenli and Tumay, 2026). This project is an application and hardware-cost study on BEED.
