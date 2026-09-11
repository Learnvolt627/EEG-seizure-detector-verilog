# Real-Time EEG Seizure Detection Hardware Accelerator

A functional pipeline translating a Python-trained Random Forest model into synthesizable Verilog HDL for real-time EEG seizure detection.

## Project Structure

- `main.py`: Trains Random Forest classifier on BEED EEG dataset.
- `extract_rules.py`: Parses decision tree structures into `tree_rules.json`.
- `generate_verilog.py`: Automatically generates synthesizable `seizure_detector.v`.
- `generate_testbench.py`: Builds `seizure_detector_tb.v` test harness using real sample data.
- `waveform.vcd`: Digital signal trace for verification in GTKWave.

## Tech Stack

- **Languages:** Python 3, Verilog HDL
- **Tools:** Scikit-Learn, Icarus Verilog (`iverilog`), GTKWave
- **Dataset:** Bangalore EEG Epilepsy Dataset (BEED)

## Simulation Instructions

```powershell
python main.py
python extract_rules.py
python generate_verilog.py
python generate_testbench.py
iverilog -o seizure_sim.out seizure_detector.v seizure_detector_tb.v
vvp seizure_sim.out
gtkwave waveform.vcd
```
