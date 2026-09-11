import pandas as pd

df = pd.read_csv('BEED_Data.csv')

# Find clear representative samples
normal_sample = df[df['y'] == 0].iloc[5]
seizure_sample = df[df['y'] == 1].iloc[5]

tb_code = f"""`timescale 1ns / 1ps

module seizure_detector_tb;

    reg clk;
    reg rst;
    reg signed [15:0] x1, x2, x3, x4, x5, x6, x7, x8;
    reg signed [15:0] x9, x10, x11, x12, x13, x14, x15, x16;

    wire seizure_alert;

    seizure_detector uut (
        .clk(clk), .rst(rst),
        .x1(x1), .x2(x2), .x3(x3), .x4(x4), .x5(x5), .x6(x6), .x7(x7), .x8(x8),
        .x9(x9), .x10(x10), .x11(x11), .x12(x12), .x13(x13), .x14(x14), .x15(x15), .x16(x16),
        .seizure_alert(seizure_alert)
    );

    always #5 clk = ~clk;

    initial begin
        // --- Setup VCD Waveform Output ---
        $dumpfile("waveform.vcd");
        $dumpvars(0, seizure_detector_tb);

        clk = 0;
        rst = 1;
        #20;
        rst = 0;

        // --- TEST CASE 1: Normal EEG Sample ---
        $display("Applying Normal EEG Sample...");
        x1  = {int(normal_sample['X1'])};  x2  = {int(normal_sample['X2'])};
        x3  = {int(normal_sample['X3'])};  x4  = {int(normal_sample['X4'])};
        x5  = {int(normal_sample['X5'])};  x6  = {int(normal_sample['X6'])};
        x7  = {int(normal_sample['X7'])};  x8  = {int(normal_sample['X8'])};
        x9  = {int(normal_sample['X9'])};  x10 = {int(normal_sample['X10'])};
        x11 = {int(normal_sample['X11'])}; x12 = {int(normal_sample['X12'])};
        x13 = {int(normal_sample['X13'])}; x14 = {int(normal_sample['X14'])};
        x15 = {int(normal_sample['X15'])}; x16 = {int(normal_sample['X16'])};

        #20;
        $display("Normal EEG Alert Output: %b (Expected: 0)", seizure_alert);

        // --- TEST CASE 2: Seizure EEG Sample ---
        $display("Applying Seizure EEG Sample...");
        x1  = {int(seizure_sample['X1'])};  x2  = {int(seizure_sample['X2'])};
        x3  = {int(seizure_sample['X3'])};  x4  = {int(seizure_sample['X4'])};
        x5  = {int(seizure_sample['X5'])};  x6  = {int(seizure_sample['X6'])};
        x7  = {int(seizure_sample['X7'])};  x8  = {int(seizure_sample['X8'])};
        x9  = {int(seizure_sample['X9'])};  x10 = {int(seizure_sample['X10'])};
        x11 = {int(seizure_sample['X11'])}; x12 = {int(seizure_sample['X12'])};
        x13 = {int(seizure_sample['X13'])}; x14 = {int(seizure_sample['X14'])};
        x15 = {int(seizure_sample['X15'])}; x16 = {int(seizure_sample['X16'])};

        #20;
        $display("Seizure EEG Alert Output: %b (Expected: 1)", seizure_alert);
        #40;
        $finish;
    end

endmodule
"""

with open('seizure_detector_tb.v', 'w') as f:
    f.write(tb_code)

print("SUCCESS: Regenerated 'seizure_detector_tb.v' with VCD waveform logging!")