`timescale 1ns / 1ps

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
        x1  = -47;  x2  = -52;
        x3  = -45;  x4  = -41;
        x5  = -29;  x6  = -24;
        x7  = -21;  x8  = -19;
        x9  = -13;  x10 = -3;
        x11 = -2; x12 = 4;
        x13 = 11; x14 = 14;
        x15 = 11; x16 = 14;

        #20;
        $display("Normal EEG Alert Output: %b (Expected: 0)", seizure_alert);

        // --- TEST CASE 2: Seizure EEG Sample ---
        $display("Applying Seizure EEG Sample...");
        x1  = 4;  x2  = 8;
        x3  = 3;  x4  = -10;
        x5  = 2;  x6  = 9;
        x7  = -5;  x8  = -1;
        x9  = -3;  x10 = 3;
        x11 = -4; x12 = 0;
        x13 = 4; x14 = -3;
        x15 = -7; x16 = 1;

        #20;
        $display("Seizure EEG Alert Output: %b (Expected: 1)", seizure_alert);
        #40;
        $finish;
    end

endmodule
