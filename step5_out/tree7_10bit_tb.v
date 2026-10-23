`timescale 1ns / 1ps
module tb;
    reg clk = 0, rst = 1;
    reg signed [9:0] x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16;
    wire seizure_alert;
    reg [15:0] mem [0:25599];
    integer i;
    seizure_detector uut (.clk(clk), .rst(rst), .x1(x1), .x2(x2), .x3(x3), .x4(x4), .x5(x5), .x6(x6), .x7(x7), .x8(x8), .x9(x9), .x10(x10), .x11(x11), .x12(x12), .x13(x13), .x14(x14), .x15(x15), .x16(x16), .seizure_alert(seizure_alert));
    always #5 clk = ~clk;
    initial begin
        $readmemh("step5_out/tree7_10bit_vectors.hex", mem);
        repeat (3) @(posedge clk);
        rst = 0;
        for (i = 0; i < 1600; i = i + 1) begin
            @(negedge clk);
            x1 = mem[i*16+0];
            x2 = mem[i*16+1];
            x3 = mem[i*16+2];
            x4 = mem[i*16+3];
            x5 = mem[i*16+4];
            x6 = mem[i*16+5];
            x7 = mem[i*16+6];
            x8 = mem[i*16+7];
            x9 = mem[i*16+8];
            x10 = mem[i*16+9];
            x11 = mem[i*16+10];
            x12 = mem[i*16+11];
            x13 = mem[i*16+12];
            x14 = mem[i*16+13];
            x15 = mem[i*16+14];
            x16 = mem[i*16+15];
            @(posedge clk);
            #1;
            $display("RESULT %0d %b", i, seizure_alert);
        end
        $finish;
    end
endmodule
