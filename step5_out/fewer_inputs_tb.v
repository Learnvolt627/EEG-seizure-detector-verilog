`timescale 1ns / 1ps
module tb;
    reg clk = 0, rst = 1;
    reg signed [5:0] x16, x8, x2, x7, x15, x11;
    wire seizure_alert;
    reg [15:0] mem [0:9599];
    integer i;
    seizure_detector uut (.clk(clk), .rst(rst), .x16(x16), .x8(x8), .x2(x2), .x7(x7), .x15(x15), .x11(x11), .seizure_alert(seizure_alert));
    always #5 clk = ~clk;
    initial begin
        $readmemh("step5_out/fewer_inputs_vectors.hex", mem);
        repeat (3) @(posedge clk);
        rst = 0;
        for (i = 0; i < 1600; i = i + 1) begin
            @(negedge clk);
            x16 = mem[i*6+0];
            x8 = mem[i*6+1];
            x2 = mem[i*6+2];
            x7 = mem[i*6+3];
            x15 = mem[i*6+4];
            x11 = mem[i*6+5];
            @(posedge clk);
            #1;
            $display("RESULT %0d %b", i, seizure_alert);
        end
        $finish;
    end
endmodule
