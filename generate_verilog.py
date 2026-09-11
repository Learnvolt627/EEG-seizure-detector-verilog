import json

with open('tree_rules.json', 'r') as f:
    trees = json.load(f)

def build_tree_verilog(node):
    if node['type'] == 'leaf':
        return f"1'b{node['vote']}"
    
    feat = node['feature'].lower()
    thresh = node['threshold']
    thresh_str = f"-16'sd{abs(thresh)}" if thresh < 0 else f"16'sd{thresh}"
        
    left_logic = build_tree_verilog(node['left'])
    right_logic = build_tree_verilog(node['right'])
    
    return f"(({feat} > {thresh_str}) ? {right_logic} : {left_logic})"

verilog_code = """// Auto-generated Machine Learning Accelerator for Seizure Detection
module seizure_detector (
    input wire clk,
    input wire rst,
    input wire signed [15:0] x1, x2, x3, x4, x5, x6, x7, x8,
    input wire signed [15:0] x9, x10, x11, x12, x13, x14, x15, x16,
    output reg seizure_alert
);

"""

for i in range(len(trees)):
    verilog_code += f"    wire tree_{i}_vote;\n"

verilog_code += "\n    // --- Decision Tree Logic Units ---\n"

for i, tree in enumerate(trees):
    verilog_code += f"    assign tree_{i}_vote = {build_tree_verilog(tree)};\n"

verilog_code += """
    // Voting Adder Unit
    wire [3:0] total_votes;
    assign total_votes = """ + " + ".join([f"tree_{i}_vote" for i in range(len(trees))]) + ";\n"

verilog_code += """
    // Output Register with Majority Threshold (>= 6 votes)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            seizure_alert <= 1'b0;
        end else begin
            seizure_alert <= (total_votes >= 4'd6) ? 1'b1 : 1'b0;
        end
    end

endmodule
"""

with open('seizure_detector.v', 'w') as f:
    f.write(verilog_code)

print("SUCCESS: Generated 'seizure_detector.v' hardware file.")