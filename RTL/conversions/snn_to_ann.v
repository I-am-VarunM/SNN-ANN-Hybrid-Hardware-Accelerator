module snn_to_ann_single_neuron #(
    parameter WIDTH = 3    // Output width: log2(4) + 1 = 3 bits
)(
    input wire clk,        // Clock signal
    input wire rst_n,      // Active-low reset
    input wire [3:0] spikes_in,  // Spikes from 4 timesteps for a single neuron
    input wire valid_in,         // Input valid signal
    output reg [WIDTH-1:0] ann_out,  // Accumulated output for ANN
    output reg valid_out         // Output valid signal
);

    // Tree adder structure for 4 timesteps
    // Level 1: Add pairs of timesteps
    wire [WIDTH-1:0] sum_01;
    wire [WIDTH-1:0] sum_23;
    
    // Level 2: Final sum
    wire [WIDTH-1:0] final_sum;
    
    // First level adders - Add timesteps 0+1 and 2+3
    assign sum_01 = {2'b00, spikes_in[0]} + {2'b00, spikes_in[1]};
    assign sum_23 = {2'b00, spikes_in[2]} + {2'b00, spikes_in[3]};
    
    // Second level adder - Combine results
    assign final_sum = sum_01 + sum_23;
    
    // Register the output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ann_out <= 0;
            valid_out <= 0;
        end else if (valid_in) begin
            ann_out <= final_sum;
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end

endmodule