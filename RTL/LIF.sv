module LIF_Model #(
    parameter T = 4,  // Number of time steps
    parameter Q = 32  // Bits for quantization
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [Q-1:0] input_data [T-1:0],  // Input data for T time steps
    input  logic [Q-1:0] threshold,           // Firing threshold
    output logic [T-1:0] spike_out            // Output spikes for T time steps
);

    logic [Q-1:0] membrane_potential;
    logic [Q-1:0] next_potential [T-1:0];

    // Calculate next potential for each time step
    always_comb begin
        for (int t = 0; t < T; t++) begin
            if (t == 0) begin
                next_potential[t] = membrane_potential + input_data[t];
            end else begin
                next_potential[t] = (spike_out[t-1]) ? input_data[t] : (next_potential[t-1] + input_data[t]);
            end
        end
    end

    // Compare with threshold and generate spikes
    always_comb begin
        for (int t = 0; t < T; t++) begin
            spike_out[t] = (next_potential[t] > threshold);
        end
    end

    // Update membrane potential
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_potential <= '0;
        end else begin
            membrane_potential <= next_potential[T-1];
        end
    end

endmodule
