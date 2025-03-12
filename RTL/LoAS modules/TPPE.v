module tppe #(
    parameter BITMASK_WIDTH = 128,
    parameter WEIGHT_WIDTH = 8,
    parameter NUM_ADDERS = 16,
    parameter TIMESTEPS = 4,
    parameter FIFO_DEPTH = 8,
    parameter PSEUDO_ACC_WIDTH = 12,
    parameter CORRECTION_ACC_WIDTH = 10,
    parameter ADDR_WIDTH = 8
)(
    input wire clk,
    input wire rst,
    
    // Input bitmasks and data
    input wire [BITMASK_WIDTH-1:0] bitmask_a,
    input wire [BITMASK_WIDTH-1:0] bitmask_b,
    input wire [BITMASK_WIDTH*WEIGHT_WIDTH-1:0] nonzero_weights_flat, // Flattened array input
    input wire valid_input,
    
    // Memory interface for accessing fibre_a data
    output wire [ADDR_WIDTH-1:0] fibre_a_addr,
    output wire fibre_a_read_en,
    input wire [TIMESTEPS-1:0] fibre_a_data,
    input wire fibre_a_valid,
    
    // Final results
    output wire [CORRECTION_ACC_WIDTH-1:0] result_0,
    output wire [CORRECTION_ACC_WIDTH-1:0] result_1, 
    output wire [CORRECTION_ACC_WIDTH-1:0] result_2,
    output wire [CORRECTION_ACC_WIDTH-1:0] result_3,
    output wire result_valid,
    output wire ready_for_input
);

    // Internal wires connecting modules
    
    // Fast prefix to laggy prefix
    wire [$clog2(BITMASK_WIDTH)-1:0] matched_position;
    wire [WEIGHT_WIDTH-1:0] matched_weight;
    wire fast_valid;
    wire ready_for_fast;
    
    // Laggy prefix to accumulator/correction
    wire [$clog2(BITMASK_WIDTH)-1:0] slow_offset;
    wire [$clog2(BITMASK_WIDTH)-1:0] current_position;
    wire [WEIGHT_WIDTH-1:0] current_weight;
    wire slow_valid;
    wire ready_for_slow;
    
    // FIFO status signals
    wire fifo_mp_empty, fifo_mp_full;
    wire fifo_weight_empty, fifo_weight_full;
    
    // FIFO control signals
    wire fifo_mp_read_en;
    wire fifo_weight_read_en;
    
    // AND result between bitmasks (shared by fast and slow prefix)
    wire [BITMASK_WIDTH-1:0] and_result;
    assign and_result = bitmask_a & bitmask_b;
    
    // Ready for new input when fast prefix is ready
    assign ready_for_input = ready_for_fast;
    
    // Fast Prefix Module
    fast_prefix #(
        .BITMASK_WIDTH(BITMASK_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH)
    ) fast_prefix_inst (
        .clk(clk),
        .rst(rst),
        .and_result(and_result),
        .bitmask_b(bitmask_b),
        .valid_match(valid_input),
        .fibre_b_data_flat(nonzero_weights_flat), // Pass flattened array
        .fast_offset(),  // Not connected - we don't need this at top level
        .matched_position(matched_position),
        .matched_weight(matched_weight),
        .fast_valid(fast_valid),
        .processing_done() // Add this output connection
    );
    
    // Laggy Prefix Module
    laggy_prefix #(
        .BITMASK_WIDTH(BITMASK_WIDTH),
        .NUM_ADDERS(NUM_ADDERS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) laggy_prefix_inst (
        .clk(clk),
        .rst(rst),
        .and_result(and_result),
        .bitmask_a(bitmask_a),
        .matched_position(matched_position),
        .matched_weight(matched_weight),
        .valid_match(fast_valid),
        .fifo_mp_read_en(fifo_mp_read_en),
        .fifo_weight_read_en(fifo_weight_read_en),
        .slow_offset(slow_offset),
        .current_position(current_position),
        .current_weight(current_weight),
        .slow_valid(slow_valid),
        .ready_for_new_calc(ready_for_slow),
        .fifo_mp_empty(fifo_mp_empty),
        .fifo_mp_full(fifo_mp_full),
        .fifo_weight_empty(fifo_weight_empty),
        .fifo_weight_full(fifo_weight_full)
    );
    
    // Accumulator and Correction Module
    accumulator_correction #(
        .TIMESTEPS(TIMESTEPS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .PSEUDO_ACC_WIDTH(PSEUDO_ACC_WIDTH),
        .CORRECTION_ACC_WIDTH(CORRECTION_ACC_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) accumulator_correction_inst (
        .clk(clk),
        .rst(rst),
        .matched_position(matched_position),
        .matched_weight(matched_weight),
        .fast_valid(fast_valid),
        .current_position(current_position),
        .current_weight(current_weight),
        .slow_offset(slow_offset),
        .slow_valid(slow_valid),
        .fifo_empty(fifo_mp_empty || fifo_weight_empty),
        .fibre_a_data(fibre_a_data),
        .fibre_a_valid(fibre_a_valid),
        .fibre_a_addr(fibre_a_addr),
        .fibre_a_read_en(fibre_a_read_en),
        .ready_for_fast(ready_for_fast),
        .ready_for_slow(ready_for_slow),
        .fifo_read_req(),  // Not used, we have separate signals now
        .result_0(result_0),
        .result_1(result_1),
        .result_2(result_2),
        .result_3(result_3),
        .result_valid(result_valid)
    );
    
    // Connect FIFO read enables from accumulator to laggy prefix
    // In the current implementation of accumulator_correction, we don't have separate
    // read enables, so we'll use the same signal for both for now
    assign fifo_mp_read_en = ready_for_slow && !fifo_mp_empty;
    assign fifo_weight_read_en = ready_for_slow && !fifo_weight_empty;

endmodule
