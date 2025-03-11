module SparTen_ComputeUnit #(
    parameter CHUNK_SIZE = 128,         // Size of SparseMap in bits
    parameter DATA_WIDTH = 8,           // Width of each data value
    parameter ACCUM_WIDTH = 32,         // Width of accumulator
    parameter FIFO_DEPTH = 16           // Depth of feeders (FIFOs)
)(
    input wire clk,
    input wire rst_n,
    
    // Control signals
    input wire enable,                  // Enable compute unit
    input wire load_filter,             // Signal to load filter data
    input wire load_feature,            // Signal to load feature map data
    input wire start_compute,           // Start computation
    output reg done_compute,            // Computation complete
    
    // Input bus connections
    input wire [CHUNK_SIZE-1:0] filter_sparse_map_in,      // Filter SparseMap from bus
    input wire [DATA_WIDTH*CHUNK_SIZE-1:0] filter_values_in, // Filter values from bus
    input wire [CHUNK_SIZE-1:0] feature_sparse_map_in,     // Feature map SparseMap from bus
    input wire [DATA_WIDTH*CHUNK_SIZE-1:0] feature_values_in, // Feature map values from bus
    
    // Output
    output reg [ACCUM_WIDTH-1:0] partial_sum,              // Accumulated partial sum
    output reg partial_sum_valid                           // Partial sum valid signal
);

    // Input buffers
    reg [CHUNK_SIZE-1:0] filter_sparse_map;
    reg [DATA_WIDTH*CHUNK_SIZE-1:0] filter_values;
    reg [CHUNK_SIZE-1:0] feature_sparse_map;
    reg [DATA_WIDTH*CHUNK_SIZE-1:0] feature_values;
    
    // Inner join computation pipeline
    reg [CHUNK_SIZE-1:0] and_result;                       // AND of SparseMaps
    reg [CHUNK_SIZE-1:0] working_and_result;               // Current working AND result
    wire [$clog2(CHUNK_SIZE)-1:0] match_position;          // Current match position
    wire match_valid;                                       // Valid match found
    
    // Parallel prefix sum signals - MODIFIED: Now using packed arrays
    wire [(CHUNK_SIZE*($clog2(CHUNK_SIZE)+1))-1:0] filter_prefix_sum_packed;
    wire [(CHUNK_SIZE*($clog2(CHUNK_SIZE)+1))-1:0] feature_prefix_sum_packed;
    
    reg [$clog2(CHUNK_SIZE):0] filter_offset;              // Filter data offset
    reg [$clog2(CHUNK_SIZE):0] feature_offset;             // Feature data offset
    
    // Control signals
    reg inner_join_valid;                                   // Inner join stage valid
    reg offset_calc_valid;                                  // Offset calculation valid
    
    // FIFO structures for matching pairs
    reg [DATA_WIDTH-1:0] filter_fifo [0:FIFO_DEPTH-1];
    reg [DATA_WIDTH-1:0] feature_fifo [0:FIFO_DEPTH-1];
    reg [$clog2(FIFO_DEPTH):0] fifo_count;
    reg [$clog2(FIFO_DEPTH)-1:0] fifo_wr_ptr;
    reg [$clog2(FIFO_DEPTH)-1:0] fifo_rd_ptr;
    wire fifo_empty;
    wire fifo_full;
    
    // MAC unit signals
    reg mac_valid;                                          // Valid input for MAC
    reg [DATA_WIDTH-1:0] mac_filter_value;                 // Filter value for MAC
    reg [DATA_WIDTH-1:0] mac_feature_value;                // Feature value for MAC
    reg [DATA_WIDTH*2-1:0] product;                        // Multiplication result
    reg mac_processing;                                     // MAC is processing
    reg all_processing_done;                                // All processing complete flag
    
    // State machine
    localparam IDLE = 2'b00;
    localparam PROCESS_MATCHES = 2'b01;
    localparam ACCUMULATE = 2'b10;
    localparam DONE = 2'b11;
    
    reg [1:0] state, next_state;
    
    // Helper function to extract prefix sum value from packed array
    function [$clog2(CHUNK_SIZE):0] get_prefix_sum;
        input [(CHUNK_SIZE*($clog2(CHUNK_SIZE)+1))-1:0] packed_array;
        input [$clog2(CHUNK_SIZE)-1:0] index;
        begin
            get_prefix_sum = packed_array[index*($clog2(CHUNK_SIZE)+1) +: ($clog2(CHUNK_SIZE)+1)];
        end
    endfunction
    
    // Assign FIFO status
    assign fifo_empty = (fifo_count == 0);
    assign fifo_full = (fifo_count == FIFO_DEPTH);
    
    // Priority encoder instantiation for finding next match efficiently
    PriorityEncoder #(
        .WIDTH(CHUNK_SIZE)
    ) priority_encoder (
        .in(working_and_result),
        .out(match_position),
        .valid(match_valid)
    );
    
    // Parallel prefix sum for filter and feature maps - MODIFIED: Now using packed arrays
    ParallelPrefixSum #(
        .WIDTH(CHUNK_SIZE)
    ) filter_prefix_sum_inst (
        .bit_array(filter_sparse_map),
        .prefix_sums_packed(filter_prefix_sum_packed)
    );
    
    ParallelPrefixSum #(
        .WIDTH(CHUNK_SIZE)
    ) feature_prefix_sum_inst (
        .bit_array(feature_sparse_map),
        .prefix_sums_packed(feature_prefix_sum_packed)
    );
    
    // State transition logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
            
            // DEBUG: State transitions
            if (state != next_state) begin
                $display("DEBUG: State transition from %d to %d at time %t", state, next_state, $time);
            end
        end
    end
    
    // Next state logic
    always @(*) begin
        case (state)
            IDLE: begin
                if (start_compute && enable)
                    next_state = PROCESS_MATCHES;
                else
                    next_state = IDLE;
            end
            
            PROCESS_MATCHES: begin
                // Only transition to DONE when both conditions are met:
                // 1. No more matches to find (AND result is empty)
                // 2. FIFO is completely empty (all matches processed)
                // 3. No MAC operation in progress
                if (!match_valid && fifo_empty && !mac_processing)
                    next_state = DONE;
                else
                    next_state = PROCESS_MATCHES;
            end
            
            DONE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    
    // Load input data
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            filter_sparse_map <= 0;
            filter_values <= 0;
            feature_sparse_map <= 0;
            feature_values <= 0;
        end else begin
            if (load_filter) begin
                filter_sparse_map <= filter_sparse_map_in;
                filter_values <= filter_values_in;
                
                // DEBUG: Display filter data being loaded
                $display("DEBUG: Loading filter - SparseMap: %b", filter_sparse_map_in);
                $display("DEBUG: Filter values[0]: %d", filter_values_in[0 +: DATA_WIDTH]);
                $display("DEBUG: Filter values[1]: %d", filter_values_in[DATA_WIDTH +: DATA_WIDTH]);
                $display("DEBUG: Filter values[2]: %d", filter_values_in[2*DATA_WIDTH +: DATA_WIDTH]);
            end
            
            if (load_feature) begin
                feature_sparse_map <= feature_sparse_map_in;
                feature_values <= feature_values_in;
                
                // DEBUG: Display feature data being loaded
                $display("DEBUG: Loading feature - SparseMap: %b", feature_sparse_map_in);
                $display("DEBUG: Feature values[0]: %d", feature_values_in[0 +: DATA_WIDTH]);
                $display("DEBUG: Feature values[1]: %d", feature_values_in[DATA_WIDTH +: DATA_WIDTH]);
                $display("DEBUG: Feature values[2]: %d", feature_values_in[2*DATA_WIDTH +: DATA_WIDTH]);
            end
        end
    end
    
    // Inner join pipeline - Stage 1: AND operation and match finding
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            and_result <= 0;
            working_and_result <= 0;
            inner_join_valid <= 1'b0;
            all_processing_done <= 1'b0;
            done_compute <= 1'b0;
            partial_sum_valid <= 1'b0;
            partial_sum <= 0;
        end else if (enable) begin
            case (state)
                IDLE: begin
                    if (start_compute) begin
                        // Initialize for a new computation
                        and_result <= filter_sparse_map & feature_sparse_map;
                        working_and_result <= filter_sparse_map & feature_sparse_map;
                        inner_join_valid <= 1'b1;
                        all_processing_done <= 1'b0;
                        done_compute <= 1'b0;
                        partial_sum_valid <= 1'b0;
                        partial_sum <= 0;
                        
                        // DEBUG: Show AND result
                        $display("DEBUG: AND result = %b", filter_sparse_map & feature_sparse_map);
                    end
                end
                
                DONE: begin
                    all_processing_done <= 1'b1;
                    partial_sum_valid <= 1'b1;
                    done_compute <= 1'b1;
                    inner_join_valid <= 1'b0;
                    
                    // DEBUG: Show final partial sum
                    $display("DEBUG: Computation done. Final partial sum = %d", partial_sum);
                end
            endcase
        end
    end
    
    // Inner join pipeline - Stage 2: Offset calculation using parallel prefix sum - MODIFIED
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            filter_offset <= 0;
            feature_offset <= 0;
            offset_calc_valid <= 1'b0;
        end else if (enable && state == PROCESS_MATCHES && match_valid && !fifo_full) begin
            // Get the prefix sum values using helper function - MODIFIED
            filter_offset <= get_prefix_sum(filter_prefix_sum_packed, match_position);
            feature_offset <= get_prefix_sum(feature_prefix_sum_packed, match_position);
            offset_calc_valid <= 1'b1;
            
            // Also clear the current match bit in working_and_result
            // This prevents double processing of the same match
            working_and_result[match_position] <= 1'b0;
            
            // DEBUG: Show calculated offsets
            $display("DEBUG: Match at position %d: Filter offset = %d, Feature offset = %d", 
                    match_position, get_prefix_sum(filter_prefix_sum_packed, match_position), 
                    get_prefix_sum(feature_prefix_sum_packed, match_position));
            $display("DEBUG: Processing match at position %d", match_position);
        end else begin
            offset_calc_valid <= 1'b0;
        end
    end
    
 // FIFO management - completely revised for robustness
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Properly initialize all FIFO-related registers
        fifo_count <= 0;
        fifo_wr_ptr <= 0;
        fifo_rd_ptr <= 0;
        mac_valid <= 1'b0;
        mac_filter_value <= 0;
        mac_feature_value <= 0;
        
        // Initialize FIFO memory to prevent 'X' values
        for (integer i = 0; i < FIFO_DEPTH; i = i + 1) begin
            filter_fifo[i] <= 0;
            feature_fifo[i] <= 0;
        end
    end else if (enable) begin
        // Reset mac_valid at the beginning of each cycle
        mac_valid <= 1'b0;
        
        // FIRST: Process read from FIFO if there are entries and we're in the right state
        if ((fifo_count > 0) && !mac_processing && (state == PROCESS_MATCHES)) begin
            // Read values from FIFO for MAC operation
            mac_filter_value <= filter_fifo[fifo_rd_ptr];
            mac_feature_value <= feature_fifo[fifo_rd_ptr];
            
            // DEBUG: Show values being read from FIFO
            $display("DEBUG: Reading from FIFO[%0d]: Filter value = %0d, Feature value = %0d, Count = %0d", 
                     fifo_rd_ptr, filter_fifo[fifo_rd_ptr], feature_fifo[fifo_rd_ptr], fifo_count);
            
            // Update read pointer and count
            fifo_rd_ptr <= (fifo_rd_ptr + 1) % FIFO_DEPTH;
            fifo_count <= fifo_count - 1;
            mac_valid <= 1'b1;
        end
        
        // SECOND: Process write to FIFO if there's a valid offset calculation
        if (offset_calc_valid && (fifo_count < FIFO_DEPTH)) begin
            // Extract values based on offsets
            filter_fifo[fifo_wr_ptr] <= extract_value(filter_values, filter_offset);
            feature_fifo[fifo_wr_ptr] <= extract_value(feature_values, feature_offset);
            
            // DEBUG: Show extracted values being added to FIFO
            $display("DEBUG: Adding to FIFO[%0d]: Filter value = %0d, Feature value = %0d, Count = %0d", 
                     fifo_wr_ptr,
                     extract_value(filter_values, filter_offset), 
                     extract_value(feature_values, feature_offset),
                     fifo_count);
            
            // Update write pointer and count
            fifo_wr_ptr <= (fifo_wr_ptr + 1) % FIFO_DEPTH;
            fifo_count <= fifo_count + 1;
        end
        
        // Print FIFO contents for debugging
        print_fifo_contents();
    end
end

// Improved FIFO contents debug printing
task print_fifo_contents;
    integer i;
    begin
        $display("DEBUG: FIFO Contents (count=%0d, wr_ptr=%0d, rd_ptr=%0d):", 
                 fifo_count, fifo_wr_ptr, fifo_rd_ptr);
        
        // Only print valid entries based on current count
        if (fifo_count > 0) begin
            for (i = 0; i < fifo_count; i = i + 1) begin
                $display("    FIFO[%0d]: Filter=%0d, Feature=%0d %s", 
                         (fifo_rd_ptr + i) % FIFO_DEPTH, 
                         filter_fifo[(fifo_rd_ptr + i) % FIFO_DEPTH], 
                         feature_fifo[(fifo_rd_ptr + i) % FIFO_DEPTH],
                         (i == 0) ? "<-- Next to Read" : "");
            end
        end
    end
endtask

// MAC operation - improved to handle edge cases
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        partial_sum <= 0;
        mac_processing <= 1'b0;
        product <= 0;
    end else if (enable) begin
        if (state == IDLE && start_compute) begin
            // Reset accumulator for new computation
            partial_sum <= 0;
            $display("DEBUG: Resetting partial sum to 0");
        end
        
        // MAC operation when valid data is available AND in PROCESS_MATCHES state
        if (mac_valid) begin
            mac_processing <= 1'b1;
            
            // Calculate product and update partial sum
            product <= mac_filter_value * mac_feature_value;
            partial_sum <= partial_sum + (mac_filter_value * mac_feature_value);
            
            // DEBUG: Show MAC operation
            $display("DEBUG: MAC operation: %0d * %0d = %0d, Accumulating to %0d", 
                     mac_filter_value, mac_feature_value, 
                     mac_filter_value * mac_feature_value,
                     partial_sum + (mac_filter_value * mac_feature_value));
                     
            mac_processing <= 1'b0;
        end
    end
end

// Extract value function - adjusted to match your data layout
function [DATA_WIDTH-1:0] extract_value;
    input [DATA_WIDTH*CHUNK_SIZE-1:0] values;
    input [$clog2(CHUNK_SIZE):0] index;
    begin
        // For values packed at LSB (beginning of array)
        // Adjust index to ensure proper mapping
        extract_value = values[(index-1)*DATA_WIDTH +: DATA_WIDTH];
    end
endfunction
    
endmodule

// Improved Priority Encoder Module
module PriorityEncoder #(
    parameter WIDTH = 128
)(
    input wire [WIDTH-1:0] in,
    output reg [$clog2(WIDTH)-1:0] out,
    output reg valid
);
    integer i;
    
    always @(*) begin
        valid = 1'b0;
        out = 0;
        
        for (i = 0; i < WIDTH; i = i + 1) begin
            if (in[i] && !valid) begin
                out = i;
                valid = 1'b1;
            end
        end
    end
endmodule

// Parallel Prefix Sum Module - MODIFIED to use packed arrays
module ParallelPrefixSum #(
    parameter WIDTH = 128
)(
    input wire [WIDTH-1:0] bit_array,
    output wire [(WIDTH*($clog2(WIDTH)+1))-1:0] prefix_sums_packed
);
    // First stage: Initialize with the bit values (0 or 1)
    wire [$clog2(WIDTH):0] stage0 [WIDTH-1:0];
    
    // Parameters for log2 calculation must be a constant at compile time
    localparam LOG2_WIDTH = $clog2(WIDTH);
    
    // Intermediate stages for parallel prefix sum calculation
    wire [$clog2(WIDTH):0] stages [0:LOG2_WIDTH][WIDTH-1:0];
    
    genvar i, j, k;
    
    // Stage 0: Initialize with the bit values (0 or 1)
    generate
        for (i = 0; i < WIDTH; i = i + 1) begin: stage0_gen
            assign stage0[i] = bit_array[i] ? 1 : 0;
        end
    endgenerate
    
    generate
        // Copy stage0 to the first stage of the computation
        for (i = 0; i < WIDTH; i = i + 1) begin: stage0_copy
            assign stages[0][i] = stage0[i];
        end
        
        // Build each subsequent stage using a parallel prefix algorithm
        for (j = 1; j <= LOG2_WIDTH; j = j + 1) begin: stages_gen
            for (k = 0; k < WIDTH; k = k + 1) begin: each_element
                if (k >= (1 << (j-1))) begin
                    // Add previous element's value that's 2^(j-1) positions away
                    assign stages[j][k] = stages[j-1][k] + stages[j-1][k - (1 << (j-1))];
                end else begin
                    // Keep the same value for elements that don't have enough preceding elements
                    assign stages[j][k] = stages[j-1][k];
                end
            end
        end
        
        // Assign the final prefix sums to a packed array
        for (i = 0; i < WIDTH; i = i + 1) begin: output_assign
            assign prefix_sums_packed[i*($clog2(WIDTH)+1) +: ($clog2(WIDTH)+1)] = stages[LOG2_WIDTH][i];
        end
    endgenerate
endmodule
