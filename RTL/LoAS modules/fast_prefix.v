module fast_prefix #(
    parameter BITMASK_WIDTH = 128,
    parameter WEIGHT_WIDTH = 8
)(
    input wire clk,
    input wire rst,
    input wire [BITMASK_WIDTH-1:0] and_result,
    input wire [BITMASK_WIDTH-1:0] bitmask_b,
    input wire valid_match,
    input wire [BITMASK_WIDTH*WEIGHT_WIDTH-1:0] fibre_b_data_flat, // Flattened array input
    output reg [$clog2(BITMASK_WIDTH)-1:0] fast_offset,
    output reg [$clog2(BITMASK_WIDTH)-1:0] matched_position,
    output reg [WEIGHT_WIDTH-1:0] matched_weight,
    output reg fast_valid,
    output reg processing_done
);

    // Create local array to work with the flattened input
    wire [WEIGHT_WIDTH-1:0] fibre_b_data [0:BITMASK_WIDTH-1];
    
    // Convert flat input to array (for internal use)
    genvar i;
    generate
        for (i = 0; i < BITMASK_WIDTH; i = i + 1) begin : gen_array
            assign fibre_b_data[i] = fibre_b_data_flat[i*WEIGHT_WIDTH +: WEIGHT_WIDTH];
        end
    endgenerate

    reg [BITMASK_WIDTH-1:0] current_and_result;
    wire [$clog2(BITMASK_WIDTH)-1:0] lowest_one_position;
    wire [WEIGHT_WIDTH-1:0] selected_weight;
    wire [$clog2(BITMASK_WIDTH)-1:0] calculated_offset;
    
    // Priority encoder for finding lowest '1' position
    function [$clog2(BITMASK_WIDTH)-1:0] find_lowest_one;
        input [BITMASK_WIDTH-1:0] bit_vector;
        integer j;
        begin
            find_lowest_one = {$clog2(BITMASK_WIDTH){1'b1}}; // Default to all 1's if no bit is set
            for (j = 0; j < BITMASK_WIDTH; j = j + 1) begin
                if (bit_vector[j] && (find_lowest_one == {$clog2(BITMASK_WIDTH){1'b1}})) begin
                    find_lowest_one = j[$clog2(BITMASK_WIDTH)-1:0];
                end
            end
        end
    endfunction
    
    // Assign lowest_one_position using the function
    assign lowest_one_position = find_lowest_one(current_and_result);
    
    // Efficient parallel prefix sum implementation
    // Create prefix sum array - each element will hold count of ones up to that position
    reg [$clog2(BITMASK_WIDTH):0] prefix_sums [0:BITMASK_WIDTH-1];
    
    integer k, j, p, idx, power_val;
    
    // Calculate prefix sums using a parallel approach
    always @(*) begin
        // Initialize prefix sums with input bits
        for (k = 0; k < BITMASK_WIDTH; k = k + 1) begin
            prefix_sums[k] = (k == 0) ? bitmask_b[0] : 0;
        end
        
        // First pass - upsweep phase
        // Level 1: Pairs
        for (k = 0; k < BITMASK_WIDTH/2; k = k + 1) begin
            if (2*k+1 < BITMASK_WIDTH) begin
                prefix_sums[2*k+1] = bitmask_b[2*k] + bitmask_b[2*k+1];
            end
        end
        
        // Level 2: Groups of 4
        for (k = 0; k < BITMASK_WIDTH/4; k = k + 1) begin
            if (4*k+3 < BITMASK_WIDTH) begin
                prefix_sums[4*k+3] = prefix_sums[4*k+1] + bitmask_b[4*k+2] + bitmask_b[4*k+3];
            end
        end
        
        // Level 3: Groups of 8
        for (k = 0; k < BITMASK_WIDTH/8; k = k + 1) begin
            if (8*k+7 < BITMASK_WIDTH) begin
                prefix_sums[8*k+7] = prefix_sums[4*k+3] + bitmask_b[8*k+4] + bitmask_b[8*k+5] + 
                                   bitmask_b[8*k+6] + bitmask_b[8*k+7];
            end
        end
        
        // Level 4: Groups of 16
        for (k = 0; k < BITMASK_WIDTH/16; k = k + 1) begin
            if (16*k+15 < BITMASK_WIDTH) begin
                prefix_sums[16*k+15] = prefix_sums[8*k+7] + 
                                     bitmask_b[16*k+8] + bitmask_b[16*k+9] + 
                                     bitmask_b[16*k+10] + bitmask_b[16*k+11] + 
                                     bitmask_b[16*k+12] + bitmask_b[16*k+13] + 
                                     bitmask_b[16*k+14] + bitmask_b[16*k+15];
            end
        end
        
        // Continue with higher levels based on BITMASK_WIDTH
        // Level 5: Groups of 32
        if (BITMASK_WIDTH >= 32) begin
            for (k = 0; k < BITMASK_WIDTH/32; k = k + 1) begin
                if (32*k+31 < BITMASK_WIDTH) begin
                    prefix_sums[32*k+31] = prefix_sums[16*k+15];
                    for (j = 16; j < 32; j = j + 1) begin
                        if (32*k+j < BITMASK_WIDTH) begin
                            prefix_sums[32*k+31] = prefix_sums[32*k+31] + bitmask_b[32*k+j];
                        end
                    end
                end
            end
        end
        
        // Level 6: Groups of 64
        if (BITMASK_WIDTH >= 64) begin
            for (k = 0; k < BITMASK_WIDTH/64; k = k + 1) begin
                if (64*k+63 < BITMASK_WIDTH) begin
                    prefix_sums[64*k+63] = prefix_sums[32*k+31];
                    for (j = 32; j < 64; j = j + 1) begin
                        if (64*k+j < BITMASK_WIDTH) begin
                            prefix_sums[64*k+63] = prefix_sums[64*k+63] + bitmask_b[64*k+j];
                        end
                    end
                end
            end
        end
        
        // Level 7: Full width (for BITMASK_WIDTH = 128)
        if (BITMASK_WIDTH >= 128) begin
            prefix_sums[127] = prefix_sums[63];
            for (k = 64; k < 128; k = k + 1) begin
                if (k < BITMASK_WIDTH) begin
                    prefix_sums[127] = prefix_sums[127] + bitmask_b[k];
                end
            end
        end
        
        // Second pass - downsweep to fill in all intermediate prefix sums
        for (k = 1; k < BITMASK_WIDTH; k = k + 1) begin
            if (!((k & (k+1)) == 0)) begin  // If k is not a power of 2 minus 1
                // Find the largest power of 2 that divides k+1
                // Using synthesis-friendly bounded logic
                p = 1;
                idx = k;
                
                // Initialize p to 1 (2^0)
                p = 1;
                
                // Calculate the largest power of 2 that is a factor of (k+1)
                // This avoids both while loops and break statements
                for (j = 0; j < $clog2(BITMASK_WIDTH); j = j + 1) begin
                    // If bit j is set in idx+1, update p and exit the effective loop
                    // by setting j to $clog2(BITMASK_WIDTH)
                    if (((idx+1) & (1 << j)) != 0) begin
                        p = (1 << j); // 2^j
                        j = $clog2(BITMASK_WIDTH) - 1; // This forces the loop to end after this iteration
                    end
                end
                
                idx = k - p;
                
                // Update the prefix sum
                prefix_sums[k] = prefix_sums[idx];
                for (j = idx+1; j <= k; j = j + 1) begin
                    prefix_sums[k] = prefix_sums[k] + bitmask_b[j];
                end
            end
        end
    end
    
    // Calculate the offset based on the match position
    // The offset is the number of 1's before the matched position
    wire [$clog2(BITMASK_WIDTH):0] ones_before_position;
    
    // If position is 0, offset is 0
    // Otherwise, it's the prefix sum at position-1
    assign ones_before_position = (lowest_one_position == 0) ? 0 : 
                                 prefix_sums[lowest_one_position-1];
    
    assign calculated_offset = ones_before_position[$clog2(BITMASK_WIDTH)-1:0];
    
    // Use direct approach for weight selection
    assign selected_weight = fibre_b_data[calculated_offset];
    
    // Sequential logic - FSM remains the same as original
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            current_and_result <= 0;
            fast_offset <= 0;
            matched_position <= 0;
            matched_weight <= 0;
            fast_valid <= 0;
            processing_done <= 1;
        end
        else if(valid_match && processing_done) begin
            // Start new processing
            current_and_result <= and_result;
            processing_done <= 0;
            fast_valid <= 0;
        end
        else if(!processing_done && current_and_result != 0) begin
            // Store results using the values calculated by parallel prefix
            matched_position <= lowest_one_position;
            fast_offset <= calculated_offset;
            matched_weight <= selected_weight;
            fast_valid <= 1;
            // Clear the processed bit in a more synthesis-friendly way
            current_and_result <= current_and_result & ~(1'b1 << lowest_one_position);
        end
        else if(!processing_done && current_and_result == 0) begin
            // All matches processed
            processing_done <= 1;
            fast_valid <= 0;
        end
    end
endmodule
