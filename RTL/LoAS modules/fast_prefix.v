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
    reg [BITMASK_WIDTH-1:0] ones_before_position;
    
    // Priority encoder for finding lowest '1' position
    // This replaces the non-synthesizable for loop with procedural break
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
    
    // Use a more direct approach for weight selection
    assign selected_weight = fibre_b_data[calculated_offset];
    
    // Calculate ones count in bitmask_b up to position
    // This uses a more synthesizable approach
    always @(*) begin
        ones_before_position = bitmask_b & ((1'b1 << (lowest_one_position + 1)) - 1);
    end
    
    // Count the ones in the masked bitmask
    function [$clog2(BITMASK_WIDTH)-1:0] count_ones;
        input [BITMASK_WIDTH-1:0] bit_vector;
        integer j;
        reg [$clog2(BITMASK_WIDTH)-1:0] count;
        begin
            count = 0;
            for (j = 0; j < BITMASK_WIDTH; j = j + 1) begin
                if (bit_vector[j]) count = count + 1;
            end
            count_ones = count;
        end
    endfunction
    
    // Calculate offset
    wire [$clog2(BITMASK_WIDTH)-1:0] ones_count;
    assign ones_count = count_ones(ones_before_position);
    assign calculated_offset = (ones_count > 0) ? (ones_count - 1) : 0;
    
    // Sequential logic
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
            // Store results using the values calculated by priority encoder
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
