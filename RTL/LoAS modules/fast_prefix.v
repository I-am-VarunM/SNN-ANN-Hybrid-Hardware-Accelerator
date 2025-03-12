module fast_prefix #(
    parameter BITMASK_WIDTH = 128,
    parameter WEIGHT_WIDTH = 8    
)(
    input wire clk,
    input wire rst,
    input wire [BITMASK_WIDTH-1:0] and_result,      
    input wire [BITMASK_WIDTH-1:0] bitmask_b,        
    input wire valid_match,
    input wire [WEIGHT_WIDTH-1:0] fibre_b_data [0:BITMASK_WIDTH-1], // Correct array indexing
    
    output reg [$clog2(BITMASK_WIDTH)-1:0] fast_offset,    
    output reg [$clog2(BITMASK_WIDTH)-1:0] matched_position,
    output reg [WEIGHT_WIDTH-1:0] matched_weight,          
    output reg fast_valid,
    output reg processing_done
);

    reg [BITMASK_WIDTH-1:0] current_and_result;
    reg [$clog2(BITMASK_WIDTH)-1:0] current_pos;
    
    // Find lowest '1' position in current_and_result
    function automatic [$clog2(BITMASK_WIDTH)-1:0] find_lowest_one;
        input [BITMASK_WIDTH-1:0] value;
        reg found;
        integer i;
        begin
            found = 0;
            find_lowest_one = 0;
            for(i = 0; i < BITMASK_WIDTH; i = i + 1) begin
                if(value[i] && !found) begin
                    find_lowest_one = i;
                    found = 1;
                end
            end
        end
    endfunction
    
    // Count ones in bitmask_b up to a position
    function automatic [$clog2(BITMASK_WIDTH)-1:0] count_ones_up_to;
        input [BITMASK_WIDTH-1:0] bitmask;
        input [$clog2(BITMASK_WIDTH)-1:0] position;
        integer i;
        begin
            count_ones_up_to = 0;
            for(i = 0; i <= position; i = i + 1) begin
                if(bitmask[i]) count_ones_up_to = count_ones_up_to + 1;
            end
        end
    endfunction

    // Find offset into compressed data array
    function automatic [$clog2(BITMASK_WIDTH)-1:0] get_offset;
        input [BITMASK_WIDTH-1:0] bitmask_b;
        input [$clog2(BITMASK_WIDTH)-1:0] position;
        begin
            // Count 1s in bitmask_b up to this position (inclusive)
            get_offset = count_ones_up_to(bitmask_b, position) - 1;
        end
    endfunction

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            current_and_result <= 0;
            current_pos <= 0;
            fast_offset <= 0;
            matched_position <= 0;
            matched_weight <= 0;
            fast_valid <= 0;
            processing_done <= 1;
            $display("Reset occurred");
        end
        else if(valid_match && processing_done) begin
            // Start new processing
            current_and_result <= and_result;
            processing_done <= 0;
            fast_valid <= 0;
            $display("Starting new processing with and_result=%b", and_result);
        end
        else if(!processing_done && current_and_result != 0) begin
            // Find next match position
            current_pos = find_lowest_one(current_and_result);
            
            // Calculate data
            matched_position <= current_pos;
            fast_offset <= get_offset(bitmask_b, current_pos);
            matched_weight <= fibre_b_data[current_pos];
            fast_valid <= 1;
            
            // Clear the processed bit
            current_and_result <= current_and_result & ~(1 << current_pos);
            
            $display("Processing match at position %0d, offset=%0d, weight=%0d", 
                    current_pos, get_offset(bitmask_b, current_pos), fibre_b_data[current_pos]);
        end
        else if(!processing_done && current_and_result == 0) begin
            // All matches processed
            processing_done <= 1;
            fast_valid <= 0;
            $display("Completed processing");
        end
    end

endmodule