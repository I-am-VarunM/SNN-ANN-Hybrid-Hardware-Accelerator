module fast_prefix_tb;
    parameter BITMASK_WIDTH = 8;
    parameter WEIGHT_WIDTH = 8;
    
    reg clk;
    reg rst;
    reg [BITMASK_WIDTH-1:0] and_result;
    reg [BITMASK_WIDTH-1:0] bitmask_b;
    reg valid_match;
    reg [WEIGHT_WIDTH-1:0] fibre_b_data [BITMASK_WIDTH-1:0];
    
    wire [$clog2(BITMASK_WIDTH)-1:0] fast_offset;
    wire [$clog2(BITMASK_WIDTH)-1:0] matched_position;
    wire [WEIGHT_WIDTH-1:0] matched_weight;
    wire fast_valid;
    wire processing_done;

    // Instantiate the fast_prefix module
    fast_prefix #(
        .BITMASK_WIDTH(BITMASK_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH)
    ) uut (
        .clk(clk),
        .rst(rst),
        .and_result(and_result),
        .bitmask_b(bitmask_b),
        .valid_match(valid_match),
        .fibre_b_data(fibre_b_data),
        .fast_offset(fast_offset),
        .matched_position(matched_position),
        .matched_weight(matched_weight),
        .fast_valid(fast_valid),
        .processing_done(processing_done)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Initialize fibre_b_data
    initial begin
        // Simple pattern: position+1 for each value
        fibre_b_data[0] = 8'd1;
        fibre_b_data[1] = 8'd2;
        fibre_b_data[2] = 8'd3;
        fibre_b_data[3] = 8'd4;
        fibre_b_data[4] = 8'd5;
        fibre_b_data[5] = 8'd6;
        fibre_b_data[6] = 8'd7;
        fibre_b_data[7] = 8'd8;
    end

    // Test stimulus
    initial begin
        $dumpfile("fast_prefix_tb.vcd");
        $dumpvars(0, fast_prefix_tb);
        
        // Initialize signals
        clk = 0;
        rst = 1;
        and_result = 0;
        bitmask_b = 0;
        valid_match = 0;
        
        // Apply reset for 3 clock cycles
        repeat(3) @(posedge clk);
        
        // Release reset
        rst = 0;
        @(posedge clk);
        
        // Single test case - exactly this AND pattern: 10101000
        and_result = 8'b10101000;  // 1s at positions 7,5,3
        bitmask_b  = 8'b10101100;  // 1s at positions 7,5,3,2
        valid_match = 1;
        
        $display("\n[TB] Starting test at time %0t", $time);
        $display("[TB] and_result = %b", and_result);
        $display("[TB] bitmask_b  = %b", bitmask_b);
        
        // Keep valid_match high for only one clock cycle
        @(posedge clk);
        valid_match = 0;
        
        // Wait for processing to complete or timeout after 15 cycles
        repeat(15) begin
            @(posedge clk);
            if (fast_valid)
                $display("[TB] Cycle output: position=%0d, offset=%0d, weight=%0d", 
                         matched_position, fast_offset, matched_weight);
        end
        
        $display("\n[TB] Test completed at time %0t", $time);
        $finish;
    end

    // Monitor key signals
    always @(posedge clk) begin
        $display("[TB] Time=%0t: processing_done=%b, fast_valid=%b, current_and=%b", 
                 $time, processing_done, fast_valid, uut.current_and_result);
    end

endmodule