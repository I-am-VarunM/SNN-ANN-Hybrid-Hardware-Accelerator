module laggy_prefix_tb;
    // Parameters
    parameter BITMASK_WIDTH = 16;  // Using 16 bits for easier testing
    parameter NUM_ADDERS = 16;
    parameter WEIGHT_WIDTH = 8;
    parameter FIFO_DEPTH = 8;
    
    // Signals
    reg clk;
    reg rst;
    reg [BITMASK_WIDTH-1:0] and_result;
    reg [BITMASK_WIDTH-1:0] bitmask_a;
    reg [$clog2(BITMASK_WIDTH)-1:0] matched_position;
    reg [WEIGHT_WIDTH-1:0] matched_weight;
    reg valid_match;
    
    wire [$clog2(BITMASK_WIDTH)-1:0] slow_offset;
    wire slow_valid;
    wire ready_for_new_calc;
    wire fifo_empty;
    wire fifo_full;

    // Instantiate the laggy_prefix module
    laggy_prefix #(
        .BITMASK_WIDTH(BITMASK_WIDTH),
        .NUM_ADDERS(NUM_ADDERS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) uut (
        .clk(clk),
        .rst(rst),
        .and_result(and_result),
        .bitmask_a(bitmask_a),
        .matched_position(matched_position),
        .matched_weight(matched_weight),
        .valid_match(valid_match),
        .slow_offset(slow_offset),
        .slow_valid(slow_valid),
        .ready_for_new_calc(ready_for_new_calc),
        .fifo_empty(fifo_empty),
        .fifo_full(fifo_full)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Test stimulus
    initial begin
        $dumpfile("laggy_prefix_tb.vcd");
        $dumpvars(0, laggy_prefix_tb);
        
        // Initialize signals
        clk = 0;
        rst = 1;
        and_result = 0;
        bitmask_a = 0;
        matched_position = 0;
        matched_weight = 0;
        valid_match = 0;
        
        // Apply reset for 3 clock cycles
        repeat(3) @(posedge clk);
        rst = 0;
        @(posedge clk);
        
        // Set input pattern
        bitmask_a = 16'b0000000000101100;   // 1s at positions 2,3,5
        
        // Push first match to FIFO
        matched_position = 5;
        matched_weight = 8'h5A;
        valid_match = 1;
        @(posedge clk);
        valid_match = 0;
        
        $display("\nPushed first matched position (5) at time %0t", $time);
        
        // Wait for result to be ready
        wait(slow_valid);
        $display("\nFirst result ready at time %0t", $time);
        $display("Position: %0d, Offset: %0d", uut.current_position, slow_offset);
        
        // Push second match to FIFO after waiting a few cycles
        repeat(3) @(posedge clk);
        matched_position = 3;
        matched_weight = 8'h3B;
        valid_match = 1;
        @(posedge clk);
        valid_match = 0;
        
        $display("\nPushed second matched position (3) at time %0t", $time);
        
        // Wait for second result to be ready
        wait(slow_valid);
        $display("\nSecond result ready at time %0t", $time);
        $display("Position: %0d, Offset: %0d", uut.current_position, slow_offset);
        
        // Push third match to FIFO
        repeat(3) @(posedge clk);
        matched_position = 2;
        matched_weight = 8'h2C;
        valid_match = 1;
        @(posedge clk);
        valid_match = 0;
        
        $display("\nPushed third matched position (2) at time %0t", $time);
        
        // Wait for third result to be ready
        wait(slow_valid);
        $display("\nThird result ready at time %0t", $time);
        $display("Position: %0d, Offset: %0d", uut.current_position, slow_offset);
        
        // Run a few more cycles and finish
        repeat(10) @(posedge clk);
        
        $display("\nSimulation complete at Time=%0t", $time);
        $finish;
    end

    // Monitor FIFO and processing status
    always @(posedge clk) begin
        if (uut.fifo_read_en)
            $display("FIFO read at time %0t, position = %0d", $time, uut.fifo_mp_out);
            
        if (valid_match)
            $display("Writing to FIFO at time %0t, position = %0d", $time, matched_position);
            
        if (slow_valid)
            $display("Slow valid at time %0t, offset = %0d", $time, slow_offset);
    end

endmodule