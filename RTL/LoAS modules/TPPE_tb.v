module tppe_tb;
    // Parameters
    parameter BITMASK_WIDTH = 16;  // Using smaller width for easier testing
    parameter WEIGHT_WIDTH = 8;
    parameter NUM_ADDERS = 16;
    parameter TIMESTEPS = 4;
    parameter FIFO_DEPTH = 8;
    parameter PSEUDO_ACC_WIDTH = 12;
    parameter CORRECTION_ACC_WIDTH = 10;
    parameter ADDR_WIDTH = 8;
    
    // Clock and reset
    reg clk;
    reg rst;
    
    // Inputs
    reg [BITMASK_WIDTH-1:0] bitmask_a;
    reg [BITMASK_WIDTH-1:0] bitmask_b;
    reg [WEIGHT_WIDTH-1:0] nonzero_weights [0:BITMASK_WIDTH-1];
    reg valid_input;
    
    // Memory interface
    wire [ADDR_WIDTH-1:0] fibre_a_addr;
    wire fibre_a_read_en;
    reg [TIMESTEPS-1:0] fibre_a_data;
    reg fibre_a_valid;
    
    // Outputs
    wire [CORRECTION_ACC_WIDTH-1:0] result_0;
    wire [CORRECTION_ACC_WIDTH-1:0] result_1;
    wire [CORRECTION_ACC_WIDTH-1:0] result_2;
    wire [CORRECTION_ACC_WIDTH-1:0] result_3;
    wire result_valid;
    wire ready_for_input;
    
    // Test control
    integer i;
    integer cycle_count;
    
    // Memory array for simulation
    reg [TIMESTEPS-1:0] memory [0:255];
    
    // Instantiate the TPPE module
    tppe #(
        .BITMASK_WIDTH(BITMASK_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .NUM_ADDERS(NUM_ADDERS),
        .TIMESTEPS(TIMESTEPS),
        .FIFO_DEPTH(FIFO_DEPTH),
        .PSEUDO_ACC_WIDTH(PSEUDO_ACC_WIDTH),
        .CORRECTION_ACC_WIDTH(CORRECTION_ACC_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) uut (
        .clk(clk),
        .rst(rst),
        .bitmask_a(bitmask_a),
        .bitmask_b(bitmask_b),
        .nonzero_weights(nonzero_weights),
        .valid_input(valid_input),
        .fibre_a_addr(fibre_a_addr),
        .fibre_a_read_en(fibre_a_read_en),
        .fibre_a_data(fibre_a_data),
        .fibre_a_valid(fibre_a_valid),
        .result_0(result_0),
        .result_1(result_1),
        .result_2(result_2),
        .result_3(result_3),
        .result_valid(result_valid),
        .ready_for_input(ready_for_input)
    );
    
    // Clock generation
    always #5 clk = ~clk;
    
    // Memory response simulation
    always @(posedge clk) begin
        if (fibre_a_read_en) begin
            #10; // Memory latency
            fibre_a_data <= memory[fibre_a_addr];
            fibre_a_valid <= 1;
            $display("[MEMORY] Reading addr %0d, data=%b", fibre_a_addr, memory[fibre_a_addr]);
        end
        else begin
            fibre_a_valid <= 0;
        end
    end
    
    // Monitor cycles and results
    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        
        if (result_valid) begin
            $display("=== RESULT READY (Cycle %0d) ===", cycle_count);
            $display("Timestep 0: %0d", result_0);
            $display("Timestep 1: %0d", result_1);
            $display("Timestep 2: %0d", result_2);
            $display("Timestep 3: %0d", result_3);
        end
    end
    
    // Test sequence
    initial begin
        $dumpfile("tppe_tb.vcd");
        $dumpvars(0, tppe_tb);
        
        // Initialize memory
        memory[5] = 4'b1010;   // Timesteps 0,2 active
        memory[8] = 4'b1111;   // All timesteps active
        memory[12] = 4'b0011;  // Timesteps 0,1 active
        
        // Initialize test control
        cycle_count = 0;
        
        // Initialize signals
        clk = 0;
        rst = 1;
        bitmask_a = 0;
        bitmask_b = 0;
        valid_input = 0;
        fibre_a_data = 0;
        fibre_a_valid = 0;
        
        // Initialize weights
        for (i = 0; i < BITMASK_WIDTH; i = i + 1) begin
            nonzero_weights[i] = i + 1;  // Simple pattern: 1,2,3,...
        end
        
        // Apply reset
        repeat(3) @(posedge clk);
        rst = 0;
        @(posedge clk);
        
        $display("\n=== TEST CASE 1: Simple pattern ===");
        
        // Wait for ready
        wait(ready_for_input);
        @(posedge clk);
        
        // Set input patterns
        bitmask_a = 16'b0000000000101100;  // 1s at positions 2,3,5
        bitmask_b = 16'b0000000000100100;  // 1s at positions 2,5
        valid_input = 1;
        
        $display("Input bitmasks:");
        $display("bitmask_a = %b", bitmask_a);
        $display("bitmask_b = %b", bitmask_b);
        $display("AND result = %b", bitmask_a & bitmask_b);
        
        @(posedge clk);
        valid_input = 0;
        
        // Wait for completion or timeout
        fork : wait_block
            begin
                repeat(500) @(posedge clk);  // Long timeout
                $display("\nTIMEOUT: Test did not complete in time");
                disable wait_block;
            end
            begin
                wait(result_valid);
                $display("\nTest completed successfully with results:");
                disable wait_block;
            end
        join
        
        // Add some delay after completion
        repeat(20) @(posedge clk);
        
        $display("\n=== Simulation Complete ===");
        $finish;
    end

endmodule