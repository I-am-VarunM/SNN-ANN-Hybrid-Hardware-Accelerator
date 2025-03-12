// Comprehensive testbench for accumulator correction module
// Compatible with older Verilog standards
module accumulator_correction_tb;
    // Parameters
    parameter TIMESTEPS = 4;
    parameter WEIGHT_WIDTH = 8;
    parameter PSEUDO_ACC_WIDTH = 12;
    parameter CORRECTION_ACC_WIDTH = 10;
    parameter ADDR_WIDTH = 8;
    
    // State definitions (for easier debugging)
    parameter CORR_IDLE = 0;
    parameter WAITING_FOR_FIFO = 1;
    parameter WAITING_FOR_DATA = 2;
    parameter CORRECTION = 3;
    parameter COMPLETE = 4;
    
    // Clock and reset
    reg clk;
    reg rst;
    
    // From fast_prefix
    reg [$clog2(128)-1:0] matched_position;
    reg [WEIGHT_WIDTH-1:0] matched_weight;
    reg fast_valid;
    
    // From laggy_prefix
    reg [$clog2(128)-1:0] current_position;
    reg [WEIGHT_WIDTH-1:0] current_weight;
    reg [$clog2(128)-1:0] slow_offset;
    reg slow_valid;
    reg fifo_empty;
    
    // Fibre A memory interface
    reg [TIMESTEPS-1:0] fibre_a_data;
    reg fibre_a_valid;
    wire [ADDR_WIDTH-1:0] fibre_a_addr;
    wire fibre_a_read_en;
    
    // Control signals
    wire ready_for_fast;
    wire ready_for_slow;
    wire fifo_read_req;
    
    // Results
    wire [CORRECTION_ACC_WIDTH-1:0] result_0;
    wire [CORRECTION_ACC_WIDTH-1:0] result_1;
    wire [CORRECTION_ACC_WIDTH-1:0] result_2;
    wire [CORRECTION_ACC_WIDTH-1:0] result_3;
    wire result_valid;
    
    // Test control variables
    integer cycle_count;
    integer timeout;
    integer test_case;
    reg test_done;
    
    // Test patterns - different fibre_a_data and weights to test
    reg [TIMESTEPS-1:0] test_patterns [0:3];
    reg [WEIGHT_WIDTH-1:0] test_weights [0:3];

    // Instantiate the module under test
    accumulator_correction #(
        .TIMESTEPS(TIMESTEPS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .PSEUDO_ACC_WIDTH(PSEUDO_ACC_WIDTH),
        .CORRECTION_ACC_WIDTH(CORRECTION_ACC_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) uut (
        .clk(clk),
        .rst(rst),
        .matched_position(matched_position),
        .matched_weight(matched_weight),
        .fast_valid(fast_valid),
        .current_position(current_position),
        .current_weight(current_weight),
        .slow_offset(slow_offset),
        .slow_valid(slow_valid),
        .fifo_empty(fifo_empty),
        .fibre_a_data(fibre_a_data),
        .fibre_a_valid(fibre_a_valid),
        .fibre_a_addr(fibre_a_addr),
        .fibre_a_read_en(fibre_a_read_en),
        .ready_for_fast(ready_for_fast),
        .ready_for_slow(ready_for_slow),
        .fifo_read_req(fifo_read_req),
        .result_0(result_0),
        .result_1(result_1),
        .result_2(result_2),
        .result_3(result_3),
        .result_valid(result_valid)
    );

    // Clock generation
    always #5 clk = ~clk;
    
    // Display state in human-readable format
    function [8*20:1] state_to_string;
        input [2:0] state;
        begin
            case(state)
                CORR_IDLE: state_to_string = "IDLE";
                WAITING_FOR_FIFO: state_to_string = "WAITING_FOR_FIFO";
                WAITING_FOR_DATA: state_to_string = "WAITING_FOR_DATA";
                CORRECTION: state_to_string = "CORRECTION";
                COMPLETE: state_to_string = "COMPLETE";
                default: state_to_string = "UNKNOWN";
            endcase
        end
    endfunction
    
    // Debug monitor to track state and signals
    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        
        // Print current cycle information
        $display("Cycle %0d: State=%s, fibre_a_valid=%b, fifo_read_req=%b, result_valid=%b",
                 cycle_count, state_to_string(uut.corr_state), fibre_a_valid, fifo_read_req, result_valid);
        
        // Print when memory request happens
        if (fibre_a_read_en) begin
            $display("  [MEM REQUEST] Address: %0d", fibre_a_addr);
        end
        
        // Print detailed information when results are ready
        if (result_valid) begin
            $display("  [RESULTS READY]");
            $display("    Timestep 0: %0d", result_0);
            $display("    Timestep 1: %0d", result_1);
            $display("    Timestep 2: %0d", result_2);
            $display("    Timestep 3: %0d", result_3);
        end
    end
    
    // Task to accumulate a weight
    task accumulate_weight;
        input [WEIGHT_WIDTH-1:0] weight;
        input [$clog2(128)-1:0] position;
        begin
            matched_weight = weight;
            matched_position = position;
            fast_valid = 1;
            @(posedge clk);
            fast_valid = 0;
            
            $display("  [ACCUMULATE] Weight=%0d, Position=%0d", weight, position);
        end
    endtask
    
    // Task to start correction process
    task start_correction;
        input [$clog2(128)-1:0] offset;
        input [WEIGHT_WIDTH-1:0] weight;
        begin
            // Make sure FIFO has data
            fifo_empty = 0;
            
            // Set up the correction request
            slow_offset = offset;
            current_weight = weight;
            current_position = offset + 10; // Just a different value for position
            slow_valid = 1;
            
            // Pulse for one cycle
            @(posedge clk);
            slow_valid = 0;
            
            $display("  [START CORRECTION] Offset=%0d, Weight=%0d", offset, weight);
        end
    endtask
    
    // Task to provide memory response
    task memory_response;
        input [TIMESTEPS-1:0] data;
        begin
            // Wait for memory request
            wait(fibre_a_read_en);
            @(posedge clk);
            
            // Provide the memory data
            fibre_a_data = data;
            fibre_a_valid = 1;
            
            // Valid for one cycle
            @(posedge clk);
            fibre_a_valid = 0;
            
            $display("  [MEMORY RESPONSE] Data=%b", data);
        end
    endtask
    
    // Task to wait for result with timeout
    task wait_for_result;
        output reg result_received;
        begin
            timeout = 0;
            result_received = 0;
            
            while (!result_valid && timeout < 20) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            
            if (result_valid) begin
                result_received = 1;
            end
        end
    endtask
    
    // Task to run a complete test case
    task run_test_case;
        input [TIMESTEPS-1:0] data_pattern;
        input [WEIGHT_WIDTH-1:0] acc_weight;
        input [WEIGHT_WIDTH-1:0] corr_weight;
        input [ADDR_WIDTH-1:0] mem_addr;
        
        reg result_received;
        begin
            $display("\n======= TEST CASE %0d =======", test_case);
            
            // Step 1: Accumulate weight
            accumulate_weight(acc_weight, mem_addr + 50);
            
            // Step 2: Start correction
            repeat(2) @(posedge clk); // Wait a bit
            start_correction(mem_addr, corr_weight);
            
            // Step 3: Provide memory response
            memory_response(data_pattern);
            
            // Step 4: Wait for result
            wait_for_result(result_received);
            
            if (result_received) begin
                $display("[TEST %0d PASSED] Results received successfully", test_case);
                
                // Verify results - display what should happen for each timestep
                $display("Expected behavior:");
                for (timeout = 0; timeout < TIMESTEPS; timeout = timeout + 1) begin
                    if (data_pattern[timeout]) begin
                        $display("  Timestep %0d: Active   - Should be %0d", 
                                timeout, acc_weight[CORRECTION_ACC_WIDTH-1:0]);
                    end else begin
                        $display("  Timestep %0d: Inactive - Should be %0d", 
                                timeout, acc_weight[CORRECTION_ACC_WIDTH-1:0] - corr_weight);
                    end
                end
            end else begin
                $display("[TEST %0d FAILED] Timeout waiting for results", test_case);
            end
            
            // Increment test case counter
            test_case = test_case + 1;
            
            // Allow time between test cases
            repeat(5) @(posedge clk);
        end
    endtask

    // Test sequence
    initial begin
        $dumpfile("accumulator_correction_tb.vcd");
        $dumpvars(0, accumulator_correction_tb);
        
        // Initialize test patterns
        test_patterns[0] = 4'b1010; // Timesteps 0,2 active
        test_patterns[1] = 4'b1111; // All timesteps active
        test_patterns[2] = 4'b0011; // Timesteps 0,1 active
        test_patterns[3] = 4'b0000; // No timesteps active
        
        test_weights[0] = 8'd25;
        test_weights[1] = 8'd40;
        test_weights[2] = 8'd15;
        test_weights[3] = 8'd35;
        
        // Initialize variables
        cycle_count = 0;
        timeout = 0;
        test_case = 0;
        test_done = 0;
        
        // Initialize signals
        clk = 0;
        rst = 1;
        matched_position = 0;
        matched_weight = 0;
        fast_valid = 0;
        current_position = 0;
        current_weight = 0;
        slow_offset = 0;
        slow_valid = 0;
        fifo_empty = 1;
        fibre_a_data = 0;
        fibre_a_valid = 0;
        
        // Apply reset
        repeat(3) @(posedge clk);
        rst = 0;
        @(posedge clk);
        
        $display("\n==== BEGINNING TEST SEQUENCE ====\n");
        
        // Run test cases
        run_test_case(test_patterns[0], test_weights[0], test_weights[0]/2, 5);
        run_test_case(test_patterns[1], test_weights[1], test_weights[1]/4, 8);
        run_test_case(test_patterns[2], test_weights[2], test_weights[2]/3, 12);
        run_test_case(test_patterns[3], test_weights[3], test_weights[3]/5, 15);
        
        // End simulation
        $display("\n==== TEST SEQUENCE COMPLETE ====\n");
        test_done = 1;
        
        repeat(5) @(posedge clk);
        $finish;
    end

endmodule