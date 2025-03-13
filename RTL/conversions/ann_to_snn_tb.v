`timescale 1ns / 1ps

module ann_to_snn_converter_tb;
    // Parameters
    parameter CLK_PERIOD = 10; // 10ns (100MHz)
    parameter DATA_WIDTH = 8;
    parameter T = 4;
    
    // Signals
    reg clk;
    reg rst_n;
    reg [DATA_WIDTH-1:0] data_in;
    reg data_valid;
    wire [T-1:0] spike_out;
    wire spike_valid;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Instantiate the DUT
    ann_to_snn_converter #(
        .DATA_WIDTH(DATA_WIDTH),
        .T(T),
        .THRESHOLD(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .data_valid(data_valid),
        .spike_out(spike_out),
        .spike_valid(spike_valid)
    );
    
    // Test case scenarios
    initial begin
        // Initialize variables
        rst_n = 0;
        data_in = 0;
        data_valid = 0;
        
        // Display header
        $display("=== ANN to SNN Converter Testbench ===");
        
        // Apply reset
        #(CLK_PERIOD*2);
        rst_n = 1;
        #(CLK_PERIOD*2);
        
        // Test Case 1: Values that should cause spikes at all timesteps
        $display("\nTest Case 1: Values that exceed threshold when accumulated");
        test_sequence({8'd6, 8'd5, 8'd6, 8'd5}, "All values should accumulate to cause spikes");
        
        // Test Case 2: Values that should cause no spikes
        $display("\nTest Case 2: Low values that don't cause spikes");
        test_sequence({8'd1, 8'd1, 8'd1, 8'd1}, "All low values, should not cause spikes");
        
        // Test Case 3: Mix of high and low values
        $display("\nTest Case 3: Mix of high and low values");
        test_sequence({8'd10, 8'd1, 8'd8, 8'd2}, "First and third should spike");
        
        // Test Case 4: Values at exact threshold
        $display("\nTest Case 4: Values at exact threshold");
        test_sequence({8'd8, 8'd8, 8'd8, 8'd8}, "All should spike");
        
        // Test Case 5: Increasing values
        $display("\nTest Case 5: Increasing values");
        test_sequence({8'd2, 8'd4, 8'd6, 8'd8}, "Accumulation should cause later spikes");
        
        // End simulation
        #(CLK_PERIOD*10);
        $display("\n=== Testbench Complete ===");
        $finish;
    end
    
    // Task to test a sequence of 4 values
    task test_sequence;
        input [DATA_WIDTH*T-1:0] values;
        input string description;
        
        reg [DATA_WIDTH-1:0] val[0:T-1];
        integer i;
        
        begin
            // Extract individual values
            val[0] = values[DATA_WIDTH-1:0];
            val[1] = values[2*DATA_WIDTH-1:DATA_WIDTH];
            val[2] = values[3*DATA_WIDTH-1:2*DATA_WIDTH];
            val[3] = values[4*DATA_WIDTH-1:3*DATA_WIDTH];
            
            $display("Testing: %s", description);
            $display("Input values: %d, %d, %d, %d", val[0], val[1], val[2], val[3]);
            
            // Send the values
            for (i = 0; i < T; i = i + 1) begin
                @(posedge clk);
                data_in = val[i];
                data_valid = 1;
                
                @(posedge clk);
                if (i == T-1) begin
                    data_valid = 0;
                end
            end
            
            // Wait for processing
            repeat(10) @(posedge clk);
            
            // Check for output
            wait(spike_valid);
            @(posedge clk);
            $display("Spike output: %b (binary), Expected behavior: %s", spike_out, description);
            
            // Wait before next test
            repeat(5) @(posedge clk);
        end
    endtask
    
    // Monitor for debugging
    always @(posedge clk) begin
        if (spike_valid) begin
            $display("Time %0t: Spike Output = %b", $time, spike_out);
        end
    end

endmodule