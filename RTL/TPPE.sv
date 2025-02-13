module TPPE #(
    parameter T = 4,  // Number of time steps
    parameter Q = 4, // Bits for quantization
    parameter N = 4, // Number of rows (width of bitmask)
    parameter ADDR_WIDTH = $clog2(N),
    parameter FIFO_DEPTH = N  // Depth of FIFO, adjust as needed
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic [N-1:0]      bitmask_A,
    input  logic [N-1:0]      bitmask_B,
    input  logic [Q-1:0]      fiber_B [N-1:0],
    output logic [Q-1:0]      result [T-1:0]
);

    // Fast prefix sum circuit
    logic [ADDR_WIDTH-1:0] fast_prefix_sum;
    logic [ADDR_WIDTH-1:0] match_positions [N-1:0];

    // Laggy prefix sum circuit
    logic [ADDR_WIDTH-1:0] laggy_prefix_sum;
    logic                  laggy_ready;

    // Pseudo-accumulator
    logic [Q-1:0] pseudo_acc [T-1:0];

    // Correction accumulators
    logic [Q-1:0] correction_acc [T-1:0];

    // FIFO for matched positions and fiber_B values
    logic [ADDR_WIDTH-1:0] fifo_mp [FIFO_DEPTH-1:0];
    logic [Q-1:0]          fifo_B [FIFO_DEPTH-1:0];
    logic [$clog2(FIFO_DEPTH):0] fifo_count;
    logic [$clog2(FIFO_DEPTH)-1:0] fifo_write_ptr, fifo_read_ptr;

    // Counter for processing matched positions
    logic [ADDR_WIDTH-1:0] match_process_counter;

    // SRAM signals
    logic [9:0] sram_read_addr;  // SRAM uses 10-bit address
    logic [31:0] sram_read_data; // SRAM uses 32-bit data width

    // Instantiate SRAM for fiber_A
    SRAM fiber_A_sram (
        .clock(clk),
        .WE(1'b0), // We're only reading from this SRAM
        .WriteAddress(10'b0),
        .ReadAddress1(sram_read_addr),
        .ReadAddress2(10'b0),
        .WriteBus(32'b0),
        .ReadBus1(sram_read_data),
        .ReadBus2()
    );

    // Fast prefix sum and matching
    always_comb begin
        fast_prefix_sum = 0;
        for (int i = 0; i < N; i++) begin
            if (bitmask_A[i] && bitmask_B[i]) begin
                match_positions[fast_prefix_sum] = i; //Stores the positions it matched, i is the bit number basically, fast_prefix sum stores #matched positions
                fast_prefix_sum++;
            end
        end
    end

    // Laggy prefix sum
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            laggy_prefix_sum <= 0;
            laggy_ready <= 0;
        end else begin
            if (!laggy_ready) begin
                // Simplified laggy prefix sum (in real hardware, this would be more complex)
                laggy_prefix_sum <= laggy_prefix_sum + 1;
                if (laggy_prefix_sum == fast_prefix_sum - 1) begin
                    laggy_ready <= 1;
                end
            end
        end
    end

    // FIFO control Basically handles variables repsponsible for popp
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_count <= 0;
            fifo_write_ptr <= 0;
            fifo_read_ptr <= 0;
        end else begin
            if (match_process_counter < fast_prefix_sum && fifo_count < FIFO_DEPTH) begin
                // Push to FIFO
                fifo_count <= fifo_count + 1;
                fifo_write_ptr <= fifo_write_ptr + 1;
            end
            if (laggy_ready && fifo_count > 0) begin
                // Pop from FIFO
                fifo_count <= fifo_count - 1;
                fifo_read_ptr <= fifo_read_ptr + 1;
            end
        end
    end

    // Main processing logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < T; i++) begin
                pseudo_acc[i] <= '0;
                correction_acc[i] <= '0;
                result[i] <= '0;
            end
            match_process_counter <= '0;
            sram_read_addr <= '0;
        end else begin
            // Process all matched positions
            if (match_process_counter < fast_prefix_sum && fifo_count < FIFO_DEPTH) begin
                automatic logic [ADDR_WIDTH-1:0] matched_pos = match_positions[match_process_counter];
                fifo_mp[fifo_write_ptr] <= matched_pos;
                fifo_B[fifo_write_ptr] <= fiber_B[matched_pos];
                
                // Update pseudo-accumulator
                for (int i = 0; i < T; i++) begin
                    pseudo_acc[i] <= pseudo_acc[i] + fiber_B[matched_pos];
                end

                match_process_counter <= match_process_counter + 1;
            end else if (fast_prefix_sum > 0 && match_process_counter == fast_prefix_sum) begin
                // Reset counter for next set of matches
                match_process_counter <= '0;
            end

            // Process laggy prefix sum results
            if (laggy_ready && fifo_count > 0) begin
                automatic logic [ADDR_WIDTH-1:0] mp = fifo_mp[fifo_read_ptr];
                automatic logic [Q-1:0] b_value = fifo_B[fifo_read_ptr];
                sram_read_addr <= mp[ADDR_WIDTH-1:0]; // Read from SRAM, using only lower 10 bits

                // In the next cycle, use the SRAM data
                #1;
                for (int i = 0; i < T; i++) begin
                    if (sram_read_data[i] == 0) begin
                        correction_acc[i] <= correction_acc[i] + b_value;
                    end
                end
            end

            // Compute final result
            for (int i = 0; i < T; i++) begin
                result[i] <= pseudo_acc[i] - correction_acc[i];
            end
        end
    end

endmodule