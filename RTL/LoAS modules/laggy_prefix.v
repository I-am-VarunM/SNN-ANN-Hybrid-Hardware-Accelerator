module laggy_prefix #(
    parameter BITMASK_WIDTH = 128,
    parameter NUM_ADDERS = 16,
    parameter WEIGHT_WIDTH = 8,
    parameter FIFO_DEPTH = 8,
    parameter CHUNK_SIZE = BITMASK_WIDTH / NUM_ADDERS
)(
    input wire clk,
    input wire rst,
    input wire [BITMASK_WIDTH-1:0] and_result,
    input wire [BITMASK_WIDTH-1:0] bitmask_a,
    input wire [$clog2(BITMASK_WIDTH)-1:0] matched_position,
    input wire [WEIGHT_WIDTH-1:0] matched_weight,
    input wire valid_match,
    
    // Separate FIFO control signals
    input wire fifo_mp_read_en,      // Read enable for matched position FIFO
    input wire fifo_weight_read_en,  // Read enable for weight FIFO
    
    output reg [$clog2(BITMASK_WIDTH)-1:0] slow_offset,
    output reg [$clog2(BITMASK_WIDTH)-1:0] current_position,
    output reg [WEIGHT_WIDTH-1:0] current_weight,
    output reg slow_valid,
    output reg ready_for_new_calc,
    
    // FIFO status outputs - separate for each FIFO
    output wire fifo_mp_empty,
    output wire fifo_mp_full,
    output wire fifo_weight_empty,
    output wire fifo_weight_full
);

    // Internal signals
    wire fifo_write_en;
    reg fifo_internal_read_en;  // Internal read enable for synchronized operations
    wire [$clog2(BITMASK_WIDTH)-1:0] fifo_mp_out;
    wire [WEIGHT_WIDTH-1:0] fifo_weight_out;
    reg [2:0] slow_counter;  // 3 bits for counting up to 8 cycles
    reg [$clog2(BITMASK_WIDTH)-1:0] adder_results [NUM_ADDERS-1:0];
    reg processing;
    
    // FIFO control signals
    assign fifo_write_en = valid_match && !fifo_mp_full && !fifo_weight_full;

    // FIFO for matched positions
    fifo #(
        .WIDTH($clog2(BITMASK_WIDTH)),
        .DEPTH(FIFO_DEPTH)
    ) matched_pos_fifo (
        .clk(clk),
        .rst(rst),
        .write_en(fifo_write_en),
        .read_en(fifo_mp_read_en || fifo_internal_read_en),  // Can be read by external OR internal control
        .data_in(matched_position),
        .data_out(fifo_mp_out),
        .full(fifo_mp_full),
        .empty(fifo_mp_empty)
    );

    // FIFO for weights
    fifo #(
        .WIDTH(WEIGHT_WIDTH),
        .DEPTH(FIFO_DEPTH)
    ) weight_fifo (
        .clk(clk),
        .rst(rst),
        .write_en(fifo_write_en),
        .read_en(fifo_weight_read_en || fifo_internal_read_en),  // Can be read by external OR internal control
        .data_in(matched_weight),
        .data_out(fifo_weight_out),
        .full(fifo_weight_full),
        .empty(fifo_weight_empty)
    );

    // Function to count 1s in a specific chunk range
    function automatic [$clog2(BITMASK_WIDTH)-1:0] count_ones_in_range;
        input [BITMASK_WIDTH-1:0] vector;
        input integer start_pos;
        input integer end_pos;
        integer i;
        begin
            count_ones_in_range = 0;
            for(i = start_pos; i <= end_pos && i < BITMASK_WIDTH; i = i + 1) begin
                if(vector[i]) count_ones_in_range = count_ones_in_range + 1;
            end
        end
    endfunction
    
    // Main processing logic
    integer i, j;
    reg [BITMASK_WIDTH-1:0] temp_vector;
    
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            slow_counter <= 0;
            slow_offset <= 0;
            slow_valid <= 0;
            processing <= 0;
            fifo_internal_read_en <= 0;
            ready_for_new_calc <= 1;
            current_position <= 0;
            current_weight <= 0;
            temp_vector <= 0;
            
            for(i = 0; i < NUM_ADDERS; i = i + 1) begin
                adder_results[i] <= 0;
            end
            $display("Laggy prefix reset");
        end
        else begin
            // Default values
            fifo_internal_read_en <= 0;
            slow_valid <= 0;
            
            // If not currently processing and FIFOs are not empty, start new calculation
            if(!processing && !fifo_mp_empty && !fifo_weight_empty && ready_for_new_calc) begin
                // Read from FIFO internally (for our own processing)
                fifo_internal_read_en <= 1;
                ready_for_new_calc <= 0;
                $display("Laggy prefix: internally popping from FIFOs at time %0t", $time);
            end
            // If we just read from FIFO, start processing
            else if(fifo_internal_read_en) begin
                // Start new calculation with the position from FIFO
                current_position <= fifo_mp_out;
                current_weight <= fifo_weight_out;  // Store weight for correction stage
                $display("Laggy prefix: popped position=%0d, weight=%0h", fifo_mp_out, fifo_weight_out);
                
                slow_counter <= 0;
                processing <= 1;
                temp_vector <= bitmask_a;
                
                // First stage: Divide input into chunks and count 1s
                for(i = 0; i < NUM_ADDERS; i = i + 1) begin
                    // Calculate start and end positions for this chunk
                    j = i * CHUNK_SIZE; // start position
                    
                    if(j > fifo_mp_out) begin
                        // This chunk is entirely past matched_position
                        adder_results[i] <= 0;
                        $display("Chunk %0d (%0d to %0d): not counting (past matched_position)", 
                                i, j, j+CHUNK_SIZE-1);
                    end
                    else if(j + CHUNK_SIZE - 1 <= fifo_mp_out) begin
                        // This chunk is entirely before or at matched_position
                        adder_results[i] <= count_ones_in_range(bitmask_a, j, j+CHUNK_SIZE-1);
                        $display("Chunk %0d (%0d to %0d): counting all", 
                                i, j, j+CHUNK_SIZE-1);
                    end
                    else begin
                        // This chunk contains matched_position
                        adder_results[i] <= count_ones_in_range(bitmask_a, j, fifo_mp_out);
                        $display("Chunk %0d (%0d to %0d): counting up to %0d", 
                                i, j, j+CHUNK_SIZE-1, fifo_mp_out);
                    end
                end
                $display("Starting laggy prefix calculation for position %0d", fifo_mp_out);
            end
            // Continue processing stages if we're in the middle of a calculation
            else if(processing) begin
                // Continue processing stages
                slow_counter <= slow_counter + 1;
                
                case(slow_counter)
                    0: begin // Second stage: combine pairs (16 -> 8)
                        for(i = 0; i < NUM_ADDERS/2; i = i + 1) begin
                            adder_results[i] <= adder_results[2*i] + adder_results[2*i+1];
                        end
                        $display("Laggy prefix stage 2 (16->8)");
                    end
                    
                    1: begin // Third stage: combine pairs (8 -> 4)
                        for(i = 0; i < NUM_ADDERS/4; i = i + 1) begin
                            adder_results[i] <= adder_results[2*i] + adder_results[2*i+1];
                        end
                        $display("Laggy prefix stage 3 (8->4)");
                    end
                    
                    2: begin // Fourth stage: combine pairs (4 -> 2)
                        for(i = 0; i < NUM_ADDERS/8; i = i + 1) begin
                            adder_results[i] <= adder_results[2*i] + adder_results[2*i+1];
                        end
                        $display("Laggy prefix stage 4 (4->2)");
                    end
                    
                    3: begin // Fifth stage: combine pairs (2 -> 1)
                        adder_results[0] <= adder_results[0] + adder_results[1];
                        $display("Laggy prefix stage 5 (2->1)");
                    end
                    
                    7: begin // Final stage: output result
                        slow_offset <= adder_results[0];
                        slow_valid <= 1;
                        processing <= 0;
                        ready_for_new_calc <= 1;  // Ready to take new position from FIFO
                        $display("Laggy prefix final: offset = %0d for position %0d, weight=%0h", 
                                adder_results[0], current_position, current_weight);
                    end
                    
                    default: begin
                        // Wait remaining cycles
                        $display("Laggy prefix waiting: cycle %0d", slow_counter);
                    end
                endcase
            end
            // If not processing and FIFOs are empty, we're ready for new calculations
            else if(!processing) begin
                ready_for_new_calc <= 1;
            end
        end
    end

endmodule

module fifo #(
    parameter WIDTH = 8,
    parameter DEPTH = 8
)(
    input wire clk,
    input wire rst,
    input wire write_en,
    input wire read_en,
    input wire [WIDTH-1:0] data_in,
    output reg [WIDTH-1:0] data_out,
    output reg full,
    output reg empty
);

    reg [WIDTH-1:0] memory [DEPTH-1:0];
    reg [$clog2(DEPTH):0] write_ptr;  // Extra bit for full/empty detection
    reg [$clog2(DEPTH):0] read_ptr;   // Extra bit for full/empty detection

    // Full and Empty flags
    always @(*) begin
        empty = (write_ptr == read_ptr);
        full = (write_ptr[$clog2(DEPTH)] != read_ptr[$clog2(DEPTH)] &&
                write_ptr[$clog2(DEPTH)-1:0] == read_ptr[$clog2(DEPTH)-1:0]);
    end

    // Write operation
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            write_ptr <= 0;
        end
        else if (write_en && !full) begin
            memory[write_ptr[$clog2(DEPTH)-1:0]] <= data_in;
            write_ptr <= write_ptr + 1;
        end
    end

    // Read operation
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            read_ptr <= 0;
            data_out <= 0;
        end
        else if (read_en && !empty) begin
            data_out <= memory[read_ptr[$clog2(DEPTH)-1:0]];
            read_ptr <= read_ptr + 1;
        end
    end

endmodule