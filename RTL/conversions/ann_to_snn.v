module ann_to_snn_converter #(
    parameter DATA_WIDTH = 8,
    parameter T = 4,
    parameter THRESHOLD = 8,
    parameter ADDR_WIDTH = $clog2(T)
)(
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] data_in,
    input wire data_valid,
    output reg [T-1:0] spike_out,
    output reg spike_valid
);

    // States for the state machine
    localparam IDLE = 2'b00;
    localparam LOAD = 2'b01;
    localparam PROCESS = 2'b10;
    localparam OUTPUT = 2'b11;

    // Registers
    reg [1:0] state, next_state;
    reg [DATA_WIDTH-1:0] mem;
    reg spike;
    reg [ADDR_WIDTH-1:0] t_counter;
    reg [DATA_WIDTH-1:0] x_buffer [0:T-1];
    reg [T-1:0] spike_buffer;
    reg [ADDR_WIDTH-1:0] input_counter;
    reg [ADDR_WIDTH-1:0] output_counter;
    reg [DATA_WIDTH-1:0] new_mem;
    reg spike_occurred;
    
    // Initial membrane value (0.5 * threshold)
    wire [DATA_WIDTH-1:0] initial_mem = THRESHOLD >> 1;
    
    // State machine for control
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    // Next state logic
    always @(*) begin
        case (state)
            IDLE: begin
                if (data_valid)
                    next_state = LOAD;
                else
                    next_state = IDLE;
            end
            
            LOAD: begin
                if (input_counter == T-1)
                    next_state = PROCESS;
                else
                    next_state = LOAD;
            end
            
            PROCESS: begin
                if (t_counter == T-1)
                    next_state = OUTPUT;
                else
                    next_state = PROCESS;
            end
            
            OUTPUT: begin
                if (output_counter == T-1)
                    next_state = IDLE;
                else
                    next_state = OUTPUT;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // Data path operations
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all registers
            mem <= initial_mem;
            spike <= 0;
            t_counter <= 0;
            input_counter <= 0;
            output_counter <= 0;
            spike_valid <= 0;
            spike_out <= 0;
            
            // Reset buffers
            for (integer i = 0; i < T; i = i + 1) begin
                x_buffer[i] <= 0;
            end
            spike_buffer <= 0;
        end else begin
            case (state)
                IDLE: begin
                    // Reset counters and membrane potential
                    mem <= initial_mem;
                    t_counter <= 0;
                    input_counter <= 0;
                    output_counter <= 0;
                    spike_valid <= 0;
                end
                
                LOAD: begin
                    // Load input data into buffer (expand temporal dimension)
                    if (data_valid) begin
                        x_buffer[input_counter] <= data_in;
                        input_counter <= input_counter + 1;
                    end
                end
                
                PROCESS: begin
                    // Calculate updated membrane potential
                    new_mem = mem + x_buffer[t_counter];
                    spike_occurred = (new_mem >= THRESHOLD);
                    
                    // 1. Update membrane (accumulate and reset if needed in one operation)
                    if (spike_occurred)
                        mem <= new_mem - THRESHOLD;
                    else
                        mem <= new_mem;
                    
                    // 2. Generate spike if membrane potential exceeds threshold
                    spike <= spike_occurred;
                    
                    // 3. Store spike in buffer
                    spike_buffer[t_counter] <= spike_occurred;
                    
                    // 4. Increment timestep counter
                    t_counter <= t_counter + 1;
                end
                
                OUTPUT: begin
                    // Output all spikes at once (T bits)
                    spike_out <= spike_buffer;
                    spike_valid <= 1;
                    output_counter <= output_counter + 1;
                    if (output_counter == 0) begin
                        // Only need one cycle to output the entire spike train
                        output_counter <= T-1;
                    end
                end
            endcase
        end
    end

endmodule