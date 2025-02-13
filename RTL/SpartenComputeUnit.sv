module SpartenComputeUnit #(
    parameter int CHUNK_SIZE = 128,
    parameter int Q = 8  // Quantization bits
) (
    input  logic clk,
    input  logic rst_n,
    input  logic enable,
    input  logic [CHUNK_SIZE-1:0] input_sparsemap,
    input  logic [CHUNK_SIZE-1:0] filter_sparsemap,
    input  logic [Q-1:0] input_data [CHUNK_SIZE],
    input  logic [Q-1:0] filter_data [CHUNK_SIZE],
    output logic [2*Q-1:0] result,
    output logic done
);

    // Inner join result
    logic [CHUNK_SIZE-1:0] and_result;

    // Priority encoder signals
    logic [$clog2(CHUNK_SIZE)-1:0] match_position;
    logic valid_match;

    // Prefix sum signals for input and filter
    logic [$clog2(CHUNK_SIZE):0] input_prefix_sum [CHUNK_SIZE];
    logic [$clog2(CHUNK_SIZE):0] filter_prefix_sum [CHUNK_SIZE];

    // Data selection signals
    logic [Q-1:0] selected_input;
    logic [Q-1:0] selected_filter;

    // FSM states
    enum logic [1:0] {IDLE, FIND_MATCH, COMPUTE, FINISH} state, next_state;

    // Inner join (AND operation)
    assign and_result = input_sparsemap & filter_sparsemap;

    // Priority encoder
    always_comb begin
        match_position = '0;
        valid_match = 1'b0;
        for (int i = 0; i < CHUNK_SIZE; i++) begin
            if (and_result[i]) begin
                match_position = i[$clog2(CHUNK_SIZE)-1:0];
                valid_match = 1'b1;
                break;
            end
        end
    end

    // Prefix sum calculation
    always_comb begin
        input_prefix_sum[0] = '0;
        filter_prefix_sum[0] = '0;
        for (int i = 1; i <= CHUNK_SIZE; i++) begin
            input_prefix_sum[i] = input_prefix_sum[i-1] + input_sparsemap[i-1];
            filter_prefix_sum[i] = filter_prefix_sum[i-1] + filter_sparsemap[i-1];
        end
    end

    // Data selection based on prefix sum
    always_comb begin
        selected_input = input_data[input_prefix_sum[match_position]];
        selected_filter = filter_data[filter_prefix_sum[match_position]];
    end

    // FSM for control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE: if (enable) next_state = FIND_MATCH;
            FIND_MATCH: next_state = COMPUTE;
            COMPUTE: next_state = valid_match ? FIND_MATCH : FINISH;
            FINISH: next_state = IDLE;
        endcase
    end

    // Multiply and Accumulate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= '0;
            done <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    result <= '0;
                    done <= 1'b0;
                end
                COMPUTE: begin
                    if (valid_match) begin
                        result <= result + selected_input * selected_filter;
                    end
                end
                FINISH: begin
                    done <= 1'b1;
                end
            endcase
        end
    end

endmodule