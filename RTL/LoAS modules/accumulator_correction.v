module accumulator_correction #(
    parameter TIMESTEPS = 4,
    parameter WEIGHT_WIDTH = 8,
    parameter PSEUDO_ACC_WIDTH = 12,
    parameter CORRECTION_ACC_WIDTH = 10,
    parameter ADDR_WIDTH = 8
)(
    input wire clk,
    input wire rst,
    
    // From fast_prefix
    input wire [$clog2(128)-1:0] matched_position,
    input wire [WEIGHT_WIDTH-1:0] matched_weight,
    input wire fast_valid,
    
    // From laggy_prefix
    input wire [$clog2(128)-1:0] current_position,
    input wire [WEIGHT_WIDTH-1:0] current_weight,
    input wire [$clog2(128)-1:0] slow_offset,
    input wire slow_valid,
    input wire fifo_empty,
    
    // Fibre A memory interface
    input wire [TIMESTEPS-1:0] fibre_a_data,
    input wire fibre_a_valid,
    output reg [ADDR_WIDTH-1:0] fibre_a_addr,
    output reg fibre_a_read_en,
    
    // Control signals
    output reg ready_for_fast,
    output reg ready_for_slow,
    output reg fifo_read_req,
    
    // Individual results (no arrays)
    output reg [CORRECTION_ACC_WIDTH-1:0] result_0,
    output reg [CORRECTION_ACC_WIDTH-1:0] result_1,
    output reg [CORRECTION_ACC_WIDTH-1:0] result_2,
    output reg [CORRECTION_ACC_WIDTH-1:0] result_3,
    output reg result_valid
);

    // States for the correction state machine
    localparam CORR_IDLE = 3'd0;
    localparam WAITING_FOR_FIFO = 3'd1;
    localparam WAITING_FOR_DATA = 3'd2;
    localparam CORRECTION = 3'd3;
    localparam COMPLETE = 3'd4;
    
    // Internal registers
    reg [2:0] corr_state;
    reg [PSEUDO_ACC_WIDTH-1:0] pseudo_accumulator;
    reg [WEIGHT_WIDTH-1:0] stored_weight;
    reg [$clog2(128)-1:0] stored_offset;
    reg [TIMESTEPS-1:0] stored_fibre_a;
    
    // Pseudo-accumulation logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pseudo_accumulator <= 0;
            ready_for_fast <= 1;
        end
        else begin
            ready_for_fast <= 1;
            
            if (fast_valid) begin
                pseudo_accumulator <= pseudo_accumulator + matched_weight;
                $display("ACC: Accumulating weight %0d, new total=%0d", 
                        matched_weight, pseudo_accumulator + matched_weight);
            end
        end
    end
    
    // Correction state machine
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            corr_state <= CORR_IDLE;
            stored_weight <= 0;
            stored_offset <= 0;
            stored_fibre_a <= 0;
            fibre_a_addr <= 0;
            fibre_a_read_en <= 0;
            ready_for_slow <= 1;
            fifo_read_req <= 0;
            result_valid <= 0;
            
            // Reset all result registers
            result_0 <= 0;
            result_1 <= 0;
            result_2 <= 0;
            result_3 <= 0;
        end
        else begin
            // Default values
            fibre_a_read_en <= 0;
            fifo_read_req <= 0;
            result_valid <= 0;
            
            case (corr_state)
                CORR_IDLE: begin
                    ready_for_slow <= 1;
                    
                    if (slow_valid && !fifo_empty) begin
                        fifo_read_req <= 1;
                        ready_for_slow <= 0;
                        corr_state <= WAITING_FOR_FIFO;
                        $display("ACC: Requesting from FIFO");
                    end
                end
                
                WAITING_FOR_FIFO: begin
                    if (!fifo_empty) begin
                        stored_weight <= current_weight;
                        stored_offset <= slow_offset;
                        
                        fibre_a_addr <= slow_offset;
                        fibre_a_read_en <= 1;
                        corr_state <= WAITING_FOR_DATA;
                        $display("ACC: FIFO data received, requesting memory at addr %0d", slow_offset);
                    end
                end
                
                WAITING_FOR_DATA: begin
                    if (fibre_a_valid) begin
                        stored_fibre_a <= fibre_a_data;
                        
                        if (fibre_a_data == {TIMESTEPS{1'b1}}) begin
                            // No correction needed - all timesteps active
                            result_0 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0];
                            result_1 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0];
                            result_2 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0];
                            result_3 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0];
                            corr_state <= COMPLETE;
                            $display("ACC: All timesteps active, no correction needed");
                        end
                        else begin
                            corr_state <= CORRECTION;
                            $display("ACC: Starting correction for fibre_a=%b", fibre_a_data);
                        end
                    end
                end
                
                CORRECTION: begin
                    // Handle each timestep individually
                    if (stored_fibre_a[0]) begin
                        result_0 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0];
                        $display("ACC: Timestep 0 active, value=%0d", pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0]);
                    end
                    else begin
                        result_0 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0] - stored_weight;
                        $display("ACC: Timestep 0 inactive, corrected value=%0d", 
                                pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0] - stored_weight);
                    end
                    
                    if (stored_fibre_a[1]) begin
                        result_1 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0];
                        $display("ACC: Timestep 1 active, value=%0d", pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0]);
                    end
                    else begin
                        result_1 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0] - stored_weight;
                        $display("ACC: Timestep 1 inactive, corrected value=%0d", 
                                pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0] - stored_weight);
                    end
                    
                    if (stored_fibre_a[2]) begin
                        result_2 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0];
                        $display("ACC: Timestep 2 active, value=%0d", pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0]);
                    end
                    else begin
                        result_2 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0] - stored_weight;
                        $display("ACC: Timestep 2 inactive, corrected value=%0d", 
                                pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0] - stored_weight);
                    end
                    
                    if (stored_fibre_a[3]) begin
                        result_3 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0];
                        $display("ACC: Timestep 3 active, value=%0d", pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0]);
                    end
                    else begin
                        result_3 <= pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0] - stored_weight;
                        $display("ACC: Timestep 3 inactive, corrected value=%0d", 
                                pseudo_accumulator[CORRECTION_ACC_WIDTH-1:0] - stored_weight);
                    end
                    
                    corr_state <= COMPLETE;
                    $display("ACC: Correction complete");
                end
                
                COMPLETE: begin
                    result_valid <= 1;
                    ready_for_slow <= 1;
                    corr_state <= CORR_IDLE;
                    
                    $display("ACC: Results ready: %0d, %0d, %0d, %0d", 
                            result_0, result_1, result_2, result_3);
                end
                
                default: corr_state <= CORR_IDLE;
            endcase
        end
    end

endmodule