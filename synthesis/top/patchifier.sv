module patchifier
(
    clk,
    reset, 
    en, 
    output_taken,
    state,
    patch_cache, 
    vectorized_patch
);


parameter CHANNEL_SIZE = 8;  
parameter NUM_CHANNELS = 3; // i.e. RGB
parameter PIXEL_WIDTH = CHANNEL_SIZE * NUM_CHANNELS; 
parameter size = 16;
parameter PATCH_VECTOR_SIZE = size*size; 

input clk, reset;
input en;
input output_taken;

input [PIXEL_WIDTH-1:0] patch_cache [size-1:0][size-1:0];

output logic [1:0] state;

// output 1D vector for each patch
output logic [PIXEL_WIDTH-1:0] vectorized_patch [PATCH_VECTOR_SIZE-1:0];


logic [PIXEL_WIDTH-1:0] reg_patch_cache [size-1:0][size-1:0];
logic [PIXEL_WIDTH-1:0] reg_vectorized_patch [PATCH_VECTOR_SIZE-1:0];
logic processing_done; // Flag to indicate processing is done

// localparam IDLE = 2'b00, PROCESSING = 2'b01, DONE = 2'b10;
localparam IDLE = 2'b00,  PROCESSING = 2'b10, DONE = 2'b11;



// State Management Block
always_ff @(posedge clk) begin
	if (reset) begin
		state <= IDLE;
	end
	else begin
		case (state)
			IDLE: if (en) state <= PROCESSING;
			PROCESSING: if (processing_done) state <= DONE;
			DONE: if (output_taken) state <= IDLE;
		endcase
	end
end



int p, q;
always_ff @(posedge clk) begin
	if (reset || (state == DONE && output_taken)) begin
		for (p = 0; p < size; p++) begin
			for (q = 0; q < size; q++) begin
				reg_patch_cache[p][q] <= 0;
			end
		end	
	end
	else if (state == IDLE && en) begin
		reg_patch_cache <= patch_cache;
    end
end


/*
PATCHIFICATION EXAMPLE:

    [ ] = 16x16 patch

    [A][B][C][D]
    [E][F][*][H] 
    [I][J][K][L]
    [M][N][O][P]


    patch_index = * = G
    --------------------------------------------
    For a given patch:

    [ 1, 2, 3, ...15, 16  ]
    [ 17, 18,  ...31, 32  ]
    [ ....................]
    [ 241, 242, ...*, 256 ]
    

    position_index = * = 255
*/

int i, j;
int patch_index, position_index;
always_ff @(posedge clk) begin
    if (reset) begin
        i <= 0;
        j <= 0;
        processing_done <= 0;
    end else if (state == PROCESSING && !processing_done) begin
        position_index <= i * size + j;
        reg_vectorized_patch[position_index] <= reg_patch_cache[i][j];
        // Increment j, and if it rolls over, increment i
        j <= j + 1;
        if (j == size - 1) begin
            j <= 0;
            i <= i + 1;
            if (i == size - 1)
                processing_done <= 1;
        end
    end else if (state != PROCESSING) begin
        processing_done <= 0; // Reset the flag when not processing
    end
end


int m;
always_ff @(posedge clk) begin
	if (reset || (state == DONE && output_taken)) begin
		for (m = 0; m < PATCH_VECTOR_SIZE; m++) begin
				vectorized_patch[m] <= {PIXEL_WIDTH{1'b0}};
		end
	end
	else if (state == DONE) begin
		vectorized_patch <= reg_vectorized_patch;
	end
end


endmodule
