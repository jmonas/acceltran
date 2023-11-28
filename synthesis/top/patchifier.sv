module patchifier
(
    clk,
    reset, 
    en, 
    output_taken,
    state,
    image_cache, 
    all_patches,
);


parameter CHANNEL_SIZE = 8;  
parameter NUM_CHANNELS = 3; // i.e. RGB
parameter PIXEL_WIDTH = CHANNEL_SIZE * NUM_CHANNELS; 

parameter IMG_WIDTH = 64;
parameter IMG_HEIGHT = 64;


parameter PATCH_SIZE = 16;
parameter PATCH_SIZE_LOG2 = 4; 

parameter PATCHES_IN_ROW = IMG_WIDTH/PATCH_SIZE;

parameter TOTAL_NUM_PATCHES = (IMG_WIDTH/PATCH_SIZE) * (IMG_HEIGHT/PATCH_SIZE);
parameter PATCH_VECTOR_SIZE = PATCH_SIZE*PATCH_SIZE; 

input clk, reset;
input en;
input output_taken;

input [PIXEL_WIDTH-1:0] image_cache [IMG_WIDTH-1:0][IMG_HEIGHT-1:0];

output logic [1:0] state;

// output 1D vector for each patch
output logic [PIXEL_WIDTH-1:0] all_patches [TOTAL_NUM_PATCHES-1:0][PATCH_VECTOR_SIZE-1:0];


logic [PIXEL_WIDTH-1:0] reg_image_cache [IMG_WIDTH-1:0][IMG_HEIGHT-1:0];
logic [PIXEL_WIDTH-1:0] reg_all_patches [TOTAL_NUM_PATCHES-1:0][PATCH_VECTOR_SIZE-1:0];
logic start_processing = 0; // Flag to start processing
logic processing_done = 0; // Flag to indicate processing is done

localparam IDLE = 2'b00, PROCESSING = 2'b01, DONE = 2'b10;



// State Management Block
always_ff @(posedge clk) begin
	if (reset) begin
		state <= IDLE;
	end
	else begin
		case (state)
			IDLE: if (en || start_processing) state <= PROCESSING;
			PROCESSING: if (processing_done) state <= DONE;
			DONE: if (output_taken) state <= IDLE;
		endcase
	end
end



// int p,q;
// always_ff @(posedge clk) begin
// 	if (reset || (state == DONE && output_taken)) begin
// 		for (p = 0; p < TOTAL_NUM_PATCHES; p++) begin
// 			for (q = 0; q < PATCH_VECTOR_SIZE; q++) begin
// 				reg_image_cache[p][q] <= 0;
// 			end
// 		end	
// 	end
// 	else if (state == IDLE && en) begin
// 		reg_image_cache <= image_cache;
// 	end
// end
always_ff @(posedge clk) begin
	 if (state == IDLE && en) begin
		reg_image_cache <= image_cache;
	end
end

/*
PATCHIFICATION EXAMPLE:

    [ ] = 16x16 patch

    [A][B][C][D]
    [E][F][*][H] < patch_row_index
    [I][J][K][L]
    [M][N][O][P]
           ^
    patch_col_index


    patch_index = * = G
    --------------------------------------------
    For a given patch:

    [ 1, 2, 3, ...15, 16  ]
    [ 17, 18,  ...31, 32  ] < position_row_index
    [ ....................]
    [ 241, 242, ...*, 256 ]
                  ^
            position_col_index
    

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
        patch_index <= (i >> PATCH_SIZE_LOG2) * PATCHES_IN_ROW + (j >> PATCH_SIZE_LOG2);
        position_index <= (i & (PATCH_SIZE - 1)) * PATCH_SIZE + (j & (PATCH_SIZE - 1));

        reg_all_patches[patch_index][position_index] <= reg_image_cache[i][j];
        // Increment j, and if it rolls over, increment i
        j <= j + 1;
        if (j == IMG_HEIGHT - 1) begin
            j <= 0;
            i <= i + 1;
            if (i == IMG_WIDTH - 1)
                processing_done <= DONE;
        end
    end else if (state != PROCESSING) begin
        processing_done <= 0; // Reset the flag when not processing
    end
end



// int l,m;
// always_ff @(posedge clk) begin
// 	if (reset || (state == DONE && output_taken)) begin
// 		for (l = 0; l < TOTAL_NUM_PATCHES; l++) begin
// 			for (m = 0; m < PATCH_VECTOR_SIZE; m++) begin
// 				all_patches[l][m] <= 0;
// 			end
// 		end	
// 	end
// 	else if (state == DONE) begin
// 		all_patches <= reg_all_patches;
// 	end
// end

always_ff @(posedge clk) begin
	if (state == DONE) begin
		all_patches <= reg_all_patches;
	end
end


endmodule
