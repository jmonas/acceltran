module patchifier
(
    clk,
    reset, 
    en, 
    output_taken,
    state,
    image_cache, 
    all_patches
);


parameter CHANNEL_SIZE = 8;  
parameter NUM_CHANNELS = 3; // i.e. RGB
parameter PIXEL_WIDTH = CHANNEL_SIZE * NUM_CHANNELS; 

parameter IMG_WIDTH = 16;
parameter IMG_HEIGHT = 16;


parameter PATCH_SIZE = 4;
parameter PATCH_SIZE_LOG2 = 2; 

parameter PATCHES_IN_ROW = IMG_WIDTH/PATCH_SIZE;

parameter TOTAL_NUM_PATCHES = (IMG_WIDTH/PATCH_SIZE) * (IMG_HEIGHT/PATCH_SIZE);
parameter PATCH_VECTOR_SIZE = PATCH_SIZE*PATCH_SIZE; 

input clk, reset;
input en;
input output_taken;

input [PIXEL_WIDTH-1:0] image_cache [IMG_WIDTH-1:0][IMG_HEIGHT-1:0];

output logic [2:0] state;

// output 1D vector for each patch
output logic [PIXEL_WIDTH-1:0] all_patches [IMG_WIDTH-1:0][IMG_HEIGHT-1:0];


// output logic [PIXEL_WIDTH-1:0] all_patches [TOTAL_NUM_PATCHES-1:0][PATCH_VECTOR_SIZE-1:0];


logic [PIXEL_WIDTH-1:0] reg_image_cache [IMG_WIDTH-1:0][IMG_HEIGHT-1:0];
logic [PIXEL_WIDTH-1:0] reg_all_patches [IMG_WIDTH-1:0][IMG_HEIGHT-1:0];

// logic [PIXEL_WIDTH-1:0] reg_all_patches [TOTAL_NUM_PATCHES-1:0][PATCH_VECTOR_SIZE-1:0];
logic pre_processing_done; // Flag to indicate processing is done
logic processing_done; // Flag to indicate processing is done
logic post_processing_done; // Flag to indicate processing is done

// localparam IDLE = 2'b00, PROCESSING = 2'b01, DONE = 2'b10;
localparam IDLE = 2'b000, PREPROCESSING = 2'b001, PROCESSING = 2'b010, POSTPROCESSING = 2'b011 , DONE = 2'b100;



// State Management Block
always_ff @(posedge clk) begin
	if (reset) begin
		state <= IDLE;
	end
	else begin
		case (state)
			IDLE: if (en) state <= PREPROCESSING;
            PREPROCESSING: if (pre_processing_done) state <= PROCESSING;       
			PROCESSING: if (processing_done) state <= POSTPROCESSING;
            POSTPROCESSING: if(post_processing_done) state <= DONE;
			DONE: if (output_taken) state <= IDLE;
		endcase
	end
end

int x, y;
always_ff @(posedge clk) begin
    if (reset) begin
        x <= 0;
        y <= 0;
        pre_processing_done <= 0;
    end else if (PREPROCESSING && !pre_processing_done) begin
        reg_image_cache[x][y] <= image_cache[x][y];
        // Increment x and y
        x <= (x == IMG_WIDTH - 1) ? 0 : x + 1;
        y <= (x == IMG_WIDTH - 1) ? ((y == IMG_HEIGHT - 1) ? 0 : y + 1) : y;
        // Check if done
        pre_processing_done <= (x == IMG_WIDTH - 1) && (y == IMG_HEIGHT - 1);
    end
end

int a, b;
always_ff @(posedge clk) begin
    if (reset) begin
        a <= 0;
        b <= 0;
        processing_done <= 0;
    end else if (PREPROCESSING && !pre_processing_done) begin
        reg_all_patches[a][b] <= reg_image_cache[a][b]+1;
        // Increment x and y
        a <= (a == IMG_WIDTH - 1) ? 0 : a + 1;
        b <= (a == IMG_WIDTH - 1) ? ((b == IMG_HEIGHT - 1) ? 0 : b + 1) : b;
        // Check if done
        processing_done <= (a == IMG_WIDTH - 1) && (b == IMG_HEIGHT - 1);
    end
end


int m, n;
always_ff @(posedge clk) begin
    if (reset) begin
        m <= 0;
        n <= 0;
        post_processing_done <= 0;
    end else if (PREPROCESSING && !pre_processing_done) begin
        all_patches[m][n] <= reg_all_patches[m][n];
        // Increment x and y
        m <= (m == IMG_WIDTH - 1) ? 0 : m + 1;
        n <= (m == IMG_WIDTH - 1) ? ((n == IMG_HEIGHT - 1) ? 0 : n + 1) : n;
        // Check if done
        post_processing_done <= (m == IMG_WIDTH - 1) && (n == IMG_HEIGHT - 1);
    end
end




// int post_position_index, post_patch_index;
// always_ff @(posedge clk) begin
//     if (reset) begin
//         post_position_index <= 0;
//         post_patch_index <= 0;
//         post_processing_done <= 0;
//     end else if (POSTPROCESSING && !post_processing_done) begin
//         all_patches[x][y] <= reg_all_patches[x][y];
//         // Increment counters
//         if (post_position_index == PATCH_VECTOR_SIZE - 1) begin
//             post_position_index <= 0;
//             post_patch_index <= (post_patch_index == TOTAL_NUM_PATCHES - 1) ? 0 : post_patch_index + 1;
//             post_processing_done <= (post_patch_index == TOTAL_NUM_PATCHES - 1);
//         end else begin
//             post_position_index <= post_position_index + 1;
//         end

//     end
// end



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
// always_ff @(posedge clk) begin
// 	 if (state == IDLE && en) begin
// 		reg_image_cache <= image_cache;
// 	end
// end

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

// int i, j;
// int patch_index, position_index;
// always_ff @(posedge clk) begin
//     if (reset) begin
//         i <= 0;
//         j <= 0;
//         processing_done <= 0;
//     end else if (state == PROCESSING && !processing_done) begin
//         patch_index <= (i >> PATCH_SIZE_LOG2) * PATCHES_IN_ROW + (j >> PATCH_SIZE_LOG2);
//         position_index <= (i & (PATCH_SIZE - 1)) * PATCH_SIZE + (j & (PATCH_SIZE - 1));

//         reg_all_patches[patch_index][position_index] <= reg_image_cache[i][j];
//         // Increment j, and if it rolls over, increment i
//         j <= j + 1;
//         if (j == IMG_HEIGHT - 1) begin
//             j <= 0;
//             i <= i + 1;
//             if (i == IMG_WIDTH - 1)
//                 processing_done <= 1;
//         end
//     end else if (state != PROCESSING) begin
//         processing_done <= 0; // Reset the flag when not processing
//     end
// end



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

// always_ff @(posedge clk) begin
// 	if (state == DONE) begin
// 		all_patches <= reg_all_patches;
// 	end
// end


endmodule
