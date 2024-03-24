# Convert a Transformer dictionary in FlexiBERT 2.0 framework to 
# software compute operations

import os
import sys
import json
import argparse
import yaml
from tqdm import tqdm
from ops import *
from enum import Enum, auto


LANGUAGE = "language"
VISION = "vision"
DUAL_ENCODER = "dual_encoder"
SINGLE_STREAM = "single_stream"
DUAL_STREAM = "dual_stream"
COMBINATION = "combination"


	
def get_vision_emb_ops(model_dict, model, config, direction):
	IMAGE_SIZE  = model_dict["image_size"]
	PATCH_SIZE = model_dict["patch_size"]
	NUM_CHANNELS = model_dict["num_channels"]
	NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE)**2
	patch_dimension = NUM_CHANNELS * PATCH_SIZE **2
	batch_size = config['batch_size']
	ops = []
	# if direction == 'fwd':
	# 	ops.append([ImagePatchify('patchify', config, IMAGE_SIZE, PATCH_SIZE, batch_size)])
	ops.append(MemoryLoadOp('emb', config, (patch_dimension + NUM_PATCHES, model['h'][0]), 'weight'))
	# patch_proj_op = MatrixMultOp(f'patch_projection', config, [], (batch_size, NUM_PATCHES, patch_dimension),(batch_size, patch_dimension, model['h'][0]))
	# ops.append(patch_proj_op)
	# ops.append(MemoryStoreOp(f'patch-projection-s', config, patch_proj_op.output_size(), 'activation'))
	return ops

	

def get_ops(model_dict, config, direction, first_layer_only, debug):
	if "seq_length" in model_dict: 
		SEQ_LENGTH = model_dict["seq_length"]
		VOCAB_SIZE = model_dict["vocab_size"]

	if "patch_size" in model_dict:
		IMAGE_SIZE  = model_dict["image_size"]
		PATCH_SIZE = model_dict["patch_size"]
		NUM_CHANNELS = model_dict["num_channels"]
		NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE)**2

	ops = []
	batch_size = config['batch_size']

	if model_dict["type"] == LANGUAGE:
		print("LANNNNNGGG")
		if direction == 'fwd':
			ops.append(MemoryLoadOp('emb', config, (VOCAB_SIZE + SEQ_LENGTH, model_dict["model"]['h'][0]), 'weight'))
		ops.extend(get_encoder_ops(model_dict["model"], config, SEQ_LENGTH, first_layer_only, debug))
			
	elif model_dict["type"] == VISION:
		ops.extend(get_vision_emb_ops(model_dict, model_dict["model"], config, direction))
		ops.extend(get_encoder_ops(model_dict["model"], config, NUM_PATCHES, first_layer_only, debug, prefix = ["vision"]))

	elif model_dict["type"] == DUAL_ENCODER:
		if direction == 'fwd':
			ops.append(MemoryLoadOp('emb', config, (VOCAB_SIZE + SEQ_LENGTH, model_dict["text_model"]['h'][0]), 'weight'))
		ops.extend(get_vision_emb_ops(model_dict, model_dict["vision_model"], config, direction))
		ops.extend(get_encoder_ops(model_dict["text_model"], config, SEQ_LENGTH, first_layer_only, debug))
		ops.extend(get_encoder_ops(model_dict["vision_model"], config, NUM_PATCHES, first_layer_only, debug, prefix = ["vision"]))
		
		# load projection matrices 
		last_hidden_text_size = model_dict["text_model"]['h'][-1]
		last_hidden_vision_size = model_dict["vision_model"]['h'][-1]
		ops.append(MemoryLoadOp(f'txt_proj-l', config, (last_hidden_text_size, model_dict["projection_dimension"]), 'weight'))
		ops.append(MemoryLoadOp(f'vis_proj-l', config,  (last_hidden_vision_size, model_dict["projection_dimension"]), 'weight'))

		text_input_size = (1, batch_size,  last_hidden_text_size)
		text_proj_matrix_size =  (1, last_hidden_text_size, model_dict["projection_dimension"])
		text_proj_op  = MatrixMultOp(f'txt_proj', config, [f'text_proj-l',], text_input_size, text_proj_matrix_size)
		
		vision_input_size = (1, batch_size, last_hidden_vision_size)
		vision_proj_matrix_size =  (1, last_hidden_vision_size, model_dict["projection_dimension"])
		vision_proj_op  = MatrixMultOp(f'vis_proj', config, [f'vision_proj-l',], vision_input_size, vision_proj_matrix_size)

		ops.append(text_proj_op)
		ops.append(vision_proj_op)

		ops.append(MemoryStoreOp(f'txt_proj-s', config, text_proj_op.output_size(), 'activation'))
		ops.append(MemoryStoreOp(f'vis_proj-s', config, vision_proj_op.output_size(), 'activation'))

		ops.append(LayerNormOp(f'ln_txt_proj', config, [],  text_proj_op.output_size()))
		ops.append(LayerNormOp(f'ln_vis_proj', config, [],  vision_proj_op.output_size()))

		# why no memory load for layernorm?
		# why no buffer depencies for and on layernorm?

		# similarity score, do I need buffer dependence or can I assume Input activations are assumed to be in the activation buffer
		print("text_proj_op", Op.transpose_size(text_proj_op.output_size()))
		print("vision_proj_op", vision_proj_op.output_size())
		similarity_op  = MatrixMultOp(f'cos', config, ['txt_proj-s', 'vis_proj-s'], vision_proj_op.output_size(), Op.transpose_size(text_proj_op.output_size()))
		ops.append(similarity_op)
		ops.append(MemoryStoreOp(f'cos-s', config, similarity_op.output_size(), 'activation'))


	elif model_dict["type"] == SINGLE_STREAM:
		if direction == 'fwd':
			ops.append(MemoryLoadOp('emb', config, (VOCAB_SIZE + SEQ_LENGTH, model_dict["model"]['h'][0]), 'weight'))
		ops.extend(get_vision_emb_ops(model_dict, model_dict["model"], config, direction))
		ops.extend(get_encoder_ops(model_dict["model"], config, SEQ_LENGTH + NUM_PATCHES, first_layer_only, debug, prefix = ["vision"]))

	elif model_dict["type"] == DUAL_STREAM:	
		if direction == 'fwd':
			ops.append(MemoryLoadOp('emb', config, (VOCAB_SIZE + SEQ_LENGTH, model_dict["text_model"]['h'][0]), 'weight'))
		ops.extend(get_vision_emb_ops(model_dict, model_dict["vision_model"], config, direction))
		ops.extend(get_encoder_ops(model_dict["text_model"], config, SEQ_LENGTH, first_layer_only, debug, ["text", "vision"]))
		ops.extend(get_encoder_ops(model_dict["vision_model"], config, NUM_PATCHES, first_layer_only, debug, prefix =["vision", "text"]))

	elif model_dict["type"] == COMBINATION:	
		if direction == 'fwd':
			ops.append(MemoryLoadOp('emb', config, (VOCAB_SIZE + SEQ_LENGTH, model_dict["text_model"]['h'][0]), 'weight'))
		ops.extend(get_vision_emb_ops(model_dict, model_dict["vision_model"], config, direction))
		ops.extend(get_encoder_ops(model_dict["text_model"], config, SEQ_LENGTH, first_layer_only, debug))
		ops.extend(get_encoder_ops(model_dict["vision_model"], config, NUM_PATCHES, first_layer_only, debug, prefix = ["vision"]))		
		# load projection matrices 
		last_hidden_text_size = model_dict["text_model"]['h'][-1]
		last_hidden_vision_size = model_dict["vision_model"]['h'][-1]
		first_hidden_fusion_size = model_dict["fusion_model"]['h'][0]
		# Should I structure as (batch_size, last_hidden_text_size, first_hidden_fusion_size) instead?
		ops.append(MemoryLoadOp(f'txt_proj-l_1', config, (last_hidden_text_size, first_hidden_fusion_size), 'weight'))
		ops.append(MemoryLoadOp(f'vis_proj-l_1', config,  (last_hidden_vision_size, first_hidden_fusion_size), 'weight'))

		text_input_size = (batch_size, SEQ_LENGTH, last_hidden_text_size)
		text_proj_matrix_size =  (batch_size, last_hidden_text_size, first_hidden_fusion_size)
		text_proj_op  = MatrixMultOp(f'txt_proj_1', config, [f'text_proj-l_1',], text_input_size, text_proj_matrix_size)
		
		vision_input_size = (batch_size, NUM_PATCHES, last_hidden_vision_size)
		vision_proj_matrix_size =  (batch_size, last_hidden_vision_size, first_hidden_fusion_size)
		vision_proj_op  = MatrixMultOp(f'vis_proj_1', config, [f'vision_proj-l_1',], vision_input_size, vision_proj_matrix_size)

		ops.append(text_proj_op)
		ops.append(vision_proj_op)

		ops.append(MemoryStoreOp(f'txt_proj-s_1', config, text_proj_op.output_size(), 'activation'))
		ops.append(MemoryStoreOp(f'vis_proj-s_1', config, vision_proj_op.output_size(), 'activation'))

		# load 2nd projection matrices 
		ops.append(MemoryLoadOp(f'txt_proj-l_2', config, (first_hidden_fusion_size, first_hidden_fusion_size), 'weight'))
		ops.append(MemoryLoadOp(f'vis_proj-l_2', config,  (first_hidden_fusion_size, first_hidden_fusion_size), 'weight'))

		text_input_size = (batch_size, SEQ_LENGTH, first_hidden_fusion_size)
		text_proj_matrix_size =  (batch_size, first_hidden_fusion_size, first_hidden_fusion_size)
		text_proj_op  = MatrixMultOp(f'txt_proj_2', config, [f'text_proj-l_1',], text_input_size, text_proj_matrix_size)
		
		vision_input_size = (batch_size, NUM_PATCHES, first_hidden_fusion_size)
		vision_proj_matrix_size =  (batch_size, first_hidden_fusion_size, first_hidden_fusion_size)
		vision_proj_op  = MatrixMultOp(f'vis_proj_2', config, [f'vision_proj-l_1',], vision_input_size, vision_proj_matrix_size)

		ops.append(text_proj_op)
		ops.append(vision_proj_op)

		ops.append(MemoryStoreOp(f'txt_proj-s_2', config, text_proj_op.output_size(), 'activation'))
		ops.append(MemoryStoreOp(f'vis_proj-s_2', config, vision_proj_op.output_size(), 'activation'))

		# run fusion encoder
		ops.extend(get_encoder_ops(model_dict["fusion_model"], config, SEQ_LENGTH + NUM_PATCHES, first_layer_only, debug, prefix = ["fusion"]))



	if direction == 'bwd':
		ops.reverse()

	return ops

def get_encoder_ops(model, config, input_embedding_size, first_layer_only, debug, prefix = ["text"]):
	"""Get forward/backward operations for the given model"""
	batch_size = config['batch_size']
	ops = []
	for layer in range(model['l'] if not first_layer_only else 1):
		layer_hidden_size = model['h'][layer]
		multihead_ops = []
		for i, attention_head in enumerate(model['o'][layer]):
			type, param, hidden = attention_head.split('_')

			op_name = prefix[0] + "_" + attention_head + '_' + str(layer + 1) + '_' + str(i + 1)

			input_size = (batch_size, input_embedding_size, layer_hidden_size)


			if type == 'sa':
				multihead_ops.append(SelfAttentionOp(op_name, config, input_size, hidden_size=int(hidden), type=param))
			elif type == 'ca':
				op_name_cross = prefix[1] + "_" + attention_head + '_' + str(layer + 1) + '_' + str(i + 1)
				multihead_ops.append(SelfAttentionOp(op_name, config, input_size, hidden_size=int(hidden), type=param, op_name_cross = op_name_cross))
			elif type == 'c':
				multihead_ops.append(ConvOp(op_name, config, input_size, hidden_size=int(hidden), kernel_size=int(param)))
			elif type == 'l':
				multihead_ops.append(LinearTransformOp(op_name, config, input_size, hidden_size=int(hidden), type=param))

			if debug: print(f'Added operation with name: {op_name}')

		ops.append(multihead_ops)

		ops.append(LayerNormOp(f'ln_{layer}_1', config, [], input_size=input_size))

		last_hidden_size = layer_hidden_size
		for i, hidden in enumerate(model['f'][layer]):
			op_name = prefix[0] + "_" + 'ff' + '_' + str(layer + 1) + '_' + str(i + 1)

			input_size = (batch_size, input_embedding_size, last_hidden_size)
			ops.append(FeedForwardOp(op_name, config, input_size, hidden_size=hidden))
			ops.append(NonLinearityOp(f'nl_{layer}_{(i+1)}', config, [f'{op_name}_f-s'], input_size, type=config['non_linearity']))
			last_hidden_size = hidden

			if debug: print(f'Added operation with name: {op_name}')

			if i == len(model['f'][layer]) - 1:
				op_name = prefix[0] + "_" + 'ff' + '_' + str(layer + 1) + '_' + str(i + 2)
				input_size = (batch_size, input_embedding_size, last_hidden_size)
				ops.append(FeedForwardOp(op_name, config, input_size, hidden_size=layer_hidden_size))
				ops.append(NonLinearityOp(f'nl_{layer}_{(i+2)}', config, [f'{op_name}_f-s'], input_size, type=config['non_linearity']))

				if debug: print(f'Added operation with name: {op_name}')

		
		input_size = (batch_size, input_embedding_size, layer_hidden_size)
		ops.append(LayerNormOp(f'ln_{layer}_2', config, [], input_size=input_size))



		projection_head = True

		if layer == model['l'] - 1:
			projection_head = False
		elif layer_hidden_size == model['h'][layer + 1]:
			projection_head = False

		if projection_head:
			op_name = prefix[0] + "_" +  'ff' + '_' + str(layer + 1) + '_' + 'proj'
			input_size = (batch_size, input_embedding_size, layer_hidden_size)
			ops.append(FeedForwardOp(op_name, config, input_size, hidden_size=model['h'][layer + 1]))
			ops.append(NonLinearityOp(f'nl_{layer}_{(i+1)}', config, [f'{op_name}_f-s'], input_size, type=config['non_linearity']))

			if debug: print(f'Added operation with name: {op_name}')
		
	return ops


def get_tiled_ops(ops, direction, tile_compute_ops, tile_memory_ops, debug):
	"""Get tiled operations in forward/backward directions"""
	memory_ops, compute_ops = [], []
	num_ops = 0
	for op in tqdm(ops, desc=f'Converting model to hardware operations in {direction} direction'):
		print(op)
		if isinstance(op, list):
			memory_multihead_ops, compute_multihead_ops = [], []
			for head_op in op:
				memory_head_ops, compute_head_ops = [], []
				if head_op.base_op:
					if head_op.compute_op:
						compute_head_ops.extend(head_op.tile_op() if tile_compute_ops else [head_op])
					else:
						memory_head_ops.extend(head_op.tile_op() if tile_memory_ops else [head_op])
				else:
					if direction == 'fwd':
						head_op.convert_to_fwd_base_ops()
					else:
						head_op.convert_to_bwd_base_ops()
					for base_op in head_op.fwd_base_ops if direction == 'fwd' else head_op.bwd_base_ops:
						if base_op.compute_op:
							compute_head_ops.extend(base_op.tile_op() if tile_compute_ops else [base_op])
						else:
							memory_head_ops.extend(base_op.tile_op() if tile_memory_ops else [base_op])
				if memory_head_ops: 
					num_ops += len(memory_head_ops)
					memory_multihead_ops.append(memory_head_ops)
				if compute_head_ops: 
					num_ops += len(compute_head_ops)
					compute_multihead_ops.append(compute_head_ops)
			if memory_multihead_ops:
				memory_ops.append(memory_multihead_ops)
			compute_ops.append(compute_multihead_ops)
		else:
			if op.base_op:
				if op.compute_op:
					new_ops = op.tile_op() if tile_compute_ops else [op]
					num_ops += len(new_ops)
					compute_ops.extend(new_ops)
				else:
					new_ops = op.tile_op() if tile_memory_ops else [op]
					num_ops += len(new_ops)
					memory_ops.extend(new_ops)
			else:
				if direction == 'fwd':
					op.convert_to_fwd_base_ops()
				else:
					op.convert_to_bwd_base_ops()
				for base_op in op.fwd_base_ops if direction == 'fwd' else op.bwd_base_ops:
					if base_op.compute_op:
						new_ops = base_op.tile_op() if tile_compute_ops else [base_op]
						num_ops += len(new_ops)
						compute_ops.extend(new_ops)
					else:
						new_ops = base_op.tile_op() if tile_memory_ops else [base_op]
						num_ops += len(new_ops)
						memory_ops.extend(new_ops)

	if debug:
		print(f'Number of operations: {num_ops}')

	return memory_ops, compute_ops, num_ops


def main(model_dict: dict, config: dict, mode='inference', tile_compute_ops=False, tile_memory_ops=False, first_layer_only=False, debug=False):
	"""Convert model dictionary to software compute operations"""
	assert 'p' not in model_dict.keys(), 'Only model dictionaries in FlexiBERT 2.0 are supported'

	fwd_ops = get_ops(model_dict, config, direction='fwd', first_layer_only=first_layer_only, debug=debug)
	bwd_ops = get_ops(model_dict, config, direction='bwd', first_layer_only=first_layer_only, debug=debug)

	memory_ops, compute_ops = [], []
	print("BITCHING")
	print(len(fwd_ops))
	print(fwd_ops)
	fwd_memory_ops, fwd_compute_ops, fwd_num_ops = get_tiled_ops(fwd_ops, direction='fwd', tile_compute_ops=tile_compute_ops, tile_memory_ops=tile_memory_ops, debug=debug)
	bwd_memory_ops, bwd_compute_ops, bwd_num_ops = get_tiled_ops(bwd_ops, direction='bwd', tile_compute_ops=tile_compute_ops, tile_memory_ops=tile_memory_ops, debug=debug)


	memory_ops.extend(fwd_memory_ops); memory_ops.extend(bwd_memory_ops)
	compute_ops.extend(fwd_compute_ops); compute_ops.extend(bwd_compute_ops)

	if mode == 'inference':
		return fwd_memory_ops, fwd_compute_ops, fwd_num_ops
	return memory_ops, compute_ops, (fwd_num_ops + bwd_num_ops)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for conversion',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model_dict_path',
		metavar='',
		type=str,
		help='path where the model dictionary file is stored')
	parser.add_argument('--config_path',
		metavar='',
		type=str,
		help='path to the configuration file')
	parser.add_argument('--tile_ops',
		dest='tile_ops',
		help='tile software operations',
		action='store_true')
	parser.add_argument('--debug',
		dest='debug',
		help='print debugging statements',
		action='store_true')
	parser.set_defaults(debug=False)
	parser.set_defaults(tile_ops=False)
	args = parser.parse_args()

	if os.path.exists(args.model_dict_path):
		model_dict = json.load(open(args.model_dict_path, 'r'))
	else:
		raise FileNotFoundError(f'Couldn\'t find JSON file for given path: {args.model_dict_path}')

	if os.path.exists(args.config_path):
		config = yaml.safe_load(open(args.config_path, 'r'))
	else:
		raise FileNotFoundError(f'Couldn\'t find JSON file for given path: {args.config_path}')

	main(model_dict, config, args.tile_ops, args.debug)
