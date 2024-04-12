import os
import sys
import numpy as np
import math 
import json 
import sys 
sys.path.insert(0, '../../training/FLAVA')
from eval import evaluate
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import torch
sys.path.append('../../transformers_new/src/transformers')
sys.path.append("/home/jmonas/acceltran/transformers/src/transformers/models/flava/")
sys.path.append("/home/jmonas/acceltran/training/FLAVA/")
sys.path.append("/home/jmonas/acceltran/")


from model import DTFlavaForVQA

from modeling_dtflava import DTFlavaModel
from datetime import datetime
import time
from transformers import FlavaMultimodalConfig, FlavaImageConfig, FlavaTextConfig, FlavaConfig, FlavaProcessor
from transformers import FlavaModel
from training.FLAVA.model import FlavaForVQA

USE_NON_PRUNED = False

f = open("id2label.json")
id2label = json.load(f)
f = open("label2id.json")
label2id = json.load(f)

def config_maker(unilayers, hidden_size, number_heads, intermediate_size):
	multi_config = FlavaMultimodalConfig(
		num_hidden_layers=unilayers//2, 
		hidden_size=hidden_size, 
		num_attention_heads=number_heads, 
		intermediate_size= intermediate_size
		)
	
	image_config = FlavaImageConfig(
		num_hidden_layers=unilayers, 
		hidden_size=hidden_size, 
		num_attention_heads=number_heads, 
		intermediate_size= intermediate_size        
		)
	text_config  = FlavaTextConfig(
		num_hidden_layers=unilayers, 
		hidden_size=hidden_size, 
		num_attention_heads=number_heads, 
		intermediate_size= intermediate_size
		)
	configuration = FlavaConfig(
		multimodal_config=multi_config.to_dict(),
		image_config=image_config.to_dict(),
		text_config=text_config.to_dict(),
		hidden_size=hidden_size
		)
	return configuration


def get_sparsity(sparsity):
	matrix_sizes, num_zeros = 0, 0
	for sp in sparsity:
		num_zeros += sp[0]
		matrix_sizes += sp[1]
	return matrix_sizes, num_zeros

def main (model_info, max_pruning_threshold):
	config = model_info['config']
	model_name = model_info['model_name']
	model_class = model_info['model_class']
	model_location = model_info['model_location']
	processor = model_info['processor']
	cache_dir = model_info['cache_dir']
	size = model_info["size"]

	output_dir = os.path.join('./results/' if USE_NON_PRUNED else './results/nn_pruning/', f'{model_name}_{size}_VQA_dp_{max_pruning_threshold}_{datetime.now()}')
	print(f'Output directory: {output_dir}')
	os.makedirs(output_dir, exist_ok=True)




	pruning_thresholds = list(np.arange(0, max_pruning_threshold, 0.005))
	ks = [None] * len(pruning_thresholds)

	results = []
	for p1 in pruning_thresholds:
		for p2 in pruning_thresholds:
			for p3 in pruning_thresholds:
						
				print(f'Running inference with pruning threshold: {p1}, {p2}, {p3}')
				result = {'text_pruning_threshold': p1, 'image_pruning_threshold': p2, 'multimodal_pruning_threshold': p3} 
				temp_dir = os.path.join(output_dir, f'p1{str(p1)[2:]}_p2{str(p2)[2:]}_p3{str(p3)[2:]}')

				config.text_config.pruning_threshold = p1
				config.image_config.pruning_threshold = p2
				config.multimodal_config.pruning_threshold = p3
				config.text_config.sparsity_file = os.path.join(temp_dir, 'text_sparsity.json')
				config.image_config.sparsity_file = os.path.join(temp_dir, 'image_sparsity.json')
				config.multimodal_config.sparsity_file = os.path.join(temp_dir, 'multimodal_sparsity.json')
				config.save_pretrained(temp_dir)

				if os.path.exists(config.text_config.sparsity_file): os.remove(config.text_config.sparsity_file)
				if os.path.exists(config.image_config.sparsity_file): os.remove(config.image_config.sparsity_file)
				if os.path.exists(config.multimodal_config.sparsity_file): os.remove(config.multimodal_config.sparsity_file)

				model = FlavaForVQA(config, len(id2label))
				model.load_state_dict(torch.load(model_location))

				# run evalutation
				ann_file = "/scratch/gpfs/jmonas/VQA/v2_mscoco_val2014_annotations.json"
				images_dir = "/scratch/gpfs/jmonas/VQA/val2014"
				questions_file  =  "/scratch/gpfs/jmonas/VQA/v2_OpenEnded_mscoco_val2014_questions.json"
				metrics = evaluate(model, processor, size, questions_file, images_dir, 32, True, ann_file, .01)
				sparsity = {
					"text_sparsity" :json.load(open(config.text_config.sparsity_file)),
					"image_sparsity" :json.load(open(config.image_config.sparsity_file)),
					"multimodal_sparsity": json.load(open(config.multimodal_config.sparsity_file))
				}
				total_matrix_sizes, total_zeros = 0
				for encoder, spars in sparsity.items():
					matrix_sizes, num_zeros = get_sparsity(sparsity)
					total_matrix_sizes+= matrix_sizes
					total_zeros += total_zeros
					print(f'Resultant {encoder} activation sparsity: {num_zeros / matrix_sizes : 0.03f}')
					result[f'{encoder}_activation_sparsity'] = num_zeros / matrix_sizes
				print(f'Resultant total activation sparsity: {total_zeros / total_matrix_sizes : 0.03f}')
				result[f'total_activation_sparsity'] = total_zeros / total_matrix_sizes

				result['overall'] = metrics[0] 
				result['perAnswerType'] = metrics[1]

				results.append(result)
				json.dump(results, open(os.path.join(output_dir, 'results.json'), 'w+'))
				print(f"p1{str(p1)[2:]}_p2{str(p2)[2:]}_p3{str(p3)[2:]}, overall_accuracy = {result['overall']}")


if __name__ == '__main__':
		config_file = '/home/jmonas/acceltran/training/FLAVA/config_tiny.json'
		config = json.load(open(config_file))
		size = f"l{config['uni_layers']}_h{config['hidden_size']}_i{config['intermediate_size']}"
		config = config_maker(config["uni_layers"], config["hidden_size"], config["number_heads"], config["intermediate_size"])

		cache_dir= "/scratch/gpfs/jmonas"
		processor = FlavaProcessor.from_pretrained("facebook/flava-full", cache_dir="/scratch/gpfs/jmonas")
		model_class = DTFlavaForVQA
		model_location = f"/scratch/gpfs/jmonas/FLAVA/Models/{size}__B/flava-saved-model-ft_v3-0-22.pt"


		model_info = {
			"config": config,
			"model_name": DTFlavaForVQA.__name__,
			"model_class" : model_class,
			"model_location": model_location,
			"processor": processor,
			"cache_dir": cache_dir,
			"size":size
		}
		main(model_info, .01)

# main(model_info, .15, None)
# main(model_info, 0, 1, "top-k")





# def throughput ():



# 	dp = .05
# 	top_k = 16

# 	# Set p and k for 30% activation sparsity
# 	cases = [(dp, None), (0, top_k)]
# 	output_dir = f'./results/throughput/{model_name}_{size}_VQA'
# 	os.makedirs(output_dir, exist_ok=True)

# 	results = []

# 	for p, k in cases:
# 		print(f'Running inference with pruning threshold: {p}, and \'k\': {k}')
# 		result = {'pruning_threshold': p, 'k': k}

# 		# Make new output directory
# 		temp_dir = os.path.join(output_dir, f'threshold_p{str(p)[2:]}_k{int(k)}_{datetime.now()}')

# 		# Load and save new config
# 		config.pruning_threshold = p
# 		config.k = k
# 		config.sparsity_file = None
# 		config.save_pretrained(temp_dir)

# 		# Load model
# 		model = model_class.from_pretrained(model_location, config=config, use_safetensors=True, cache_dir=cache_dir)

# 		# Run evaluation on the SST-2 task or the SQuAD task
# 		start_time = time.time()
# 		metrics = run_glue(training_args)
# 		end_time = time.time()
# 		print(metrics)

# 		result['throughput'] = 256 / (end_time - start_time)
# 		print(f'Throughput: {256 / (end_time - start_time)} seq/sec')

# 		results.append(result)
# 		json.dump(results, open(os.path.join(output_dir, 'results.json'), 'w+'))

# 	return
		