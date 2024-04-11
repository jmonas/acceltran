import os
import sys
import numpy as np
import math 
import json 
import sys 
sys.path.insert(0, '../../training/ViLT/test')
from eval import evaluate
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import torch

USE_NON_PRUNED = False

def main (model_name, model, processor, weights_pruned = False, max_pruning_threshold = 0, min_k = None, method = "dynatran"):
	output_dir = os.path.join('./results/' if USE_NON_PRUNED else './results/nn_pruning/', f'{model_name}_VQA_{"dp" if max_pruning_threshold > 0 else "top-k"}')
	print(f'Output directory: {output_dir}')
	os.makedirs(output_dir, exist_ok=True)


	if method == "dynatran":
		assert(max_pruning_threshold > 0 and min_k is None, 'Either min_k has to be None or max_pruning_threshold has to be zero')
		pruning_thresholds = list(np.arange(0, max_pruning_threshold, 0.005))
		ks = [None] * len(pruning_thresholds)

	else:
		assert(max_pruning_threshold == 0 and min_k,'Either min_k has to be None or max_pruning_threshold has to be zero')
		logk = math.log(min_k, 2)
		ks = list(np.logspace(logk, 9, num=10, base=2))
		pruning_thresholds = [0] * len(ks)

	results = []
	for p, k in zip(pruning_thresholds, ks):
		print(f'Running inference with pruning threshold: {p}, and \'k\': {k}')
		result = {'pruning_threshold': p, 'k': k} 
		temp_dir = os.path.join(output_dir, f'threshold_p{str(p)[2:]}_k{int(k)}')

		config = model.config
		config.pruning_threshold = p
		config.k = k
		config.sparsity_file = os.path.join(temp_dir, 'sparsity.json')
		config.save_pretrained(temp_dir)

		if os.path.exists(config.sparsity_file): os.remove(config.sparsity_file)

		# run evalutation
		ann_file = "/scratch/network/jmonas/VQA/v2_mscoco_val2014_annotations.json"
		images_dir = "/scratch/network/jmonas/VQA/val2014"
		questions_file  =  "/scratch/network/jmonas/VQA/v2_OpenEnded_mscoco_val2014_questions.json"
		metrics = evaluate(model, processor, config, questions_file, images_dir, 32, True, ann_file, .01)
		if p > 0 or k is not None:
			sparsity = json.load(open(config.sparsity_file))
			print("sparsity: ", sparsity)

			matrix_sizes, num_zeros = 0, 0
			for sp in sparsity:
				num_zeros += sp[0]
				matrix_sizes += sp[1]

			print(f'Resultant activation sparsity: {num_zeros / matrix_sizes : 0.03f}')
			result['activation_sparsity'] = num_zeros / matrix_sizes
		else:
			result['activation_sparsity'] = 0

		result['overall'] = metrics['overall'] 
		result['perAnswerType'] = metrics['perAnswerType']

		results.append(result)
		json.dump(results, open(os.path.join(output_dir, 'results.json'), 'w+'))
		print("p = {p}, k = {k}, overall_accuracy = {}")


if __name__ == '__main__':
		config_file = 'config_medium_plus.json'
		config = json.load(open(config_file))
		size = f"l{config['num_hidden_layers']}_h{config['hidden_size']}_i{config['intermediate_size']}"
		cache_dir= "/scratch/gpfs/jmonas"
		model_location = f"{cache_dir}/ViLT/Models/{size}/vilt-saved-model-ft-93-0"
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		configuration = ViltConfig(**config)
		processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=cache_dir)
		model =ViltForQuestionAnswering.from_pretrained(model_location, config=configuration, use_safetensors=True, cache_dir=cache_dir)
		main(model, processor, False, 0.1, None)