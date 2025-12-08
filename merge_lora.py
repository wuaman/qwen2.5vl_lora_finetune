import argparse
import pathlib

from modelscope import AutoTokenizer
from peft import PeftModel
import torch
from transformers import (
	AutoProcessor,
	Qwen2_5_VLForConditionalGeneration,
)


def merge_lora_weights(base_model_path, lora_checkpoint_path, output_path):
	"""
	将LoRA权重合并到基础模型中

	Args:
		base_model_path: 基础模型路径
		lora_checkpoint_path: LoRA checkpoint路径
		output_path: 合并后模型的保存路径
	"""
	print(f'Loading base model from {base_model_path}...')
	base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
		base_model_path,
		torch_dtype=torch.bfloat16,
		device_map='auto',
	)

	print(f'Loading LoRA weights from {lora_checkpoint_path}...')
	peft_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)

	print('Merging LoRA weights into base model...')
	merged_model = peft_model.merge_and_unload()

	# 加载tokenizer和processor
	print('Loading tokenizer and processor...')
	tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
	processor = AutoProcessor.from_pretrained(base_model_path)

	# 保存合并后的模型
	output_path = pathlib.Path(output_path)
	output_path.mkdir(parents=True, exist_ok=True)

	print(f'Saving merged model to {output_path}...')
	merged_model.save_pretrained(output_path)
	tokenizer.save_pretrained(output_path)
	processor.save_pretrained(output_path)

	print(f'✅ Merged model saved successfully to {output_path}')


def find_latest_checkpoint(checkpoint_dir):
	"""
	查找最新的checkpoint

	Args:
		checkpoint_dir: checkpoint目录路径

	Returns:
		最新checkpoint的路径
	"""
	checkpoint_paths = list(pathlib.Path(checkpoint_dir).glob('checkpoint-*'))
	if not checkpoint_paths:
		raise ValueError(f'No checkpoint found in {checkpoint_dir}!')
	# 分离数字和非数字后缀的checkpoint
	numeric_checkpoints = []
	non_numeric_checkpoints = []

	for checkpoint_path in checkpoint_paths:
		suffix = checkpoint_path.name.split('-')[-1]
		try:
			# 尝试转换为数字
			num = int(suffix)
			numeric_checkpoints.append((num, checkpoint_path))
		except ValueError:
			# 非数字后缀（如 'best', 'last'）
			non_numeric_checkpoints.append(checkpoint_path)

	# 优先选择特殊标记的checkpoint（如 checkpoint-best, checkpoint-last）
	# 这些通常表示最佳或最后的checkpoint
	if non_numeric_checkpoints:
		# 优先选择 'best'，其次是 'last'，最后是其他
		priority_order = ['best', 'last']
		for priority in priority_order:
			for checkpoint in non_numeric_checkpoints:
				if checkpoint.name.endswith(f'-{priority}'):
					return checkpoint
		# 如果没有找到优先的，返回第一个非数字checkpoint
		return non_numeric_checkpoints[0]

	# 如果没有特殊checkpoint，选择数字最大的
	if numeric_checkpoints:
		numeric_checkpoints.sort(key=lambda x: x[0], reverse=True)
		return numeric_checkpoints[0][1]


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Merge LoRA weights into base model')
	parser.add_argument(
		'--base_model_path',
		type=str,
		default='./MonkeyOCR/Recognition',
		help='Path to the base pretrained model',
	)
	parser.add_argument(
		'--lora_checkpoint_path',
		type=str,
		default=None,
		help='Path to the LoRA checkpoint directory. If not specified, will use --checkpoint_dir to find latest checkpoint',
	)
	parser.add_argument(
		'--checkpoint_dir',
		type=str,
		default='./output/Qwen2.5-VL-MonkeyOCR',
		help='Directory containing checkpoints (used to find latest checkpoint if --lora_checkpoint_path not specified)',
	)
	parser.add_argument(
		'--output_path',
		type=str,
		required=True,
		help='Path to save the merged model',
	)

	args = parser.parse_args()

	# 如果没有指定具体的checkpoint路径，则查找最新的checkpoint
	if args.lora_checkpoint_path is None:
		print(f'Finding latest checkpoint in {args.checkpoint_dir}...')
		args.lora_checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
		print(f'Using checkpoint: {args.lora_checkpoint_path}')

	merge_lora_weights(args.base_model_path, args.lora_checkpoint_path, args.output_path)
