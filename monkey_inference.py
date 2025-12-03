import argparse
import json
from pathlib import Path
from typing import Any, Optional

from peft import LoraConfig, PeftModel, TaskType
from qwen_vl_utils import process_vision_info
import torch
from transformers import (
	AutoProcessor,
	AutoTokenizer,
	Qwen2_5_VLForConditionalGeneration,
)


def load_model(
	model_path: str,
	lora_checkpoint: Optional[str] = None,
	lora_rank: int = 16,
	lora_alpha: int = 4,
	lora_dropout: float = 0.05,
):
	"""
	加载模型（基础模型或带LoRA的模型）

	Args:
	model_path: 模型路径
	lora_checkpoint: LoRA checkpoint路径（可选）
	lora_rank: LoRA rank
	lora_alpha: LoRA alpha
	lora_dropout: LoRA dropout

	Returns:
	model: 模型实例
	processor: AutoProcessor实例
	tokenizer: AutoTokenizer实例
	"""
	print(f'正在加载模型: {model_path}')

	# 加载tokenizer和processor
	tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
	processor = AutoProcessor.from_pretrained(model_path)

	# 加载基础模型
	model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
		model_path,
		torch_dtype=torch.bfloat16,
		device_map='auto',
	)

	# 如果提供了LoRA checkpoint，加载LoRA权重
	if lora_checkpoint:
		print(f'正在加载LoRA权重: {lora_checkpoint}')
		val_config = LoraConfig(
			task_type=TaskType.CAUSAL_LM,
			target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
			inference_mode=True,
			r=lora_rank,
			lora_alpha=lora_alpha,
			lora_dropout=lora_dropout,
			bias='none',
		)
		model = PeftModel.from_pretrained(model, model_id=lora_checkpoint, config=val_config)
		print('LoRA权重加载完成')

	model.eval()
	print('模型加载完成')

	return model, processor, tokenizer


def predict(
	messages: list[dict[str, Any]],
	model: Qwen2_5_VLForConditionalGeneration,
	processor: AutoProcessor,
	max_new_tokens: int = 512,
	temperature: float = 0.7,
	top_p: float = 0.9,
) -> str:
	"""
	推理函数

	Args:
		messages: 消息列表，格式为 [{"role": "user", "content": [...]}]
		model: 模型实例
		processor: AutoProcessor实例
		max_new_tokens: 最大生成token数
		temperature: 温度参数
		top_p: top_p采样参数

	Returns:
		生成的文本
	"""
	# 准备推理
	text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	image_inputs, video_inputs = process_vision_info(messages)
	inputs = processor(
		text=[text],
		images=image_inputs,
		videos=video_inputs,
		padding=True,
		return_tensors='pt',
	)
	inputs = inputs.to(model.device)

	# 生成输出
	with torch.no_grad():
		generated_ids = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			temperature=temperature,
			top_p=top_p,
			do_sample=temperature > 0,
		)

	generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
	output_text = processor.batch_decode(
		generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)

	return output_text[0]


def inference_single_image(
	image_path: str,
	instruction: str,
	model: Qwen2_5_VLForConditionalGeneration,
	processor: AutoProcessor,
	max_new_tokens: int = 512,
	temperature: float = 0.7,
	top_p: float = 0.9,
) -> str:
	"""
	单张图片推理

	Args:
		image_path: 图片路径
		instruction: 提示词
		model: 模型实例
		processor: AutoProcessor实例
		max_new_tokens: 最大生成token数
		temperature: 温度参数
		top_p: top_p采样参数

	Returns:
		生成的文本
	"""
	if not Path(image_path).exists():
		raise FileNotFoundError(f'图片文件不存在: {image_path}')

	messages = [
		{
			'role': 'user',
			'content': [
				{
					'type': 'image',
					'image': image_path,
					'resized_height': 280,
					'resized_width': 280,
				},
				{'type': 'text', 'text': instruction},
			],
		}
	]

	return predict(messages, model, processor, max_new_tokens, temperature, top_p)


def inference_batch(
	image_paths: list[str],
	instruction: str,
	model: Qwen2_5_VLForConditionalGeneration,
	processor: AutoProcessor,
	max_new_tokens: int = 512,
	temperature: float = 0.7,
	top_p: float = 0.9,
) -> list[dict[str, str]]:
	"""
	批量推理

	Args:
		image_paths: 图片路径列表
		instruction: 提示词
		model: 模型实例
		processor: AutoProcessor实例
		max_new_tokens: 最大生成token数
		temperature: 温度参数
		top_p: top_p采样参数

	Returns:
		结果列表，每个元素包含image_path和result
	"""
	results = []
	for i, image_path in enumerate(image_paths, 1):
		print(f'正在处理第 {i}/{len(image_paths)} 张图片: {image_path}')
		try:
			result = inference_single_image(
				image_path, instruction, model, processor, max_new_tokens, temperature, top_p
			)
			results.append({'image_path': image_path, 'result': result})
			print(f'结果: {result}\n')
		except Exception as e:
			print(f'处理图片 {image_path} 时出错: {e}\n')
			results.append({'image_path': image_path, 'result': f'Error: {str(e)}'})

	return results


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
	"""
	查找最新的checkpoint

	Args:
		checkpoint_dir: Checkpoint目录

	Returns:
		最新checkpoint路径，如果未找到则返回None
	"""
	checkpoint_dir_path = Path(checkpoint_dir)
	checkpoint_paths = list(checkpoint_dir_path.glob('checkpoint-*'))
	if checkpoint_paths:
		latest_checkpoint = max(checkpoint_paths, key=lambda x: int(x.name.split('-')[-1]))
		return str(latest_checkpoint)
	return None


def main():
	parser = argparse.ArgumentParser(description='MonkeyOCR推理脚本')

	# 模型相关参数
	parser.add_argument('--model_path', type=str, default='./MonkeyOCR/Recognition', help='基础模型路径')
	parser.add_argument('--lora_checkpoint', type=str, default=None, help='LoRA checkpoint路径（可选）')
	parser.add_argument(
		'--checkpoint_dir', type=str, default=None, help='Checkpoint目录，如果指定会自动查找最新的checkpoint'
	)
	parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
	parser.add_argument('--lora_alpha', type=int, default=4, help='LoRA alpha')
	parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')

	# 输入相关参数
	parser.add_argument('--image_path', type=str, default=None, help='单张图片路径')
	parser.add_argument('--image_dir', type=str, default=None, help='图片目录（批量推理）')
	parser.add_argument('--image_list', type=str, default=None, help='图片路径列表文件（每行一个路径）')
	parser.add_argument(
		'--instruction', type=str, default='Please output the text content from the image.', help='提示词'
	)

	# 生成参数
	parser.add_argument('--max_new_tokens', type=int, default=512, help='最大生成token数')
	parser.add_argument('--temperature', type=float, default=0.7, help='温度参数')
	parser.add_argument('--top_p', type=float, default=0.9, help='top_p采样参数')

	# 输出相关参数
	parser.add_argument('--output', type=str, default=None, help='输出文件路径（JSON格式）')
	parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='设备')

	args = parser.parse_args()

	# 确定LoRA checkpoint
	lora_checkpoint = args.lora_checkpoint
	if lora_checkpoint is None and args.checkpoint_dir:
		lora_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
		if lora_checkpoint:
			print(f'自动找到最新checkpoint: {lora_checkpoint}')

	# 加载模型
	model, processor, tokenizer = load_model(
		args.model_path,
		lora_checkpoint,
		args.lora_rank,
		args.lora_alpha,
		args.lora_dropout,
		args.device,
	)

	# 收集图片路径
	image_paths = []
	if args.image_path:
		image_paths = [args.image_path]
	elif args.image_dir:
		# 从目录中读取所有图片
		image_dir = Path(args.image_dir)
		extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
		for ext in extensions:
			image_paths.extend([str(p) for p in image_dir.glob(f'*{ext}')])
		image_paths.sort()
	elif args.image_list:
		# 从文件中读取图片路径列表
		with Path(args.image_list).open(encoding='utf-8') as f:
			image_paths = [line.strip() for line in f if line.strip()]
	else:
		parser.error('必须指定 --image_path、--image_dir 或 --image_list 之一')

	if not image_paths:
		parser.error('未找到任何图片文件')

	print(f'找到 {len(image_paths)} 张图片')
	print(f'使用提示词: {args.instruction}\n')

	# 执行推理
	if len(image_paths) == 1:
		# 单张图片推理
		print(f'正在处理图片: {image_paths[0]}')
		result = inference_single_image(
			image_paths[0],
			args.instruction,
			model,
			processor,
			args.max_new_tokens,
			args.temperature,
			args.top_p,
		)
		print(f'\n结果:\n{result}')

		# 保存结果
		if args.output:
			output_data = [{'image_path': image_paths[0], 'instruction': args.instruction, 'result': result}]
			with Path(args.output).open('w', encoding='utf-8') as f:
				json.dump(output_data, f, ensure_ascii=False, indent=2)
			print(f'\n结果已保存到: {args.output}')
	else:
		# 批量推理
		results = inference_batch(
			image_paths,
			args.instruction,
			model,
			processor,
			args.max_new_tokens,
			args.temperature,
			args.top_p,
		)

		# 保存结果
		output_data = {'instruction': args.instruction, 'results': results}

		if args.output:
			with Path(args.output).open('w', encoding='utf-8') as f:
				json.dump(output_data, f, ensure_ascii=False, indent=2)
			print(f'\n所有结果已保存到: {args.output}')
		else:
			# 如果没有指定输出文件，打印到控制台
			print('\n=== 推理结果汇总 ===')
			for item in results:
				print(f'\n图片: {item["image_path"]}')
				print(f'结果: {item["result"]}')


if __name__ == '__main__':
	main()
