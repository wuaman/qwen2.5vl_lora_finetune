import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import pathlib

from modelscope import AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
import torch
from torch.utils.data import Dataset
from transformers import (
	AutoProcessor,
	Qwen2_5_VLForConditionalGeneration,
	Trainer,
	TrainingArguments,
)

IGNORE_INDEX = -100


class Qwen2_5_VL_Dataset(Dataset):
	"""Qwen2.5-VL 数据集类，用于加载和预处理 MonkeyOCR 格式的训练数据"""

	def __init__(self, data_path: str, processor, tokenizer, max_length: int = 8192):
		"""
		初始化数据集

		Args:
			data_path: JSON 数据文件路径
			processor: AutoProcessor 实例
			tokenizer: AutoTokenizer 实例
			max_length: 最大序列长度
		"""
		self.processor = processor
		self.tokenizer = tokenizer
		self.max_length = max_length

		# 加载 JSON 数据
		with pathlib.Path(data_path).open(encoding='utf-8') as f:
			self.data = json.load(f)

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
		"""
		获取单个样本并进行预处理

		Args:
			idx: 样本索引

		Returns:
			处理后的数据字典
		"""
		example = self.data[idx]
		conversation = example['conversations']
		input_content = conversation[0]['value']  # user输入
		output_content = conversation[1]['value']  # assistant输出

		# 解析输入内容：格式为 "instruction <|vision_start|>image_path<|vision_end|>"
		if '<|vision_start|>' not in input_content or '<|vision_end|>' not in input_content:
			raise ValueError(f'Invalid input format: {input_content}')

		parts = input_content.split('<|vision_start|>')
		instruction_text = parts[0].strip()  # instruction部分
		file_path = parts[1].split('<|vision_end|>')[0].strip()  # 图像路径

		# 构建messages格式
		messages = [
			{
				'role': 'user',
				'content': [
					{
						'type': 'image',
						'image': file_path,
					},
					{'type': 'text', 'text': instruction_text},
				],
			}
		]

		# 应用chat template获取文本
		text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

		# 处理视觉信息
		image_inputs, video_inputs = process_vision_info(messages)

		# 使用processor处理输入
		inputs = self.processor(
			text=[text],
			images=image_inputs,
			videos=video_inputs,
			padding=True,
			return_tensors='pt',
			do_resize=False,
		)

		# 转换为list以便拼接
		inputs = {key: value.tolist() for key, value in inputs.items()}
		instruction = inputs

		# 处理输出内容
		response = self.tokenizer(f'{output_content}', add_special_tokens=False)

		# 拼接input_ids
		input_ids = instruction['input_ids'][0] + response['input_ids'] + [self.tokenizer.pad_token_id]

		# 构建labels（instruction部分用-100掩码，只计算response部分的loss）
		labels = (
			[IGNORE_INDEX] * len(instruction['input_ids'][0]) + response['input_ids'] + [self.tokenizer.pad_token_id]
		)

		# 截断到最大长度
		if len(input_ids) > self.max_length:
			input_ids = input_ids[: self.max_length]
			labels = labels[: self.max_length]

		# 转换为tensor
		input_ids = torch.tensor(input_ids)
		labels = torch.tensor(labels)
		pixel_values = torch.tensor(inputs['pixel_values'])
		image_grid_thw = torch.tensor(inputs['image_grid_thw'])

		return {
			'input_ids': input_ids,
			'labels': labels,
			'pixel_values': pixel_values,
			'image_grid_thw': image_grid_thw,
		}


@dataclass
class Qwen2_5_VL_DataCollator:
	"""Qwen2.5-VL 数据整理器，用于批量处理数据"""

	tokenizer: object

	def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
		"""
		整理批量数据

		Args:
			instances: 批量样本列表

		Returns:
			整理后的批量数据字典
		"""
		# 提取 input_ids 和 labels
		input_ids, labels = tuple([instance[key] for instance in instances] for key in ('input_ids', 'labels'))

		# 使用 pad_sequence 进行 padding
		input_ids = torch.nn.utils.rnn.pad_sequence(
			input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
		)
		labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

		# 动态生成 attention_mask
		attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

		# 处理视觉输入
		images = [inst['pixel_values'] for inst in instances if 'pixel_values' in inst]
		if len(images) != 0:
			concat_images = torch.cat(images, dim=0)
			grid_thw = torch.cat([inst['image_grid_thw'] for inst in instances if 'image_grid_thw' in inst], dim=0)
		else:
			concat_images = None
			grid_thw = None

		return {
			'input_ids': input_ids,
			'labels': labels,
			'attention_mask': attention_mask,
			'pixel_values': concat_images,
			'image_grid_thw': grid_thw,
		}


def predict(messages, model, processor):
	"""
	推理函数

	Args:
	messages: 消息列表
	model: 模型实例
	processor: AutoProcessor实例

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
	inputs = inputs.to('cuda')

	# 生成输出
	generated_ids = model.generate(**inputs, max_new_tokens=128)
	generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
	output_text = processor.batch_decode(
		generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)

	return output_text[0]


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='MonkeyOCR LoRA Training Script')
	parser.add_argument(
		'--model_path', type=str, default='./MonkeyOCR/Recognition', help='Path to the pretrained model'
	)
	parser.add_argument(
		'--train_data_path',
		type=str,
		required=True,
		default='./monkey_data/train_data.json',
		help='Path to training data JSON file',
	)
	parser.add_argument(
		'--output_dir', type=str, default='./output/Qwen2.5-VL-MonkeyOCR', help='Output directory for checkpoints'
	)
	parser.add_argument('--per_device_train_batch_size', type=int, default=4, help='Batch size per device')
	parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
	parser.add_argument('--num_train_epochs', type=int, default=2, help='Number of training epochs')
	parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
	parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank')
	parser.add_argument('--lora_alpha', type=int, default=4, help='LoRA alpha')
	parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
	parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
	parser.add_argument('--save_steps', type=int, default=100, help='Save steps')
	parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
	parser.add_argument('--test_data_path', type=str, default=None, help='Path to test data JSON file (optional)')

	args = parser.parse_args()

	# 加载tokenizer和processor
	print(f'Loading tokenizer and processor from {args.model_path}')
	tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
	processor = AutoProcessor.from_pretrained(args.model_path)

	# 加载模型
	print(f'Loading model from {args.model_path}')
	model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
		args.model_path,
		torch_dtype=torch.bfloat16,
		device_map='auto',
	)

	if args.gradient_checkpointing:
		model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

	# 加载训练数据
	print(f'Loading training data from {args.train_data_path}')
	train_dataset = Qwen2_5_VL_Dataset(
		data_path=args.train_data_path,
		processor=processor,
		tokenizer=tokenizer,
	)

	# 配置LoRA
	config = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
		inference_mode=False,  # 训练模式
		r=args.lora_rank,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		bias='none',
	)

	# 获取LoRA模型
	print('Applying LoRA configuration...')
	peft_model = get_peft_model(model, config)

	# 配置训练参数
	training_args = TrainingArguments(
		output_dir=args.output_dir,
		per_device_train_batch_size=args.per_device_train_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		logging_steps=args.logging_steps,
		logging_first_step=True,
		num_train_epochs=args.num_train_epochs,
		save_steps=args.save_steps,
		learning_rate=args.learning_rate,
		save_on_each_node=True,
		gradient_checkpointing=args.gradient_checkpointing,
		report_to='none',
		bf16=True,
	)

	# 配置Trainer
	data_collator = Qwen2_5_VL_DataCollator(tokenizer=tokenizer)
	trainer = Trainer(
		model=peft_model,
		args=training_args,
		train_dataset=train_dataset,
		data_collator=data_collator,
	)

	# 开始训练
	print('Starting training...')
	trainer.train()

	print(f'Training completed! Checkpoints saved to {args.output_dir}')

	# 如果提供了测试数据，进行推理测试
	if args.test_data_path:
		print(f'\nLoading test data from {args.test_data_path}')

		# 配置测试参数
		val_config = LoraConfig(
			task_type=TaskType.CAUSAL_LM,
			target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
			inference_mode=True,  # 测试模式
			r=args.lora_rank,
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout,
			bias='none',
		)

		# 获取最新的checkpoint
		checkpoint_dir = args.output_dir
		checkpoint_paths = pathlib.Path(checkpoint_dir).glob('checkpoint-*')
		if checkpoint_paths:
			latest_checkpoint = max(checkpoint_paths, key=lambda x: int(x.split('-')[-1]))
			print(f'Loading latest checkpoint: {latest_checkpoint}')
		else:
			raise ValueError('No checkpoint found!')

		# 加载LoRA权重
		val_peft_model = PeftModel.from_pretrained(model, model_id=latest_checkpoint, config=val_config)
		val_peft_model = val_peft_model.to('cuda')

		# 读取测试数据
		with pathlib.Path(args.test_data_path).open('r', encoding='utf-8') as f:
			test_dataset = json.load(f)

		print(f'\nRunning inference on {len(test_dataset)} test samples...')
		for item in test_dataset:
			input_content = item['conversations'][0]['value']

			# 解析输入内容
			parts = input_content.split('<|vision_start|>')
			instruction_text = parts[0].strip()
			image_path = parts[1].split('<|vision_end|>')[0].strip()

			messages = [
				{
					'role': 'user',
					'content': [{'type': 'image', 'image': image_path}, {'type': 'text', 'text': instruction_text}],
				}
			]

			response = predict(messages, val_peft_model, processor)
			print(f'Input: {instruction_text}')
			print(f'Output: {response}\n')
