import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

# 定义三个提示词
INSTRUCTION_TEXT = 'Please output the text content from the image.'
INSTRUCTION_FORMULA = 'Please write out the expression of the formula in the image using LaTeX format.'
INSTRUCTION_TABLE = 'This is the image of a table. Please output the table in html format.'


def get_instruction_by_category(category: str) -> str:
	"""
	根据category返回对应的instruction

	Args:
		category: 类别（文本、公式、表格）

	Returns:
		对应的instruction字符串
	"""
	category_map = {
		'文本': INSTRUCTION_TEXT,
		'公式': INSTRUCTION_FORMULA,
		'表格': INSTRUCTION_TABLE,
	}
	instruction = category_map.get(category)
	if instruction is None:
		raise ValueError(f"Unknown category '{category}'. Supported categories: {list(category_map.keys())}")
	return instruction


def find_image_file(image_dir: str, base_name: str) -> Optional[str]:
	"""
	查找图像文件（支持多种扩展名）

	Args:
		image_dir: 图像目录
		base_name: 基础文件名（不含扩展名）

	Returns:
		找到的图像文件路径，如果未找到则返回None
	"""
	extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
	for ext in extensions:
		image_path = Path(image_dir) / f'{base_name}{ext}'
		if image_path.exists():
			return str(image_path)
	return None


def get_bbox_from_points(points: list[dict[str, float]]) -> tuple[int, int, int, int]:
	"""
	从points中提取边界框坐标

	Args:
		points: 点的列表，每个点包含x和y坐标

	Returns:
		(x_min, y_min, x_max, y_max) 边界框坐标
	"""
	x_coords = [p['x'] for p in points]
	y_coords = [p['y'] for p in points]

	x_min = int(min(x_coords))
	y_min = int(min(y_coords))
	x_max = int(max(x_coords))
	y_max = int(max(y_coords))

	return x_min, y_min, x_max, y_max


def crop_image_by_points(image_path: str, points: list[dict[str, float]], output_path: str) -> bool:
	"""
	根据points坐标切分图像

	Args:
		image_path: 原始图像路径
		points: 点的列表，每个点包含x和y坐标
		output_path: 切分后图像的保存路径

	Returns:
		是否成功切分
	"""
	try:
		# 打开原始图像
		img = Image.open(image_path)
		img_width, img_height = img.size

		# 获取边界框
		x_min, y_min, x_max, y_max = get_bbox_from_points(points)

		# 确保坐标在图像范围内
		x_min = max(0, x_min)
		y_min = max(0, y_min)
		x_max = min(img_width, x_max)
		y_max = min(img_height, y_max)

		# 检查边界框是否有效
		if x_max <= x_min or y_max <= y_min:
			print(f'Warning: Invalid bbox for {image_path}: ({x_min}, {y_min}, {x_max}, {y_max})')
			return False

		# 切分图像
		cropped_img = img.crop((x_min, y_min, x_max, y_max))

		# 处理图像模式：如果是RGBA或P模式（带透明通道），转换为RGB
		if cropped_img.mode in ('RGBA', 'LA', 'P'):
			# 创建白色背景
			rgb_img = Image.new('RGB', cropped_img.size, (255, 255, 255))
			if cropped_img.mode == 'P':
				cropped_img = cropped_img.convert('RGBA')
			rgb_img.paste(cropped_img, mask=cropped_img.split()[-1] if cropped_img.mode in ('RGBA', 'LA') else None)
			cropped_img = rgb_img
		elif cropped_img.mode != 'RGB':
			# 其他模式也转换为RGB
			cropped_img = cropped_img.convert('RGB')

		# 确保输出目录存在
		output_path_obj = Path(output_path)
		output_path_obj.parent.mkdir(exist_ok=True, parents=True)

		# 根据输出路径的扩展名决定保存格式
		output_ext = output_path_obj.suffix.lower()
		if output_ext in ('.jpg', '.jpeg'):
			cropped_img.save(str(output_path_obj), 'JPEG', quality=95)
		elif output_ext == '.png':
			cropped_img.save(str(output_path_obj), 'PNG')
		else:
			# 默认保存为JPEG
			if not output_ext:
				output_path_obj = output_path_obj.with_suffix('.jpg')
			cropped_img.save(str(output_path_obj), 'JPEG', quality=95)

		return True
	except Exception as e:
		print(f'Error cropping image {image_path}: {e}')
		import traceback

		traceback.print_exc()
		return False


def convert_monkeyocr_to_training_data(
	label_json_path: str,
	image_dir: str,
	output_json_path: str,  # noqa: ARG001
	cropped_image_dir: str,
	image_base_path: Optional[str] = None,
) -> list[dict[str, Any]]:
	"""
	将MonkeyOCR格式的数据转换为Qwen2.5-VL训练格式
	对每个目标进行图像切分，每个目标生成一个训练样本

	Args:
		label_json_path: MonkeyOCR的JSON标注文件路径
		image_dir: 原始图像文件所在目录
		output_json_path: 输出的训练数据JSON文件路径
		cropped_image_dir: 切分后图像保存目录
		image_base_path: 图像的基础路径（用于生成绝对路径，如果为None则使用相对路径）

	Returns:
		转换后的训练样本列表
	"""
	# 读取MonkeyOCR标注文件
	with Path(label_json_path).open(encoding='utf-8') as f:
		monkey_data = json.load(f)

	# 获取图像文件名（假设JSON文件名和图像文件名对应）
	label_file_name = Path(label_json_path).stem

	# 查找原始图像文件
	original_image_path = find_image_file(image_dir, label_file_name)
	if original_image_path is None:
		print(f'Warning: Image file not found for {label_file_name} in {image_dir}')
		return []

	# 处理每个label_ocr项
	training_samples = []
	label_ocr_list = monkey_data.get('label_ocr', [])

	if not label_ocr_list:
		print(f"Warning: No 'label_ocr' found in {label_json_path}")
		return []

	for idx, label_item in enumerate(label_ocr_list):
		category = label_item.get('category', '')
		text = label_item.get('text', '')
		points = label_item.get('points', [])

		# 跳过空文本
		if not text or not text.strip():
			continue

		# 检查points是否有效
		if not points or len(points) < 3:
			print(f'Warning: Invalid points for item {idx}, skipping')
			continue

		# 根据category选择对应的instruction
		try:
			instruction = get_instruction_by_category(category)
		except ValueError as e:
			print(f'Warning: {e}, skipping item {idx}')
			continue

		# 生成切分后图像的保存路径
		instance_id = label_item.get('instanceId', f'item_{idx}')
		# 获取原始图像的扩展名，保持相同的格式
		original_ext = Path(original_image_path).suffix.lower()
		if original_ext in ('.png', '.PNG'):
			cropped_image_filename = f'{label_file_name}_{instance_id}.png'
		else:
			cropped_image_filename = f'{label_file_name}_{instance_id}.jpg'
		cropped_image_path = Path(cropped_image_dir) / cropped_image_filename

		# 切分图像
		if not crop_image_by_points(original_image_path, points, str(cropped_image_path)):
			print(f'Warning: Failed to crop image for item {idx}, skipping')
			continue

		# 构建图像路径（用于训练数据）
		if image_base_path:
			# 使用绝对路径
			final_image_path = cropped_image_path.resolve()
		else:
			# 使用相对路径
			final_image_path = cropped_image_path

		# 构建训练数据格式
		sample = {
			'id': f'{label_file_name}_{idx}_{instance_id}',
			'conversations': [
				{'from': 'user', 'value': f'{instruction} <|vision_start|>{final_image_path}<|vision_end|>'},
				{'from': 'assistant', 'value': text},
			],
		}
		training_samples.append(sample)

	# 保存转换后的数据
	if training_samples:
		# with open(output_json_path, 'w', encoding='utf-8') as f:
		#     json.dump(training_samples, f, ensure_ascii=False, indent=2)
		print(f'Successfully converted {len(training_samples)} samples from {label_json_path}')
	else:
		print(f'Warning: No valid samples converted from {label_json_path}')

	return training_samples


def batch_convert_monkeyocr_data(
	label_dir: str, image_dir: str, output_json_path: str, cropped_image_dir: str, image_base_path: Optional[str] = None
) -> list[dict[str, Any]]:
	"""
	批量转换MonkeyOCR数据

	Args:
		label_dir: 包含MonkeyOCR JSON文件的目录
		image_dir: 原始图像文件所在目录
		output_json_path: 输出的合并后的训练数据JSON文件路径
		cropped_image_dir: 切分后图像保存目录
		image_base_path: 图像的基础路径

	Returns:
		所有转换后的训练样本列表
	"""
	label_dir = Path(label_dir)
	all_samples = []

	# 查找所有JSON文件
	json_files = list(label_dir.glob('*.json'))

	if not json_files:
		print(f'No JSON files found in {label_dir}')
		return []

	print(f'Found {len(json_files)} JSON files to process')

	# 确保切分图像目录存在
	Path(cropped_image_dir).mkdir(exist_ok=True, parents=True)

	for json_file in json_files:
		try:
			samples = convert_monkeyocr_to_training_data(
				str(json_file),
				image_dir,
				str(json_file.with_suffix('.train.json')),  # 临时文件,可以不用
				cropped_image_dir,
				image_base_path,
			)
			if samples:
				all_samples.extend(samples)
		except Exception as e:
			print(f'Error processing {json_file}: {e}')
			import traceback

			traceback.print_exc()
			continue

	# 保存所有样本
	if all_samples:
		with Path(output_json_path).open('w', encoding='utf-8') as f:
			json.dump(all_samples, f, ensure_ascii=False, indent=2)
		print(f'\nTotal: {len(all_samples)} training samples saved to {output_json_path}')
		print(f'Cropped images saved to {cropped_image_dir}')
	else:
		print('\nWarning: No samples were converted')

	return all_samples


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Convert MonkeyOCR data to Qwen2.5-VL training format')
	parser.add_argument('--label_json', type=str, help='Path to MonkeyOCR label JSON file')
	parser.add_argument(
		'--label_dir', type=str, help='Directory containing MonkeyOCR label JSON files (for batch processing)'
	)
	parser.add_argument('--image_dir', type=str, required=True, help='Directory containing original image files')
	parser.add_argument(
		'--cropped_image_dir', type=str, default='monkey_data/cropped_images', help='Directory to save cropped images'
	)
	parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
	parser.add_argument('--image_base_path', type=str, default=None, help='Base path for images (absolute path)')

	args = parser.parse_args()

	if args.label_json:
		# 单文件转换
		convert_monkeyocr_to_training_data(
			args.label_json, args.image_dir, args.output, args.cropped_image_dir, args.image_base_path
		)
	elif args.label_dir:
		# 批量转换
		batch_convert_monkeyocr_data(
			args.label_dir, args.image_dir, args.output, args.cropped_image_dir, args.image_base_path
		)
	else:
		parser.print_help()
		print('\nError: Either --label_json or --label_dir must be specified')
