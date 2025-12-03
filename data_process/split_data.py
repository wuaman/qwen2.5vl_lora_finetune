import json
import pathlib

if __name__ == '__main__':
	# 处理数据集：读取json文件
	# 拆分成训练集和测试集，保存为data_vl_train.json和data_vl_test.json
	train_json_path = 'coco_2014/data_vl.json'
	with pathlib.Path(train_json_path).open('r') as f:
		data = json.load(f)
		train_data = data[:-50]
		test_data = data[-50:]

	with pathlib.Path('coco_2014/data_vl_train.json').open('w') as f:
		json.dump(train_data, f)

	with pathlib.Path('coco_2014/data_vl_test.json').open('w') as f:
		json.dump(test_data, f)
