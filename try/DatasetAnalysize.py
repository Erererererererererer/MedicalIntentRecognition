import pandas as pd
import matplotlib.pyplot as plt
import json
from DataPrepare import data_path
import pickle


# 设置后端为Agg，避免交互式问题
plt.switch_backend('agg')


# 标签id
dict_label2index = {}
with open("../" + data_path + "label2index.pkl", "rb") as file:
    dict_label2index = pickle.load(file)


# 读json文件，并转为DataFrame格式
def read_json_file(path):
    with open(path, "r", encoding="utf-8") as file:
        data_dict = json.load(file)  # dict
        data_list = []
        for item in data_dict.items():
            # 一个对话
            key = item[0]  # 对话id
            value = item[1]  # json列表
            for sentence in value:
                sentence["qa_id"] = key
                sentence["sentence"] = sentence["speaker"] + " " + sentence["sentence"]
                sentence["label"] = dict_label2index[sentence["dialogue_act"]]
                data_list.append(sentence)
        data_df = pd.DataFrame(data_list)
        return data_df

# 读取训练数据
print("读取训练数据...")
data_train = read_json_file("../" + data_path + "IMCS-DAC_train.json")  # train数据(DataFrame形式，对话拆分成单句)
data_dev = read_json_file("../" + data_path + "IMCS-DAC_dev.json")  # dev数据(DataFrame形式，对话拆分成单句)
print("Train: " + str(len(data_train)) + " | Dev: " + str(len(data_dev)))

# 假设df是你的训练DataFrame，label列包含0-15的标签
class_dist = data_dev['label'].value_counts(normalize=True)

plt.figure(figsize=(12, 6))
class_dist.sort_index().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class ID')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig("dataset.png")

# 识别长尾类别
tail_classes = class_dist[class_dist < 0.02].index.tolist()
print(f"长尾类别(占比<2%): {tail_classes}")
