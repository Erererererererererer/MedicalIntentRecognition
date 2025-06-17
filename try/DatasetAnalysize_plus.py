import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import json
from DataPrepare import data_path

# 设置后端为Agg，避免交互式问题
plt.switch_backend('agg')


# 标签id
dict_index2label = {}
with open("../" + data_path + "index2label.pkl", "rb") as file:
    dict_index2label = pickle.load(file)
dict_label2index = {}
with open("../" + data_path + "label2index.pkl", "rb") as file:
    dict_label2index = pickle.load(file)
# print(dict_index2label)


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

# 合并数据集以获得整体分布
all_labels = pd.concat([data_train['label'], data_dev['label']])

# 计算类别分布
class_counts = Counter(all_labels)
sorted_classes = sorted(class_counts.keys())
class_names = [dict_index2label[i] for i in range(len(dict_index2label))]
class_freq = [class_counts[c] for c in sorted_classes]
class_percent = [count / len(all_labels) * 100 for count in class_freq]

# 打印基本统计信息
print(f"总样本数: {len(all_labels)}")
print(f"类别数: {len(class_counts)}")
print("\n类别分布统计:")
for c, count, percent in zip(sorted_classes, class_freq, class_percent):
    print(f"类别 {c}-{dict_index2label[c]}: {count} 条 ({percent:.2f}%)")

# 识别长尾类别
long_tail_threshold = 0.02  # 定义长尾阈值为2%
long_tail_classes = [c for c, p in zip(sorted_classes, class_percent) if p < long_tail_threshold * 100]
print(f"\n长尾类别(占比<{long_tail_threshold * 100:.1f}%): {long_tail_classes}")

# 创建更美观的分布可视化
plt.figure(figsize=(14, 8))

# 使用条形图显示类别分布
ax = sns.barplot(x=class_names, y=class_freq, palette="viridis")
plt.title('class_distribution', fontsize=16)
plt.xlabel('label', fontsize=14)
plt.ylabel('amount', fontsize=14)
plt.xticks(rotation=45, ha='right')

# 在条形上方添加百分比标签
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 50,
            f'{class_percent[i]:.1f}%',
            ha='center', fontsize=10)

# 标记长尾类别
for i, c in enumerate(sorted_classes):
    if c in long_tail_classes:
        ax.patches[i].set_edgecolor('red')
        ax.patches[i].set_linewidth(2)

plt.tight_layout()

# 保存图像到文件
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
print("\n类别分布图已保存为 'class_distribution.png'")

# 显示长尾类别的详细信息
if long_tail_classes:
    print("\n长尾类别详细分析:")
    long_tail_df = data_train[data_train['label'].isin(long_tail_classes)]

    # 打印每个长尾类别的样本示例
    for c in long_tail_classes:
        samples = long_tail_df[long_tail_df['label'] == c]['sentence'].head(3).tolist()
        print(f"\n类别 {c} 的示例样本:")
        for i, sample in enumerate(samples, 1):
            print(f"  {i}. {sample}")