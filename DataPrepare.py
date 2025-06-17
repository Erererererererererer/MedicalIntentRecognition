# -*- coding: utf-8 -*-

import os
from collections import Counter

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import json
import pandas as pd
import numpy as np
import pickle
import re
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader

# from torchtext.data.functional import to_map_style_dataset  # 只支持到torch 2.3.0

data_path = "./data/"
save_path = "./save/"
pretrain_model = "google-bert/bert-base-chinese"
# pretrain_model = "nghuyong/ernie-3.0-base-zh"


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
                data_list.append(sentence)
        data_df = pd.DataFrame(data_list)
        return data_df


def text_clean(text):
    # 去除特定符号
    cleaned_text = re.sub('[，。]', ' ', text)
    # 替换行内多余的空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text


# 将DataFrame中指定列转为list
def df2list(df, col):
    data = df[col]
    return data.tolist()


# 获取类别集合
def build_label_list(data_train, data_dev):
    label_train_list = df2list(data_train, "dialogue_act")
    label_dev_list = df2list(data_dev, "dialogue_act")
    label_all_list = label_train_list + label_dev_list
    label_set = set(label_all_list)
    return list(label_set)


def tokenize(data, context_length):
    outputs = tokenizer(
        data,
        truncation=True,  # 允许截断
        max_length=context_length,
        padding="max_length",
        return_attention_mask=True,  # 标记哪些位置是填充的，哪些不是
        # return_overflowing_tokens=True,
        # return_length=True,
    )
    return outputs["input_ids"], outputs["attention_mask"]


if __name__ == "__main__":
    params = {}
    # 读取训练数据
    print("读取训练数据...")
    data_train = read_json_file(data_path + "IMCS-DAC_train.json")  # train数据(DataFrame形式，对话拆分成单句)
    data_dev = read_json_file(data_path + "IMCS-DAC_dev.json")  # dev数据(DataFrame形式，对话拆分成单句)
    data_test = read_json_file(data_path + "IMCS-DAC_test.json")  # test数据(DataFrame形式，对话拆分成单句)
    print("Train: " + str(len(data_train)) + " | Dev: " + str(len(data_dev)) + " | Test: " + str(len(data_test)))
    # print(data_train[0:3])

    # 数据清洗
    data_train["sentence"] = data_train["sentence"].apply(text_clean)
    data_dev["sentence"] = data_dev["sentence"].apply(text_clean)
    data_test["sentence"] = data_test["sentence"].apply(text_clean)
    # print(data_train[0:3])

    # 获取分类标签
    dict_index2label = {}  # {(int)index: (String)label}
    dict_label2index = {}  # {(String)label: (int)index}
    if os.path.exists(data_path + "index2label.pkl"):
        # 若有，则直接用缓存
        with open(data_path + "index2label.pkl", "rb") as file:
            dict_index2label = pickle.load(file)
        with open(data_path + "label2index.pkl", "rb") as file:
            dict_label2index = pickle.load(file)
    else:
        label_list = build_label_list(data_train, data_dev)
        # 制作映射
        dict_index2label = dict(enumerate(label_list, 0))
        dict_label2index = {label: int(index) for index, label in
                            dict_index2label.items()}
        # 打包
        with open(data_path + "index2label.pkl", "wb") as file:
            pickle.dump(dict_index2label, file)
        with open(data_path + "label2index.pkl", "wb") as file:
            pickle.dump(dict_label2index, file)
    print("意图类别: " + str(len(dict_index2label)))
    # print(dict_index2label)
    # print(dict_label2index)

    # 绑定标签
    label_train_list = df2list(data_train, "dialogue_act")  # 意图分类列表
    label_dev_list = df2list(data_dev, "dialogue_act")
    label_train_index = [dict_label2index[l] for l in label_train_list]  # 意图分类(转index)列表
    label_dev_index = [dict_label2index[l] for l in label_dev_list]
    # 标签统计
    labels = label_train_index + label_dev_index
    classes_count = Counter(labels)
    classes_count = [classes_count[i] for i in range(len(classes_count))]
    params["classes_count"] = classes_count  # 保存打包

    # tokenize
    print("分词...")
    context_length = 32
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model, cache_dir=save_path)
    data_train_list = df2list(data_train, "sentence")
    data_dev_list = df2list(data_dev, "sentence")
    data_test_list = df2list(data_test, "sentence")
    data_train_tokens, data_train_mask = tokenize(data_train_list, context_length)
    data_dev_tokens, data_dev_mask = tokenize(data_dev_list, context_length)
    data_test_tokens, data_test_mask = tokenize(data_test_list, context_length)
    # print(input_train_tokens)
    vocab_size = tokenizer.vocab_size
    print("词表大小: " + str(vocab_size))
    params["vocab_size"] = vocab_size
    with open(data_path + "param.pkl", "wb") as file:
        pickle.dump(params, file)

    # 数据集封装
    print("数据集封装...")
    data_train_tensor = torch.tensor(data_train_tokens)
    data_dev_tensor = torch.tensor(data_dev_tokens)
    data_test_tensor = torch.tensor(data_test_tokens)
    mask_train_tensor = torch.tensor(data_train_mask)
    mask_dev_tensor = torch.tensor(data_dev_mask)
    mask_test_tensor = torch.tensor(data_test_mask)
    label_train_tensor = torch.from_numpy(np.array(label_train_index))
    label_dev_tensor = torch.from_numpy(np.array(label_dev_index))
    # 封装进Dataset
    dataset_train = TensorDataset(data_train_tensor, mask_train_tensor, label_train_tensor)
    dataset_dev = TensorDataset(data_dev_tensor, mask_dev_tensor, label_dev_tensor)
    dataset_test = TensorDataset(data_test_tensor, mask_test_tensor)
    # 定义DataLoader
    batch_size = 128
    # shuffle=False不打乱，drop_last=False若最后不足64也不舍弃
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)
    # 打印第一个批次来查看结构
    # for batch in train_loader:
    #     print(batch)
    #     break

    # 保存
    print("数据集保存...")
    with open(data_path + "trainLoader.pkl", "wb") as file:
        pickle.dump(train_loader, file)
    with open(data_path + "devLoader.pkl", "wb") as file:
        pickle.dump(dev_loader, file)
    with open(data_path + "testLoader.pkl", "wb") as file:
        pickle.dump(test_loader, file)
