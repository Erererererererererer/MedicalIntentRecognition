# -*- coding: utf-8 -*-

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 零样本分类
from transformers import pipeline
from DataPrepare import df2list, read_json_file, data_path, save_path
import pickle

model = "IDEA-CCNL/Erlangshen-Roberta-110M-NLI"

data_train = read_json_file(data_path + "IMCS-DAC_train.json")
data_train_list = df2list(data_train, "sentence")[:10]

with open(data_path + "index2label.pkl", "rb") as file:
    dict_index2label = pickle.load(file)

# labels = list(dict_index2label.values())
# labels = ['Inform-Drug_Recommendation', 'Inform-Basic_Information', 'Request-Medical_Advice',
#           'Request-Basic_Information', 'Inform-Medical_Advice', 'Request-Drug_Recommendation', 'Other', 'Diagnose',
#           'Request-Symptom', 'Inform-Precautions', 'Inform-Etiology', 'Inform-Symptom', 'Request-Etiology',
#           'Request-Precautions', 'Request-Existing_Examination_and_Treatment',
#           'Inform-Existing_Examination_and_Treatment']
labels = ["告知药物建议", "告知基本信息", "请求医疗建议", "请求基本信息", "告知医疗建议", "请求药物建议", "其他",
          "诊断", "请求症状信息", "告知注意事项", "告知病因", "告知症状", "请求病因信息", "请求注意事项",
          "请求现有检查和治疗信息", "告知现有检查和治疗"]

classifier = pipeline("zero-shot-classification", model=model, cache_dir=save_path)
output = classifier(data_train_list,
                    candidate_labels=labels)
print(output)
