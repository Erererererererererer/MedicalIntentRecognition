import torch
import pickle
import json
from model.BERTClassifier import BERTClassifier
from DataPrepare import data_path, save_path


save_file = "sample_model-best.pt"


def read_json_file(path):
    with open(path, "r", encoding="utf-8") as file:
        data_dict = json.load(file)  # dict
        return data_dict


def save_json_file(dict):
    with open("output/predict.json", "w", encoding="utf-8") as file:
        json.dump(dict, file, indent=2, ensure_ascii=False)


def predict(model, device, data_loader):
    model.to(device)
    model.eval()
    target = list()

    with torch.no_grad():
        for data, mask in data_loader:
            data, mask = data.to(device), mask.to(device)
            # 获得输出结果
            output = model(x=data, mask=mask)
            # 在每一行找到最大概率的索引，这个索引即为模型的预测类别
            pred = output.argmax(dim=1)
            target.extend(pred.cpu().numpy().tolist())

    return target


if __name__ == "__main__":
    # 读取预测数据
    print("读取预测数据...")
    data_test = read_json_file(data_path + "IMCS-DAC_test.json")

    with open(data_path + "param.pkl", "rb") as file:
        vocab_size = pickle.load(file)
    with open(data_path + "testLoader.pkl", "rb") as file:
        test_loader = pickle.load(file)  # 类型：torch.utils.data.DataLoader

    # 意图预测
    print("加载模型...")
    # model = AttentionLSTM(d_emb=32, d_hidden=32, d_output=16, vocab=vocab_size)
    model_bert = BERTClassifier(d_output=16, dropout=0.5)
    model = model_bert
    model.load_state_dict(torch.load(save_path + save_file))
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    print("预测...")
    predict_list = predict(model, device, test_loader)

    # 保存预测数据
    print("保存数据...")
    dict_index2label = {}
    with open(data_path + "index2label.pkl", "rb") as file:
        dict_index2label = pickle.load(file)

    p_predict = 0  # 数据组织不一样，data_test以对话为第一维，每段对话中的句子为第二维；predict_list直接平铺成一维，所以用一个指针
    for qa_id, sentence_list in data_test.items():
        for data in sentence_list:
            data["dialogue_act"] = dict_index2label[predict_list[p_predict]]
            p_predict += 1

    # 写入文件
    save_json_file(data_test)
