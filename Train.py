import csv
import os

import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pickle
from DataPrepare import data_path, save_path
import torch
from torch import optim
from model.BERTClassifier import BERTClassifier
from utils import print_progress_bar, draw_batch, draw_epoch


loss_epoch_train = []
acc_epoch_train = []
loss_epoch_dev = []
acc_epoch_dev = []


def train(model, device, train_loader, dev_loader, criterion, optimizer, num_epoch):
    model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=[2, 3, 4, 5])  # 多卡并行

    # lr = 5e-5
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        loss_train = []
        loss_ave_train = []
        acc_train = []
        acc_ave_train = []

        # optimizer = optim.AdamW(model.parameters(), lr=lr * 0.8, weight_decay=0.01)

        for batch, (data, mask, target) in enumerate(train_loader):
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(x=data, mask=mask)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            total += target.size(0)

            # 保存loss和acc
            loss_train.append(loss.item())
            loss_ave_train.append(train_loss / (batch + 1))
            acc_train.append((pred == target).sum().item() / target.size(0))
            acc_ave_train.append(train_correct / total)
            # 打印训练进度
            print_progress_bar(batch + 1, len(train_loader), prefix=f"Epoch{epoch + 1}:", length=50,
                               suffix=f" | Train Loss: {train_loss / (batch + 1):.5f}"
                                      f" | Train Accuracy: {train_correct / total * 100:.2f}%")

        train_accuracy = train_correct / total * 100

        # 在每个 epoch 中后调用测试模型返回的结果，以计算测试损失和测试准确率
        test_loss, test_accuracy = test(model, dev_loader, device, criterion)

        # 保存模型
        torch.save(model.state_dict(), save_path + "sample_model-" + str(epoch) + '.pt')
        # 可视化
        draw_batch(epoch, loss_train, "loss")
        draw_batch(epoch, loss_ave_train, "loss_average")
        draw_batch(epoch, acc_train, "accuracy")
        draw_batch(epoch, acc_ave_train, "accuracy_average")
        # 保存原数据
        with open(save_path + "results/results_epoch" + str(epoch) + ".csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["loss", "loss_ave", "acc", "acc_ave"])
            for i in range(len(loss_train)):
                writer.writerow([loss_train[i], loss_ave_train[i], acc_train[i], acc_ave_train[i]])


        # 保存epoch级别的loss和acc
        loss_epoch_train.append(train_loss / len(train_loader))
        acc_epoch_train.append(train_accuracy)

        print(f'Epoch: {epoch + 1}/{num_epoch} | '
              f'Train Loss: {train_loss / len(train_loader):.5f} | '
              f'Train Accuracy: {train_accuracy:.2f}% | '
              f'Test Loss: {test_loss:.5f} | '
              f'Test Accuracy: {test_accuracy:.2f}%')

    # 可视化
    draw_epoch(loss_epoch_train, loss_epoch_dev, "loss")
    draw_epoch(acc_epoch_train, acc_epoch_dev, "accuracy")


def test(model, data_loader, device, criterion):
    loss_dev = []
    acc_dev = []

    model.eval()
    test_loss = 0
    test_correct = 0
    total = 0

    # 测试函数无需梯度计算
    with torch.no_grad():
        for data, mask, target in data_loader:
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            # output = model(data)
            output = model(x=data, mask=mask)
            # 计算损失
            loss = criterion(output, target)
            # 损失累加
            test_loss += loss.item()
            # 在每一行找到最大概率的索引，这个索引即为模型的预测类别
            pred = output.argmax(dim=1)
            # 预测正确计数
            test_correct += (pred == target).sum().item()
            total += target.size(0)

            # 保存loss和acc
            loss_dev.append(loss.item())
            acc_dev.append((pred == target).sum().item())

    # 计算评价损失和平均准确率
    average_loss = test_loss / len(data_loader)
    average_accuracy = test_correct / total * 100
    # 保存epoch级别的loss和acc
    loss_epoch_dev.append(average_loss)
    acc_epoch_dev.append(average_accuracy)
    return average_loss, average_accuracy


if __name__ == "__main__":
    print("加载数据...")
    with open(data_path + "param.pkl", "rb") as file:
        params = pickle.load(file)
        vocab_size = params["vocab_size"]
        classes_count = params["classes_count"]
    with open(data_path + "trainLoader.pkl", "rb") as file:
        train_loader = pickle.load(file)  # 类型：torch.utils.data.DataLoader
    with open(data_path + "devLoader.pkl", "rb") as file:
        dev_loader = pickle.load(file)  # 类型：torch.utils.data.DataLoader

    print("加载模型...")
    # model_lstm = AttentionLSTM(d_emb=32, d_hidden=32, d_output=16, vocab=vocab_size)
    model_bert = BERTClassifier(d_output=16, dropout=0.1)
    model = model_bert
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    num_epoch = 5

    # 定义交叉熵损失函数
    # total_samples = sum(classes_count)  # 总样本数
    # CE_weight = [min(np.sqrt(total_samples) / (np.sqrt(count) * 16) * 10, 2) for count in classes_count]
    # print(CE_weight)
    # CE_weight_torch = torch.tensor(CE_weight, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss()  # 加权交叉熵weight=CE_weight_torch
    # 定义 Adam 优化器
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    print("开始训练...")
    train(model, device, train_loader, dev_loader, criterion, optimizer, num_epoch)
