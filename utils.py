import time
import sys
import matplotlib.pyplot as plt

from DataPrepare import save_path


# 训练进度可视化
def print_progress_bar(
    iteration, total, prefix="", suffix="", decimals=1, length=50, fill="█"
):
    """
    打印进度条
    @params:
        iteration   - 必需  : 当前迭代（Int）
        total       - 必需  : 总迭代（Int）
        prefix      - 可选  : 前缀字符串（Str）
        suffix      - 可选  : 后缀字符串（Str）
        decimals    - 可选  : 有效的十进制数（Int）
        length      - 可选  : 进度条的长度（Int）
        fill        - 可选  : 填充字符（Str）
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    sys.stdout.write("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # 打印新的一行
    if iteration == total:
        print()


# 训练loss&acc可视化（可视化每个epoch中数值随batch的变化情况）
def draw_batch(epoch, result_list, title):
    batch = [i + 1 for i in range(len(result_list))]  # 横坐标
    # batch = batch[:-1]  # 最后一个batch数据量不足batch_size，不算
    # result_list = result_list[:-1]  # 最后一个batch数据量不足batch_size，不算

    plt.figure()
    plt.title(title + " in epoch" + str(epoch + 1))
    plt.plot(batch, result_list, label="train_" + title)
    plt.legend()
    plt.grid()
    plt.show()

    # 保存
    plt.savefig(save_path + "imgs/" + title + "_epoch-" + str(epoch) + ".png")

    plt.close("all")


# 训练loss&acc可视化（可视化整个训练中数值随epoch的变化情况）
def draw_epoch(train_list, dev_list, title):
    epoch = [i + 1 for i in range(len(train_list))]

    plt.figure()
    plt.title(title)
    plt.plot(epoch, train_list, label="train_" + title)
    plt.plot(epoch, dev_list, label="dev_" + title)
    plt.legend()
    plt.grid()
    plt.show()

    # 保存
    plt.savefig(save_path + "imgs/" + title + ".png")

    plt.close("all")
