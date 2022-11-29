import datetime
import os

import matplotlib.pyplot as plt
from cnn import *
from data.data_utils import get_CIFAR10_data
from solver import Solver
# from options import Options

def plot_fig(loss_history, train_acc_history, val_acc_history):
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(loss_history, 'o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(train_acc_history, '-o', label='train')
    plt.plot(val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()


def log_file_create():
    today = str(datetime.date.today())
    filename = f'{today}.log'
    filedirs = './log'

    log_filepath = str(filedirs+'/'+filename)
    if not os.path.exists(filedirs):
        os.makedirs(filedirs)

    time = str(datetime.datetime.now())
    msg = str("训练时间：") + time
    file = open(log_filepath, 'a')
    file.writelines(msg + '\n')
    file.close()


if __name__ == '__main__':
    # Augment
    log_file_create()

    cifar10_dir = '../cifar-10-batches-py'  # cifar文件路径
    batchsize = 20                                      # 设置批量大小
    epochs = 2                                          # 设置训练轮数
    learning_rate = 1e-3                                # 设置学习率

    # 获取cifar10里面的数据
    data = get_CIFAR10_data(cifar10_dir)

    # 获得模型
    model = ThreeLayerConvNet(reg=0.001)

    solver = Solver(model, data,
                    lr_decay=0.95,                                                       # lr_decay学习率减少步长
                    print_every=10, num_epochs=epochs, batch_size=batchsize,             # print_every 每40次打印一次
                    update_rule='adam',
                    optim_config={'learning_rate': learning_rate, 'momentum': 0.9})

    solver.train()

    plot_fig(solver.loss_history, solver.train_acc_history, solver.val_acc_history)

    best_model = model
    y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
    y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
    print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
    print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())