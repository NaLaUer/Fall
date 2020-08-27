import timeit
import os

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from config.args import setting_args
from src.util import *


print ("Device being used", torch.cuda.is_available())

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def main():
    parser = setting_args()
    args = parser.parse_args()

    # 1 设置训练集和测试机
    train_val_loaders, train_val_sizes, test_data_loader,\
                            test_size = prepare_dataset(args=args)
    # 2 模型初始化
    model, criterion, scheduler, optimizer= build(args=args)
    # 3 开始训练模型
    for epoch in range(0, args.epochs):
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()
            if phase == 'train':
                epoch_loss, epoch_acc, model, optimizer = train_model(
                                                    model, criterion, scheduler,
                                                    optimizer,train_val_loaders[phase],
                                                    train_val_sizes[phase])
            if phase == 'val':
                epoch_loss, epoch_acc = test_model(model, criterion, scheduler,
                                                    optimizer,train_val_loaders[phase],
                                                    train_val_sizes[phase])

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1,
                                            args.epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
        # 3.1 保存模型
        save_model(args, epoch, model, optimizer)
        # 3.2 训练验证集
        if args.useTest and epoch % args.nTestInterval == (args.nTestInterval - 1):
            start_time = timeit.default_timer()
            epoch_loss, epoch_acc = test_model(model, criterion, scheduler,
                                               optimizer, test_data_loader,
                                               test_size)
            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format("test", epoch + 1,
                                            args.epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

if __name__ == "__main__":
    main()
