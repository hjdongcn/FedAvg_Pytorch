import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')  # 可以不让全部的client进行训练，每次只随机选取0.1*num个client
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')  # 发送1000次参数
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])  # 新建目录

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # dev = cuda

    net = None
    if args['model_name'] == 'mnist_2nn':  # 模型名字 默认为2nn
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)  # 并行
    net = net.to(dev)  # 将模型载入，在Models中

    loss_func = F.cross_entropy  # 计算交叉熵损失
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])  # SGD 梯度下降法

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)  # client组，默认为100个client
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))  # 参与communicate的数量 10

    global_parameters = {}
    for key, var in net.state_dict().items():  # 神经网络的各参数
        global_parameters[key] = var.clone()

    for i in range(args['num_comm']):  # communicate 数量，默认为1000
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args['num_of_clients'])  # 随机得到1-100的序列
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]  # 相当于随机选择num_in_comm=10个client参与communicate

        sum_parameters = None
        for client in tqdm(clients_in_comm):  # 处于communicate的client导入深度学习的超参
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,  # epoch默认为5 batchsize默认为10
                                                                         loss_func, opti, global_parameters)  # 损失函数，优化方法，全局参数
                                                                         # 返回学习epoch次后的参数
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:  # 5次输出一次准确率
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))

        if (i + 1) % args['save_freq'] == 0:  # 20次保存一次模型
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_accu{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                sum_accu / num,
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

