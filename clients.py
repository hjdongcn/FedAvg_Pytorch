import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet


class client(object):
    def __init__(self, trainDataSet, dev):  # 构造函数，设置训练集和设备（gpu）
        self.train_ds = trainDataSet  # 训练集
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):  # 深度学习并更新参数
        Net.load_state_dict(global_parameters, strict=True)  # 加载模型
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)  # 导入训练集
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def local_val(self):
        pass  # 不做任何事情


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):  # 数据集名称，是否为独立同分布，client 数量，设备
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)  # 转成张量 tensor
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)  # 返回指定维度最大值的序号，dim=1表示行，也就是对应的label，因为表示成了one-hot
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)  # TensorDataset 可以用来对 tensor 进行打包，这样子test_data和test_label对应
        # DataLoader 载入数据，shuffle是每次epoch会不会打乱顺序
        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2  # // 向下取整，表示每个client分配了shard_size*2个train_data
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)  # 2*num_of_clients
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))  # vstack 垂直(行)按顺序堆叠数组
            local_label = np.argmax(local_label, axis=1)
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)  # 相当于随机找两段，连接起来
            self.clients_set['client{}'.format(i)] = someone  # client i所使用的数据集就是这一段

if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])

# client 这个类所做的就是分配数据集给这几个client


