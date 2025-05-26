import logging
import random

import numpy as np
import torch.utils.data as data

from .datasets import Yummly28k, Yummly28k_truncated, Yummly28k_truncated_WO_reload
from .transform import data_transforms_yummly28k


def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/Yummly28k/distribution.txt'):
    """读取数据分布文件"""
    distribution = {}
    with open(filename, 'r') as data_file:
        for x in data_file.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(
                        tmp[1].strip().replace(',', '')
                    )
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/Yummly28k/net_dataidx_map.txt'):
    """读取网络数据索引映射文件"""
    net_dataidx_map = {}
    with open(filename, 'r') as data_file:
        for x in data_file.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    """记录网络数据统计信息"""
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def load_yummly28k_data(datadir, resize=224, augmentation=True, 
                        classification_type="cuisine", args=None):
    """加载Yummly28k数据集"""
    train_transform, test_transform = data_transforms_yummly28k(
        resize=resize, augmentation=augmentation
    )

    if args and args.data_efficient_load:
        yummly_train_ds = Yummly28k(
            datadir, train=True, download=True, transform=train_transform,
            classification_type=classification_type
        )
        yummly_test_ds = Yummly28k(
            datadir, train=False, download=True, transform=test_transform,
            classification_type=classification_type
        )
    else:
        yummly_train_ds = Yummly28k_truncated(
            datadir, train=True, download=True, transform=train_transform,
            classification_type=classification_type
        )
        yummly_test_ds = Yummly28k_truncated(
            datadir, train=False, download=True, transform=test_transform,
            classification_type=classification_type
        )

    X_train, y_train = yummly_train_ds.data, yummly_train_ds.targets
    X_test, y_test = yummly_test_ds.data, yummly_test_ds.targets

    return (X_train, y_train, X_test, y_test, yummly_train_ds, yummly_test_ds)


def partition_data(dataset, datadir, partition, n_nets, alpha, resize=224, 
                   augmentation=True, classification_type="cuisine", args=None):
    """数据分区函数，支持多种分区策略"""
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test, yummly_train_ds, yummly_test_ds = (
        load_yummly28k_data(
            datadir, resize=resize, augmentation=augmentation,
            classification_type=classification_type, args=args
        )
    )

    X_train_np = np.array(range(len(X_train)))  # 使用索引而不是实际数据
    y_train_np = np.array(y_train)
    n_train = len(X_train)

    if partition == "homo":
        # 同构数据分布
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        # 异构数据分布（迪利克雷分布）
        min_size = 0
        K = yummly_train_ds.get_num_classes()  # 获取类别数
        N = y_train_np.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # 对每个类别进行分配
            for k in range(K):
                idx_k = np.where(y_train_np == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                
                if args and args.dirichlet_balance:
                    argsort_proportions = np.argsort(proportions, axis=0)
                    if k != 0:
                        used_p = np.array([len(idx_j) for idx_j in idx_batch])
                        argsort_used_p = np.argsort(used_p, axis=0)
                        inv_argsort_proportions = argsort_proportions[::-1]
                        proportions[argsort_used_p] = proportions[
                            inv_argsort_proportions
                        ]
                else:
                    proportions = np.array([
                        p * (len(idx_j) < N / n_nets) 
                        for p, idx_j in zip(proportions, idx_batch)
                    ])

                # 设置最小值以平滑分布
                if args and args.dirichlet_min_p is not None:
                    proportions += float(args.dirichlet_min_p)
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist() 
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        # 非IID标签分布
        num = eval(partition[13:])
        K = yummly_train_ds.get_num_classes()
        
        if num == K:
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_nets)}
            for i in range(K):
                idx_k = np.where(y_train_np == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_nets)
                for j in range(n_nets):
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
        else:
            times = [0 for i in range(K)]
            contain = []
            for i in range(n_nets):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while j < num:
                    ind = random.randint(0, K - 1)
                    if ind not in current:
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_nets)}
            for i in range(K):
                idx_k = np.where(y_train_np == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(n_nets):
                    if i in contain[j]:
                        net_dataidx_map[j] = np.append(
                            net_dataidx_map[j], split[ids]
                        )
                        ids += 1

    elif partition == "long-tail":
        # 长尾分布
        if n_nets == 10 or n_nets == 100:
            pass
        else:
            raise NotImplementedError
        
        # 主要客户端占alpha比例的数据
        main_prop = alpha / (n_nets // 10)
        # 尾部客户端分享剩余数据
        tail_prop = (1 - main_prop) / (n_nets - n_nets // 10)

        net_dataidx_map = {}
        K = yummly_train_ds.get_num_classes()
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train_np == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.array([tail_prop for _ in range(n_nets)])
            main_clients = np.array([k + i * K for i in range(n_nets // K)])
            proportions[main_clients] = main_prop
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist() 
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        # 固定异构分布
        dataidx_map_file_path = (
            './data_preprocessing/non-iid-distribution/Yummly28k/net_dataidx_map.txt'
        )
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = (
            './data_preprocessing/non-iid-distribution/Yummly28k/distribution.txt'
        )
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train_np, net_dataidx_map)

    return (X_train, y_train, X_test, y_test, net_dataidx_map, 
            traindata_cls_counts, yummly_train_ds, yummly_test_ds)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None,
                   resize=224, augmentation=True, classification_type="cuisine",
                   args=None, full_train_dataset=None, full_test_dataset=None):
    """获取数据加载器"""
    return get_dataloader_yummly28k(
        datadir, train_bs, test_bs, dataidxs,
        resize=resize, augmentation=augmentation,
        classification_type=classification_type, args=args,
        full_train_dataset=full_train_dataset,
        full_test_dataset=full_test_dataset
    )


def get_dataloader_yummly28k(datadir, train_bs, test_bs, dataidxs=None,
                             resize=224, augmentation=True,
                             classification_type="cuisine", args=None,
                             full_train_dataset=None, full_test_dataset=None):
    """获取Yummly28k数据加载器"""
    train_transform, test_transform = data_transforms_yummly28k(
        resize=resize, augmentation=augmentation
    )

    if args and args.data_efficient_load:
        dl_obj = Yummly28k_truncated_WO_reload
        train_ds = dl_obj(
            datadir, dataidxs=dataidxs, train=True, transform=train_transform,
            full_dataset=full_train_dataset
        )
        test_ds = dl_obj(
            datadir, train=False, transform=test_transform,
            full_dataset=full_test_dataset
        )
    else:
        dl_obj = Yummly28k_truncated
        train_ds = dl_obj(
            datadir, dataidxs=dataidxs, train=True, transform=train_transform,
            download=True, classification_type=classification_type
        )
        test_ds = dl_obj(
            datadir, train=False, transform=test_transform,
            download=True, classification_type=classification_type
        )

    drop_last = True
    if args and args.batch_size > len(train_ds):
        drop_last = False

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last,
        num_workers=0, pin_memory=False  # 禁用多进程和pin_memory
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False  # 禁用多进程和pin_memory
    )

    return train_dl, test_dl


def load_partition_data_distributed_yummly28k(process_id, dataset, data_dir,
                                              partition_method, partition_alpha,
                                              client_number, batch_size,
                                              classification_type="cuisine",
                                              args=None):
    """分布式加载分区数据"""
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, \
        yummly_train_ds, yummly_test_ds = partition_data(
            dataset, data_dir, partition_method, client_number, partition_alpha,
            classification_type=classification_type, args=args
        )
    
    class_num = yummly_train_ds.get_num_classes()
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # 获取全局测试数据
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(
            dataset, data_dir, batch_size, batch_size,
            classification_type=classification_type
        )
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # 获取本地数据集
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, dataidxs,
            classification_type=classification_type
        )
        logging.info(
            "process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                process_id, len(train_data_local), len(test_data_local)
            )
        )
        train_data_global = None
        test_data_global = None
    
    return (train_data_num, train_data_global, test_data_global, local_data_num,
            train_data_local, test_data_local, class_num)


def load_partition_data_yummly28k(dataset, data_dir, partition_method,
                                  partition_alpha, client_number, batch_size,
                                  classification_type="cuisine", args=None):
    """加载分区数据"""
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, \
        yummly_train_ds, yummly_test_ds = partition_data(
            dataset, data_dir, partition_method, client_number, partition_alpha,
            classification_type=classification_type, args=args
        )

    class_num = yummly_train_ds.get_num_classes()
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        dataset, data_dir, batch_size, batch_size, args=args,
        classification_type=classification_type,
        full_train_dataset=yummly_train_ds,
        full_test_dataset=yummly_test_ds
    )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # 获取本地数据集
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_index in range(client_number):
        dataidxs = net_dataidx_map[client_index]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_index] = local_data_num
        logging.info(
            "client_index = %d, local_sample_number = %d" % (
                client_index, local_data_num
            )
        )

        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, dataidxs, args=args,
            classification_type=classification_type,
            full_train_dataset=yummly_train_ds,
            full_test_dataset=yummly_test_ds
        )
        logging.info(
            "client_index = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_index, len(train_data_local), len(test_data_local)
            )
        )
        train_data_local_dict[client_index] = train_data_local
        test_data_local_dict[client_index] = test_data_local
    
    return (train_data_num, test_data_num, train_data_global, test_data_global,
            data_local_num_dict, train_data_local_dict, test_data_local_dict,
            class_num) 