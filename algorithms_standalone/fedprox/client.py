import copy
import logging
import torch

from algorithms_standalone.fedavg.client import FedAVGClient


class FedProxClient(FedAVGClient):
    def __init__(self, client_index, train_ori_data, train_ori_targets,
                 test_dataloader, train_data_num, test_data_num,
                 train_cls_counts_dict, device, args, model_trainer,
                 vae_model=None, dataset_num=None, mu=0.1):
        super().__init__(client_index, train_ori_data, train_ori_targets,
                         test_dataloader, train_data_num, test_data_num,
                         train_cls_counts_dict, device, args, model_trainer,
                         vae_model, dataset_num)
        self.mu = mu  # FedProx近端项系数
        self.global_model_params = None  # 存储全局模型参数
        logging.info(f"FedProx Client {client_index} initialized with mu={mu}")

    def set_model_params(self, model_parameters):
        """设置模型参数，同时保存全局模型参数用于近端项计算"""
        super().set_model_params(model_parameters)
        # 保存全局模型参数的副本
        self.global_model_params = copy.deepcopy(model_parameters)

    def _compute_proximal_loss(self, model):
        """计算FedProx的近端项损失"""
        if self.global_model_params is None:
            return 0.0
        
        proximal_loss = 0.0
        global_params_dict = dict(self.global_model_params)
        
        for name, param in model.named_parameters():
            if name in global_params_dict:
                global_param = global_params_dict[name].to(param.device)
                proximal_loss += torch.norm(param - global_param) ** 2
        
        return (self.mu / 2.0) * proximal_loss

    def fedprox_train(self, shared_data_x, shared_data_y, shared_targets,
                      round_idx, named_params, params_type="model",
                      global_other_params=None, traininig_start=False,
                      shared_params_for_simulation=None):
        """FedProx训练方法，添加近端项"""
        
        # 设置模型参数
        self.set_model_params(named_params)
        
        # 使用父类的fedavg_train方法，它会自动处理数据加载器
        return super().fedavg_train(
            shared_data_x, shared_data_y, shared_targets,
            round_idx, global_other_params, shared_params_for_simulation)

    def train(self, shared_data_x, shared_data_y, shared_targets, round_idx,
              named_params, params_type="model", global_other_params=None,
              shared_params_for_simulation=None):
        """训练入口方法"""
        return self.fedprox_train(
            shared_data_x, shared_data_y, shared_targets, round_idx,
            named_params, params_type, global_other_params,
            traininig_start=(round_idx == 0),
            shared_params_for_simulation=shared_params_for_simulation) 