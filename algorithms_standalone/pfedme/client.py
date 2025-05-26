import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms_standalone.fedavg.client import FedAVGClient


class PFedMeClient(FedAVGClient):
    def __init__(self, client_index, train_ori_data, train_ori_targets,
                 test_dataloader, train_data_num, test_data_num,
                 train_cls_counts_dict, device, args, model_trainer,
                 vae_model=None, dataset_num=None, beta=1.0, lamda=15.0,
                 K=5, personal_lr=0.01):
        super().__init__(client_index, train_ori_data, train_ori_targets,
                         test_dataloader, train_data_num, test_data_num,
                         train_cls_counts_dict, device, args, model_trainer,
                         vae_model, dataset_num)
        
        # PFedMe特有参数
        self.beta = beta
        self.lamda = lamda
        self.K = K  # 本地更新步数
        self.personal_lr = personal_lr
        
        # 个性化模型参数
        self.personal_model_params = None
        self.global_model_params = None
        
        logging.info(f"PFedMe Client {client_index} initialized with "
                    f"beta={beta}, lamda={lamda}, K={K}, personal_lr={personal_lr}")

    def _get_local_dataloader(self):
        """获取本地数据加载器"""
        return self.local_train_dataloader

    def set_model_params(self, model_parameters):
        """设置模型参数，同时初始化个性化模型"""
        super().set_model_params(model_parameters)
        self.global_model_params = copy.deepcopy(model_parameters)
        
        # 如果是第一次，初始化个性化模型参数
        if self.personal_model_params is None:
            self.personal_model_params = copy.deepcopy(model_parameters)

    def _update_personal_model(self, train_dataloader):
        """更新个性化模型参数"""
        if self.personal_model_params is None:
            return
        
        # 设置个性化模型参数
        self.trainer.set_model_params(self.personal_model_params)
        model = self.trainer.get_model()
        
        # 创建个性化优化器
        personal_optimizer = optim.SGD(model.parameters(), lr=self.personal_lr)
        
        # 个性化训练
        model.train()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            if batch_idx >= self.K:  # 限制本地更新步数
                break
                
            data, target = data.to(self.device), target.to(self.device)
            personal_optimizer.zero_grad()
            
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            # 添加正则化项
            reg_loss = 0.0
            global_params_dict = dict(self.global_model_params)
            for name, param in model.named_parameters():
                if name in global_params_dict:
                    global_param = global_params_dict[name].to(param.device)
                    reg_loss += torch.norm(param - global_param) ** 2
            
            total_loss = loss + (self.lamda / 2.0) * reg_loss
            total_loss.backward()
            personal_optimizer.step()
        
        # 保存更新后的个性化模型参数
        self.personal_model_params = self.trainer.get_model_params()

    def _compute_gradient_step(self, train_dataloader):
        """计算梯度步骤，用于服务器聚合"""
        if self.personal_model_params is None or self.global_model_params is None:
            return self.trainer.get_model_params()
        
        # 计算个性化模型与全局模型的差异
        gradient_params = {}
        personal_dict = dict(self.personal_model_params)
        global_dict = dict(self.global_model_params)
        
        for name in personal_dict:
            if name in global_dict:
                gradient_params[name] = (
                    global_dict[name] - personal_dict[name]) / self.beta
        
        return gradient_params

    def pfedme_train(self, shared_data_x, shared_data_y, shared_targets,
                     round_idx, named_params, params_type="model",
                     global_other_params=None, traininig_start=False,
                     shared_params_for_simulation=None):
        """PFedMe训练方法"""
        
        # 设置全局模型参数
        self.set_model_params(named_params)
        
        # 获取数据加载器
        if self.args.VAE and shared_data_x is not None and len(shared_data_x) > 0:
            train_dataloader = self._get_shared_dataloader(
                shared_data_x, shared_data_y, shared_targets)
        else:
            train_dataloader = self._get_local_dataloader()
        
        # 更新个性化模型
        self._update_personal_model(train_dataloader)
        
        # 计算梯度步骤用于聚合
        gradient_params = self._compute_gradient_step(train_dataloader)
        
        return gradient_params, None, self.local_sample_number, {}, shared_params_for_simulation

    def train(self, shared_data_x, shared_data_y, shared_targets, round_idx,
              named_params, params_type="model", global_other_params=None,
              shared_params_for_simulation=None):
        """训练入口方法"""
        return self.pfedme_train(
            shared_data_x, shared_data_y, shared_targets, round_idx,
            named_params, params_type, global_other_params,
            traininig_start=(round_idx == 0),
            shared_params_for_simulation=shared_params_for_simulation)

    def test_personal_model(self):
        """测试个性化模型性能"""
        if self.personal_model_params is None:
            return self.trainer.test(self.test_dataloader, self.device, self.args)
        
        # 设置个性化模型参数进行测试
        original_params = self.trainer.get_model_params()
        self.trainer.set_model_params(self.personal_model_params)
        
        test_results = self.trainer.test(self.test_dataloader, self.device, self.args)
        
        # 恢复原始参数
        self.trainer.set_model_params(original_params)
        
        return test_results 