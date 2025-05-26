import copy
import logging

from .client import PFedMeClient
from .aggregator import PFedMeAggregator
from utils.data_utils import (
    get_avg_num_iterations,
)
from algorithms_standalone.basePS.basePSmanager import BasePSManager
from model.build import create_model
from trainers.build import create_trainer
from model.FL_VAE import *


class PFedMeManager(BasePSManager):
    def __init__(self, device, args):
        # PFedMe特有参数 - 必须在super().__init__之前设置
        self.beta = getattr(args, 'pfedme_beta', 1.0)  # 个性化参数
        self.lamda = getattr(args, 'pfedme_lamda', 15.0)  # 正则化参数
        self.K = getattr(args, 'pfedme_K', 5)  # 本地更新步数
        self.personal_lr = getattr(args, 'pfedme_personal_lr', 0.01)  # 个性化学习率
        
        super().__init__(device, args)
        self.global_epochs_per_round = self.args.global_epochs_per_round
        
        logging.info(f"PFedMe initialized with beta={self.beta}, "
                    f"lamda={self.lamda}, K={self.K}, personal_lr={self.personal_lr}")

    def _setup_server(self):
        logging.info("############_setup_server (START)#############")
        model = create_model(self.args, model_name=self.args.model, 
                           output_dim=self.args.model_output_dim,
                           device=self.device, **self.other_params)
        init_state_kargs = {} 
        if self.args.VAE:
            VAE_model = FL_CVAE_cifar(args=self.args, d=self.args.VAE_d, 
                                    z=self.args.VAE_z, device=self.device)
        else:
            VAE_model = None

        model_trainer = create_trainer(   
            self.args, self.device, model,
            train_data_global_num=self.train_data_global_num,
            test_data_global_num=self.test_data_global_num, 
            train_data_global_dl=self.train_data_global_dl,
            test_data_global_dl=self.test_data_global_dl, 
            train_data_local_num_dict=self.train_data_local_num_dict,
            class_num=self.class_num, server_index=0, role='server',
            **init_state_kargs
        )

        self.aggregator = PFedMeAggregator(
            self.train_data_global_dl, self.test_data_global_dl, 
            self.train_data_global_num, self.test_data_global_num, 
            self.train_data_local_num_dict, self.args.client_num_in_total, 
            self.device, self.args, model_trainer, VAE_model)

        logging.info("############_setup_server (END)#############")

    def _setup_clients(self):
        logging.info("############setup_clients (START)#############")
        init_state_kargs = self.get_init_state_kargs()  
        
        for client_index in range(self.number_instantiated_client):
            if self.args.VAE:
                VAE_model = FL_CVAE_cifar(args=self.args, d=self.args.VAE_d, 
                                        z=self.args.VAE_z, device=self.device)
            else:
                VAE_model = None

            model = create_model(self.args, model_name=self.args.model, 
                               output_dim=self.args.model_output_dim,
                               device=self.device, **self.other_params)

            num_iterations = get_avg_num_iterations(
                self.train_data_local_num_dict, self.args.batch_size)   
            model_trainer = create_trainer(
                self.args, self.device, model, class_num=self.class_num,
                other_params=self.other_params, client_index=client_index, 
                role='client', **init_state_kargs)

            client = PFedMeClient(
                client_index,
                train_ori_data=self.train_data_local_ori_dict[client_index],
                train_ori_targets=self.train_targets_local_ori_dict[client_index],
                test_dataloader=self.test_data_local_dl_dict[client_index],
                train_data_num=self.train_data_local_num_dict[client_index],
                test_data_num=self.test_data_local_num_dict[client_index],
                train_cls_counts_dict=self.train_cls_local_counts_dict[client_index],
                device=self.device, args=self.args, model_trainer=model_trainer,
                vae_model=VAE_model, dataset_num=self.train_data_global_num,
                beta=self.beta, lamda=self.lamda, K=self.K, 
                personal_lr=self.personal_lr)
            
            self.client_list.append(client)
        logging.info("############setup_clients (END)#############")

    def check_end_epoch(self):
        return True

    def algorithm_train(self, round_idx, client_indexes, named_params, 
                       params_type, global_other_params, update_state_kargs, 
                       shared_params_for_simulation):
        
        for i, client_index in enumerate(client_indexes):
            copy_global_other_params = copy.deepcopy(global_other_params)
            if self.args.exchange_model:
                copy_named_model_params = copy.deepcopy(named_params)

            if self.args.instantiate_all:
                client = self.client_list[client_index]
            else:
                client = self.client_list[i]

            traininig_start = (round_idx == 0)

            # 客户端训练
            model_params, model_indexes, local_sample_number, \
            client_other_params, shared_params_for_simulation = \
                client.train(
                    self.global_share_dataset1, self.global_share_dataset2, 
                    self.global_share_data_y, round_idx, copy_named_model_params, 
                    params_type, copy_global_other_params,
                    shared_params_for_simulation=shared_params_for_simulation)

            self.aggregator.add_local_trained_result(
                client_index, model_params, model_indexes, 
                local_sample_number, client_other_params)

        # 聚合
        global_model_params, global_other_params, \
        shared_params_for_simulation = self.aggregator.aggregate()

        params_type = 'model'

        # 分发更新后的模型
        logging.info("distribute the updated model to all clients")
        for client_index in range(len(self.client_list)):
            self.client_list[client_index].set_model_params(global_model_params)
    
        return (global_model_params, params_type, global_other_params, 
                shared_params_for_simulation) 

    def train_with_progress(self, progress_tracker):
        """带进度跟踪的训练方法"""
        for round_idx in range(self.args.comm_round):
            # 更新进度跟踪器
            progress_tracker.update_round(round_idx)
            
            # 执行一轮训练
            self.train_one_round(round_idx)
            
            # 获取当前性能指标
            train_loss, train_acc = self.aggregator.test_on_server_for_all_clients("train")
            test_loss, test_acc = self.aggregator.test_on_server_for_all_clients("test")
            
            # 记录训练结果
            progress_tracker.record_round_results(
                round_idx, train_loss, train_acc, test_loss, test_acc
            )
        
        # 完成训练
        progress_tracker.finish_training()

    def train_one_round(self, round_idx):
        """执行一轮训练"""
        logging.info("################Communication round : {}".format(round_idx))
        
        # 采样客户端
        client_indexes = self.aggregator.client_sampling(
            round_idx, self.args.client_num_in_total, self.args.client_num_per_round)
        logging.info("client_indexes = " + str(client_indexes))

        # 获取全局模型参数
        global_model_params = self.aggregator.get_global_model_params()
        
        # 执行算法训练
        global_model_params, params_type, global_other_params, \
        shared_params_for_simulation = self.algorithm_train(
            round_idx, client_indexes, global_model_params, "model", 
            {}, {}, None)

        # 更新聚合器的全局模型
        self.aggregator.set_global_model_params(global_model_params) 