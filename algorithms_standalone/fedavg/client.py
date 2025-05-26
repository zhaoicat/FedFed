import logging
import copy

from algorithms_standalone.basePS.client import Client
from model.build import create_model

class FedAVGClient(Client):
    def __init__(self, client_index, train_ori_data, train_ori_targets,test_dataloader, train_data_num,
                test_data_num, train_cls_counts_dict, device, args, model_trainer, vae_model, dataset_num):
        super().__init__(client_index, train_ori_data, train_ori_targets, test_dataloader, train_data_num,
                test_data_num,  train_cls_counts_dict, device, args, model_trainer, vae_model, dataset_num)
        local_num_iterations_dict = {}
        local_num_iterations_dict[self.client_index] = self.local_num_iterations

        self.global_epochs_per_round = self.args.global_epochs_per_round
        local_num_epochs_per_comm_round_dict = {}
        local_num_epochs_per_comm_round_dict[self.client_index] = self.args.global_epochs_per_round



        #========================SCAFFOLD=====================#
        if self.args.scaffold:
            self.c_model_local = create_model(self.args,
                model_name=self.args.model, output_dim=self.args.model_output_dim)
            for name, params in self.c_model_local.named_parameters():
                params.data = params.data*0


    # override
    def lr_schedule(self, num_iterations, warmup_epochs):
        epoch = None
        iteration = None
        round_idx = self.client_timer.local_comm_round_idx
        if self.args.sched == "no":
            pass
        else:
            if round_idx < warmup_epochs:
                # Because gradual warmup need iterations updates
                self.trainer.warmup_lr_schedule(round_idx*num_iterations)
            else:
                self.trainer.lr_schedule(round_idx)


    def test(self, epoch):
        acc_avg = self.trainer.test(epoch, self.test_dataloader, self.device)
        return acc_avg

    def fedavg_train(self, share_data1, share_data2, share_y, round_idx=None, global_other_params=None, 
                    shared_params_for_simulation=None,
                     **kwargs):

        client_other_params = {}
        train_kwargs = {}

        # ========================SCAFFOLD/FedProx=====================#
        if self.args.fedprox or self.args.scaffold:
            previous_model = copy.deepcopy(self.trainer.get_model_params())
            train_kwargs['previous_model'] = previous_model
        # ========================SCAFFOLD/FedProx=====================#

        # ========================SCAFFOLD=====================#
        if self.args.scaffold:
            c_model_global = global_other_params["c_model_global"]
            for name in c_model_global:
                c_model_global[name] = c_model_global[name].to(self.device)
            self.c_model_local.to(self.device)
            c_model_local = self.c_model_local.state_dict()

            train_kwargs['c_model_global'] = c_model_global
            train_kwargs['c_model_local'] = c_model_local
        # ========================SCAFFOLD=====================#

        iteration_cnt = 0
        for epoch in range(self.args.global_epochs_per_round):
            # 检查是否启用VAE和是否有共享数据
            if self.args.VAE and share_data1 is not None and share_data2 is not None and share_y is not None:
                # 使用混合数据加载器（包含共享数据）
                self.construct_mix_dataloader(share_data1, share_data2, share_y, round_idx)
                self.trainer.train_mix_dataloader(epoch, self.local_train_mixed_dataloader, self.device, **train_kwargs)
                logging.info("#############train finish for {epoch} epoch with mixed data on client {index} ########".format(
                        epoch=epoch, index=self.client_index))
            else:
                # 使用普通的本地数据加载器，调用简化的训练方法
                self._train_with_local_data(epoch, **train_kwargs)
                logging.info("#############train finish for {epoch} epoch with local data only on client {index} ########".format(
                        epoch=epoch, index=self.client_index))

        # ========================SCAFFOLD=====================#
        if self.args.scaffold:
            # refer to https://github.com/Xtra-Computing/NIID-Bench/blob/HEAD/experiments.py#L403-L411

            c_new_para = self.c_model_local.state_dict()
            c_delta_para = copy.deepcopy(self.c_model_local.state_dict())
            global_model_para = previous_model
            net_para = self.trainer.model.state_dict()
            if self.trainer.lr_scheduler is not None:
                current_lr = self.trainer.lr_scheduler.lr
            else:
                current_lr = self.args.lr

            logging.debug(f"current_lr is {current_lr}")
            for key in net_para:
                c_new_para[key] = c_new_para[key] - c_model_global[key] + \
                    (global_model_para[key].to(self.device) - net_para[key]) / (iteration_cnt * current_lr)
                c_delta_para[key] = (c_new_para[key] - c_model_local[key]).to('cpu')

            self.c_model_local.load_state_dict(c_new_para)
            self.trainer.model.to('cpu')
            self.c_model_local.to('cpu')
            client_other_params["c_delta_para"] = c_delta_para
        # ========================SCAFFOLD=====================#

        weights, model_indexes = self.get_model_params()

        return weights, model_indexes, self.test_data_num, client_other_params, shared_params_for_simulation  # 用于train的数据量

    def _train_with_local_data(self, epoch, **kwargs):
        """使用本地数据进行训练的简化方法"""
        from utils.set import AverageMeter, accuracy
        from utils.log_info import log_info
        
        self.trainer.model.to(self.device)
        self.trainer.model.train()
        
        loss_avg = AverageMeter()
        acc = AverageMeter()

        logging.info('\n=> Training Epoch #%d, LR=%.4f' % (epoch, self.trainer.optimizer.param_groups[0]['lr']))
        
        for batch_idx, (x, y) in enumerate(self.local_train_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)

            self.trainer.optimizer.zero_grad()
            out = self.trainer.model(x)
            loss = self.trainer.criterion(out, y)

            # FedProx正则化
            if self.args.fedprox and 'previous_model' in kwargs:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
                for name, param in self.trainer.model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(self.device)))**2)
                loss += fed_prox_reg

            loss.backward()
            self.trainer.optimizer.step()

            # ========================SCAFFOLD=====================#
            if self.args.scaffold and 'c_model_global' in kwargs:
                c_model_global = kwargs['c_model_global']
                c_model_local = kwargs['c_model_local']
                for name, param in self.trainer.model.named_parameters():
                    param.data = param.data - 0.000001 * \
                                 (c_model_global[name] - c_model_local[name]).to(param.data.device)
            # ========================SCAFFOLD=====================#

            prec1, prec5, correct, pred, _ = accuracy(out.data, y.data, topk=(1, 5))

            loss_avg.update(loss.data.item(), batch_size)
            acc.update(prec1.data.item(), batch_size)

            n_iter = (epoch - 1) * len(self.local_train_dataloader) + batch_idx

            log_info('scalar', 'client_{index}_train_loss_epoch_{epoch}'.format(index=self.client_index, epoch=epoch),
                     loss_avg.avg, step=n_iter, record_tool=self.args.record_tool, 
                     wandb_record=self.args.wandb_record)
            log_info('scalar', 'client_{index}_train_acc_epoch_{epoch}'.format(index=self.client_index, epoch=epoch),
                     acc.avg, step=n_iter, record_tool=self.args.record_tool,
                     wandb_record=self.args.wandb_record)

    def algorithm_on_train(self, share_data1, share_data2, share_y,round_idx,
            named_params, params_type='model',
            global_other_params=None,
            shared_params_for_simulation=None):

        if params_type == 'model':
            self.set_model_params(named_params)

        model_params, model_indexes, local_sample_number, client_other_params, shared_params_for_simulation = self.fedavg_train(
                share_data1, share_data2, share_y,
                round_idx,
                global_other_params,
                shared_params_for_simulation)
        return model_params, model_indexes, local_sample_number, client_other_params, shared_params_for_simulation

















