import logging
import time
from algorithms_standalone.fedavg.aggregator import FedAVGAggregator


class PFedMeAggregator(FedAVGAggregator):
    """PFedMe聚合器，使用梯度聚合方式"""
    
    def __init__(self, train_global_dl, test_global_dl, train_data_num,
                 test_data_num, train_data_local_num_dict, client_num,
                 device, args, model_trainer, vae_model=None):
        super().__init__(train_global_dl, test_global_dl, train_data_num,
                         test_data_num, train_data_local_num_dict, client_num,
                         device, args, model_trainer, vae_model)
    
    def aggregate(self):
        """PFedMe聚合方法，基于梯度的聚合"""
        start_time = time.time()
        
        model_list = []
        training_num = 0
        
        for idx in self.selected_clients:
            if idx in self.sample_num_dict and self.sample_num_dict[idx] > 0:
                model_list.append((self.sample_num_dict[idx], 
                                   self.model_dict[idx]))
                training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + 
                     str(len(self.model_dict)))

        if not model_list:
            # 如果没有有效的模型，返回当前全局模型
            global_model_params = self.trainer.get_model_params()
            return global_model_params, {}, None

        # 加权平均聚合梯度
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # 更新全局模型参数
        global_model_params = self.trainer.get_model_params()
        global_params_dict = dict(global_model_params)
        
        # 应用聚合后的梯度更新
        for name in averaged_params:
            if name in global_params_dict:
                global_params_dict[name] = (
                    global_params_dict[name] - averaged_params[name])

        # 清空本轮聚合数据
        self.model_dict = {}
        self.sample_num_dict = {}
        self.flag_client_model_uploaded_dict = {}
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        
        return global_params_dict, {}, None 