from algorithms_standalone.fedavg.aggregator import FedAVGAggregator


class FedProxAggregator(FedAVGAggregator):
    """FedProx聚合器，与FedAvg聚合方式相同"""
    
    def __init__(self, train_global_dl, test_global_dl, train_data_num,
                 test_data_num, train_data_local_num_dict, client_num,
                 device, args, model_trainer, vae_model=None):
        super().__init__(train_global_dl, test_global_dl, train_data_num,
                         test_data_num, train_data_local_num_dict, client_num,
                         device, args, model_trainer, vae_model)
    
    def aggregate(self):
        """FedProx使用与FedAvg相同的聚合方式"""
        return super().aggregate() 