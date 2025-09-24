from torch.utils.data import DistributedSampler as TorchDistributedSampler

class DistributedSampler(TorchDistributedSampler):
    def __init__(self, dataset, shuffle=True):
        super().__init__(dataset, shuffle=shuffle)