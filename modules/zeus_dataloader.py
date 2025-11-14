from torch.utils.data import DataLoader

class Zeus_DataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        # self.dataset = dataset  # ensure reference
    
    def __getattr__(self, name):
        # If an attribute isn't found in DataLoader, look in dataset
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'ZeusLoader' object has no attribute '{name}'")
    
if __name__ == "__main__":
    
    import os, sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    from models.config_copy import config
    from modules.params_copy import Params
    from modules.zeus_dataset_copy import Zeus_Dataset
    
    pm = Params(config)
    dataset = Zeus_Dataset(pm, Path(config['dataset_path']), Path(config['management_path']))
    
    dataloader = Zeus_DataLoader(dataset, batch_size=None, shuffle=None)
    