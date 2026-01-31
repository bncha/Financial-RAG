from typing import Optional, Union
from config.settings import Settings 
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict


class HFDatasetService:
    """
    A handler class for managing connections and operations with Hugging Face Datasets.
    """

    def __init__(self, settings: Settings):
        token = settings.HF_APIKEY
        login(token)

    
    def load_data(self, path: str, name: Optional[str] = None, split : Optional[str] = None) -> Union[Dataset, DatasetDict]:
        
        dataset = load_dataset(path=path, name=name, split=split)
        
        return dataset

    @staticmethod
    def get_split(dataset: DatasetDict, split: str) -> Dataset:
        
        if isinstance(dataset, DatasetDict):
            if split in dataset:
                return dataset[split]
            else : 
                raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
        
        elif isinstance(dataset, Dataset):
            print("Can't split a single dataset. Returning original dataset")
            return dataset
        
        else: 
            raise TypeError(f"Expected DatasetDict, got {type(dataset)}")