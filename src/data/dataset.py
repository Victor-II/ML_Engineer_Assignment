from transformers import AutoTokenizer
from src.data.utils import read_dataset, split_data, get_encodings_and_labels, convert_to_hf_dataset
import json

def build_datasets(data_path:str,
                   tokenizer:AutoTokenizer,
                   train_split:float=0.8,
                   test_split:float=0.1,
                   dev:bool=True,
                   seed:int=42
                   ):
    datasets = []
    data = read_dataset(path=data_path, annotated_only=True)
    splits = split_data(data=data, train_split=train_split, test_split=test_split, dev=dev, seed=seed)
    for split in splits:
        split_encodings, split_labels = get_encodings_and_labels(data=split, tokenizer=tokenizer)
        split_dataset = convert_to_hf_dataset(encodings=split_encodings, labels=split_labels)
        datasets.append(split_dataset)

    return datasets

def get_config(path:str|None='config.json'):
    with open(path) as f:
        config = json.load(f)
    return config

if __name__ == '__main__':

    config = get_config('/home/victor-ii/Desktop/ML_Assignment/src/configs/config.json')
    tokenizer=AutoTokenizer.from_pretrained(config['model']['pretrained_model_checkpoint'])
    datasets = build_datasets(data_path=config['paths']['dataset'],
                              tokenizer=tokenizer,
                              train_split=config['dataset']['train_split'],
                              test_split=config['dataset']['test_split'])