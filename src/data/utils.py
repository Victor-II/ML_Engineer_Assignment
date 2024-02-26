from bs4 import BeautifulSoup
import json
import numpy as np
import requests
import spacy
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Any


def get_urls(path:str, to_list:bool=True) -> pd.DataFrame | list:
    urls_df = pd.read_csv(path)
    if to_list == False:
        return urls_df
    else:
        return urls_df[urls_df.columns[0]].to_list()

def extract_and_filter_text(response_content:str) -> list[str]:
    soup = BeautifulSoup(response_content, 'html.parser')
    for script in soup(['meta', 'head', 'input', 'script', 'style', 'noscript', 'footer', 'header']):
        script.extract()

    # get text
    text = soup.get_text(separator="['SEP']", strip=True)
    res = ["".join([char for char in sentence if char.isalpha() or char.isnumeric() or char in " .,"]) for sentence in text.split("['SEP']") if "".join([char for char in sentence if char.isalpha()])]
        
    return res if len(res) > 0 else None

def get_filtered_text_from_url(url: str) -> list[str] | None:
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return extract_and_filter_text(resp.content) 
        else:
            return None
    except Exception as e:
        print(f'Second_Block_Exception: {e}')
    
def tokenize(sentences: list[str]) -> list[list]:
    tokenizer = spacy.load('en_core_web_sm').tokenizer
    return [[token.text for token in tokenizer(sentence) if not token.text.isspace()] for sentence in sentences]

def create_dataset(urls: list) -> list[dict]:
    dataset = []
    if isinstance(urls, str):
        urls = [urls]
    for url_id, url in enumerate(urls):
        sentences = get_filtered_text_from_url(url)[0]
        if sentences == None:
            continue
        else:
            sentences = tokenize(sentences)
            for sentence in sentences:
                item = {'url_id': url_id, 'tokens': sentence, 'ner_tags': []}
                dataset.append(item)
    return dataset

def save_dataset(dataset: list[dict], path:str) -> None:
    with open(path, 'w') as f:
        for item in dataset:      
            json.dump(item, f)
            f.write('\n')

def read_dataset(path:str, annotated_only:bool=False) -> list[dict]:
    dataset = []
    with open(path, 'r') as f:
        if annotated_only == True:
            for line in f:
                if json.loads(line)['ner_tags'] != []:
                    dataset.append(json.loads(line))
                else:
                    break
        else:
            for line in f:
                dataset.append(json.loads(line))
    return dataset

def split_data(data:list[dict], train_split:float, test_split:float, dev:bool=True, seed:int=42) -> tuple[list[dict], list[dict], list[dict]] | tuple[list[dict], list[dict]]:
    np.random.seed(seed)
    train_ds = []
    test_ds = []
    dev_ds = []
    indices = np.random.choice(len(data), len(data), replace=False)
    for i in indices[:int(len(indices)*train_split)]:
        train_ds.append(data[i])
    for i in indices[int(len(indices)*train_split):int(len(indices)*(train_split+test_split))]:
        test_ds.append(data[i])
    if dev == True:
        for i in indices[int(len(indices)*(train_split+test_split)):]:
            dev_ds.append(data[i])
        return train_ds, test_ds, dev_ds
    
    return train_ds, test_ds

def get_encodings_and_labels(data:list[dict], tokenizer:AutoTokenizer | Any) -> tuple[list[dict], list[int]]:
    encodings = []
    labels = []
    for i, elem in enumerate(data):
        tokenized_input = tokenizer(elem["tokens"], truncation=True, is_split_into_words=True)

        label = []
        word_ids = tokenized_input.word_ids()  # Map tokens to their respective word.
        previous_word_idx = None
        for word_id in word_ids:  # Set the special tokens to -100.
            if word_id is None:
                label.append(-100)
            elif word_id != previous_word_idx:  # Only label the first token of a given word.
                try:
                    label.append(elem['ner_tags'][word_id])
                except IndexError:
                    print(len(elem['ner_tags']))
                    print(word_id)
                    print(f'Record at index {i} may not be annotated')
                    break
            else:
                label.append(-100)
            previous_word_idx = word_id
        labels.append(label)
        encodings.append(tokenized_input)
    
    return encodings, labels

def convert_to_hf_dataset(encodings:list[dict], labels:list[int]) -> Dataset:
    dataset = []
    for encoding, label in zip(encodings, labels):
        item = {key: val for key, val in encoding.items()}
        item['labels'] = label
        dataset.append(item)
    return Dataset.from_list(dataset)

# def get_dataset_statistics(dataset:list[dict]):
#     url_ids = set()
#     for item in dataset:
#         url_ids.add(item['url_id'])
#     return url_ids

def annotate_dataset(data:list[dict], start_index:int|None=None):
    urls = get_urls("/home/victor-ii/Desktop/ML_Assignment/urls.csv")
    if start_index == None:
        for idx, elem in enumerate(data):
            if elem['ner_tags'] == []:
                i = idx
                break
    else:
        i = start_index

    while True:
        res = ''
        tokens = data[i]['tokens']
        for token_id, token in enumerate(tokens):
            res += f'\033[32m[{token_id}]\033[00m{token}'
        print(f'\n<{data[i]["url_id"]}> {urls[data[i]["url_id"]]}\n[{i}]{res}\n')
        ner_tags = [0] * len(tokens)
        tag_ids_pairs = [pair.split() for pair in input('Enter tag indices: ').split(',')]
        try:
            if len(tag_ids_pairs[0]) == 0:
                pass
            elif tag_ids_pairs[0][0] == 'q':
                break
            elif tag_ids_pairs[0][0] == 's':
                i += 1
                print("\033c", end="", flush=True)
                continue
            elif tag_ids_pairs[0][0] == 'a':
                ner_tags = [1] + [2] * (len(ner_tags) - 1)
            elif tag_ids_pairs[0][0] == 'b':
                i -= 2
            else:
                for tag_ids in tag_ids_pairs:
                    if len(tag_ids) == 1:
                        ner_tags[int(tag_ids[0])] = 1
                    elif len(tag_ids) == 2:
                        tag_ids = [int(idx) for idx in tag_ids]
                        if tag_ids[1] >= len(tokens) - 1:
                            tag_ids[1] = None
                        else:
                            tag_ids[1] = tag_ids[1] + 1
                        ner_tags[tag_ids[0]:tag_ids[1]] = [1] + [2] * len(ner_tags[tag_ids[0]+1:tag_ids[1]])
                    else:
                        print('Error: Too many arguments')
        except Exception as e:
            print(e)
            continue

        data[i]['ner_tags']= ner_tags
        i += 1
        print("\033c", end="", flush=True)

    return data

if __name__ == '__main__':

    pass
    
    
   
    



