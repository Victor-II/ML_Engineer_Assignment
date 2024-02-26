import torch
from transformers import pipeline
from src.data.dataset import *
from src.data.utils import read_dataset, get_urls, save_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
import json

# model_checkpoint = '/home/victor-ii/Desktop/ML_Assignment/models/checkpoint-5740'
# data_path = '/home/victor-ii/Desktop/ML_Assignment/data/dataset.json'
# urls_path = '/home/victor-ii/Desktop/ML_Assignment/urls.csv'

def inference(model_checkpoint:str, data_path:str, urls_path:str, save_path:str, lines:bool=True):

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    classifier = pipeline(task='token-classification', model=model, tokenizer=tokenizer)
    data = read_dataset(data_path, annotated_only=False)[:2000]

    urls = get_urls(urls_path)
    if lines == True:
        results = []
    else:
        results = dict()

    for item in data:
        sentence = ' '.join(item['tokens'])

        res = classifier(sentence)
        
        if res != []:
            products = ''

            if res[0]['entity'] == 'B-product':
                    word = res[0]['word'] if not res[0]['word'].startswith('##') else res[0]['word'][2:]
                    products = products + word

            elif res[0]['entity'] == 'I-product':
                if res[0]['word'].startswith('##'):
                    products = products + res[0]['word'][2:]
                else:
                    products = products + ' ' + res[0]['word']

            for entry in res[1:]:
                if entry['entity'] == 'B-product':
                    products = products + '<SEP>'
                    word = entry['word'] if not entry['word'].startswith('##') else entry['word'][2:]
                    products = products + word

                elif entry['entity'] == 'I-product':
                    if entry['word'].startswith('##'):
                        products = products + entry['word'][2:]
                    else:
                        products = products + ' ' + entry['word']
                
            product_list = products.split('<SEP>')

            if lines == True:
                for product in product_list:
                    results.append({"url_id":item['url_id'], "url": urls[item['url_id']], "product": product}) 
            else:
                value = results.get(urls[item['url_id']], list())
                value.extend(product_list)
                results[urls[item['url_id']]] = value

    if lines == True:
        save_dataset(results, save_path)
    else:
        with open(save_path, 'w') as f:
            json.dump(results, f)

    




        
        
        