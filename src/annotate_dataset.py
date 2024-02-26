from data.utils import read_dataset, save_dataset, annotate_dataset
from config import get_config

def annotate_and_save(path, start_index:int|None=None):
    data = read_dataset(path, annotated_only=False)
    annotated_data = annotate_dataset(data, start_index)
    save_dataset(annotated_data, path)

if __name__ == '__main__':

    path = get_config()['paths']['dataset']
    annotate_and_save(path, start_index=0)
    # data = read_dataset(path, annotated_only=True)
    # urls = set()
    # for item in data:
    #     urls.add(item['url_id'])
    # print(len(urls))


    

    
