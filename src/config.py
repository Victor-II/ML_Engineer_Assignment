import json

CONFIG_PATH = './src/configs/'

def get_config(path:str|None='config.json'):
    with open(CONFIG_PATH+path) as f:
        config = json.load(f)
    return config

if __name__ == '__main__':
    pass
