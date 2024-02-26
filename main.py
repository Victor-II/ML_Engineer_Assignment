from src.config import get_config
from src.train import train
from src.inference import inference
from src.visualize import get_metrics, plot_metrics
from src.data.utils import *


if __name__ == '__main__':
    
    config = get_config()

    train(
        model_checkpoint=config['model']['pretrained_model_checkpoint'],
        num_labels=config['model']['num_labels'],
        id2label=config['model']['id2label'],
        label2id=config['model']['label2id'],
        data_path=config['paths']['dataset'],
        train_split=config['dataset']['train_split'],
        test_split=config['dataset']['test_split'],
        dev=config['dataset']['dev_split'],
        training_args=config['training_args'],
    )

    inference(
        model_checkpoint=config['model']['inference_model_checkpoint'],
        data_path=config['paths']['dataset'],
        urls_path=config['paths']['urls'],
        save_path=config['paths']['results'],
        lines=True
    )

    plot_metrics(get_metrics(config['model']['inference_model_checkpoint']),
                 save=True)
    

    # data = read_dataset(config['paths']['dataset'],annotated_only=True)
    # urls = get_dataset_statistics(data)
    # print(len(data))