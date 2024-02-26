import matplotlib.pyplot as plt
import numpy as np
import json

model_checkpoint = 'checkpoint-5740'
models_path = '/home/victor-ii/Desktop/ML_Assignment/models/'


def get_metrics(model_checkpoint:str):
    trainer_state = f'{model_checkpoint}/trainer_state.json'

    with open(trainer_state, 'r') as f:
        log_history = json.load(f)['log_history']

    metrics = {
        'eval_accuracy': [],
        'eval_f1': [],
        'eval_loss': [],
        'eval_precision': [],
        'eval_recall': []
    }

    for i in range(len(log_history)):
        for key, val in metrics.items():
            try:
                val.append(log_history[i][key])
            except KeyError:
                continue

    return metrics
            
def plot_metrics(metrics:dict[list], save:bool=True):
    x = np.linspace(1, len(metrics['eval_accuracy']), len(metrics['eval_accuracy']))
    plt.figure()
    for key, val in metrics.items():
        plt.plot(x, val, label=key, marker='o', markersize=3)
    plt.legend()
    if save == True:
        plt.savefig('metrics_plot')
    plt.show()


if __name__ == '__main__':

    metrics = get_metrics(model_checkpoint, models_path)
    plot_metrics(metrics)