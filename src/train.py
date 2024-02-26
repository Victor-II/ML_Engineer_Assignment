import torch
from src.data.dataset import build_datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import evaluate
import numpy as np


def train(model_checkpoint:str, num_labels:int, id2label:dict, label2id:dict,
          data_path:str, train_split:float, test_split:float, dev:bool,
          training_args:dict):
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    label_list = [key for key in label2id.keys()]
    seqeval = evaluate.load("seqeval")

    train_ds, test_ds = build_datasets(
        data_path=data_path,
        tokenizer=tokenizer,
        train_split=train_split,
        test_split=test_split,
        dev=dev
    )
    print(f'train_ds: {len(train_ds)}')
    print(f'test_ds: {len(test_ds)}')
    print(f'total: {len(test_ds)+len(train_ds)}')

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels, 
        id2label=id2label, 
        label2id=label2id
    )

    def compute_metrics(p):
        
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    args = TrainingArguments(
        output_dir = training_args.get('output_dir'),
        evaluation_strategy = training_args.get('evaluation_strategy', 'epoch'),
        save_strategy = training_args.get('save_strategy', 'epoch'),
        learning_rate = training_args.get('learning_rate', 1e-4),
        per_device_train_batch_size = training_args.get('per_device_train_batch_size', 32),
        per_device_eval_batch_size = training_args.get('per_device_eval_batch_size', 32),
        num_train_epochs = training_args.get('num_training_epochs', 1),
        weight_decay = training_args.get('weight_decay', 1e-5),
        load_best_model_at_end = training_args.get('load_best_model_at_end', True),
        auto_find_batch_size = training_args['auto_find_batch_size'],
        gradient_checkpointing = training_args['gradient_checkpointing']
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    if training_args.get('evaluate') == True:
        trainer.evaluate()


if __name__ == '__main__':

    pass