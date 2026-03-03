import site
site.ENABLE_USER_SITE = False

import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import watch
from collections import Counter

import os
import platform
import sys
from packaging import version
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score

import copy
import argparse
from glob import glob
import random
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback

)

from transformers.trainer_callback import TrainerCallback
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, ClassLabel
import evaluate

os.environ["COMET_API_KEY"] = "xxxxxxxxxxxxxxxxx"
os.environ["COMET_PROJECT_NAME"] = "polar"

experiment = Experiment(
    api_key="LifP49eHZV49oQCylB3x2Sy2n",
    project_name="polar",
)
experiment.set_name("XLM SemEval")


timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
print(timestamp)
print(f"Script Timestamp: {timestamp}")


##### Check PyTorch has access to GPU #####
def check_gpu_access():
    # Check the current operating system
    os_name = platform.system()
    device = 'cpu'

    # Switch case based on the operating system
    if os_name == 'Windows':
        # Windows specific GPU check
        if torch.cuda.is_available():
            device = 'cuda'
    elif os_name == 'Linux':
        # Linux specific GPU check
        if torch.cuda.is_available():
            device = 'cuda'
    elif os_name == 'Darwin':
        # Mac specific GPU check
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'

    return {
        "Operating System": os_name,
        "Device": device
    }


def load_and_label_csv(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    datasets_list = []

    for csv_file in csv_files:
        #splits csv file name in half and takes language abbreviation
        lang_code = csv_file.split(".")[0]
        #creates path for CSV
        file_path = os.path.join(folder_path, csv_file)

        #loads individual csv
        ds = load_dataset("csv", data_files=file_path)["train"] 
        #add language column
        ds = ds.map(lambda x: {"lang": lang_code})

        #adds to a dataset list
        datasets_list.append(ds)
    
    #combines datasets together
    combined_dataset = concatenate_datasets(datasets_list).shuffle(seed=1)
    return combined_dataset


class BalancedTrainer(Trainer):
    #weigh polarization for each language equally
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

    #weight batch sizes for each language individually
    def get_train_dataloader(self):
        langs = self.train_dataset["lang"]
        counts = Counter(langs)
        weight = np.array([1.0 / counts[l] for l in langs])

        sampler = WeightedRandomSampler(
            weights=weight,
            num_samples=len(self.train_dataset),
            replacement=True
        )
        if self.args.remove_unused_columns:
            dataset = self._remove_unused_columns(self.train_dataset)
        else:
            dataset = self.train_dataset

        return DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator
        )


class MultilingualMetrics:
    def __init__(self, eval_dataset):
        self.eval_dataset = eval_dataset

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        global_f1 = f1_score(labels, predictions, average="macro")
        global_acc = accuracy_score(labels, predictions)

        metrics = {
            "macro_f1": global_f1,
            "accuracy": global_acc
        }

        if "lang" in self.eval_dataset.column_names:
            lang_column = np.array(self.eval_dataset["lang"])
            unique_langs = np.unique(lang_column)

            for lang in unique_langs:
                mask = (lang_column == lang)
                lang_preds = predictions[mask]
                lang_labels = labels[mask]

                if len(lang_labels) > 0:
                    metrics[f"f1_{lang}"] = f1_score(lang_labels, lang_preds, average="macro")
                    metrics[f"acc_{lang}"] = accuracy_score(lang_labels, lang_preds)

        return metrics


def main(args):
    #check gpu access
    gpu_access = check_gpu_access()
    device = gpu_access['Device']
    device = torch.device(device)

    #experiment logging
    experiment.log_parameters(args)

    print(f"Using device: {device}")
    print(args)

    #seed for reproducibility
    some_seed = 1
    random.seed(some_seed)

    #get path to csvs for train and dev
    train_path = os.path.join(os.getcwd(), "test_phase", args.task, "train")
    dev_path = os.path.join(os.getcwd(), "test_phase", args.task, "dev")
    test_path = os.path.join(os.getcwd(), "test_phase", args.task, "test")

    #get csvs and combine them/sort by language
    train_dataset = load_and_label_csv(train_path)
    dev_dataset = load_and_label_csv(dev_path)

    num_labels = 2
    full_train_dataset = train_dataset.cast_column("polarization",ClassLabel(num_classes=num_labels)) 
    full_val_dataset = dev_dataset.cast_column("polarization", ClassLabel(num_classes= num_labels))

    #load huggingface model
    modelName = 'FacebookAI/xlm-roberta-large'
    #import tokenizer
    tokenizer = AutoTokenizer.from_pretrained(modelName)

    #tokenizer settings
    def tokenize(batch):
        #all batches will take in text
        #truncation is true 
        #all inputs have max of 128 tokens, then cut off
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256
        )

    #map each csv to a dataset that tokenizes and batches examples
    full_train_dataset = full_train_dataset.map(tokenize, batched=True)
    full_val_dataset = full_val_dataset.map(tokenize, batched=True)

    #Since dataset is tokenized and is in tensors, remove text column
    full_train_dataset = full_train_dataset.remove_columns(["text", "id"])
    full_val_dataset = full_val_dataset.remove_columns(["text", "id"])

    #rename labels
    full_train_dataset = full_train_dataset.rename_column("polarization", "labels")
    full_val_dataset = full_val_dataset.rename_column("polarization", "labels")

    #set format to pytorch tensors
    full_train_dataset.set_format("torch")
    full_val_dataset.set_format("torch")

    #load hugging face metric
    metric = evaluate.load("f1")

    #data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir = f"./outputs_{timestamp}",
        run_name=f"polar_{timestamp}",

        learning_rate = args.learningRate,
        per_device_train_batch_size=args.batchSize,
        per_device_eval_batch_size=args.batchSize,
        weight_decay=0.01, #COULD CHANGE?
        num_train_epochs=args.epochs,
        warmup_ratio=0.1, #COULD CHANGE?

        
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better = True,

        logging_dir="./logs",
        logging_steps=1,
        logging_strategy="steps", 
        report_to=["comet_ml"],
        max_grad_norm=1.0,
        gradient_accumulation_steps=2
    )

    #load model and add classification head
    model = AutoModelForSequenceClassification.from_pretrained(modelName, num_labels=2)

    compute_metrics_handler = MultilingualMetrics(full_val_dataset)

    trainer = BalancedTrainer(
        model = model,
        args = training_args,
        train_dataset = full_train_dataset,
        eval_dataset = full_val_dataset,
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics_handler,
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.005
            )
        ]
    )

    print(f"Dataset: {args.data}")
    print(f"Model: {args.modelName}")

    #training
    trainer.train()

    #evaluate
    eval_results = trainer.evaluate()
    print(eval_results)

    #save model
    trainer.save_model(f"./outputs/best_model_{timestamp}")
    tokenizer.save_pretrained(f"./outputs/best_model_{timestamp}")


    dev_csv_files = [f for f in os.listdir(test_path) if f.endswith(".csv")]

    for csv_file in dev_csv_files:
        lang_code = csv_file.split(".")[0]
        file_path = os.path.join(test_path, csv_file)

        # Load the individual CSV
        ds = load_dataset("csv", data_files=file_path)["train"]

        # Tokenize
        ds = ds.map(tokenize, batched=True)
        ds = ds.remove_columns(["text"])  # remove original text column
        ds.set_format("torch")

        # Make predictions
        preds = trainer.predict(ds).predictions
        pred_labels = np.argmax(preds, axis=-1)

        # Save individual CSV
        submission_df = pd.DataFrame({
            "id": ds["id"],  # or whatever column identifies rows
            "polarization": pred_labels
        })

        submission_file = f"./pred_{lang_code}.csv"
        submission_df.to_csv(submission_file, index=False)
        print(f"Saved predictions for {lang_code} to {submission_file}")

    #TODO
    #log training accuracy to comet (later if necessary)
    #optimizer/schedueler customization (later if necessary)


#code always runs, can customize easily here
batch = 16
lr = 1e-5
epoch = 8
task = 1
dataset = "SemEval2025"
strModel = "FacebookAI/xlm-roberta-large"

strTask = "subtask" + str(task)
print(f"batch size: {batch}, lr: {lr}, epoch: {epoch},  subtask: {strTask}")
main(argparse.Namespace(batchSize= batch, learningRate=lr, epochs=epoch, task=strTask, data=dataset, modelName=strModel))

print(f"Finished Job {timestamp}")
