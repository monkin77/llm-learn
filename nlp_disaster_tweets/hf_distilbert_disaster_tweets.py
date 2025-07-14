import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

def main():
    # 1. Load the data
    train_path = 'input/train.csv'
    test_path = 'input/test.csv'  # Not used for training, but can be used for inference
    train_df = pd.read_csv(train_path)
    print(f'Train shape: {train_df.shape}')
    print(train_df.head())

    # 2. Preprocess and split the data
    df = train_df[['text', 'target']].copy()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['target'].tolist(),
        test_size=0.1,
        random_state=42,
    )
    print(f'Train size: {len(train_texts)}, Validation size: {len(val_texts)}')

    # 3. Tokenize the data
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # 4. Load model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # 7. Evaluation
    preds = trainer.predict(val_dataset)
    print('Validation Accuracy:', preds.metrics.get('test_accuracy', preds.metrics.get('eval_accuracy')))
    print(classification_report(val_labels, preds.predictions.argmax(-1)))

    # 8. Save model and tokenizer
    model.save_pretrained('./distilbert-disaster-tweets')
    tokenizer.save_pretrained('./distilbert-disaster-tweets')
    print('Model and tokenizer saved!')

if __name__ == '__main__':
    main() 