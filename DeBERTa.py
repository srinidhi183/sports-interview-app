# train_model.py

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from imblearn.over_sampling import RandomOverSampler
import json

# 1. Load and prepare data
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')

unique_labels = sorted(train_df['Labels'].unique())
label2id = {str(lbl): i for i, lbl in enumerate(unique_labels)}
id2label = {i: str(lbl) for i, lbl in enumerate(unique_labels)}

train_df['label'] = train_df['Labels'].map(lambda x: label2id[str(x)])
val_df['label'] = val_df['Labels'].map(lambda x: label2id[str(x)])

# 2. Oversample
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(train_df[['Interview Text']], train_df['label'])
train_df_balanced = pd.DataFrame(X_resampled, columns=['Interview Text'])
train_df_balanced['label'] = y_resampled

# 3. Tokenize
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')  # Changed to DeBERTa

def tokenize(batch):
    return tokenizer(batch['Interview Text'], truncation=True, padding=True)

train_ds = Dataset.from_pandas(train_df_balanced)
val_ds = Dataset.from_pandas(val_df[['Interview Text', 'label']])
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 4. Load model
model = AutoModelForSequenceClassification.from_pretrained(
    'microsoft/deberta-base',  # Changed to DeBERTa
    num_labels=8,
    id2label=id2label,
    label2id=label2id
)

# 5. Training arguments (updated output_dir and other hyperparameters)
training_args = TrainingArguments(
    output_dir="./best_model",  # ✅ Updated
    eval_strategy="steps",  # Evaluate every 'eval_steps' steps
    save_strategy="steps",  # Save checkpoints every 'save_steps' steps
    save_steps=100,
    eval_steps=100,  # Evaluate every 100 steps
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=12,  # Increased epochs
    weight_decay=0.01,
    learning_rate=5e-5,  # Increased learning rate
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_weighted_f1",  # Metric to load the best model
    greater_is_better=True,
    lr_scheduler_type="linear",
    warmup_steps=500,
    save_total_limit=1
)

# 6. Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'weighted_f1': f1_score(labels, preds, average='weighted')
    }

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 8. Train
trainer.train()

# 9. Evaluate on validation set
eval_results = trainer.evaluate()

# Save evaluation results to JSON
eval_results_dict = {
    "accuracy": eval_results["eval_accuracy"],
    "weighted_f1": eval_results["eval_weighted_f1"]
}

with open("results.json", "w") as f:
    json.dump(eval_results_dict, f, indent=4)



# Print evaluation results
print("\nEvaluation Results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Weighted F1 Score: {eval_results['eval_weighted_f1']:.4f}")