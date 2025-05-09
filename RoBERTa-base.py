# RoBERTa-base with Oversampling and Advanced Training Settings

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from imblearn.over_sampling import RandomOverSampler

# ---------- 1. Load and Prepare Data ----------
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

# Label mapping
unique_labels = sorted(train_df['Labels'].unique())
label2id = {str(lbl): i for i, lbl in enumerate(unique_labels)}
id2label = {i: str(lbl) for i, lbl in enumerate(unique_labels)}
train_df['label'] = train_df['Labels'].map(lambda x: label2id[str(x)])
val_df['label'] = val_df['Labels'].map(lambda x: label2id[str(x)])

# ---------- 2. Oversample ----------
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(train_df[['Interview Text']], train_df['label'])
train_df_balanced = pd.DataFrame(X_resampled, columns=['Interview Text'])
train_df_balanced['label'] = y_resampled
print("Balanced class distribution:")
print(train_df_balanced['label'].value_counts())

# ---------- 3. Tokenize ----------
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def tokenize(batch):
    return tokenizer(batch['Interview Text'], truncation=True, padding=True)

train_ds = Dataset.from_pandas(train_df_balanced)
val_ds = Dataset.from_pandas(val_df[['Interview Text', 'label']])
test_ds = Dataset.from_pandas(test_df[['ID', 'Interview Text']])

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_ds.set_format('torch', columns=['input_ids', 'attention_mask'])

# ---------- 4. Load Model ----------
model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=8,
    id2label=id2label,
    label2id=label2id
)

# ---------- 5. Training Arguments ----------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",  # renamed argument
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_weighted_f1",
    greater_is_better=True,
    lr_scheduler_type="linear",
    warmup_steps=500
)

# ---------- 6. Evaluation Metrics ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'weighted_f1': f1_score(labels, preds, average='weighted')
    }

# ---------- 7. Trainer ----------
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

# ---------- 8. Train ----------
trainer.train()

# ---------- 9. Evaluate ----------
eval_results = trainer.evaluate()
print("\n📊 Evaluation Results:")
print("Validation Accuracy:", eval_results['eval_accuracy'])
print("Validation Weighted F1:", eval_results['eval_weighted_f1'])

predictions = trainer.predict(val_ds)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(8)]))

# ---------- 10. Confusion Matrix ----------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[id2label[i] for i in range(8)], yticklabels=[id2label[i] for i in range(8)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ---------- 11. Predict Test Set ----------
test_logits = trainer.predict(test_ds).predictions
test_preds = np.argmax(test_logits, axis=1)
predicted_labels = [id2label[p] for p in test_preds]

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Labels': predicted_labels
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv saved!")