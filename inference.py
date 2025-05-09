''' 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./best_model")
tokenizer = AutoTokenizer.from_pretrained("./best_model")

# Label map
id2label = model.config.id2label
label2id = model.config.label2id

# Load validation data
val_df = pd.read_csv('val.csv')
val_df['label'] = val_df['Labels'].map(lambda x: label2id[str(x)])

# Tokenize
def tokenize(batch):
    return tokenizer(batch['Interview Text'], truncation=True, padding=True)

val_ds = Dataset.from_pandas(val_df[['Interview Text', 'label']])
val_ds = val_ds.map(tokenize, batched=True)
val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Setup trainer for inference
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# Predict
predictions = trainer.predict(val_ds)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

# Classification report
print("\n📊 Classification Report:\n")
report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))])
print(report)

# Calculate and print specific evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f"\n📈 Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Weighted F1 Score: {f1:.2f}")
print(f"Weighted Precision: {precision:.2f}")
print(f"Weighted Recall: {recall:.2f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[id2label[i] for i in range(len(id2label))],
            yticklabels=[id2label[i] for i in range(len(id2label))])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


'''

# evaluate_model.py

import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# 1. Load and prepare data
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')  # Loading test data

unique_labels = sorted(val_df['Labels'].unique())
label2id = {str(lbl): i for i, lbl in enumerate(unique_labels)}
id2label = {i: str(lbl) for i, lbl in enumerate(unique_labels)}

val_df['label'] = val_df['Labels'].map(lambda x: label2id[str(x)])

# 2. Tokenize
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')  # Using DeBERTa

def tokenize(batch):
    return tokenizer(batch['Interview Text'], truncation=True, padding=True)

val_ds = Dataset.from_pandas(val_df[['Interview Text', 'label']])
test_ds = Dataset.from_pandas(test_df[['Interview Text']])  # No label for test data, adjust if needed
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)
val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_ds.set_format('torch', columns=['input_ids', 'attention_mask'])

# 3. Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    './best_model',  # Load the trained model from the saved directory
    num_labels=8,
    id2label=id2label,
    label2id=label2id
)

# 4. Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'weighted_f1': f1_score(labels, preds, average='weighted')
    }

# 5. Trainer
training_args = TrainingArguments(
    output_dir="./best_model",  # This is for saving but not used in evaluation
    per_device_eval_batch_size=8,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# 6. Evaluate on validation data
eval_results = trainer.evaluate()
print("\nValidation Evaluation Results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Weighted F1 Score: {eval_results['eval_weighted_f1']:.4f}")

# 7. Save evaluation results to results.json
eval_results_df = pd.DataFrame([eval_results])
eval_results_df.to_json('results.json', orient='records')

# 8. Make predictions on the test set
test_predictions = trainer.predict(test_ds)
test_preds = np.argmax(test_predictions.predictions, axis=1)

# 9. Create submission file for the test data
submission = pd.DataFrame({
    'ID': test_df['ID'],  # Ensure that the test set has an 'ID' column
    'Labels': [id2label[pred] for pred in test_preds]  # Convert numerical labels back to original labels
})
submission.to_csv('submission.csv', index=False)

print("\nSubmission and results have been saved.")
