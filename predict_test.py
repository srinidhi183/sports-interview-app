# evaluate_model.py

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# 1. Load test data
test_df = pd.read_csv('test.csv')  # Must contain 'Interview Text' and 'ID'

# 2. Load tokenizer from trained model
tokenizer = AutoTokenizer.from_pretrained('./best_model')

# 3. Tokenize test data
def tokenize(batch):
    return tokenizer(batch['Interview Text'], truncation=True, padding=True)

test_ds = Dataset.from_pandas(test_df[['Interview Text']])
test_ds = test_ds.map(tokenize, batched=True)
test_ds.set_format('torch', columns=['input_ids', 'attention_mask'])

# 4. Load trained model (must have correct label mappings)
model = AutoModelForSequenceClassification.from_pretrained('./best_model')

# 5. Create Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# 6. Make predictions on test set
test_predictions = trainer.predict(test_ds)
test_preds = np.argmax(test_predictions.predictions, axis=1)

# 7. Convert predicted labels back to original string labels using model config
id2label = model.config.id2label
predicted_labels = [id2label[pred] for pred in test_preds]


# 8. Create submission file
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Labels': predicted_labels
})
submission.to_csv('submission.csv', index=False)

print("\n✅ submission.csv has been saved with predictions.")
