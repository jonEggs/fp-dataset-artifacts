import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from helpers import prepare_dataset_nli, compute_accuracy
from dataclasses import dataclass
from typing import Any, Dict, List

NUM_PREPROCESSING_WORKERS = 2  # From run.py

# Custom data collator that preserves hypothesis text for bias model
@dataclass
class DataCollatorWithHypothesis:
    tokenizer: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract hypothesis texts before processing
        hypothesis_texts = [f['hypothesis'] for f in features]
        
        # Create a clean feature dict for padding (only tensor fields)
        clean_features = []
        for f in features:
            clean_f = {k: v for k, v in f.items() if k not in ['hypothesis', 'premise']}
            clean_features.append(clean_f)
        
        # Use default collation for tensor fields
        batch = self.tokenizer.pad(
            clean_features,
            padding=True,
            return_tensors='pt'
        )
        
        # Add hypothesis texts back (as list of strings)
        batch['hypothesis_text'] = hypothesis_texts
        return batch

# Load the bias model (hypothesis-only)
bias_model_path = '/content/drive/MyDrive/nli_models/hypothesis_only_model'  # Path to hypothesis-only model
bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_path)
bias_model.eval()  # Set to eval mode
for param in bias_model.parameters():
    param.requires_grad = False  # Freeze it

# Custom Trainer that uses product of experts
class DebiasedTrainer(Trainer):
    def __init__(self, *args, bias_model, bias_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_model = bias_model
        self.bias_weight = bias_weight
        # Move bias model to same device as main model
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        hypothesis_texts = inputs.pop("hypothesis_text", None)
        
        # Get predictions from main model (full premise+hypothesis)
        outputs = model(**inputs)
        main_logits = outputs.logits
        
        # Move bias model to same device if needed
        if self.bias_model.device != main_logits.device:
            self.bias_model = self.bias_model.to(main_logits.device)
        
        # Get predictions from bias model (hypothesis-only)
        with torch.no_grad():
            if hypothesis_texts is not None:
                # Re-tokenize hypothesis-only for the bias model
                hypothesis_inputs = self.tokenizer(
                    hypothesis_texts,
                    truncation=True,
                    max_length=128,
                    padding='max_length',
                    return_tensors='pt'
                ).to(main_logits.device)
                bias_outputs = self.bias_model(**hypothesis_inputs)
            else:
                # Fallback: use same inputs (less accurate debiasing)
                bias_outputs = self.bias_model(**inputs)
            bias_logits = bias_outputs.logits
        
        # Product of Experts: combine logits
        # Subtract bias model's confident predictions
        combined_logits = main_logits - self.bias_weight * bias_logits
        
        # Compute loss on combined logits
        loss = F.cross_entropy(combined_logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# Main training
# Load dataset (from run.py)
dataset = datasets.load_dataset('snli')

# Initialize tokenizer and model (from run.py)
tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    'google/electra-small-discriminator', 
    num_labels=3
)

# Make tensor contiguous if needed (from run.py lines 84-88)
if hasattr(model, 'electra'):
    for param in model.electra.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")

# Remove SNLI examples with no label (from run.py lines 104-105)
dataset = dataset.filter(lambda ex: ex['label'] != -1)

# Prepare datasets (FULL premise+hypothesis for debiased model)
# We need to keep the hypothesis text for the bias model
def prepare_with_hypothesis(examples):
    # Get tokenized features
    tokenized = prepare_dataset_nli(examples, tokenizer, 128, hypothesis_only=False)
    # Explicitly add hypothesis text back for bias model (keep as list)
    tokenized['hypothesis'] = examples['hypothesis']
    tokenized['premise'] = examples['premise']  # Keep premise too in case needed
    return tokenized

# Map datasets and set format to keep string columns
train_dataset = dataset['train'].map(
    prepare_with_hypothesis,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
)

eval_dataset = dataset['validation'].map(
    prepare_with_hypothesis,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
)

# Set the dataset format to keep the hypothesis column
# We specify which columns should be formatted as PyTorch tensors
train_dataset.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'label'],
    output_all_columns=True  # This keeps other columns like 'hypothesis' as regular Python objects
)

eval_dataset.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'label'],
    output_all_columns=True
)

training_args = TrainingArguments(
    output_dir='./debiased_model',  # Fixed path - removed nested structure
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Default from run.py
    per_device_eval_batch_size=8,
    do_train=True,
    do_eval=True,
)

trainer = DebiasedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithHypothesis(tokenizer=tokenizer),  # Custom collator to preserve hypothesis text
    compute_metrics=compute_accuracy,
    bias_model=bias_model,
    bias_weight=1.0  # Tune this: try 0.5, 1.0, 2.0
)

# Train and save (from run.py)
print("Starting training...")
trainer.train()
print("Training complete. Saving model...")
trainer.save_model()
print("Model saved!")