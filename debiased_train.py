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
        # Extract hypothesis texts and remove string fields we don't need
        hypothesis_texts = [f.pop('hypothesis') for f in features]
        # Remove premise if it exists (we don't need it)
        for f in features:
            f.pop('premise', None)
        
        # Use default collation for tensor fields
        batch = self.tokenizer.pad(
            features,
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
        # Store tokenizer from parent class for hypothesis re-tokenization
        # self.tokenizer is set by parent Trainer.__init__
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        hypothesis_texts = inputs.pop("hypothesis_text", None)
        
        # Get predictions from main model (full premise+hypothesis)
        outputs = model(**inputs)
        main_logits = outputs.logits
        
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
    # Explicitly add hypothesis text back for bias model
    tokenized['hypothesis'] = examples['hypothesis']
    return tokenized

train_dataset = dataset['train'].map(
    prepare_with_hypothesis,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
    # Don't remove any columns - let the data collator handle it
)

eval_dataset = dataset['validation'].map(
    prepare_with_hypothesis,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
    # Don't remove any columns - let the data collator handle it
)

training_args = TrainingArguments(
    output_dir='./content/drive/MyDrive/nli_models/debiased_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Default from run.py
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
trainer.train()
trainer.save_model()