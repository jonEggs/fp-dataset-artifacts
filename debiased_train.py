import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
import torch.nn.functional as F
from helpers import prepare_dataset_nli, compute_accuracy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

NUM_PREPROCESSING_WORKERS = 2

# Minimal custom collator - just preserves the extra hyp_* fields
@dataclass
class DataCollatorForDebiasing(DataCollatorWithPadding):
    """
    Simple data collator that preserves hypothesis-only fields.
    Extends the default collator to handle our extra fields.
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate out the hypothesis fields
        hyp_input_ids = [f.pop('hyp_input_ids') for f in features]
        hyp_attention_mask = [f.pop('hyp_attention_mask') for f in features]
        
        # Use parent class to handle the main fields (input_ids, attention_mask, labels)
        batch = super().__call__(features)
        
        # Add hypothesis fields back as tensors
        batch['hyp_input_ids'] = torch.tensor(hyp_input_ids, dtype=torch.long)
        batch['hyp_attention_mask'] = torch.tensor(hyp_attention_mask, dtype=torch.long)
        
        return batch

# Load the bias model (hypothesis-only)
bias_model_path = '/content/drive/MyDrive/nli_models/hypothesis_only_model'  # CHANGE THIS PATH
print(f"Loading bias model from {bias_model_path}...")
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
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        # Extract hypothesis-only inputs (we added these during preprocessing)
        hyp_input_ids = inputs.pop("hyp_input_ids")
        hyp_attention_mask = inputs.pop("hyp_attention_mask")
        
        # Get predictions from main model (full premise+hypothesis)
        outputs = model(**inputs)
        main_logits = outputs.logits
        
        # Move bias model to same device if needed
        if self.bias_model.device != main_logits.device:
            self.bias_model = self.bias_model.to(main_logits.device)
        
        # Get predictions from bias model (hypothesis-only)
        with torch.no_grad():
            bias_outputs = self.bias_model(
                input_ids=hyp_input_ids,
                attention_mask=hyp_attention_mask
            )
            bias_logits = bias_outputs.logits
        
        # Product of Experts: combine logits
        # Subtract bias model's confident predictions
        combined_logits = main_logits - self.bias_weight * bias_logits
        
        # Compute loss on combined logits
        loss = F.cross_entropy(combined_logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# Main training
print("Loading dataset...")
dataset = datasets.load_dataset('snli')

# Initialize tokenizer and model
print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    'google/electra-small-discriminator', 
    num_labels=3
)

# Make tensor contiguous if needed
if hasattr(model, 'electra'):
    for param in model.electra.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

print("Preprocessing data...")
# Remove SNLI examples with no label
dataset = dataset.filter(lambda ex: ex['label'] != -1)

# SIMPLE APPROACH: Tokenize BOTH versions during preprocessing
def prepare_dual_dataset(examples):
    # Tokenize premise+hypothesis for main model
    main_tokenized = prepare_dataset_nli(examples, tokenizer, 128, hypothesis_only=False)
    
    # Tokenize hypothesis-only for bias model
    hyp_tokenized = prepare_dataset_nli(examples, tokenizer, 128, hypothesis_only=True)
    
    # Combine both - prefix hypothesis-only fields with 'hyp_'
    combined = {
        'input_ids': main_tokenized['input_ids'],
        'attention_mask': main_tokenized['attention_mask'],
        'label': main_tokenized['label'],
        'hyp_input_ids': hyp_tokenized['input_ids'],
        'hyp_attention_mask': hyp_tokenized['attention_mask'],
    }
    return combined

print("Tokenizing training data (both versions)...")
train_dataset = dataset['train'].map(
    prepare_dual_dataset,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
    remove_columns=dataset['train'].column_names
)

print("Tokenizing validation data (both versions)...")
eval_dataset = dataset['validation'].map(
    prepare_dual_dataset,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
    remove_columns=dataset['validation'].column_names
)

training_args = TrainingArguments(
    output_dir='./debiased_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    do_train=True,
    do_eval=True,
)

# Create the custom data collator
data_collator = DataCollatorForDebiasing(tokenizer=tokenizer, padding=True)

print("Initializing trainer...")
trainer = DebiasedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # Use our custom collator
    compute_metrics=compute_accuracy,
    bias_model=bias_model,
    bias_weight=1.0,  # Tune this: try 0.5, 1.0, 2.0
)

# Train and save
print("Starting training...")
print(f"Training on {len(train_dataset)} examples")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Bias weight: {trainer.bias_weight}")
trainer.train()
print("Training complete. Saving model...")
trainer.save_model()
print("Model saved to ./debiased_model/")