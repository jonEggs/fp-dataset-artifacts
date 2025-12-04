import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
import torch
import torch.nn.functional as F
from helpers import prepare_dataset_nli, compute_accuracy

NUM_PREPROCESSING_WORKERS = 2

# Simple data collator that preserves idx field
def data_collator_with_idx(features):
    # Extract idx if present
    if 'idx' in features[0]:
        idx_list = [f.pop('idx') for f in features]
    else:
        idx_list = None
    
    # Use default collator for standard fields
    batch = default_data_collator(features)
    
    # Add idx back if it existed
    if idx_list is not None:
        batch['idx'] = torch.tensor(idx_list, dtype=torch.long)
    
    return batch

# Global cache for hypothesis texts
train_hypotheses = []

# Load the bias model (hypothesis-only)
bias_model_path = '/content/drive/MyDrive/nli_models/hypothesis_only_model'  # CHANGE THIS
print(f"Loading bias model from {bias_model_path}...")
bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_path)
bias_model.eval()
for param in bias_model.parameters():
    param.requires_grad = False

# Custom Trainer
class DebiasedTrainer(Trainer):
    def __init__(self, *args, bias_model, bias_weight=1.0, train_hypotheses=None, val_hypotheses=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_model = bias_model
        self.bias_weight = bias_weight
        self.train_hypotheses = train_hypotheses
        self.val_hypotheses = val_hypotheses
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        idx = inputs.pop("idx", None)
        
        # Get predictions from main model
        outputs = model(**inputs)
        main_logits = outputs.logits
        
        # Move bias model to same device
        if self.bias_model.device != main_logits.device:
            self.bias_model = self.bias_model.to(main_logits.device)
        
        # Get predictions from bias model (hypothesis-only)
        if idx is not None:
            # Determine which hypothesis cache to use based on whether we're training or evaluating
            # During training, use train_hypotheses; during eval, use val_hypotheses
            hypotheses_cache = self.train_hypotheses if model.training else self.val_hypotheses
            
            if hypotheses_cache is not None:
                # Look up hypotheses using indices
                batch_hypotheses = [hypotheses_cache[i.item()] for i in idx]
                
                # Tokenize hypothesis-only
                with torch.no_grad():
                    hyp_inputs = self.tokenizer(
                        batch_hypotheses,
                        truncation=True,
                        max_length=128,
                        padding='max_length',
                        return_tensors='pt'
                    ).to(main_logits.device)
                    
                    bias_outputs = self.bias_model(**hyp_inputs)
                    bias_logits = bias_outputs.logits
            else:
                # Fallback: use same inputs
                with torch.no_grad():
                    bias_outputs = self.bias_model(**inputs)
                    bias_logits = bias_outputs.logits
        else:
            # Fallback: use same inputs
            with torch.no_grad():
                bias_outputs = self.bias_model(**inputs)
                bias_logits = bias_outputs.logits
        
        # Product of Experts
        combined_logits = main_logits - self.bias_weight * bias_logits
        loss = F.cross_entropy(combined_logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# Main training
print("Loading dataset...")
dataset = datasets.load_dataset('snli')

print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    'google/electra-small-discriminator',
    num_labels=3
)

if hasattr(model, 'electra'):
    for param in model.electra.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

print("Preprocessing...")
dataset = dataset.filter(lambda ex: ex['label'] != -1)

# Cache hypotheses BEFORE adding indices
print("Caching hypotheses...")
train_hypotheses = dataset['train']['hypothesis']
val_hypotheses = dataset['validation']['hypothesis']
print(f"Cached {len(train_hypotheses)} train hypotheses")
print(f"Cached {len(val_hypotheses)} validation hypotheses")

# Add index field
print("Adding indices...")
def add_idx(examples, indices):
    return {'idx': indices}

dataset['train'] = dataset['train'].map(
    add_idx,
    batched=True,
    with_indices=True
)

dataset['validation'] = dataset['validation'].map(
    add_idx,
    batched=True,
    with_indices=True
)

# Tokenize
print("Tokenizing...")
def prepare_with_idx(examples):
    tokenized = prepare_dataset_nli(examples, tokenizer, 128, hypothesis_only=False)
    tokenized['idx'] = examples['idx']
    print(f"Tokenized batch with idx: {tokenized['idx'][:5]}")  # Debug print
    return tokenized

train_dataset = dataset['train'].map(
    prepare_with_idx,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
    remove_columns=[c for c in dataset['train'].column_names if c != 'idx']
)

eval_dataset = dataset['validation'].map(
    prepare_with_idx,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
    remove_columns=[c for c in dataset['validation'].column_names if c != 'idx']
)

bias_coefficient = 0.22 # Define bias coefficient once
training_args = TrainingArguments(
    output_dir=f'./debiased_model_{bias_coefficient}',
    num_train_epochs=0.01,
    per_device_train_batch_size=8
)

print("Initializing trainer...")
trainer = DebiasedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator_with_idx,  # Custom collator for idx
    compute_metrics=compute_accuracy,
    bias_model=bias_model,
    bias_weight=bias_coefficient,
    train_hypotheses=train_hypotheses,
    val_hypotheses=val_hypotheses
)

print("\nStarting training with proper hypothesis-only debiasing...")
trainer.train()
print("Saving...")
trainer.save_model()
print("Done!")