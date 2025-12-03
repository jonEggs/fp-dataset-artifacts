import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from helpers import prepare_dataset_nli, compute_accuracy

NUM_PREPROCESSING_WORKERS = 2

# Global cache for hypothesis texts (indexed by position in dataset)
train_hypotheses = []

# Load the bias model (hypothesis-only)
bias_model_path = '/content/drive/MyDrive/nli_models/hypothesis_only_model'  # CHANGE THIS PATH
print(f"Loading bias model from {bias_model_path}...")
bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_path)
bias_model.eval()  # Set to eval mode
for param in bias_model.parameters():
    param.requires_grad = False  # Freeze it

# Custom Trainer that uses product of experts
class DebiasedTrainer(Trainer):
    def __init__(self, *args, bias_model, bias_weight=1.0, train_hypotheses=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_model = bias_model
        self.bias_weight = bias_weight
        self.train_hypotheses = train_hypotheses
        
        # Move bias model to same device as main model will be
        if torch.cuda.is_available():
            self.bias_model = self.bias_model.cuda()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        idx = inputs.pop("idx")  # Get the indices
        batch_size = labels.shape[0]
        
        # Get predictions from main model (full premise+hypothesis)
        outputs = model(**inputs)
        main_logits = outputs.logits
        
        # Move bias model to same device if needed
        if self.bias_model.device != main_logits.device:
            self.bias_model = self.bias_model.to(main_logits.device)
        
        # Get hypotheses for this batch using indices
        batch_hypotheses = [self.train_hypotheses[i.item()] for i in idx]
        
        # Tokenize hypotheses on-the-fly
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

# Cache hypotheses (we'll use them later if needed)
print("Caching hypothesis texts...")
train_hypotheses = dataset['train']['hypothesis']
print(f"Cached {len(train_hypotheses)} training hypotheses")

# Add index to dataset so we can look up hypotheses
def add_index(examples, idx):
    examples['idx'] = idx
    return examples

print("Adding indices to training data...")
dataset['train'] = dataset['train'].map(
    add_index,
    batched=True,
    with_indices=True
)

# Standard preprocessing (just like run.py)
print("Tokenizing training data...")
train_dataset = dataset['train'].map(
    lambda exs: prepare_dataset_nli(exs, tokenizer, 128, hypothesis_only=False),
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
    remove_columns=[col for col in dataset['train'].column_names if col != 'idx']  # Keep idx
)

print("Tokenizing validation data...")
eval_dataset = dataset['validation'].map(
    lambda exs: prepare_dataset_nli(exs, tokenizer, 128, hypothesis_only=False),
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

print("Initializing trainer...")
trainer = DebiasedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_accuracy,
    bias_model=bias_model,
    bias_weight=1.0,  # Tune this: try 0.5, 1.0, 2.0
    train_hypotheses=train_hypotheses
)

# Train and save
print("Starting training...")
print(f"Training on {len(train_dataset)} examples")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Bias weight: {trainer.bias_weight}")
print("\nUsing index-based hypothesis lookup (proper debiasing)")
print("Tokenizing hypotheses on-the-fly during training\n")
trainer.train()
print("Training complete. Saving model...")
trainer.save_model()
print("Model saved to ./debiased_model/")