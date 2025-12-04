import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
import torch
import torch.nn.functional as F
from helpers import compute_accuracy

def data_collator_with_hyp(features):
    """Collate main inputs + separate hypothesis inputs"""
    # Debug: Print features to inspect missing keys
    for idx, f in enumerate(features):
        if 'hyp_input_ids' not in f:
            print(f"[DEBUG] Feature at index {idx} missing 'hyp_input_ids': {f}")
        if 'hyp_attention_mask' not in f:
            print(f"[DEBUG] Feature at index {idx} missing 'hyp_attention_mask': {f}")

    try:
        hyp_input_ids = [f.pop('hyp_input_ids') for f in features]
        hyp_attention_mask = [f.pop('hyp_attention_mask') for f in features]
    except KeyError as e:
        print(f"[ERROR] KeyError in data_collator_with_hyp: {e}")
        print(f"[ERROR] Features: {features}")
        raise

    # Use default collator for main fields
    batch = default_data_collator(features)

    # Use stack for tensors, tensor for lists
    if isinstance(hyp_input_ids[0], torch.Tensor):
        batch['hyp_input_ids'] = torch.stack(hyp_input_ids)
        batch['hyp_attention_mask'] = torch.stack(hyp_attention_mask)
    else:
        batch['hyp_input_ids'] = torch.tensor(hyp_input_ids, dtype=torch.long)
        batch['hyp_attention_mask'] = torch.tensor(hyp_attention_mask, dtype=torch.long)

    return batch

# Load the bias model
bias_model_path = '/content/drive/MyDrive/nli_models/hypothesis_only_model'  # CHANGE THIS
print(f"Loading bias model from {bias_model_path}...")
bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_path)
bias_model.eval()
for param in bias_model.parameters():
    param.requires_grad = False

class DebiasedTrainer(Trainer):
    def __init__(self, *args, bias_model, bias_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_model = bias_model
        self.bias_weight = bias_weight
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        hyp_input_ids = inputs.pop("hyp_input_ids")
        hyp_attention_mask = inputs.pop("hyp_attention_mask")
        
        # Main model prediction
        outputs = model(**inputs)
        main_logits = outputs.logits
        
        # Get bias model's confidence (frozen, no gradients)
        if self.bias_model.device != main_logits.device:
            self.bias_model = self.bias_model.to(main_logits.device)
        with torch.no_grad():
            test_outputs = self.bias_model(
                input_ids=hyp_input_ids[:5],
                attention_mask=hyp_attention_mask[:5]
            )
            print("Bias model predictions:", test_outputs.logits.argmax(dim=-1))
        with torch.no_grad():
            bias_logits = self.bias_model(
                input_ids=hyp_input_ids,
                attention_mask=hyp_attention_mask
            ).logits
            bias_probs = F.softmax(bias_logits, dim=-1)
            bias_confidence = bias_probs.max(dim=-1)[0]
        
        # Debiased Focal Loss: down-weight high-confidence bias examples
        bias_weight = self.bias_weight
        example_weights = (1.0 - bias_confidence) ** bias_weight

        
        # Compute weighted loss
        ce_loss = F.cross_entropy(main_logits, labels, reduction='none')
        loss = (example_weights * ce_loss).mean()
        
        # Right before computing loss, add:
        if self.state.global_step % 100 == 0:  # Log every 100 steps
            print(f"\n=== Step {self.state.global_step} ===")
            print(f"Bias confidence - Min: {bias_confidence.min():.3f}, "
                f"Mean: {bias_confidence.mean():.3f}, Max: {bias_confidence.max():.3f}")
            print(f"Example weights - Min: {example_weights.min():.3f}, "
                f"Mean: {example_weights.mean():.3f}, Max: {example_weights.max():.3f}")
            print(f"% heavily downweighted (<0.1): {(example_weights < 0.1).float().mean():.1%}")
            print(f"Weighted loss: {loss.item():.4f}")
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

def prepare_both_inputs(examples):
    """Pre-tokenize both premise+hypothesis AND hypothesis-only"""
    # Main model inputs: premise + hypothesis
    main_tok = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=128,
        padding='max_length'
    )
    
    # Bias model inputs: hypothesis only
    hyp_tok = tokenizer(
        examples['hypothesis'],
        truncation=True,
        max_length=128,
        padding='max_length'
    )

    result = {
        'input_ids': main_tok['input_ids'],
        'attention_mask': main_tok['attention_mask'],
        'label': examples['label'],
        'hyp_input_ids': hyp_tok['input_ids'],
        'hyp_attention_mask': hyp_tok['attention_mask'],
    }
    
    # DEBUG: Print once
    if len(examples['premise']) > 0 and examples['premise'][0].startswith('A'):
        print(f"prepare_both_inputs returning keys: {result.keys()}")
    
    return result

print("Tokenizing train set...")
train_dataset = dataset['train'].map(
    prepare_both_inputs,
    batched=True,
    num_proc=1,
    remove_columns=dataset['train'].column_names
)

# Explicitly set which columns to keep and convert to torch
train_dataset.set_format(
    type='torch', 
    columns=['input_ids', 'attention_mask', 'label', 'hyp_input_ids', 'hyp_attention_mask']
)

# Verify
print(f"Format: {train_dataset.format}")
print(f"Columns: {train_dataset.column_names}")

# DEBUG: Verify columns exist
print(f"Columns after map: {train_dataset.column_names}")
print(f"First example keys: {train_dataset[0].keys()}")
print(f"hyp_input_ids sample: {train_dataset[0]['hyp_input_ids'][:10]}")

print("Tokenizing validation set...")
eval_dataset = dataset['validation'].map(
    prepare_both_inputs,
    batched=True,
    num_proc=1,
    remove_columns=dataset['validation'].column_names
)

# Explicitly set which columns to keep and convert to torch
eval_dataset.set_format(
    type='torch', 
    columns=['input_ids', 'attention_mask', 'label', 'hyp_input_ids', 'hyp_attention_mask']
)

BIAS_CONST = 0.5 # Define how aggressively to downweight examples that hypoth only model is confident on.
training_args = TrainingArguments(
    output_dir=f'/content/drive/MyDrive/nli_models/debiased_reweight_model_{BIAS_COEFFICIENT}',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    logging_steps=10000,
    remove_unused_columns=False
)

print("Initializing trainer...")
trainer = DebiasedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator_with_hyp,
    compute_metrics=compute_accuracy,
    bias_model=bias_model,
    bias_weight=BIAS_COEFFICIENT
)

print("\nStarting training with proper hypothesis-only debiasing...")
trainer.train()
print("Saving...")
trainer.save_model()
print("Done!")