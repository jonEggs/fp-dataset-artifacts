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
        
         # DEBUG: Decode and compare for first few steps
        if self.state.global_step < 3:
            for i in range(min(2, len(hyp_input_ids))):  # Check first 2 examples in batch
                # Decode main model input (premise + hypothesis)
                main_text = self.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
                
                # Decode bias model input (hypothesis only)
                hyp_text = self.tokenizer.decode(hyp_input_ids[i], skip_special_tokens=True)
                
                print(f"\n=== Step {self.state.global_step}, Example {i} ===")
                print(f"Main model sees: {main_text}")
                print(f"Bias model sees: {hyp_text}")
                print(f"Hypothesis in main? {hyp_text in main_text}")
        # Main model: premise + hypothesis
        outputs = model(**inputs)
        main_logits = outputs.logits
        
        # Move bias model to same device if needed
        if self.bias_model.device != main_logits.device:
            self.bias_model = self.bias_model.to(main_logits.device)
        
        # Bias model: hypothesis only (pre-tokenized)
        with torch.no_grad():
            bias_logits = self.bias_model(
                input_ids=hyp_input_ids,
                attention_mask=hyp_attention_mask
            ).logits
        
        # Product of Experts
        combined_logits = main_logits - self.bias_weight * bias_logits
        loss = F.cross_entropy(combined_logits, labels)
        
        # Debug prints for first few steps
        if self.state.global_step < 3:
            print(f"Step {self.state.global_step}")
            print(f"  bias_logits[0]: {bias_logits[0].tolist()}")
            print(f"  main_logits[0]: {main_logits[0].tolist()}")
            print(f"  combined_logits[0]: {combined_logits[0].tolist()}")
        
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

bias_coefficient = 0.5 # Define bias coefficient once
training_args = TrainingArguments(
    output_dir=f'/content/drive/MyDrive/nli_models/debiased_model_{bias_coefficient}',
    num_train_epochs=3,
    per_device_train_batch_size=16,
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
    bias_weight=bias_coefficient
)

print("\nStarting training with proper hypothesis-only debiasing...")
trainer.train()
print("Saving...")
trainer.save_model()
print("Done!")