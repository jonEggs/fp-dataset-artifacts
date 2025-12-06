
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bias_const', type=float, default=1.0, help='Bias downweighting constant')
args = parser.parse_args()

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
        
        outputs = model(**inputs)
        main_logits = outputs.logits
        
        if self.bias_model.device != main_logits.device:
            self.bias_model = self.bias_model.to(main_logits.device)
        
        with torch.no_grad():
            bias_logits = self.bias_model(
                input_ids=hyp_input_ids,
                attention_mask=hyp_attention_mask
            ).logits
        
        # PoE: subtract bias log-probs
        bias_log_probs = F.log_softmax(bias_logits, dim=-1)
        debiased_logits = main_logits - self.bias_weight * bias_log_probs
        loss = F.cross_entropy(debiased_logits, labels)
        
        if self.state.global_step % 2000 == 0:
            with torch.no_grad():
                # Predictions from each
                main_preds = main_logits.argmax(dim=-1)
                bias_preds = bias_logits.argmax(dim=-1)
                debiased_preds = debiased_logits.argmax(dim=-1)
                
                # Accuracies
                main_acc = (main_preds == labels).float().mean()
                bias_acc = (bias_preds == labels).float().mean()
                debiased_acc = (debiased_preds == labels).float().mean()
                
                # Agreement: is main model just copying bias model?
                main_bias_agree = (main_preds == bias_preds).float().mean()
                
                # How often does PoE change the prediction?
                poe_flips = (main_preds != debiased_preds).float().mean()
                
                # How often does PoE flip to correct answer?
                main_wrong = (main_preds != labels)
                debiased_right = (debiased_preds == labels)
                poe_saves = (main_wrong & debiased_right).float().mean()
                
                # Loss comparison
                main_loss = F.cross_entropy(main_logits, labels)
                
                print(f"\n=== Step {self.state.global_step} ===")
                print(f"Batch accuracy  - Main: {main_acc:.1%}, Bias: {bias_acc:.1%}, Debiased: {debiased_acc:.1%}")
                print(f"Main-Bias agreement: {main_bias_agree:.1%}")
                print(f"PoE flips prediction: {poe_flips:.1%}")
                print(f"PoE saves (wrong→right): {poe_saves:.1%}")
                print(f"Loss - PoE: {loss.item():.4f}, Main only: {main_loss.item():.4f}")
        
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


training_args = TrainingArguments(
    output_dir=f'/content/drive/MyDrive/nli_models/debiased_poe_model_{args.bias_const}',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=10000,
    learning_rate=1e-5,
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
    bias_weight=args.bias_const
)

print("\nStarting training with proper hypothesis-only debiasing...")
trainer.train()
print("Saving...")
trainer.save_model()
print("Done!")