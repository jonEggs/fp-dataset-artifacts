import datasets
from datasets import DatasetDict  # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from helpers import prepare_dataset_nli, compute_accuracy

# Load the bias model (hypothesis-only)
bias_model_path = '/content/drive/MyDrive/nli_models/hypothesis_only_model'
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
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # Get predictions from main model
        outputs = model(**inputs)
        main_logits = outputs.logits
        
        # Get predictions from bias model (hypothesis-only)
        with torch.no_grad():
            bias_outputs = self.bias_model(**inputs)
            bias_logits = bias_outputs.logits
        
        # Product of Experts: combine logits
        # Subtract bias model's confident predictions
        combined_logits = main_logits - self.bias_weight * bias_logits
        
        # Compute loss on combined logits
        loss = F.cross_entropy(combined_logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# Main training
dataset = datasets.load_dataset('snli')

tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
model = AutoModelForSequenceClassification.from_pretrained(
    'google/electra-small-discriminator', 
    num_labels=3
)

# Prepare datasets (FULL premise+hypothesis for debiased model)
# Filter out examples with no label and tokenize
train_split = dataset['train']  # type: ignore
train_dataset = train_split.filter(lambda ex: ex['label'] != -1).map(
    lambda ex: prepare_dataset_nli(ex, tokenizer, 128, hypothesis_only=False),
    batched=True,
    remove_columns=train_split.column_names
)

eval_split = dataset['validation']  # type: ignore
eval_dataset = eval_split.filter(lambda ex: ex['label'] != -1).map(
    lambda ex: prepare_dataset_nli(ex, tokenizer, 128, hypothesis_only=False),
    batched=True,
    remove_columns=eval_split.column_names
)

training_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/nli_models/debiased_model',
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
    compute_metrics=compute_accuracy,
    bias_model=bias_model,
    bias_weight=1.0  # Tune this: try 0.5, 1.0, 2.0
)

trainer.train()
trainer.save_model()