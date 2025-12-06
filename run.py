#Modifications to the code below was written with Copilot assistance
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy, compute_accuracy_hans
import os
import json

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--hypothesis_only', action='store_true',
                      help='If set, train/evaluate NLI models using only the hypothesis text (omit the premise).')
    argp.add_argument('--eval_dataset', type=str, default=None,
                      help='Override the evaluation dataset. For ANLI, use "anli" to eval on all rounds, or "anli:r1", "anli:r2", "anli:r3" for specific rounds.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset is not None and (args.dataset.endswith('.json') or args.dataset.endswith('.jsonl')):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        # Pass the hypothesis_only flag through so we can train/eval a hypothesis-only baseline.
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length, args.hypothesis_only)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        hans_eval = False
        if args.eval_dataset is not None:
            if args.eval_dataset.startswith('anli'):
                print(f"Loading ANLI evaluation dataset: {args.eval_dataset}")
                parts = args.eval_dataset.split(':')
                anli_dataset = datasets.load_dataset('facebook/anli')
                if len(parts) == 1:
                    # Load all ANLI rounds combined
                    from datasets import Dataset
                    eval_dataset = datasets.concatenate_datasets([
                        anli_dataset['test_r1'],  # type: ignore
                        anli_dataset['test_r2'],  # type: ignore
                        anli_dataset['test_r3']   # type: ignore
                    ])
                    print("Loaded ANLI test sets (all rounds combined)")
                elif len(parts) == 2 and parts[1] in ['r1', 'r2', 'r3']:
                    # Load specific round
                    round_num = parts[1]
                    eval_dataset = anli_dataset[f'test_{round_num}']
                    print(f"Loaded ANLI test set (round {round_num})")
                else:
                    raise ValueError(f"Invalid ANLI specification: {args.eval_dataset}. Use 'anli', 'anli:r1', 'anli:r2', or 'anli:r3'")
            elif args.eval_dataset == 'hans':
                hans_eval = True
                print("Loading HANS evaluation dataset")
                hans_dataset = datasets.load_dataset('SebastiaanBeekman/hans', split='test')
                # Map SebastiaanBeekman/hans columns to expected NLI columns
                def map_hans_columns(example):
                    input_str = example['input']
                    # Handle multiline input with 'Premise:' and 'Hypothesis:' on separate lines
                    lines = input_str.splitlines()
                    premise = ''
                    hypothesis = ''
                    for i, line in enumerate(lines):
                        if line.strip().startswith('Premise:'):
                            premise = lines[i+1].strip() if i+1 < len(lines) else ''
                        if line.strip().startswith('Hypothesis:'):
                            hypothesis = lines[i+1].strip() if i+1 < len(lines) else ''
                    # Map string label to int: 'entailment'->0, 'non-entailment'->2
                    ref = example['reference']
                    if isinstance(ref, str):
                        label = 0 if ref.strip() == 'entailment' else 2
                    else:
                        label = 0 if ref == 0 else 2
                    return {'premise': premise, 'hypothesis': hypothesis, 'label': label}
                eval_dataset = hans_dataset.map(map_hans_columns)
            else:
                # Use the same dataset as training
                eval_dataset = dataset[eval_split]
        else:
            # Use the same dataset as training
            eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = evaluate.load('squad')   # datasets.load_metric() deprecated
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        if hans_eval:
            compute_metrics = compute_accuracy_hans
        else:
            compute_metrics = compute_accuracy

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:

        results = trainer.evaluate(**eval_kwargs)
        print('Evaluation results:')
        print(results)
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        # Save all predictions as before
        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')

        # For SNLI/NLI: Print and save 100 right and 100 wrong answers
        if args.task == 'nli':
            if hans_eval:
                hans_correct = []
                for i, example in enumerate(eval_dataset):
                    pred_label = int(eval_predictions.predictions[i].argmax())
                    # HANS labels are grouped: 0 = entailment, 2 = non-entailment
                    true_label = example['label']
                    record = {
                        'premise': example.get('premise', ''),
                        'hypothesis': example.get('hypothesis', ''),
                        'true_label': true_label,
                        'predicted_label': pred_label,
                        'predicted_scores': eval_predictions.predictions[i].tolist()
                    }
                    if pred_label == true_label and len(hans_correct) < 100:
                        hans_correct.append(record)
                    if len(hans_correct) >= 100:
                        break
                print("\nFirst 5 correct HANS predictions:")
                for ex in hans_correct[:5]:
                    print(json.dumps(ex, indent=2))
                hans_save_path = os.path.join(training_args.output_dir, 'hans_correct_100.json')
                with open(hans_save_path, 'w', encoding='utf-8') as f:
                    json.dump({'correct': hans_correct}, f, ensure_ascii=False, indent=2)
                print(f"Saved 100 correct HANS predictions to {hans_save_path}")
            else:
                correct = []
                wrong = []
                label_counts = {0: 0, 1: 0, 2: 0}
                label_correct = {0: 0, 1: 0, 2: 0}
                for i, example in enumerate(eval_dataset):
                    pred_label = int(eval_predictions.predictions[i].argmax())
                    true_label = example['label']
                    label_counts[true_label] += 1
                    if pred_label == true_label:
                        label_correct[true_label] += 1
                        if len(correct) < 100:
                            correct.append({
                                'premise': example.get('premise', ''),
                                'hypothesis': example.get('hypothesis', ''),
                                'true_label': true_label,
                                'predicted_label': pred_label,
                                'predicted_scores': eval_predictions.predictions[i].tolist()
                            })
                    elif len(wrong) < 100:
                        wrong.append({
                            'premise': example.get('premise', ''),
                            'hypothesis': example.get('hypothesis', ''),
                            'true_label': true_label,
                            'predicted_label': pred_label,
                            'predicted_scores': eval_predictions.predictions[i].tolist()
                        })
                print("\nFirst 5 correct predictions:")
                for ex in correct[:5]:
                    print(json.dumps(ex, indent=2))
                print("\nFirst 5 wrong predictions:")
                for ex in wrong[:5]:
                    print(json.dumps(ex, indent=2))
                # Print per-label accuracy
                print("\nSNLI Per-label accuracy (whole dataset):")
                label_names = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
                for label in [0, 1, 2]:
                    total = label_counts[label]
                    correct_n = label_correct[label]
                    acc = (correct_n / total * 100) if total > 0 else 0.0
                    print(f"  {label_names[label]}: {correct_n}/{total} = {acc:.2f}%")
                # Save to JSON for Colab/Google Drive
                save_path = os.path.join(training_args.output_dir, 'snli_right_wrong_100.json')
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump({'right': correct, 'wrong': wrong, 'per_label_accuracy': {
                        label_names[label]: {
                            'correct': label_correct[label],
                            'total': label_counts[label],
                            'accuracy': (label_correct[label] / label_counts[label] * 100) if label_counts[label] > 0 else 0.0
                        } for label in [0, 1, 2]
                    }}, f, ensure_ascii=False, indent=2)
                print(f"Saved 100 right and 100 wrong SNLI predictions to {save_path}")


if __name__ == "__main__":
    main()
