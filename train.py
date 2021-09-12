import os
import argparse
import random
import math
import datetime

import textattack
import transformers
import datasets
import pandas as pd

from configs import DATASET_CONFIGS

LOG_TO_WANDB = True


def filter_fn(x):
    """Filter bad samples."""
    if x["label"] == -1:
        return False
    if "premise" in x:
        if x["premise"] is None or x["premise"] == "":
            return False
    if "hypothesis" in x:
        if x["hypothesis"] is None or x["hypothesis"] == "":
            return False
    return True


def main(args):

    if args.train not in DATASET_CONFIGS:
        raise ValueError()
    dataset_config = DATASET_CONFIGS[args.train]

    if "local_path" in dataset_config:
        train_dataset = datasets.load_dataset(
            "csv",
            data_files=os.path.join(dataset_config["local_path"], "train.tsv"),
            delimiter="\t",
        )["train"]
    else:
        train_dataset = datasets.load_dataset(
            dataset_config["remote_name"], split="train"
        )

    if "local_path" in dataset_config:
        eval_dataset = datasets.load_dataset(
            "csv",
            data_files=os.path.join(dataset_config["local_path"], "val.tsv"),
            delimiter="\t",
        )["train"]
    else:
        eval_dataset = datasets.load_dataset(
            dataset_config["remote_name"], split="validation"
        )

    if args.augmented_data:
        pd_train_dataset = train_dataset.to_pandas()
        feature = train_dataset.features
        augmented_dataset = datasets.load_dataset(
            "csv",
            data_files=args.augmented_data,
            delimiter="\t",
            features=feature,
        )["train"]
        augmented_dataset = augmented_dataset.filter(filter_fn)
        sampled_indices = list(range(len(augmented_dataset)))
        random.shuffle(sampled_indices)
        sampled_indices = sampled_indices[
            : math.ceil(len(sampled_indices) * args.pct_of_augmented)
        ]
        augmented_dataset = augmented_dataset.select(
            sampled_indices, keep_in_memory=True
        ).to_pandas()
        train_dataset = datasets.Dataset.from_pandas(
            pd.concat((pd_train_dataset, augmented_dataset))
        )

    train_dataset = train_dataset.filter(lambda x: x["label"] != -1)
    eval_dataset = eval_dataset.filter(lambda x: x["label"] != -1)

    train_dataset = textattack.datasets.HuggingFaceDataset(
        train_dataset,
        dataset_columns=dataset_config["dataset_columns"],
        label_names=dataset_config["label_names"],
    )

    eval_dataset = textattack.datasets.HuggingFaceDataset(
        eval_dataset,
        dataset_columns=dataset_config["dataset_columns"],
        label_names=dataset_config["label_names"],
    )

    if args.model_type == "bert":
        pretrained_name = "bert-base-uncased"
    elif args.model_type == "roberta":
        pretrained_name = "roberta-base"

    if args.model_chkpt_path:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            args.model_chkpt_path
        )
    else:
        num_labels = dataset_config["labels"]
        config = transformers.AutoConfig.from_pretrained(
            pretrained_name, num_labels=num_labels
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_name, config=config
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_name, use_fast=True
    )
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    if args.attack == "a2t":
        attack = textattack.attack_recipes.A2TYoo2021.build(model_wrapper, mlm=False)
    elif args.attack == "a2t_mlm":
        attack = textattack.attack_recipes.A2TYoo2021.build(model_wrapper, mlm=True)
    else:
        raise ValueError(f"Unknown attack {args.attack}.")

    training_args = textattack.TrainingArgs(
        num_epochs=args.num_epochs,
        num_clean_epochs=args.num_clean_epochs,
        attack_epoch_interval=args.attack_epoch_interval,
        parallel=args.parallel,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accumu_steps,
        num_warmup_steps=args.num_warmup_steps,
        learning_rate=args.learning_rate,
        num_train_adv_examples=args.num_adv_examples,
        attack_num_workers_per_device=1,
        query_budget_train=200,
        checkpoint_interval_epochs=args.checkpoint_interval_epochs,
        output_dir=args.model_save_path,
        log_to_wandb=LOG_TO_WANDB,
        wandb_project="nlp-robustness",
        load_best_model_at_end=True,
        logging_interval_step=10,
        random_seed=args.seed,
    )
    trainer = textattack.Trainer(
        model_wrapper,
        "classification",
        attack,
        train_dataset,
        eval_dataset,
        training_args,
    )
    trainer.train()


if __name__ == "__main__":

    def int_or_float(v):
        try:
            return int(v)
        except ValueError:
            return float(v)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        choices=sorted(list(DATASET_CONFIGS.keys())),
        help="Name of dataset for training.",
    )
    parser.add_argument(
        "--augmented-data",
        type=str,
        required=False,
        default=None,
        help="Path of augmented data (in TSV).",
    )
    parser.add_argument(
        "--pct-of-augmented",
        type=float,
        required=False,
        default=0.2,
        help="Percentage of augmented data to use.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        required=True,
        choices=sorted(list(DATASET_CONFIGS.keys())),
        help="Name of huggingface dataset for validation",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run training with multiple GPUs."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["bert", "roberta"],
        help='Type of model (e.g. "bert", "roberta").',
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="./saved_model",
        help="Directory to save model checkpoint.",
    )
    parser.add_argument(
        "--model-chkpt-path",
        type=str,
        default=None,
        help="Directory of model checkpoint to resume from.",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=4, help="Number of epochs to train."
    )
    parser.add_argument(
        "--num-clean-epochs", type=int, default=1, help="Number of clean epochs"
    )
    parser.add_argument(
        "--num-adv-examples",
        type=int_or_float,
        help="Number (or percentage) of adversarial examples for training.",
    )
    parser.add_argument(
        "--attack-epoch-interval",
        type=int,
        default=1,
        help="Attack model to generate adversarial examples every N epochs.",
    )
    parser.add_argument(
        "--attack", type=str, choices=["a2t", "a2t_mlm"], help="Name of attack."
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Train batch size (per GPU device).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num-warmup-steps", type=int, default=500, help="Number of warmup steps."
    )
    parser.add_argument(
        "--grad-accumu-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--checkpoint-interval-epochs",
        type=int,
        default=None,
        help="If set, save model checkpoint after every `N` epochs.",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()
    main(args)
