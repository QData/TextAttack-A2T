import argparse
import functools
import json
import os
import random
import math
import multiprocessing as mp

import datasets
import numpy as np
import textattack
import torch
import tqdm
import transformers
from lime.lime_text import LimeTextExplainer, IndexedString

from configs import DATASET_CONFIGS


NUM_SAMPLES_FOR_EVALUATION = 1000


class AOPC:
    def __init__(self, model, tokenizer, labels):
        self.interpreter = LimeTextExplainer(
            class_names=labels, bow=False, mask_string=tokenizer.unk_token
        )
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.tokenizer = tokenizer
        self.K = 10
        self.num_samples = 1000

    def pred_fn_nli(self, premise, texts, batch_size=128):
        texts = [(premise, t) for t in texts]
        all_probs = []
        for i in range(0, len(texts), batch_size):
            inputs = texts[i : i + batch_size]
            input_ids = self.tokenizer(
                inputs,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            input_ids.to("cuda")
            with torch.no_grad():
                logits = self.model(**input_ids)[0]
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        probs = np.concatenate(all_probs, axis=0)

        return probs

    def pred_fn(self, texts, batch_size=128):
        all_probs = []
        for i in range(0, len(texts), batch_size):
            inputs = texts[i : i + batch_size]
            input_ids = self.tokenizer(
                inputs, padding="max_length", truncation=True, return_tensors="pt"
            )
            input_ids.to("cuda")
            with torch.no_grad():
                logits = self.model(**input_ids)[0]
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        probs = np.concatenate(all_probs, axis=0)

        return probs

    def calc_aopc_dataset(self, dataset):
        aopc_scores = []
        for row in tqdm.tqdm(dataset):
            if "content" in row:
                text = row["content"]
            elif "hypothesis" in row:
                text = (row["premise"], row["hypothesis"])
            else:
                text = row["text"]
            label = row["label"]
            num_words = IndexedString(text, bow=False).num_words()
            K = min(max(self.K, math.ceil(num_words * 0.1)), num_words)
            exp = self.interpreter.explain_instance(
                text, self.pred_fn, num_features=K, num_samples=self.num_samples
            )
            exp = exp.as_map()[1]
            perturbed_texts = [text]
            for k in range(1, K + 1):
                top_exp = [e[0] for e in exp[:k]]
                x = IndexedString(text, bow=False, mask_string="").inverse_removing(
                    top_exp
                )
                perturbed_texts.append(x)
            probs = self.pred_fn(perturbed_texts)
            probs_diff = (probs[0] - probs)[1:, label]
            aopc_scores.append(probs_diff.sum())
        avg_aopc = sum(aopc_scores) / (len(aopc_scores) * (1 + self.K))
        return avg_aopc

    def calc_aopc_instance(self, text, label, nli=False):
        if nli:
            premise = text[0]
            text = text[1]

        num_words = IndexedString(text, bow=False).num_words()
        K = min(self.K, num_words)
        if nli:
            pred_fn = functools.partial(self.pred_fn_nli, premise)
        else:
            pred_fn = self.pred_fn

        exp = self.interpreter.explain_instance(
            text, pred_fn, num_features=K, num_samples=self.num_samples
        )

        exp = exp.as_map()[1]
        perturbed_texts = [text]
        for k in range(1, K + 1):
            top_exp = [e[0] for e in exp[:k]]
            x = IndexedString(text, bow=False, mask_string="").inverse_removing(top_exp)
            perturbed_texts.append(x)

        probs = pred_fn(perturbed_texts)
        probs_diff = (probs[0] - probs)[1:, label]
        score = probs_diff.sum() / (1 + K)
        return score


def calc_aopc(model_type, model_path, labels, num_gpus, in_queue, out_queue, nli):
    gpu_id = (torch.multiprocessing.current_process()._identity[0] - 1) % num_gpus
    torch.cuda.set_device(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
    if model_type == "roberta":
        model_type = "roberta-base"
    else:
        model_type = "bert-base-uncased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
    aopc = AOPC(model, tokenizer, labels)
    while True:
        try:
            i, input_text, label = in_queue.get()
            if i == "END" and example == "END" and ground_truth_output == "END":
                # End process when sentinel value is received
                break
            else:
                aopc_score = aopc.calc_aopc_instance(input_text, label, nli=nli)
                out_queue.put((i, aopc_score))
        except Exception as e:
            out_queue.put((i, e))


# Helper functions for collating data
def collate_fn(input_columns, data):
    input_texts = []
    labels = []
    for d in data:
        label = d["label"]
        _input = tuple(d[c] for c in input_columns)
        if len(_input) == 1:
            _input = _input[0]
        input_texts.append(_input)
        labels.append(label)
    return input_texts, torch.tensor(labels)


def load_dataset(name):
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {name}")
    dataset_config = DATASET_CONFIGS[name]
    if "local_path" in dataset_config:
        dataset = datasets.load_dataset(
            "csv",
            data_files=os.path.join(dataset_config["local_path"], "test.tsv"),
            delimiter="\t",
        )["train"]
    else:
        if "split" in dataset_config:
            dataset = datasets.load_dataset(
                dataset_config["remote_name"], split=dataset_config["split"]
            )
        else:
            dataset = datasets.load_dataset(dataset_config["remote_name"], split="test")

    dataset = dataset.filter(lambda x: x["label"] != -1)

    return dataset


def calc_attack_stats(results):
    total_attacks = len(results)

    all_num_words = np.zeros(total_attacks)
    perturbed_word_percentages = np.zeros(total_attacks)
    failed_attacks = 0
    skipped_attacks = 0
    successful_attacks = 0

    for i, result in enumerate(results):
        all_num_words[i] = len(result.original_result.attacked_text.words)
        if isinstance(result, textattack.attack_results.FailedAttackResult):
            failed_attacks += 1
            continue
        elif isinstance(result, textattack.attack_results.SkippedAttackResult):
            skipped_attacks += 1
            continue
        else:
            successful_attacks += 1
        num_words_changed = len(
            result.original_result.attacked_text.all_words_diff(
                result.perturbed_result.attacked_text
            )
        )
        if len(result.original_result.attacked_text.words) > 0:
            perturbed_word_percentage = (
                num_words_changed
                * 100.0
                / len(result.original_result.attacked_text.words)
            )
        else:
            perturbed_word_percentage = 0
        perturbed_word_percentages[i] = perturbed_word_percentage

    attack_success_rate = successful_attacks * 100.0 / total_attacks
    attack_success_rate = round(attack_success_rate, 2)

    perturbed_word_percentages = perturbed_word_percentages[
        perturbed_word_percentages > 0
    ]
    average_perc_words_perturbed = round(perturbed_word_percentages.mean(), 2)

    num_queries = np.array(
        [
            r.num_queries
            for r in results
            if not isinstance(r, textattack.attack_results.SkippedAttackResult)
        ]
    )
    avg_num_queries = round(num_queries.mean(), 2)

    return attack_success_rate, avg_num_queries, average_perc_words_perturbed


#####################################################################################


def evaluate_interpretability(args):
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError()
    dataset_config = DATASET_CONFIGS[args.dataset]
    test_dataset = load_dataset(args.dataset)

    all_correct_indices = set(range(len(test_dataset)))
    for path in args.checkpoint_paths:
        with open(os.path.join(path, "test_logs.json"), "r") as f:
            logs = json.load(f)
            correct_indices = logs[f"checkpoint-epoch-{args.epoch}"][args.dataset][
                "correct_indices"
            ]
            all_correct_indices = all_correct_indices.intersection(correct_indices)

    all_correct_indices = list(all_correct_indices)
    random.shuffle(all_correct_indices)
    indices = all_correct_indices[:NUM_SAMPLES_FOR_EVALUATION]

    test_dataset = test_dataset.select(indices)

    if args.model_type == "bert":
        model_type = "bert-base-uncased"
    elif args.model_type == "roberta":
        model_type = "roberta-base"

    num_gpus = torch.cuda.device_count()
    nli = args.dataset == "snli"

    print("Evaluating interpretability (this might take a long time)")

    for path in args.checkpoint_paths:
        logs = {}
        logs["indices"] = indices
        logs[f"checkpoint-epoch-{args.epoch}"] = {}
        model_path = f"{path}/checkpoint-epoch-{args.epoch}"

        print(f"====== {path}/checkpoint-epoch-{args.epoch} =====")

        if num_gpus > 1:
            torch.multiprocessing.set_start_method("spawn", force=True)
            torch.multiprocessing.set_sharing_strategy("file_system")

            in_queue = torch.multiprocessing.Queue()
            out_queue = torch.multiprocessing.Queue()
            label_names = dataset_config["label_names"]
            for i, row in enumerate(test_dataset):
                if "content" in row:
                    text = row["content"]
                elif "hypothesis" in row:
                    text = (row["premise"], row["hypothesis"])
                else:
                    text = row["text"]
                label = row["label"]
                if label == -1:
                    print("Warning: Found label==-1")
                in_queue.put((i, text, label))

            # Start workers.
            worker_pool = torch.multiprocessing.Pool(
                num_gpus,
                calc_aopc,
                (
                    model_type,
                    model_path,
                    label_names,
                    num_gpus,
                    in_queue,
                    out_queue,
                    nli,
                ),
            )
            scores = []
            pbar = tqdm.tqdm(total=len(test_dataset), smoothing=0)
            for _ in range(len(test_dataset)):
                idx, score = out_queue.get(block=True)
                pbar.update()
                if isinstance(score, Exception):
                    raise score
                scores.append(score)
            aopc_score = np.array(scores).mean()

            # Send sentinel values to worker processes
            for _ in range(num_gpus):
                in_queue.put(("END", "END", "END"))
            worker_pool.terminate()
            worker_pool.join()

        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_path
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_type, use_fast=True
            )
            aopc = AOPC(model, tokenizer, dataset_config["label_names"])
            aopc_score = aopc.calc_aopc_dataset(test_dataset)

        aopc_score = round(aopc_score, 4)
        logs[f"checkpoint-epoch-{args.epoch}"]["aopc"] = aopc_score

        print(f"AOPC: {aopc_score}")

        with open(os.path.join(path, "interpretability_eval_logs.json"), "w") as f:
            json.dump(logs, f)


def eval_robustness(args):
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError()
    dataset_config = DATASET_CONFIGS[args.dataset]

    test_dataset = load_dataset(args.dataset)

    all_correct_indices = set(range(len(test_dataset)))
    for path in args.checkpoint_paths:
        with open(os.path.join(path, "test_logs.json"), "r") as f:
            logs = json.load(f)
            correct_indices = logs[f"checkpoint-epoch-{args.epoch}"][args.dataset][
                "correct_indices"
            ]
            all_correct_indices = all_correct_indices.intersection(correct_indices)

    all_correct_indices = list(all_correct_indices)
    random.shuffle(all_correct_indices)
    indices_to_test = all_correct_indices[:NUM_SAMPLES_FOR_EVALUATION]

    test_dataset = test_dataset.select(indices_to_test)
    test_dataset = textattack.datasets.HuggingFaceDataset(
        test_dataset,
        dataset_columns=dataset_config["dataset_columns"],
        label_names=dataset_config["label_names"],
    )

    if args.model_type == "bert":
        model_type = "bert-base-uncased"
    elif args.model_type == "roberta":
        model_type = "roberta-base"
    else:
        raise ValueError(f"Unknown model type {args.model_type}.")

    print("Evaluating robustness (this might take a long time)...")

    for path in args.checkpoint_paths:
        logs = {}
        logs["indices"] = indices_to_test
        logs[f"checkpoint-epoch-{args.epoch}"] = {}
        model_path = f"{path}/checkpoint-epoch-{args.epoch}"
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_type, use_fast=True
        )
        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
            model, tokenizer
        )
        print(f"====== {path}/checkpoint-epoch-{args.epoch} =====")
        for attack_name in args.attacks:
            log_file_name = f"{path}/{attack_name}-test-{args.epoch}"
            attack_args = textattack.AttackArgs(
                num_examples=NUM_SAMPLES_FOR_EVALUATION,
                parallel=(torch.cuda.device_count() > 1),
                disable_stdout=True,
                num_workers_per_device=1,
                query_budget=10000,
                shuffle=False,
                log_to_txt=log_file_name + ".txt",
                log_to_csv=log_file_name + ".csv",
                silent=True,
            )
            if attack_name == "a2t":
                attack = textattack.attack_recipes.A2TYoo2021.build(
                    model_wrapper, mlm=False
                )
            elif attack_name == "a2t_mlm":
                attack = textattack.attack_recipes.A2TYoo2021.build(
                    model_wrapper, mlm=True
                )
            elif attack_name == "textfooler":
                attack = textattack.attack_recipes.TextFoolerJin2019.build(
                    model_wrapper
                )
            elif attack_name == "bae":
                attack = textattack.attack_recipes.BAEGarg2019.build(model_wrapper)
            elif attack_name == "pwws":
                attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
            elif attack_name == "pso":
                attack = textattack.attack_recipes.PSOZang2020.build(model_wrapper)

            attacker = textattack.Attacker(attack, test_dataset, attack_args)
            results = attacker.attack_dataset()

            (
                attack_success_rate,
                avg_num_queries,
                avg_pct_perturbed,
            ) = calc_attack_stats(results)
            logs[f"checkpoint-epoch-{args.epoch}"][attack_name] = {
                "attack_success_rate": attack_success_rate,
                "avg_num_queries": avg_num_queries,
                "avg_pct_perturbed": avg_pct_perturbed,
            }

            print(
                f"{attack_name}: {round(attack_success_rate, 1)} (attack success rate) | {avg_num_queries} (avg num queries) | {avg_pct_perturbed} (avg pct perturbed)"
            )

        with open(os.path.join(path, "robustness_eval_logs.json"), "w") as f:
            json.dump(logs, f)


def eval_accuracy(args):
    print("Evaluating accuarcy")
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError()
    dataset_config = DATASET_CONFIGS[args.dataset]
    test_datasets = dataset_config["eval_datasets"]
    eval_datasets = [
        (test_datasets[key], load_dataset(test_datasets[key])) for key in test_datasets
    ]

    for path in args.checkpoint_paths:
        logs = {}
        model_save_path = os.path.join(path, f"checkpoint-epoch-{args.epoch}")
        if args.model_type == "bert":
            model = transformers.BertForSequenceClassification.from_pretrained(
                model_save_path
            )
            tokenizer = transformers.BertTokenizerFast.from_pretrained(
                "bert-base-uncased"
            )
        elif args.model_type == "roberta":
            model = transformers.RobertaForSequenceClassification.from_pretrained(
                model_save_path
            )
            tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
                "roberta-base"
            )
        else:
            raise ValueError()

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)

        model.eval()
        model.cuda()

        if isinstance(model, torch.nn.DataParallel):
            eval_batch_size = 128 * num_gpus
        else:
            eval_batch_size = 128

        logs[f"checkpoint-epoch-{args.epoch}"] = {}
        print(f"====== {path}/checkpoint-epoch-{args.epoch} =====")

        for dataset_name, dataset in eval_datasets:
            input_columns = DATASET_CONFIGS[dataset_name]["dataset_columns"][0]
            collate_func = functools.partial(collate_fn, input_columns)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=eval_batch_size, collate_fn=collate_func
            )

            preds_list = []
            labels_list = []

            with torch.no_grad():
                for batch in dataloader:
                    input_texts, labels = batch
                    input_ids = tokenizer(
                        input_texts,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                    )
                    for key in input_ids:
                        if isinstance(input_ids[key], torch.Tensor):
                            input_ids[key] = input_ids[key].cuda()
                    logits = model(**input_ids)[0]

                    preds = logits.argmax(dim=-1).detach().cpu()
                    preds_list.append(preds)
                    labels_list.append(labels)

            preds = torch.cat(preds_list)
            labels = torch.cat(labels_list)

            compare = preds == labels
            num_correct = compare.sum().item()
            accuracy = round(num_correct / len(labels), 4)
            correct = torch.nonzero(compare, as_tuple=True)[0].tolist()

            logs[f"checkpoint-epoch-{args.epoch}"][dataset_name] = {
                "accuracy": accuracy,
                "correct_indices": correct,
            }

            print(f"{dataset_name}: {accuracy}")

        if args.save_log:
            with open(
                os.path.join(
                    os.path.dirname(model_save_path), "accuracy_eval_logs.json"
                ),
                "w",
            ) as f:
                json.dump(logs, f)


def main(args):
    for path in args.checkpoint_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path {path} not found.")
    if args.accuracy:
        eval_accuracy(args)

    if args.robustness:
        eval_robustness(args)

    if args.interpretability:
        evaluate_interpretability(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(list(DATASET_CONFIGS.keys())),
        help="Name train dataset.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["bert", "roberta"],
        help="Type of model. Choices: `bert` and `robert`.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        type=str,
        nargs="*",
        default=None,
        help="Path of model checkpoint",
    )
    parser.add_argument(
        "--epoch", type=int, default=4, help="Epoch of model to evaluate."
    )
    parser.add_argument(
        "--save-log", action="store_true", help="Save evaluation result as log."
    )
    parser.add_argument("--accuracy", action="store_true", help="Evaluate accuracy.")
    parser.add_argument(
        "--robustness", action="store_true", help="Evaluate robustness."
    )
    attack_choices = ["a2t", "at2_mlm", "textfooler", "bae", "pwws", "pso"]
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="*",
        default=None,
        help=f"Attacks to use to measure robustness. Choices are {attack_choices}.",
    )
    parser.add_argument(
        "--interpretability",
        action="store_true",
        help="Evaluate interpretability using AOPC metric.",
    )

    args = parser.parse_args()
    main(args)
