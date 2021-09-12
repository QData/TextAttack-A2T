import os
import argparse
import datasets
import tqdm
import torch
import random
import numpy as np

from augmenters import BackTranslationAugmenter, SSMBA
from configs import DATASET_CONFIGS


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def augment(args, num_gpus, in_queue, out_queue):
    gpu_id = (torch.multiprocessing.current_process()._identity[0] - 1) % num_gpus
    set_seed(args.seed)
    torch.cuda.set_device(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)

    if args.augmentation == "backtranslation":
        augmenter = BackTranslationAugmenter()
    elif args.augmentation == "ssmba":
        augmenter = SSMBA()
    else:
        raise ValueError(f"Unknown augmentation {augmentation}.")

    while True:
        try:
            i, inputs, label = in_queue.get()
            if i == "END" and example == "END" and ground_truth_output == "END":
                # End process when sentinel value is received
                break
            else:
                if isinstance(inputs, tuple):
                    text_to_augment = inputs[1]
                else:
                    text_to_augment = inputs

                augmented_text = ""
                tries = 0
                while augmented_text == "" and tries < 10:
                    augmented_text = augmenter(text_to_augment)
                    augmented_text = augmented_text.strip()
                    tries += 1

                if isinstance(inputs, tuple):
                    augmented_text = (inputs[0], augmented_text)

                out_queue.put((i, augmented_text, label))
        except Exception as e:
            out_queue.put((i, e, e))


def main(args):
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {args.dataset}")
    dataset_config = DATASET_CONFIGS[args.dataset]

    if "local_path" in dataset_config:
        dataset = datasets.load_dataset(
            "csv",
            data_files=os.path.join(dataset_config["local_path"], "train.tsv"),
            delimiter="\t",
        )["train"]
    else:
        dataset = datasets.load_dataset(dataset_config["remote_name"], split="train")

    augmented_text = []
    augmented_label = []
    augmented_indices = []
    num_workers = torch.cuda.device_count()
    assert num_workers >= 1, "You need at least one GPU to perform augmentation."

    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")

    in_queue = torch.multiprocessing.Queue()
    out_queue = torch.multiprocessing.Queue()
    input_columns = dataset_config["dataset_columns"][0]
    for i, row in enumerate(dataset):
        input_text = tuple(row[col] for col in input_columns)
        if len(input_text) == 1:
            input_text = input_text[0]
        in_queue.put((i, input_text, row["label"]))

    # Start workers.
    worker_pool = torch.multiprocessing.Pool(
        num_workers,
        augment,
        (
            args,
            num_workers,
            in_queue,
            out_queue,
        ),
    )
    pbar = tqdm.tqdm(total=len(dataset), smoothing=0)
    for _ in range(len(dataset)):
        idx, aug_text, aug_label = out_queue.get(block=True)
        pbar.update()
        if isinstance(aug_text, Exception):
            continue
        if aug_text == "":
            continue
        augmented_indices.append(idx)
        augmented_text.append(aug_text)
        augmented_label.append(aug_label)

    # Send sentinel values to worker processes
    for _ in range(num_workers):
        in_queue.put(("END", "END", "END"))
    worker_pool.terminate()
    worker_pool.join()

    augmented_indices = np.array(augmented_indices)
    argsort_indices = np.argsort(augmented_indices)
    augmented_text = [augmented_text[i] for i in argsort_indices]
    augmented_label = [augmented_label[i] for i in argsort_indices]

    if isinstance(augmented_text[0], tuple):
        augmented_data = {
            col: [t[i] for t in augmented_text] for i, col in enumerate(input_columns)
        }
        augmented_data["label"] = augmented_label
    else:
        augmented_data = {input_columns[0]: augmented_text, "label": augmented_label}

    augmented_dataset = datasets.Dataset.from_dict(augmented_data)
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    augmented_dataset.to_csv(args.output_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation",
        type=str,
        required=True,
        choices=["ssmba", "backtranslation"],
        help="Augmentation to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(list(DATASET_CONFIGS.keys())),
        help="Name of dataset to augment",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for augmented data (in TSV format).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    main(args)
