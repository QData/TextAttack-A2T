DATASET_CONFIGS = {
    "imdb": {
        "labels": 2,
        "label_names": ["negative", "positive"],
        "dataset_columns": (["text"], "label"),
        "local_path": "./data/imdb",
        "eval_datasets": {"test": "imdb", "cross_domain": "yelp"},
    },
    "yelp": {
        "labels": 2,
        "label_names": ["negative", "positive"],
        "dataset_columns": (["text"], "label"),
        "local_path": "./data/yelp",
        "eval_datasets": {"test": "yelp", "cross_domain": "imdb"},
    },
    "rt": {
        "labels": 2,
        "label_names": ["negative", "positive"],
        "dataset_columns": (["text"], "label"),
        "remote_name": "rotten_tomatoes",
        "eval_datasets": {"test": "rt", "cross_domain": "yelp"},
    },
    "snli": {
        "labels": 3,
        "label_names": ["entailment", "neutral", "contradiction"],
        "dataset_columns": (["premise", "hypothesis"], "label"),
        "remote_name": "snli",
        "eval_datasets": {"test": "snli", "cross_domain": "mnli"},
    },
    "mnli": {
        "labels": 3,
        "label_names": ["entailment", "neutral", "contradiction"],
        "dataset_columns": (["premise", "hypothesis"], "label"),
        "remote_name": "multi_nli",
        "split": "validation_mismatched+validation_matched",
    },
}
