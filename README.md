# A2T: Towards Improving Adversarial Training of NLP Models

This is the source code for the EMNLP 2021 (Findings) paper ["Towards Improving Adversarial Training of NLP Models"](https://arxiv.org/abs/2109.00544).

If you use the code, please cite the paper:
```
@misc{yoo2021improving,
      title={Towards Improving Adversarial Training of NLP Models}, 
      author={Jin Yong Yoo and Yanjun Qi},
      year={2021},
      eprint={2109.00544},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Prerequisites
The work heavily relies on the [TextAttack](https://github.com/QData/TextAttack) package. In fact, the main training code is implemented in the TextAttack package.

Required packages are listed in the `requirements.txt` file.
```
pip install -r requirements.txt
```

## Data
All of the data used for the paper are available from HuggingFace's [Datasets](https://huggingface.co/datasets).

For IMDB and Yelp datasets, because there are no official validation splits, we randomly sampled 5k and 10k, respectively, from the training set and used them as valid splits. The splits are available in `data`.

Also, augmented training data generated using SSMBA and back-translation are available under `data/augmented_data`.

## Training
To train BERT model on IMDB dataset with A2T attack for 4 epochs and 1 clean epoch with gamma of 0.2:
```
python train.py \
    --train imdb \
    --eval imdb \
    --model-type bert \
    --model-save-path ./example \
    --num-epochs 4 \
    --num-clean-epochs 1 \
    --num-adv-examples 0.2 \
    --attack-epoch-interval 1 \
    --attack a2t \
    --learning-rate 5e-5 \
    --num-warmup-steps 100 \
    --grad-accumu-steps 1 \
    --checkpoint-interval-epochs 1 \
    --seed 42
```

You can also pass `roberta` to train RoBERTa model instead of BERT model. To select other datasets from the paper, pass `rt` (MR), `yelp`, or `snli` for `--train` and `--eval`.

This script is actually just to run the `Trainer` class from the TextAttack package. To checkout how training is performed, please checkout the `Trainer` [class](https://github.com/QData/TextAttack/blob/master/textattack/trainer.py).

## Evaluation
To evalute the accuracy, robustness, and interpretability of our trained model from above, run
```
python evaluate.py \
    --dataset imdb \
    --model-type bert \
    --checkpoint-paths ./example_run \
    --epoch 4 \
    --save-log \
    --accuracy \
    --robustness \
    --attacks a2t a2t_mlm textfooler bae pwws pso \
    --interpretability 
```

This takes the last checkpoint model (`--epoch 4`) and evaluates its accuracy on both IMDB and Yelp dataset (for cross-domain accuracy). It also evalutes the model's robustness against A2T, A2T-MLM, TextFooler, BAE, PWWS, and PSO attacks. Lastly, with the `--interpretability` flag, AOPC scores are calculated. 

Note that you will have to run `--robustness` and `--interpretability` with `--accuracy` (or after you separately evaluate accuracy) since both robustness and intepretability evaluations rely on the accuracy evaluation to know which samples the model was able to predict correctly.
By default 1000 samples are attacked to evaluate robustness. Likewise, 1000 samples are used to calculate AOPC score for interpretability.

If you're evaluating multiple models for comparison, it's also advised that you provide all the checkpoint paths together to `--checkpoint-paths`. This is because the samples that are correctly by each model will be different, so we first need to identify the intersection of the all correct predictions before using them to evaluate robustness for all the models. This will allow fairer comparison of models' robustness rather than using attack different samples for each model.

## Data Augmentation
Lastly, we also provide `augment.py` which we used to perform data augmentation methods such as SSMBA and back-translation.

Following is an example command for augmenting imdb dataset with SSMBA method.
```
python augment.py \
    --dataset imdb \
    --augmentation ssmba \
    --output-path ./augmented_data \
    --seed 42 
```

You can also pass `backtranslation` to `--augmentation`.
