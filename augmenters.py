import transformers
import torch
import nltk
import numpy as np


class BackTranslationAugmenter:
    def __init__(self, sample_temp=0.8, batch_size=16):
        self.en2de_model = transformers.FSMTForConditionalGeneration.from_pretrained(
            "facebook/wmt19-en-de"
        )
        self.en2de_tokenizer = transformers.FSMTTokenizer.from_pretrained(
            "facebook/wmt19-en-de"
        )
        self.de2en_model = transformers.FSMTForConditionalGeneration.from_pretrained(
            "facebook/wmt19-de-en"
        )
        self.de2en_tokenizer = transformers.FSMTTokenizer.from_pretrained(
            "facebook/wmt19-de-en"
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.en2de_model.eval()
        self.de2en_model.eval()
        self.en2de_model.to(self._device)
        self.de2en_model.to(self._device)
        self.sample_temp = sample_temp
        self.batch_size = batch_size

    def __call__(self, text):
        """
        Generate list of augmented data based off of `text`

        Args:
            text (str): seed text
        Returns:
            augmented_text (list[str]): List of augmented text
        """
        return self.batch_call([text])[0]

    def batch_call(self, texts):
        # First split paragraphs into sentences:
        texts_as_sent = []
        num_sents = []
        for text in texts:
            sentences = nltk.sent_tokenize(text)
            texts_as_sent.extend(sentences)
            num_sents.append(len(sentences))

        i = 0
        translated_texts = []
        while i < len(texts_as_sent):
            batch = texts_as_sent[i : i + self.batch_size]
            i += self.batch_size

            with torch.no_grad():
                en_inputs = self.en2de_tokenizer.batch_encode_plus(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self._device)
                de_outputs = self.en2de_model.generate(
                    en_inputs["input_ids"], do_sample=True, temperature=self.sample_temp
                )
                de_texts = [
                    self.en2de_tokenizer.decode(output, skip_special_tokens=True)
                    for output in de_outputs
                ]
                de_inputs = self.de2en_tokenizer.batch_encode_plus(
                    de_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self._device)
                en_outputs = self.de2en_model.generate(
                    de_inputs["input_ids"], do_sample=True, temperature=self.sample_temp
                )
                en_texts = [
                    self.de2en_tokenizer.decode(output, skip_special_tokens=True)
                    for output in en_outputs
                ]

            translated_texts.extend(en_texts)

        augmented_data = []
        j = 0
        for n in num_sents:
            augmented_data.append(" ".join(translated_texts[j : j + n]))
            j += n

        return augmented_data


class SSMBA:
    """
    Data augmentation method proposed by "SSMBA: Self-Supervised Manifold Based Data Augmentation forImproving Out-of-Domain Robustness" (Ng et. al., 2020)
    Most of the code has been adapted or copied from https://github.com/nng555/ssmba

    Args:
        model (str): name of masked language model from Huggingface's `transformers`
        noise_prob (float): Probability for selecting a token for noising. Selected tokens are then masked, randomly replaced, or left the same.
            Default is 0.15.
        random_token_prob (float): Probability of a selected token being replaced randomly from the vocabulary. Default is 0.1
        leave_unmasked_prob (float): Probability of a selected o tken being left unmasked and unchanged. Default is 0.1
        max_tries (int): Num of tries to generate a unique sample before giving up Default is 10.
        num_samples (float): Number of augmented samples to generate for each sample. Default is 4.
        top_k (int): Top k to use for sampling reconstructed tokens from the BERT model. -1 indicates unrestricted sampling. Default is -1.
        min_seq_len (int): Minimum sequence length of the input for agumentation. Default is 4
        max_seq_len (int): Maximum sequence length of the input for augmentation. Default is 512
    """

    def __init__(
        self,
        model="bert-base-uncased",
        noise_prob=0.15,
        random_token_prob=0.1,
        leave_unmasked_prob=0.1,
        max_tries=10,
        num_samples=1,
        top_k=-1,
        min_seq_len=4,
        max_seq_len=512,
    ):
        self.mlm_model = transformers.AutoModelForMaskedLM.from_pretrained(model).cuda()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        self.noise_prob = noise_prob
        self.random_token_prob = random_token_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.max_tries = max_tries
        self.num_samples = num_samples
        self.top_k = top_k
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        self._softmax_mask = np.full(len(self.tokenizer.vocab), False)
        self._softmax_mask[self.tokenizer.all_special_ids] = True
        self._weights = np.ones(len(self.tokenizer.vocab))
        self._weights[self.tokenizer.all_special_ids] = 0
        for k, v in self.tokenizer.vocab.items():
            if "[unused" in k:
                self._softmax_mask[v] = True
                self._weights[v] = 0

        self._weights = self._weights / self._weights.sum()

    def _mask_and_corrupt(self, tokens):
        """
        Main corruption function that (1) randomly masks tokens
        and (2) randomly switches tokens with another random token sampled from the vocabulary.

        Args:
            tokens (np.ndarray): numpy array of input tokens
        Returns:
            masked_tokens, mask_targets (tuple[torch.Tensor, torch.Tensor]):
                `masked_tokens` is tensor of tokenized `text` after being corrupted, while `mask_targets` is a tensor storing the original values of tokens
                that have been corrupted.
        """
        if self.noise_prob == 0.0:
            return tokens

        seq_len = len(tokens)
        mask = np.full(seq_len, False)
        # number of tokens to mask
        num_mask = int(self.noise_prob * seq_len + np.random.rand())

        mask_choice_p = np.ones(seq_len)
        for i in range(seq_len):
            if tokens[i] in self.tokenizer.all_special_ids:
                mask_choice_p[i] = 0
        mask_choice_p = mask_choice_p / mask_choice_p.sum()

        mask[np.random.choice(seq_len, num_mask, replace=False, p=mask_choice_p)] = True

        # decide unmasking and random replacement
        rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (np.random.rand(seq_len) < rand_or_unmask_prob)
            if self.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
            elif self.leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
            else:
                unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                decision = np.random.rand(seq_len) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
            unmask = rand_mask = None

        if unmask is not None:
            mask = mask ^ unmask

        tokens[mask] = self.tokenizer.mask_token_id
        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                tokens[rand_mask] = np.random.choice(
                    len(self.tokenizer.vocab),
                    num_rand,
                    p=self._weights,
                )

        mask_targets = np.full(len(mask), self.tokenizer.pad_token_id)
        mask_targets[mask] = tokens[mask == 1]

        return torch.tensor(tokens).long(), torch.tensor(mask_targets).long()

    def _reconstruction_prob_tok(self, masked_tokens, target_tokens):
        single = masked_tokens.dim() == 1

        # expand batch size 1
        if single:
            masked_tokens = masked_tokens.unsqueeze(0)
            target_tokens = target_tokens.unsqueeze(0)

        masked_index = (target_tokens != self.tokenizer.pad_token_id).nonzero(
            as_tuple=True
        )
        masked_orig_index = target_tokens[masked_index]

        # edge case of no masked tokens
        if len(masked_orig_index) == 0:
            return masked_tokens

        masked_orig_enum = [list(range(len(masked_orig_index))), masked_orig_index]

        masked_tokens = masked_tokens.cuda()
        target_tokens = target_tokens.cuda()
        outputs = self.mlm_model(masked_tokens, labels=target_tokens)

        features = outputs[1]

        logits = features[masked_index]

        for i in range(len(logits)):
            logits[i][self._softmax_mask] = float("-inf")
        probs = logits.softmax(dim=-1)

        # sample from topk
        if self.top_k != -1:
            values, indices = probs.topk(k=self.top_k, dim=-1)
            kprobs = values.softmax(dim=-1)
            if len(masked_index) > 1:
                samples = torch.cat(
                    [
                        idx[torch.multinomial(kprob, 1)]
                        for kprob, idx in zip(kprobs, indices)
                    ]
                )
            else:
                samples = indices[torch.multinomial(kprobs, 1)]

        # unrestricted sampling
        else:
            if len(masked_index) > 1:
                samples = torch.cat([torch.multinomial(prob, 1) for prob in probs])
            else:
                samples = torch.multinomial(probs, 1)

        # set samples
        masked_tokens[masked_index] = samples

        if single:
            return masked_tokens[0]
        else:
            return masked_tokens

    def _decode_tokens(self, tokens):
        """
        Decode tokens into string

        Args:
            tokens (torch.Tensor): tokens of ids
        Returns:
            text (str): decoded string
        """
        # remove [CLS] and [SEP] tokens
        tokens = tokens[1:-1]
        # remove [PAD] tokens
        tokens = tokens[tokens != self.tokenizer.pad_token_id]
        return self.tokenizer.decode(tokens).strip()

    def __call__(self, text):
        """
        Generate list of augmented data based off of `text`

        Args:
            text (str): seed text
        Returns:
            augmented_text (list[str]): List of augmented text
        """
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            return_tensors="np",
            truncation=True,
            max_length=self.max_seq_len,
        )[0]
        if len(tokens) < self.min_seq_len or len(tokens) > self.max_seq_len:
            raise ValueError(
                f"Given input of sequence length {len(tokens)} is too short. Minimum sequence length is {self.min_seq_len} "
                f"and maximum sequence length is {self.max_seq_len}."
            )

        num_tries = 0
        new_samples = []

        while num_tries < self.max_tries:
            masked_tokens, target_tokens = self._mask_and_corrupt(np.copy(tokens))
            new_sample = self._reconstruction_prob_tok(masked_tokens, target_tokens)
            num_tries += 1
            new_sample = self._decode_tokens(new_sample)

            # check if identical reconstruction or empty
            if new_sample != text and new_sample != "":
                new_samples.append(new_sample)
                break

        return new_samples[0]
