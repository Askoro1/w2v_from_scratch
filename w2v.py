import numpy as np
from tqdm import tqdm
from datasets import load_dataset


def sigmoid(x):
    # I use the standard formula for x > 0
    # In case of x < 0, I use the alternative formula to avoid the overflow of e^(-x)
    return np.where(x >= 0., 1. / (1. + np.exp(-x)), np.exp(x) / (1. + np.exp(x)))


class Word2Vec:
    def __init__(self, corpus, embedding_size, window_size=5, num_negative_samples=5):
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.num_negative_samples = num_negative_samples

        self._preprocess_data(corpus)
        self._create_dicts()
        self._create_lookup_tables()
        self._build_negative_samples_table()

    def _preprocess_data(self, corpus, min_count=5):
        self.vocab = {}
        self.corpus_tokenized = []
        self.num_tokens = 0

        print("Building corpus + vocab...")
        for text in tqdm(corpus["text"]):
            text = text.strip().lower().split()
            self.corpus_tokenized.append(text)
            self.num_tokens += len(text)
            for word in text:
                if word not in self.vocab:
                    self.vocab[word] = 0
                self.vocab[word] += 1
        print("Built corpus + vocab")

        print("Filtering rare words...")
        self.vocab = {word: (freq / self.num_tokens) for word, freq in self.vocab.items() if freq >= min_count}
        print("Filtered rare words")

        print("Subsampling...")
        discard_const = 1e-5
        stop_list = []
        for word in self.vocab.keys():
            discard_prob = np.clip(1. - np.sqrt(discard_const / self.vocab[word]), 0., 1.)
            if np.random.random() < discard_prob:
                stop_list += [word]

        for word in stop_list:
            del self.vocab[word]
        print("Performed subsampling")

        print("Deleting out-of-vocab tokens")
        for text_i in tqdm(range(len(self.corpus_tokenized))):
            clear_text = []
            for word in self.corpus_tokenized[text_i]:
                if word in self.vocab:
                    clear_text += [word]
            self.corpus_tokenized[text_i] = clear_text
        print("Deleted all out-of-vocab tokens")

        self.vocab_size = len(self.vocab)

    def _create_dicts(self):
        print("Building mappings...")
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab.keys())}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        print("Built mappings")

    def _create_lookup_tables(self):
        print("Initializing embed lookup tables...")

        self.W_input = np.random.uniform(
            low=(-0.5 / self.embedding_size),
            high=(0.5 / self.embedding_size),
            size=(self.vocab_size, self.embedding_size)
        ).astype(np.float32)

        self.W_output = np.zeros(
            shape=(self.vocab_size, self.embedding_size),
            dtype=np.float32
        )

        print("Initialized embed lookup tables")

    def _build_negative_samples_table(self):
        print("Building negative samples table...")

        self.negative_samples_table_size = int(1e8)
        freq_power = 0.75

        freq_table = np.zeros(self.vocab_size)

        for (idx, word) in self.idx_to_word.items():
            freq_table[idx] = self.vocab[word]

        prob_table = np.power(freq_table, freq_power)
        prob_table /= np.sum(prob_table)

        self.negative_samples_table = np.random.choice(
            np.arange(self.vocab_size, dtype=np.int32),
            size=self.negative_samples_table_size,
            p=prob_table
        )

        print("Built negative samples table")

    def train(self, base_lr=0.025, min_lr=0.0000025, epochs=5, iters_per_epoch=10000000, valid_sim=None,
              valid_analogies=None):
        print("Training...")
        for epoch in range(epochs):
            print(f"epoch {epoch + 1}/{epochs}")
            np.random.shuffle(self.corpus_tokenized)
            iters_performed = 0
            running_loss = 0.

            with tqdm(total=iters_per_epoch, position=0, ncols=120) as pbar:
                for text_i in range(len(self.corpus_tokenized)):
                    for j, center_word in enumerate(self.corpus_tokenized[text_i]):
                        center_word_idx = self.word_to_idx[center_word]
                        for k in range(-self.window_size, self.window_size + 1):
                            if k == 0:
                                continue
                            if j + k < 0 or j + k >= len(self.corpus_tokenized[text_i]):
                                continue
                            context_word = self.corpus_tokenized[text_i][j + k]
                            context_word_idx = self.word_to_idx[context_word]

                            curr_lr = np.clip(base_lr * (
                                        1. - (iters_per_epoch * epoch + iters_performed) / (iters_per_epoch * epochs)),
                                              a_min=min_lr, a_max=None)

                            loss = self._train_on_pair(center_word_idx, context_word_idx, curr_lr)
                            running_loss += loss

                            if iters_performed % 10000 == 0:
                                pbar.set_postfix({"loss (curr iter)": f"{loss:.4f}",
                                                  "loss (avg)": f"{(running_loss / (iters_performed + 1)):.3f}",
                                                  "lr": f"{curr_lr:.4f}"})

                            if iters_performed % 1000000 == 0:
                                if valid_sim is not None:
                                    pbar.write("\n")
                                    pbar.write("-" * 80)
                                    pbar.write("\n")
                                    pbar.write("Validation (Similarity Test):")
                                    for a in valid_sim:
                                        topk_s, topk_similarities = self.get_topk_similar(a)
                                        topk_str = ', '.join(
                                            [f'vec({topk_s_i}) (sim: {topk_sim_i:.4f})' for (topk_s_i, topk_sim_i) in
                                             list(zip(topk_s, topk_similarities))])
                                        pbar.write(f"\t vec({a}) is most similar to: {topk_str}")

                                if valid_analogies is not None:
                                    pbar.write("\n")
                                    pbar.write("-" * 80)
                                    pbar.write("\n")
                                    pbar.write("Validation (Analogy Test):")
                                    for (a, b, c) in valid_analogies:
                                        topk_d, _ = self.get_topk_analogies(a, b, c)
                                        topk_d = ', '.join([f'vec({topk_d_i})' for topk_d_i in topk_d])
                                        pbar.write(f"\t vec({b}) - vec({a}) + vec({c}) is most similar to: {topk_d}")

                            iters_performed += 1
                            pbar.update(1)
                            if iters_performed == iters_per_epoch:
                                break
                        if iters_performed == iters_per_epoch:
                            break
                    if iters_performed == iters_per_epoch:
                        break
        print("\nFinished training")

        if valid_sim is not None:
            print()
            print("-" * 80)
            print()
            print("Final Eval (Similarity Test):")
            for a in valid_sim:
                topk_s, topk_similarities = self.get_topk_similar(a)
                topk_str = ', '.join(
                    [f'vec({topk_s_i}) (sim: {topk_sim_i:.4f})' for (topk_s_i, topk_sim_i) in
                     list(zip(topk_s, topk_similarities))])
                print(f"\t vec({a}) is most similar to: {topk_str}")

        if valid_analogies is not None:
            print()
            print("-" * 80)
            print()
            print("Final Eval (Analogy Test):")
            for (a, b, c) in valid_analogies:
                topk_d, _ = self.get_topk_analogies(a, b, c)
                topk_d = ', '.join([f'vec({topk_d_i})' for topk_d_i in topk_d])
                print(f"\t vec({b}) - vec({a}) + vec({c}) is most similar to: {topk_d}")

    def _get_negative_samples(self):
        return self.negative_samples_table[
            np.random.randint(0, self.negative_samples_table_size, size=self.num_negative_samples, dtype=np.int32)]

    def _train_on_pair(self, center_word_idx, context_word_idx, lr):
        center_embed = self.W_input[center_word_idx]
        context_embed = self.W_output[context_word_idx]

        negative_sample_indices = self._get_negative_samples()
        negative_embeds = self.W_output[negative_sample_indices]

        pos_dotp = np.dot(center_embed, context_embed)
        neg_dotp = (center_embed[np.newaxis, :] * negative_embeds).sum(axis=1)

        pos_prob = sigmoid(pos_dotp)
        neg_prob = sigmoid(-neg_dotp)
        neg_prob_grad = sigmoid(neg_dotp)

        eps = 1e-12
        loss = -np.log(pos_prob + eps) - np.sum(np.log(neg_prob + eps))

        center_grad = (pos_prob - 1.) * context_embed + np.sum(neg_prob_grad[:, np.newaxis] * negative_embeds, axis=0)
        context_grad = (pos_prob - 1.) * center_embed
        negative_grads = neg_prob_grad[:, np.newaxis] * center_embed[np.newaxis, :]

        self.W_input[center_word_idx] -= lr * center_grad
        self.W_output[context_word_idx] -= lr * context_grad
        np.add.at(self.W_output, negative_sample_indices, -lr * negative_grads)

        return loss

    def get_topk_similar(self, word, topk=5):
        if word not in self.word_to_idx:
            return None
        word_idx = self.word_to_idx[word]
        word_embed = self.W_input[word_idx]

        dotp = np.dot(self.W_input, word_embed)
        norm_prod = np.linalg.norm(self.W_input, axis=1) * np.linalg.norm(word_embed)
        similarities = dotp / norm_prod

        topk_indices = np.argsort(similarities)[::-1][1:(topk+1)]
        topk_words = [self.idx_to_word[idx] for idx in topk_indices]
        topk_similarities = similarities[topk_indices]

        return topk_words, topk_similarities

    def get_topk_analogies(self, a, b, c, topk=5):
        if a not in self.word_to_idx or b not in self.word_to_idx or c not in self.word_to_idx:
            return None
        a_idx = self.word_to_idx[a]
        b_idx = self.word_to_idx[b]
        c_idx = self.word_to_idx[c]
        a_embed = self.W_input[a_idx]
        b_embed = self.W_input[b_idx]
        c_embed = self.W_input[c_idx]

        d_embed = b_embed - a_embed + c_embed

        dotp = np.dot(self.W_input, d_embed)
        norm_prod = np.linalg.norm(self.W_input, axis=1) * np.linalg.norm(d_embed)
        similarities = dotp / norm_prod

        topk_indices = np.argsort(similarities)[::-1][:topk]
        topk_words = [self.idx_to_word[idx] for idx in topk_indices]
        topk_similarities = similarities[topk_indices]

        return topk_words, topk_similarities


def main():
    np.random.seed(0)
    corpus = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    w2v = Word2Vec(corpus, embedding_size=256, window_size=5, num_negative_samples=5)

    valid_sim = ["apple", "europe", "mathematics", "essay", "stanford"]

    valid_analogies = [
        ("baghdad", "iraq", "berlin"),
        ("finland", "helsinki", "greece"),
        ("walk", "walked", "swim"),
        ("wise", "wiser", "smart"),
        ("japan", "yen", "usa")
    ]

    w2v.train(valid_sim=valid_sim, valid_analogies=valid_analogies)


if __name__ == "__main__":
    main()