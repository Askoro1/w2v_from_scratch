# Word2Vec SkipGram Train Loop Implementation in pure numpy

This repository contains an implementation of the Word2Vec Skip-Gram with Negative Sampling method for training and generating embeddings, created using only numpy.

I trained the Skip-Gram model I developed on a text corpus compiled from Wikipedia, `WikiText/wikitext-103-raw-v1`. I also conducted two tests to evaluate the quality of the resulting embeddings: Similarity Test - finding the top-K embeddings most similar to a given embedding - and Analogy Test - for a triplet of tokens A, B, C, where A and B are related by a certain relationship (e.g., A=country $\leftrightarrow$ B=capital), finding a token D related to C by the same relationship.

## Logic of the Method

I followed the standard approach and techniques described in the paper where SkipGram with Negative Sampling was introduced ([link](https://arxiv.org/pdf/1310.4546)). 

The model is defined by two trainable matrices: `W_input` (the matrix of central words) and `W_output` (the matrix of context words).

The text corpus is scanned during training. For each text, every word is treated as the central word, and the words to the left and right of it, up to a fixed distance (the window size), are considered as context words. Additionally, N negative examples are sampled for each pair consisting of the central and context word. A model training step is performed based on the resulting triplet.

The model is trained to minimize the following loss:

$$-\log \sigma ({v_{w_O}}^{\top} v_{w_I}) - \sum_{i=1}^{k}\mathbb{E}_{w_i \sim P_n(w)}\big[\log \sigma (-{v_{w_i}}^{\top} v_{w_I})\big]$$

Thus, we maximize the probability of a given context word appearing near the central word and minimize the probability of negative examples appearing near the central word (i.e., words that, in theory, should not appear near the central word). Here we allow some simplification of sampling procedure: we sample arbitrary negative examples, which may include the context word, but the probability of this happening is extremely low, so we ignore it.

## Implementation Details

- Following the classic implementation by the authors of Skip-Gram, I implement a streaming version of the training process; that is, I do not pre-compute triplets for training, but generate them on the fly. I am training in batches of size 1 for simplicity.

- To speed up the sampling of negative examples, I replace on-the-fly sampling with a precomputed table of negative examples, from which values can then be retrieved in $O(1)$ time.

- For sampling (to generate a table of negative examples), I use the Modified Unigram Distribution, just like the authors of the original paper:

    $P(w_i) = \dfrac{\text{freq}(w_i)^{0.75}}{\sum_{j = 1}^{k} \text{freq}(w_j)^{0.75}}$

- Unlike the authors, who chose a random size for each window, I use a fixed window size for simplicity (which is generally also a valid approach for Word2Vec).

- During the vocabulary and text corpus preparation, I filter out rare words (those that appear fewer than 5 times in the text), just as in the original implementation, and I also use subsampling to discard frequent words with a certain probability. In subsampling, I use the following probability to discard a word:

    $$P(w_i) = 1 - \sqrt{\dfrac{t}{\text{freq}(w_i)}}$$, where $$t = 10^{-5}$$

- I initialize matrix `W_input` with samples from $U\left(\left[-\frac{0.5}{\text{dim}}, \frac{0.5}{\text{dim}}\right]\right)$ and matrix `W_output` with zeros, just as the authors of the original implementation did.

- During training, I use linearly decaying learning rate scheduling based on the number of tokens the model is already trained on, just as the authors of the original paper did, to ensure more stable training.

## Evaluation Results

I trained 256-dimensional Skip-Gram embeddings. I trained them for 5 epochs, each consisting of 10 million iterations.

The test results are shown below (I used cosine similarity as a measure of embedding similarity):

### Similarity Test 

| Query Word  | 1st Similar        | 2nd Similar       | 3rd Similar        | 4th Similar       | 5th Similar          |
| ----------- | ------------------ | ----------------- | ------------------ | ----------------- | -------------------- |
| apple       | app (0.9534)       | google (0.9506)   | update (0.9498)    | ipad (0.9429)     | marketplace (0.9428) |
| europe      | asia (0.9113)      | italy (0.8976)    | denmark (0.8946)   | finland (0.8932)  | sweden (0.8865)      |
| mathematics | economics (0.9704) | academic (0.9635) | sociology (0.9509) | sciences (0.9482) | thesis (0.9448)      |
| essay       | essays (0.9841)    | bible (0.9789)    | biblical (0.9759)  | hymns (0.9751)    | authorship (0.9748)  |
| stanford    | yale (0.9712)      | alumnus (0.9700)  | gibbs (0.9662)     | ucla (0.9658)     | hopkins (0.9633)     |

### Analogy Test

| Analogy                     | 1st Result | 2nd Result  | 3rd Result | 4th Result | 5th Result   |
| --------------------------- | ---------- | ----------- | ---------- | ---------- | ------------ |
| iraq − baghdad + berlin     | overseas   | afghanistan | nato       | berlin     | humanitarian |
| helsinki − finland + greece | crusader   | antioch     | pasha      | mehmed     | saladin      |
| walked − walk + swim        | dash       | misses      | sweeps     | concussion | quick        |
| wiser − wise + smart        | twists     | filler      | lush       | eyed       | crisp        |
| yen − japan + usa           | robinson   | martyn      | ward       | leigh      | warren       |

As it can be seen above, Similarity Test produced meaningful neighbors for all five examples. Analogy Test, however, demonstrated limited results. Nevertheless, better results can be expected with more extensive training.

## Usage

I’ve attached the notebook with usage scenario in the repository.

Alternatively, one may run the training of this Word2Vec implementation using the following script:

```
git clone https://github.com/Askoro1/w2v_from_scratch
cd w2v_from_scratch
pip install -r requirements.txt
python w2v.py
```

