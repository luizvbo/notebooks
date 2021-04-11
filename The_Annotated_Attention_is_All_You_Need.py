# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="cSTdMB0-dSxy"
# # Attention is all you need
#

# %% [markdown] id="tSWEk4ttUgQH"
# > When teaching, I emphasize implementation as a way to understand recent developments in ML. This post is an attempt to keep myself honest along this goal. The recent ["Attention is All You Need"]
# (https://arxiv.org/abs/1706.03762) paper from NIPS 2017 has been instantly impactful paper as a new method for machine translation and potentiall NLP generally. The paper is very clearly written, but the conventional wisdom has been that it is quite difficult to implement correctly.
# >
# > In this post I follow the paper through from start to finish and try to implement each component in code.
# (I have done some minor reordering and skipping from the original paper). This document itself is a working notebook, and should be a completely usable and efficient implementation. To follow along you will first need to install [PyTorch](http://pytorch.org/) and [torchtext](https://github.com/pytorch/text). The complete code is available on [github](https://github.com/harvardnlp/annotated-transformer).
# >- Alexander "Sasha" Rush ([@harvardnlp](https://twitter.com/harvardnlp))
#

# %% id="ZaYyfFUqUnGY"
# # !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext

# %% id="4LTc4HW7UgQI"
# Standard PyTorch imports
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import tensorflow as tf
import math
from copy import deepcopy
# from torch.autograd import Variable

# For plots
# %matplotlib inline
import matplotlib.pyplot as plt


# %% [markdown] id="jKALH2Rlc1FC"
# > The comments from the original post author are blockquoted. The main text is from the paper itself.

# %% [markdown] id="esxhOQubUgQL"
# # Background

# %% [markdown] id="1M-PiEMOUgQM"
# The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU
# [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building
# block, computing hidden representations in parallel for all input and output positions. In these models,
# the number of operations required to relate signals from two arbitrary input or output positions grows
# in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes
# it more difficult to learn dependencies between distant positions [12]. In the Transformer this is
# reduced to a constant number of operations, albeit at the cost of reduced effective resolution due
# to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as
# described in section 3.2.
#
# Self-attention, sometimes called intra-attention is an attention mechanism relating different positions
# of a single sequence in order to compute a representation of the sequence. Self-attention has been
# used successfully in a variety of tasks including reading comprehension, abstractive summarization,
# textual entailment and learning task-independent sentence representations [4, 27, 28, 22].
# End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned
# recurrence and have been shown to perform well on simple-language question answering and
# language modeling tasks [34].
#
# To the best of our knowledge, however, the Transformer is the first transduction model relying
# entirely on self-attention to compute representations of its input and output without using sequencealigned
# RNNs or convolution. In the following sections, we will describe the Transformer, motivate
# self-attention and discuss its advantages over models such as [17, 18] and [9].

# %% [markdown] id="84vTAA5TUgQM"
# # Model Architecture

# %% [markdown] id="VBOvyU9BUgQN"
# Most competitive neural sequence transduction models have an encoder-decoder structure [(cite)](cho2014learning,bahdanau2014neural,sutskever14). Here, the encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous representations $\mathbf{z} = (z_1, ..., z_n)$. Given $\mathbf{z}$, the decoder then generates an output sequence $(y_1,...,y_m)$ of symbols one element at a time. At each step the model is auto-regressive [(cite)](graves2013generating), consuming the previously generated symbols as additional input when generating the next.

# %% id="1AC8KeDJUgQO"
class Transformer(tf.keras.Model):
    """
    A standard Encoder-Decoder architecture. Base model for this and many
    other models.
    """
    def __init__(
            self,
            input_vocab,
            output_vocab,
            N=6,  # Number of layers               L
            d_model=512,  # Hidden size            H
            d_ff=2048,
            h=8,  # Number o attention heads       A
            dropout=0.1
            # encoder, decoder, input_embed, output_embed, generator
    ):
        super().__init__()
        self.input_embed = tf.keras.Sequential(
            Embeddings(d_model, input_vocab),
            PositionalEncoding(d_model, dropout)
        )
        self.encoder = Encoder(h, N, d_model, d_ff, dropout)
        self.decoder = Decoder(h, N, d_model, d_ff, dropout)
        self.output_embed = tf.keras.Sequential(
            Embeddings(d_model, output_vocab),
            PositionalEncoding(d_model, dropout)
        )
        self.generator = Generator(d_model, output_vocab)

    def call(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encoder(self.input_embed(src), src_mask)
        output = self.decoder(
            self.output_embed(tgt), memory, src_mask, tgt_mask
        )
        return output


# %% id="XtJ9EDY17TtX"
class LayerList(tf.keras.Model):
    "Produce N identical layers."
    def __init__(self, main_layer, N):
        super().__init__()
        assert N > 0, "N must be greater than 0"
        self.layer_list = [deepcopy(main_layer) for _ in range(N)]

    def call(self, *args, **kwargs):
        output = self.layer_list[0](*args, **kwargs)
        for layer in self.layer_list[1:]:
            output = layer(output)
        return output

# %% [markdown] id="ip0EXqvEUgQQ"
# The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

# %% [markdown] id="q9dznVhQUgQQ"
# <img src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png" width=400px>

# %% [markdown] id="euyRXbaMUgQR"
# ## Encoder and Decoder Stacks
#
# %% [markdown] id="KWq05WUwHPJi"
#
# ### Encoder
#
# The encoder is composed of a stack of $N=6$ identical layers.
#
# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.

# %% id="dependent-granny"
class Encoder(tf.keras.layers.Layer):
    "Core encoder is a stack of N layers"
    def __init__(self, h, N, d_model, d_ff, dropout):
        super(Encoder, self).__init__()
        self.layers = LayerList(
            EncoderLayer(h, d_model, d_ff, dropout), N
        )
        self.norm = LayerNorm(d_model)

    def call(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        output = self.layers(x, mask)
        # for layer in self.layers:
        #     x = layer(x, mask)
        return self.norm(output)


class EncoderLayer(tf.keras.layers.Layer):
    "Encoder is made up of two sublayers, self-attn and feed forward (defined below)"
    def __init__(self, h, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = LayerList(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# %% [markdown] id="Mie9sUeSUgQV"
# We employ a residual connection [(cite)](he2016deep) around each of the two sub-layers, followed by layer normalization [(cite)](layernorm2016).

# %% id="sEz9kLClUgQV"
class LayerNorm(tf.keras.layers.Layer):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = tf.ones(size)
        self.b_2 = tf.zeros(size)
        self.eps = eps

    def call(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# %% [markdown] id="nx4On5PCUgQY"
# That is, the output of each sub-layer is $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function implemented by the sub-layer itself.  We apply dropout [(cite)](srivastava2014dropout) to the output of each sub-layer, before it is added to the sub-layer input and normalized.
#
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}}=512$.

# %% id="zx9JBwAcUgQY"
class SublayerConnection(tf.keras.layers.Layer):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, sublayer):
        "Apply residual connection to any sublayer function that maintains the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# %% [markdown] id="professional-scott"
# ### Decoder
#
# The decoder is also composed of a stack of $N=6$ identical layers.
#
# In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.  Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.


# %% id="3o_ZB42sUgQd"
class Decoder(tf.keras.layers.Layer):
    "Generic N layer decoder with masking."
    def __init__(self, h, N, d_model, d_ff, dropout):
        super(Decoder, self).__init__()
        self.layers = LayerList(
            DecoderLayer(h, d_model, d_ff, dropout), N
        )
        self.norm = LayerNorm(d_model)

    def call(self, x, memory, src_mask, tgt_mask):
        output = self.layers(x, memory, src_mask, tgt_mask)
        # for layer in self.layers:
        #     x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(output)


class DecoderLayer(tf.keras.layers.Layer):
    "Decoder is made up of three sublayers, self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, h, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.src_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = LayerList(SublayerConnection(d_model, dropout), 3)
        self.size = d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# %% [markdown] id="3Nen9h7wUgQi"
# We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.  This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

# %% id="RyQKI9AgUgQj"
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return tf.convert_to_tensor(subsequent_mask) == 0


# %% [markdown] id="Hr8oFHr8GijK"
# > Below the attention mask shows the position each tgt word (row) is allowed to look at (column). Words are blocked for attending to future words during training.

# %% id="fgQMtvM-UgQl" colab={"base_uri": "https://localhost:8080/", "height": 338} outputId="814260b1-7e9b-47ad-f910-15e505309a18"
# The attention mask shows the position each tgt word (row) is allowed to look at (column).
# Words are blocked for attending to future words during training.
plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])


# %% [markdown] id="fl2E1IaqUgQq"
#

# %% [markdown] id="3twSbimFUgQq"
# ### Attention:
# An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.  The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
#
# We call our particular attention "Scaled Dot-Product Attention".   The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$.  We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.
# <img width="220px" src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_33_0.png">
#
# In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$.   The keys and values are also packed together into matrices $K$ and $V$.  We compute the matrix of outputs as:
#
# $$
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
# $$
#

# %% id="wlZ8zw9PUgQr"
def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = (
        tf.linalg.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    )
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = tf.nn.softmax(scores, dim=-1)
    # (Dropout described below)
    p_attn = tf.nn.dropout(p_attn, rate=dropout)
    return tf.linalg.matmul(p_attn, value), p_attn


# %% [markdown] id="AV7cIqbrUgQs"
# The two most commonly used attention functions are additive attention [(cite)](bahdanau2014neural), and dot-product (multiplicative) attention.  Dot-product attention is identical to our algorithm, **except for the scaling factor of $\frac{1}{\sqrt{d_k}}$**. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.  While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
#
#
# While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$ [(cite)](DBLP:journals/corr/BritzGLL17). **We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients** (To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$.  Then their dot product, $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$, has mean $0$ and variance $d_k$.). To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

# %% [markdown] id="OV7kNMbKUgQt"
#

# %% [markdown] id="uiaCxaGGUgQt"
# ### Multi-Head Attention
#
# Instead of performing a single attention function with $d_{\text{model}}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively.
# On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values:
#
# <img width="270px" src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_38_0.png">
#
# Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.
#
#
#
# $$
# \mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O    \\
#     \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $$
#
# Where the projections are parameter matrices $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$.
#
#
#
# In this work we employ $h=8$ parallel attention layers, or heads. For each of these we use $d_k=d_v=d_{\text{model}}/h=64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

# %% id="_ea0UrEgUgQt"
class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.rate = dropout
        self.linears = LayerList(tf.keras.layers.Dense(d_model), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            layer(x).view(nbatches, -1, self.d_model, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.rate
        )

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.d_model * self.d_k
        )
        return self.linears[-1](x)


# %% [markdown] id="aVdpQ_KwUgQv"
# ### Applications of Attention in our Model
# The Transformer uses multi-head attention in three different ways:
# 1) In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.   This allows every position in the decoder to attend over all positions in the input sequence.  This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [(cite)](wu2016google, bahdanau2014neural,JonasFaceNet2017).
#
#
# 2) The encoder contains self-attention layers.  In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder.   Each position in the encoder can attend to all positions in the previous layer of the encoder.
#
#
# 3) Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.  We need to prevent leftward information flow in the decoder to preserve the auto-regressive property.  **We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections**.

# %% [markdown] id="gERLhK-FUgQw"
# ## Position-wise Feed-Forward Networks
#
# In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.  This consists of two linear transformations with a ReLU activation in between.
#
# $$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$
#
# While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1.  The dimensionality of input and output is $d_{\text{model}}=512$, and the inner-layer has dimensionality $d_{ff}=2048$.

# %% id="HuDPthO2UgQx"
class PositionwiseFeedForward(tf.keras.layers.Layer):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.w_1 = tf.keras.layers.Dense(d_ff)
        self.w_2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(tf.nn.relu(self.w_1(x))))


# %% [markdown] id="68VLwifsUgQz"
# ## Embeddings and Softmax
# Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{\text{model}}$.  We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.  In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [(cite)](press2016using). In the embedding layers, we multiply those weights by $\sqrt{d_{\text{model}}}$.

# %% id="sl5JzPeGUgQz"
class Embeddings(tf.keras.layers.Layer):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = tf.keras.layers.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# %% [markdown] id="F_hw5TyCUgQ1"
# ## Positional Encoding
# Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.  To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.  The positional encodings have the same dimension $d_{\text{model}}$ as the embeddings, so that the two can be summed.   There are many choices of positional encodings, learned and fixed [(cite)](JonasFaceNet2017).
#
# In this work, we use sine and cosine functions of different frequencies:
# $$
#     PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{\text{model}}}) \\
#     PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{\text{model}}})
# $$
# where $pos$ is the position and $i$ is the dimension.  That is, each dimension of the positional encoding corresponds to a sinusoid.  The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.  We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.
#
# In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.  For the base model, we use a rate of $P_{drop}=0.1$.
#
#

# %% id="MVsjhp6uUgQ1"
class PositionalEncoding(tf.keras.layers.Layer):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

        # Compute the positional encodings once in log space.
        pe = tf.Variable(tf.zeros((max_len, d_model)))
        position = tf.expand_dims(tf.range(0, max_len, dtype=np.float32), 1)
        div_term = tf.math.exp(
            tf.range(0, d_model, 2, dtype=np.float32) *
            -(math.log(10000.0) / d_model)
        )
        pe = pe[:, 0::2].assign(tf.math.sin(position * div_term))
        pe = pe[:, 1::2].assign(tf.math.cos(position * div_term))

        self.pe = tf.Variable(tf.expand_dims(pe, 0))

    def call(self, x):
        x = x + tf.Variable(self.pe[:, :x.shape[1]])
        return self.dropout(x)


# %% id="qMsBRCuLUgQ3" colab={"base_uri": "https://localhost:8080/", "height": 321} outputId="0ba52f87-93d6-441b-b6c6-b724981758db"
# The positional encoding will add in a sine wave based on position.
# The frequency and offset of the wave is different for each dimension.
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe(tf.Variable(tf.zeros((1, 100, 20))))
plt.plot(np.arange(100), y[0, :, 4:8].numpy())
plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
None


# %% [markdown] id="6HjtmgH1UgQ5"
# We also experimented with using learned positional embeddings [(cite)](JonasFaceNet2017) instead, and found that the two versions produced nearly identical results.  We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

# %% [markdown] id="ZtnFnHH9UgQ6"
# ## Generation

# %% id="heaKRIaZUgQ6"
class Generator(tf.keras.layers.Layer):
    "Standard generation step. (Not described in the paper.)"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = tf.keras.layers.Dense(vocab)

    def call(self, x):
        return tf.nn.log_softmax(self.proj(x), dim=-1)


# %% id="0i97-Y7AUgQ8"

# %% [markdown] id="dd3lP9fTUgQ9"
# ## Full Model

# %% id="b-LDhRoaUgQ-"
def make_model(
            input_vocab, output_vocab, N=6, d_model=512,
            d_ff=2048, h=8, dropout=0.1
        ):
    "Construct a model object based on hyperparameters."
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(
            EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N
        ),
        Decoder(
            DecoderLayer(
                d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout
            ), N
        ),
        nn.Sequential(Embeddings(d_model, input_vocab), deepcopy(position)),
        nn.Sequential(Embeddings(d_model, output_vocab), deepcopy(position)),
        Generator(d_model, output_vocab))

    # This was important from their code. Initialize parameters with Glorot or fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


# %% id="qP-g4KfhUgQ_" colab={"base_uri": "https://localhost:8080/", "height": 3023} outputId="e5d671a2-a9d4-461b-f08c-bf5b7a4ddb73"
# Small example model.
tmp_model = Transformer(10, 10, 2)
tmp_model


# %% [markdown] id="MuV1e8nEUgRB"
# # Training
#
# This section describes the training regime for our models.
#

# %% id="OhVxtlZaUgRC"

# %% id="naFpt_GUUgRD"

# %% id="X60DPvyaUgRF"

# %% [markdown] id="Lz6n3REAUgRH"
# ## Training Data and Batching
#
# We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.  Sentences were encoded using byte-pair encoding \citep{DBLP:journals/corr/BritzGLL17}, which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [(cite)](wu2016google).
#
#
# Sentence pairs were batched together by approximate sequence length.  Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

# %% id="PRoJURieUgRH"

# %% [markdown] id="52xwbbb4UgRK"
# ## Hardware and Schedule
# We trained our models on one machine with 8 NVIDIA P100 GPUs.  For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds.  We trained the base models for a total of 100,000 steps or 12 hours.  For our big models, step time was 1.0 seconds.  The big models were trained for 300,000 steps (3.5 days).

# %% [markdown] id="MPp__T_uUgRK"
# ## Optimizer
#
# We used the Adam optimizer [(cite)](kingma2014adam) with $\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$.  We varied the learning rate over the course of training, according to the formula:
# $$
# lrate = d_{\text{model}}^{-0.5} \cdot
#   \min({step\_num}^{-0.5},
#     {step\_num} \cdot {warmup\_steps}^{-1.5})
# $$
# This corresponds to increasing the learning rate linearly for the first $warmup\_steps$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.  We used $warmup\_steps=4000$.

# %% id="Q5yV0f2QUgRL"
# Note: This part is incredibly important.
# Need to train with this setup of the model is very unstable.
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup**(-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# %% [markdown] id="FM5n1PJxUgRN"
#

# %% id="PC2wCVEFUgRO" colab={"base_uri": "https://localhost:8080/", "height": 266} outputId="b3fc5e15-3be4-4e1b-fb3b-fd722207a9a9"
# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
None


# %% [markdown] id="8tkzxQYKUgRQ"
# ## Regularization
#
# ### Label Smoothing
#
# During training, we employed label smoothing of value $\epsilon_{ls}=0.1$ [(cite)](DBLP:journals/corr/SzegedyVISW15).  This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

# %% id="IVWDJLXrUgRQ"
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# %% id="_gXwt5HVUgRT" colab={"base_uri": "https://localhost:8080/", "height": 254} outputId="3ea1a660-0bc7-4b1e-ba8e-7d3fbf560640"
#Example
crit = LabelSmoothing(5, 0, 0.5)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()),
         Variable(torch.LongTensor([2, 1, 0])))

# Show the target distributions expected by the system.
plt.imshow(crit.true_dist)
None

# %% id="sHBHTwmjUgRU" colab={"base_uri": "https://localhost:8080/", "height": 288} outputId="4086db46-2dd9-4566-804e-8e5f80641a69"
# Label smoothing starts to penalize the model
# if it gets very confident about a given choice
crit = LabelSmoothing(5, 0, 0.2)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                 ])
    #print(predict)
    return crit(Variable(predict.log()),
                 Variable(torch.LongTensor([1]))).data[0]
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])


# %% [markdown] id="XoyfFgLoUgRW"
# ### Memory Optimization

# %% id="yVKyONFsUgRW"
def loss_backprop(generator, criterion, out, targets, normalize):
    """
    Memory optmization. Compute each timestep separately and sum grads.
    """
    assert out.size(1) == targets.size(1)
    total = 0.0
    out_grad = []
    for i in range(out.size(1)):
        out_column = Variable(out[:, i].data, requires_grad=True)
        gen = generator(out_column)
        loss = criterion(gen, targets[:, i]) / normalize
        total += loss.data[0]
        loss.backward()
        out_grad.append(out_column.grad.data.clone())
    out_grad = torch.stack(out_grad, dim=1)
    out.backward(gradient=out_grad)
    return total


# %% id="LiTlbMq2UgRY"
def make_std_mask(src, tgt, pad):
    src_mask = (src != pad).unsqueeze(-2)
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return src_mask, tgt_mask


# %% id="sSF9AaKJUgRZ"
def train_epoch(train_iter, model, criterion, opt, transpose=False):
    model.train()
    for i, batch in enumerate(train_iter):
        src, trg, src_mask, trg_mask = \
            batch.src, batch.trg, batch.src_mask, batch.trg_mask
        out = model.forward(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
        loss = loss_backprop(model.generator, criterion, out, trg[:, 1:], batch.ntokens)

        model_opt.step()
        model_opt.optimizer.zero_grad()
        if i % 10 == 1:
            print(i, loss, model_opt._rate)


# %% id="q4kFYs7nUgRb"
def valid_epoch(valid_iter, model, criterion, transpose=False):
    model.test()
    total = 0
    for batch in valid_iter:
        src, trg, src_mask, trg_mask = \
            batch.src, batch.trg, batch.src_mask, batch.trg_mask
        out = model.forward(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
        loss = loss_backprop(model.generator, criterion, out, trg[:, 1:], batch.ntokens)



# %% id="uJo5AZasUgRd"
class Batch:
    def __init__(self, src, trg, src_mask, trg_mask, ntokens):
        self.src = src
        self.trg = trg
        self.src_mask = src_mask
        self.trg_mask = trg_mask
        self.ntokens = ntokens

def data_gen(V, batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        src_mask, tgt_mask = make_std_mask(src, tgt, 0)
        yield Batch(src, tgt, src_mask, tgt_mask, (tgt[1:] != 0).data.sum())


# %% id="vrpU5b2sUgRe" colab={"base_uri": "https://localhost:8080/", "height": 604} outputId="00871f60-2fcb-4235-c0ab-8de02a0c069c"
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = get_std_opt(model)
for epoch in range(2):
    train_epoch(data_gen(V, 30, 20), model, criterion, model_opt)

# %% [markdown] id="5S2BcIoOUgRg"
# # A Real World Example

# %% id="aP_oq0kLUgRh"
# For data loading.
from torchtext import data, datasets

# %% id="eKynVliVXg6F" colab={"base_uri": "https://localhost:8080/", "height": 989} outputId="f471f71a-d329-4fa9-d188-b4dcc9283234"
# !pip install torchtext spacy
# !python -m spacy download en
# !python -m spacy download de

# %% id="lXtYwdHqUgRj" colab={"base_uri": "https://localhost:8080/", "height": 395} outputId="31ee7819-a5eb-4cf2-cb06-cc7932bf535a"
# Load words from IWSLT

# #!pip install torchtext spacy
# #!python -m spacy download en
# #!python -m spacy download de

import spacy
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
                 eos_token = EOS_WORD, pad_token=BLANK_WORD)

MAX_LEN = 100
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                         len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 1
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

# %% id="F8MTIJTWUgRl" colab={"base_uri": "https://localhost:8080/", "height": 244} outputId="bd6c163a-14a2-4f8c-b0d5-01a033f278c4"
# Detail. Batching seems to matter quite a bit.
# This is temporary code for dynamic batching based on number of tokens.
# This code should all go away once things get merged in this library.

BATCH_SIZE = 4096
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    src_mask, trg_mask = make_std_mask(src, trg, pad_idx)
    return Batch(src, trg, src_mask, trg_mask, (trg[1:] != pad_idx).data.sum())

train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)

# %% id="wamR3SPdUgRo" colab={"base_uri": "https://localhost:8080/", "height": 1900} outputId="b3b92f29-2560-4dbb-c2b0-f2a0b71db37c"
# Create the model an load it onto our GPU.
pad_idx = TGT.vocab.stoi["<blank>"]
model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
model_opt = get_std_opt(model)
model.cuda()

# %% id="SStCCZoiUgRp" outputId="a551e444-2fe5-4cd5-f46a-3de3505a2545"

criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.cuda()
for epoch in range(15):
    train_epoch((rebatch(pad_idx, b) for b in train_iter), model, criterion, model_opt)
    valid_epoch((rebatch(pad_idx, b) for b in valid_iter), model, criterion)

# %% [markdown] id="1NO9lsw2UgRt"
#
# OTHER

# %% id="B8BVm-hEUgRw"
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field()
TGT = data.Field(init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD) # only target needs BOS/EOS

MAX_LEN = 100
train = datasets.TranslationDataset(path="/n/home00/srush/Data/baseline-1M_train.tok.shuf",
                                    exts=('.en', '.fr'),
                                    fields=(SRC, TGT),
                                    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                         len(vars(x)['trg']) <= MAX_LEN)
SRC.build_vocab(train.src, max_size=50000)
TGT.build_vocab(train.trg, max_size=50000)

# %% id="eFdZyOIzUgRx" outputId="bc543dba-c343-47bc-e81e-f74a25688668"
pad_idx = TGT.vocab.stoi["<blank>"]
print(pad_idx)
model = make_model(len(SRC.vocab), len(TGT.vocab), pad_idx, N=6)
model_opt = get_opt(model)
model.cuda()

# %% id="cinuTkbtUgRz"
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, label_smoothing=0.1)
criterion.cuda()
for epoch in range(15):
    train_epoch(train_iter, model, criterion, model_opt)
    valid_epoch()

# %% id="PsoeJn4bUgR1"

# %% id="4LadFBIEUgR3" outputId="ba9a812f-b5f6-4972-af91-215d91a223d1"
print(pad_idx)
print(len(SRC.vocab))

# %% id="kP_Au0bHUgR7" outputId="d5ad5d88-d512-4947-b5b8-f18eaba4f9ff"
torch.save(model, "/n/rush_lab/trans_ipython.pt")

# %% id="nqKKIhoOUgR-" outputId="2c087978-9803-4639-886b-88df91e62096"
#weight = torch.ones(len(TGT.vocab))
#weight[pad_idx] = 0
#criterion = nn.NLLLoss(size_average=False, weight=weight.cuda())
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, label_smoothing=0.1)
criterion.cuda()
for epoch in range(15):
    train_epoch(train_iter, model, criterion, model_opt)

# %% id="35JC6i9QUgSB"
1 10.825187489390373 6.987712429686844e-07
101 9.447168171405792 3.56373333914029e-05
201 7.142856806516647 7.057589553983712e-05
301 6.237934365868568 0.00010551445768827134
401 5.762486848048866 0.00014045301983670557
501 5.415792358107865 0.00017539158198513977
601 5.081815680023283 0.000210330144133574
701 4.788327748770826 0.00024526870628200823
801 4.381739928154275 0.0002802072684304424
901 4.55433791608084 0.00031514583057887664
1001 4.911875109748507 0.0003500843927273108
1101 4.0579032292589545 0.0003850229548757451
1201 4.2276234351193125 0.0004199615170241793
1301 3.932735869428143 0.00045490007917261356
1401 3.8179439397063106 0.0004898386413210477
1501 3.3608515430241823 0.000524777203469482
1601 3.832796103321016 0.0005597157656179162
1701 2.907085266895592 0.0005946543277663504
1801 3.5280659823838505 0.0006295928899147847
1901 2.895841649500653 0.0006645314520632189
2001 3.273784235585481 0.000699470014211653
2101 3.181488689899197 0.0007344085763600873
2201 3.4151616653980454 0.0007693471385085215
2301 3.4343731447588652 0.0008042857006569557
2401 3.0505455391539726 0.0008392242628053899
2501 2.8089329147478566 0.0008741628249538242
2601 2.7827929875456903 0.0009091013871022583
2701 2.4428516102489084 0.0009440399492506926
2801 2.4015486147254705 0.0009789785113991267
2901 2.3568112018401735 0.001013917073547561
3001 2.6349758653668687 0.0010488556356959952
3101 2.5981983028614195 0.0010837941978444295
3201 2.666826274838968 0.0011187327599928637
3301 3.0092043554177508 0.0011536713221412978
3401 2.4580375660589198 0.0011886098842897321
3501 2.586465588421561 0.0012235484464381662
3601 2.5663993963389657 0.0012584870085866006
3701 2.9430236657499336 0.0012934255707350347
3801 2.464644919440616 0.001328364132883469
3901 2.7124062888276512 0.0013633026950319032
4001 2.646443709731102 0.0013971932312809247
4101 2.7294750874862075 0.001380057517579748
4201 2.1295202329056337 0.0013635372009002666
4301 2.596563663915731 0.001347596306985731
4401 2.1265982036820787 0.0013322017384983986
4501 2.3880532500334084 0.0013173229858148
4601 2.6129120760888327 0.0013029318725783852
4701 2.2873719420749694 0.001289002331178292
4801 2.4949760700110346 0.0012755102040816328
4901 2.496607314562425 0.001262433067573089
5001 2.1889712483389303 0.0012497500749750088
5101 1.8677761815488338 0.0012374418168536253
5201 2.2992054556962103 0.0012254901960784316
5301 2.664361578106707 0.0012138783159049418
5401 2.705850490485318 0.0012025903795063202
5501 2.581445264921058 0.0011916115995949978
5601 2.2480602325085783 0.0011809281169581616
5701 1.9289666265249252 0.0011705269268863989
5801 2.4863578918157145 0.0011603958126073107
5901 2.632946971571073 0.0011505232849492607
6001 2.496141305891797 0.0011408985275576757
6101 2.6422974687084206 0.0011315113470699342
6201 2.448802186456305 0.0011223521277270118
