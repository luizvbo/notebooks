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

# %% [markdown] id="sharing-musician"
# # Attention is all you need
#

# %% [markdown] id="selective-lighter"
# ## Note from the author
#
# This notebook is a "translation" from the post ["The Annotated Transformer"](https://nlp.seas.harvard.edu/2018/04/03/attention.html) to TensorFlow.
#
# On top of that, I added some notes that helped me to understand the original paper.
#
# The main text is from the original paper. My notes are in <font color='red'>red</font> and the comments from the original author are in <font color='blue'>blue</font>.
#
# ## Note from the original post author
#
# When teaching, I emphasize implementation as a way to understand recent developments in ML. This post is an attempt to keep myself honest along this goal. The recent ["Attention is All You Need"]
# (https://arxiv.org/abs/1706.03762) paper from NIPS 2017 has been instantly impactful paper as a new method for machine translation and potentiall NLP generally. The paper is very clearly written, but the conventional wisdom has been that it is quite difficult to implement correctly.
#
# In this post I follow the paper through from start to finish and try to implement each component in code.
# (I have done some minor reordering and skipping from the original paper). This document itself is a working notebook, and should be a completely usable and efficient implementation.
#
# > Alexander "Sasha" Rush ([@harvardnlp](https://twitter.com/harvardnlp))
#

# %% id="smaller-central" colab={"base_uri": "https://localhost:8080/"} outputId="65de8c8c-0c2a-4366-f130-d931977ec7f7"
# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext

# %% id="single-heaven"
# Standard PyTorch imports
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import tensorflow as tf
import math
from copy import deepcopy
import time

# For plots
# %matplotlib inline
import matplotlib.pyplot as plt


# %% [markdown] id="pending-postcard"
# # Background

# %% [markdown] id="coupled-development"
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

# %% [markdown] id="muslim-pitch"
# # Model Architecture

# %% [markdown] id="ignored-container"
# Most competitive neural sequence transduction models have an encoder-decoder structure [(cite)](cho2014learning,bahdanau2014neural,sutskever14). Here, the encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous representations $\mathbf{z} = (z_1, ..., z_n)$. Given $\mathbf{z}$, the decoder then generates an output sequence $(y_1,...,y_m)$ of symbols one element at a time. At each step the model is auto-regressive [(cite)](graves2013generating), consuming the previously generated symbols as additional input when generating the next.

# %% id="outdoor-story"
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
        self.input_embed = tf.keras.Sequential((
            Embeddings(d_model, input_vocab),
            PositionalEncoding(d_model, dropout)
        ))
        self.encoder = Encoder(h, N, d_model, d_ff, dropout)
        self.decoder = Decoder(h, N, d_model, d_ff, dropout)
        self.output_embed = tf.keras.Sequential((
            Embeddings(d_model, output_vocab),
            PositionalEncoding(d_model, dropout)
        ))
        self.generator = Generator(d_model, output_vocab)

    def call(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encoder(self.input_embed(src), src_mask)
        output = self.decoder(
            self.output_embed(tgt), memory, src_mask, tgt_mask
        )
        return output


# %% id="electoral-malawi"
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

# %% [markdown] id="lonely-durham"
# The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

# %% [markdown] id="intense-crossing"
# <img src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png" width=400px>

# %% [markdown] id="southern-evening"
# ## Encoder and Decoder Stacks
#
# %% [markdown] id="delayed-louisville"
#
# ### Encoder
#
# The encoder is composed of a stack of $N=6$ identical layers.
#
# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
#

# %% id="interior-backing"
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
        return self.norm(output)


class EncoderLayer(tf.keras.layers.Layer):
    "Encoder is made up of two sublayers, self-attn and feed forward (defined below)"
    def __init__(self, h, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = LayerList(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def call(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer.layer_list[0](
            x, lambda x: self.self_attn(x, x, x, mask)
        )
        return self.sublayer.layer_list[1](x, self.feed_forward)


# %% [markdown] id="passive-easter"
# We employ a residual connection [(cite)](he2016deep) around each of the two sub-layers, followed by layer normalization [(cite)](layernorm2016).

# %% id="egyptian-perth"
class LayerNorm(tf.keras.layers.Layer):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = tf.ones(size)
        self.b_2 = tf.zeros(size)
        self.eps = eps

    def call(self, x):
        mean = tf.math.reduce_mean(x, -1, keepdims=True)
        # In order to get an unbiased std, we need to multiply by the
        # Bessel's correction factor
        br = tf.sqrt(x.shape[-1]/(x.shape[-1] - 1))
        std = br * tf.math.reduce_std(x, -1, keepdims=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# %% [markdown] id="educational-emergency"
# That is, the output of each sub-layer is $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function implemented by the sub-layer itself.  We apply dropout [(cite)](srivastava2014dropout) to the output of each sub-layer, before it is added to the sub-layer input and normalized.
#
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}}=512$.

# %% id="molecular-summer"
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


# %% [markdown] id="olympic-plenty"
# ### Decoder
#
# The decoder is also composed of a stack of $N=6$ identical layers.
#
# In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.  Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.


# %% id="instrumental-begin"
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

    def call(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# %% [markdown] id="bronze-transcript"
# We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.  This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

# %% id="devoted-steel"
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return tf.convert_to_tensor(subsequent_mask) == 0


# %% [markdown] id="solar-annual"
# <font color='blue'>
#     Below the attention mask shows the position each tgt word (row) is allowed to look at (column). Words are blocked for attending to future words during training.
# </font>

# %% colab={"base_uri": "https://localhost:8080/", "height": 337} id="lesbian-cosmetic" outputId="617be236-ec5e-4d97-fa8c-bd5ea99c50d8"
# The attention mask shows the position each tgt word (row) is allowed to look at (column).
# Words are blocked for attending to future words during training.
plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])


# %% [markdown] id="elementary-prague"
# ### Attention:
# An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.  The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
#
# We call our particular attention "Scaled Dot-Product Attention".   The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$.  We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.
#
# <center>
#     <img width="220px" src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_33_0.png">
# </center>
#
# In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$.   The keys and values are also packed together into matrices $K$ and $V$.  We compute the matrix of outputs as:
#
# $$
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
# $$
#
#
# <font color='red'>
#     The attention computation is depicted in the picture bellow. The attention weight is computed from the query ($q$) and key ($k_i$). The weight is used to compute the resulting value as a dot product between the attention weigths and the values.
# </font>
#
# <center>
#     <img src="https://github.com/luizvbo/notebooks/raw/master/img/attention-computation.jpg" width=800px>
# </center>

# %% id="sophisticated-brisbane"
def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.shape[-1]
    idx_t = list(range(len(key.shape)))
    idx_t[-2], idx_t[-1] = idx_t[-1], idx_t[-2]
    scores = (
        tf.linalg.matmul(query, tf.transpose(key, idx_t)) / math.sqrt(d_k)
    )
    if mask is not None:
        scores = tf.where(tf.cast(mask, tf.int32) == 0, -1e9, scores)
    p_attn = tf.nn.softmax(scores, axis=-1)
    # (Dropout described below)
    p_attn = tf.nn.dropout(p_attn, rate=dropout)
    return tf.linalg.matmul(p_attn, value), p_attn


# %% [markdown] id="norwegian-forestry"
# The two most commonly used attention functions are additive attention [(cite)](bahdanau2014neural), and dot-product (multiplicative) attention.  Dot-product attention is identical to our algorithm, **except for the scaling factor of $\frac{1}{\sqrt{d_k}}$**. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.  While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
#
#
# While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$ [(cite)](DBLP:journals/corr/BritzGLL17). **We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients** (To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$.  Then their dot product, $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$, has mean $0$ and variance $d_k$.). To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

# %% [markdown] id="comic-combat"
#

# %% [markdown] id="seventh-vault"
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
#
#
#

# %% id="devoted-shift"
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

    def call(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = tf.expand_dims(mask, 1)
        nbatches = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            tf.transpose(
                tf.reshape(layer(x), (nbatches, -1, self.h, self.d_k)),
                (0, 2, 1, 3)
            )
            for layer, x in zip(self.linears.layer_list, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.rate
        )

        # 3) "Concat" using a view and apply a final linear.
        idx_t = list(range(len(x.shape)))
        idx_t[1], idx_t[2] = (2, 1)
        x = tf.reshape(
            tf.transpose(x, idx_t),
            (nbatches, -1, self.h * self.d_k)
        )
        return self.linears.layer_list[-1](x)


# %% [markdown] id="neutral-manhattan"
# ### Applications of Attention in our Model
# The Transformer uses multi-head attention in three different ways:
# 1) In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.   This allows every position in the decoder to attend over all positions in the input sequence.  This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [(cite)](wu2016google, bahdanau2014neural,JonasFaceNet2017).
#
#
# 2) The encoder contains self-attention layers.  In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder.   Each position in the encoder can attend to all positions in the previous layer of the encoder.
#
#
# 3) Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.  We need to prevent leftward information flow in the decoder to preserve the auto-regressive property.  **We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections**.

# %% [markdown] id="rational-physics"
# ## Position-wise Feed-Forward Networks
#
# In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.  This consists of two linear transformations with a ReLU activation in between.
#
# $$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$
#
# While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1.  The dimensionality of input and output is $d_{\text{model}}=512$, and the inner-layer has dimensionality $d_{ff}=2048$.

# %% id="dying-duration"
class PositionwiseFeedForward(tf.keras.layers.Layer):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.w_1 = tf.keras.layers.Dense(d_ff)
        self.w_2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        return self.w_2(self.dropout(tf.nn.relu(self.w_1(x))))


# %% [markdown] id="interested-software"
# ## Embeddings and Softmax
# Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{\text{model}}$.  We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.  In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [(cite)](press2016using). In the embedding layers, we multiply those weights by $\sqrt{d_{\text{model}}}$.

# %% id="perceived-aquarium"
class Embeddings(tf.keras.layers.Layer):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = tf.keras.layers.Embedding(vocab, d_model)
        self.d_model = d_model

    def call(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# %% [markdown] id="tracked-england"
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

# %% id="fantastic-essence"
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


# %% colab={"base_uri": "https://localhost:8080/", "height": 320} id="pretty-escape" outputId="2baa8afe-d26c-41bd-dfba-c6a5577176d3"
# The positional encoding will add in a sine wave based on position.
# The frequency and offset of the wave is different for each dimension.
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe(tf.Variable(tf.zeros((1, 100, 20))))
plt.plot(np.arange(100), y[0, :, 4:8].numpy())
plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
None


# %% [markdown] id="complimentary-cooler"
# We also experimented with using learned positional embeddings [(cite)](JonasFaceNet2017) instead, and found that the two versions produced nearly identical results.  We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

# %% [markdown] id="controlling-motorcycle"
# ## Generation

# %% id="legal-airport"
class Generator(tf.keras.layers.Layer):
    "Standard generation step. (Not described in the paper.)"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = tf.keras.layers.Dense(vocab)

    def call(self, x):
        return tf.nn.log_softmax(self.proj(x), dim=-1)


# %% [markdown] id="circular-retro"
# ## Full Model

# %% id="electrical-consortium"
# Small example model.
tmp_model = Transformer(10, 10, 2)


# %% [markdown] id="rising-extension"
# # Training
#
# This section describes the training regime for our models.
#
#
# <font color='blue'>
#     We stop for a quick interlude to introduce some of the tools needed to train a standard encoder decoder model. First we define a batch object that holds the src and target sentences for training, as well as constructing the masks.
# </font>

# %% [markdown] id="acceptable-saskatchewan"
# ## Batches and Masking

# %% id="convinced-providence"
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = tf.expand_dims((src != pad), -2)

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = tf.reduce_sum(
                tf.cast((self.trg_y != pad), tf.float32)
            )

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = tf.expand_dims((tgt != pad), -2)
        tgt_mask = tgt_mask & tf.Variable(
            tf.cast(subsequent_mask(tgt.shape[-1]), tgt_mask.dtype)
        )
        return tgt_mask


# %% [markdown] id="touched-survey"
# <font color='blue'>
#     Next we create a generic training and scoring function to keep track of loss. We pass in a generic loss compute function that also handles parameter updates.
# </font>

# %% [markdown] id="XBLLvishLGdM"
# ## Training Loop

# %% id="9xMJBQ1iLMHD"
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.call(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# %% [markdown] id="vanilla-sheep"
# ## Training Data and Batching
#
# We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.  Sentences were encoded using byte-pair encoding \citep{DBLP:journals/corr/BritzGLL17}, which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [(cite)](wu2016google).
#
#
# Sentence pairs were batched together by approximate sequence length.  Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

# %% id="third-chile"
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


# %% [markdown] id="accessible-floating"
# ## Hardware and Schedule
# We trained our models on one machine with 8 NVIDIA P100 GPUs.  For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds.  We trained the base models for a total of 100,000 steps or 12 hours.  For our big models, step time was 1.0 seconds.  The big models were trained for 300,000 steps (3.5 days).

# %% [markdown] id="executed-senegal"
# ## Optimizer
#
# We used the Adam optimizer [(cite)](kingma2014adam) with $\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$.  We varied the learning rate over the course of training, according to the formula:
# $$
# lrate = d_{\text{model}}^{-0.5} \cdot
#   \min({step\_num}^{-0.5},
#     {step\_num} \cdot {warmup\_steps}^{-1.5})
# $$
# This corresponds to increasing the learning rate linearly for the first $warmup\_steps$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.  We used $warmup\_steps=4000$.

# %% id="tender-melbourne"
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
        self.optimizer.learning_rate = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (
            self.factor * (
                self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup**(-1.5))
            )
        )

def get_std_opt(model):
    return NoamOpt(
        model.input_embed.layers[0].d_model, 2, 4000,
        tf.keras.optimizers.Adam(
            lr=0, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )
    )


# %% [markdown] id="recreational-advantage"
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="differential-delhi" outputId="7a8e524c-fc5c-4ac9-c687-cd04b9ba5b43"
# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
None


# %% [markdown] id="TyFsf39LmWrq"
# ## Regularization

# %% [markdown] id="conscious-botswana"
#
#
# ### Label Smoothing
#
# During training, we employed label smoothing of value $\epsilon_{ls}=0.1$ [(cite)](DBLP:journals/corr/SzegedyVISW15).  This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

# %% id="nutritional-movie"
class LabelSmoothing(tf.keras.layers.Layer):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = tf.keras.losses.KLDivergence()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def call(self, x, target):
        assert x.shape[1] == self.size

        true_dist = tf.fill(x.shape, self.smoothing / (self.size - 2))
        indices = tf.concat(
            [
                tf.expand_dims(tf.range(0, true_dist.shape[0]), 1),
                tf.expand_dims(target, 1)
            ], axis=1
        )
        true_dist = tf.Variable(
            tf.tensor_scatter_nd_update(
                true_dist, indices, [self.confidence] * indices.shape[0]
            )
        )
        true_dist = true_dist[:, self.padding_idx].assign(
            tf.fill(true_dist[:, self.padding_idx].shape, 0.)
        )
        mask = tf.where(target == self.padding_idx)
        if len(mask.shape) > 0 and not tf.equal(tf.size(mask), 0):
            true_dist = true_dist[tf.squeeze(mask), :].assign(
                tf.fill(true_dist.shape[1], 0.)
            )
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


# %% [markdown] id="QfT3NbP-jEOb"
# <font color='blue'>
#     Here we can see an example of how the mass is distributed to the words based on confidence.
# </font>

# %% colab={"base_uri": "https://localhost:8080/", "height": 253} id="meaningful-dodge" outputId="c6f48cdd-b4d9-46ad-c4b4-dc9e666b5328"
#Example
crit = LabelSmoothing(5, 0, 0.4)
predict = tf.convert_to_tensor(
    [[0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0]]
)
v = crit(
    tf.Variable(predict),
    tf.Variable([2, 1, 0])
)

# Show the target distributions expected by the system.
plt.imshow(crit.true_dist)
None

# %% [markdown] id="l_nzzjAOVdRK"
# <font color='blue'>
#     Label smoothing starts to penalize the model if it gets very confident about a given choice
# </font>

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="manufactured-ministry" outputId="8d2bd248-39d4-4d44-ce42-44d164922232"
crit = LabelSmoothing(5, 0, 0.2)
def loss(x):
    d = x + 3 * 1
    predict = tf.Variable(
        [[0, x / d, 1 / d, 1 / d, 1 / d],]
    )
    a, b = (
        tf.Variable(predict),
        tf.Variable([1])
    )
    return crit(a,b)
    # return crit(
    #     tf.Variable(tf.math.log(predict)),
    #     tf.Variable([1])
    # )
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])


# %% [markdown] id="jlkHgcPpmNMn"
# # A First Example

# %% [markdown] id="XVV34WrHmcbJ"
# <font color='blue'>
#     We can begin by trying out a simple copy-task. Given a random set of input symbols from a small vocabulary, the goal is to generate back those same symbols.
# </font>
#

# %% [markdown] id="xUNowmcImmWm"
# ## Synthetic Data

# %% id="Rt-6IB2cmofa"
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = np.random.randint(1, V, size=(batch, 10))
        data[:, 0] = 1
        data = tf.convert_to_tensor(data)
        src = tf.Variable(data)
        tgt = tf.Variable(data)
        yield Batch(src, tgt, 0)


# %% [markdown] id="2e_d9QRKmqR0"
# ## Loss Computation

# %% id="db8l26a3mtCv"
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm


# %% [markdown] id="yLC_yVqCmv88"
# ## Greedy Decoding

# %% colab={"base_uri": "https://localhost:8080/", "height": 749} id="ky_TNTuYmwYq" outputId="fcbb036b-f8d9-4a0a-d7d0-9c28ea73f011"
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = Transformer(V, V, N=2)

model_opt = NoamOpt(
    model.input_embed.layers[0].d_model, 1, 400,
    tf.keras.optimizers.Adam(lr=0, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
)

for epoch in range(10):
    run_epoch(data_gen(V, 30, 20), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))


# %% [markdown] id="7W-DUarFnAz5"
# <font color='blue'>
#     This code predicts a translation using greedy decoding for simplicity.
# </font>
#

# %% id="iH8xFRAbm74h"
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


# %% [markdown] id="twelve-button"
# ### Memory Optimization

# %% id="beautiful-quebec"
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


# %% id="blank-handling"
def make_std_mask(src, tgt, pad):
    src_mask = (src != pad).unsqueeze(-2)
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return src_mask, tgt_mask


# %% id="natural-registration"
def train_epoch(train_iter, model, criterion, opt, transpose=False):
    model.train()
    for i, batch in enumerate(train_iter):
        src, trg, src_mask, trg_mask = \
            batch.src, batch.trg, batch.src_mask, batch.trg_mask
        out = model.call(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
        loss = loss_backprop(model.generator, criterion, out, trg[:, 1:], batch.ntokens)

        model_opt.step()
        model_opt.optimizer.zero_grad()
        if i % 10 == 1:
            print(i, loss, model_opt._rate)


# %% id="meaning-license"
def valid_epoch(valid_iter, model, criterion, transpose=False):
    model.test()
    total = 0
    for batch in valid_iter:
        src, trg, src_mask, trg_mask = \
            batch.src, batch.trg, batch.src_mask, batch.trg_mask
        out = model.call(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
        loss = loss_backprop(model.generator, criterion, out, trg[:, 1:], batch.ntokens)



# %% id="grateful-third"

# %% id="chinese-springfield"
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = get_std_opt(model)
for epoch in range(2):
    train_epoch(data_gen(V, 30, 20), model, criterion, model_opt)

# %% [markdown] id="ceramic-provider"
# # A Real World Example

# %% id="prospective-natural"
# For data loading.
from torchtext import data, datasets

# %% id="talented-lunch"
# !pip install torchtext spacy
# !python -m spacy download en
# !python -m spacy download de

# %% id="gross-monroe"
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

# %% id="hindu-junior"
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

# %% id="objective-bronze"
# Create the model an load it onto our GPU.
pad_idx = TGT.vocab.stoi["<blank>"]
model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
model_opt = get_std_opt(model)
model.cuda()

# %% id="retired-microwave"

criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.cuda()
for epoch in range(15):
    train_epoch((rebatch(pad_idx, b) for b in train_iter), model, criterion, model_opt)
    valid_epoch((rebatch(pad_idx, b) for b in valid_iter), model, criterion)

# %% [markdown] id="attached-assurance"
#
# OTHER

# %% id="temporal-assistant"
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

# %% id="departmental-removal"
pad_idx = TGT.vocab.stoi["<blank>"]
print(pad_idx)
model = make_model(len(SRC.vocab), len(TGT.vocab), pad_idx, N=6)
model_opt = get_opt(model)
model.cuda()

# %% id="closing-devices"
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, label_smoothing=0.1)
criterion.cuda()
for epoch in range(15):
    train_epoch(train_iter, model, criterion, model_opt)
    valid_epoch()

# %% id="governing-click"

# %% id="roman-force"
print(pad_idx)
print(len(SRC.vocab))

# %% id="front-ceramic"
torch.save(model, "/n/rush_lab/trans_ipython.pt")

# %% id="selected-shaft"
#weight = torch.ones(len(TGT.vocab))
#weight[pad_idx] = 0
#criterion = nn.NLLLoss(size_average=False, weight=weight.cuda())
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, label_smoothing=0.1)
criterion.cuda()
for epoch in range(15):
    train_epoch(train_iter, model, criterion, model_opt)
