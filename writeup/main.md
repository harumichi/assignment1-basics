# Assignment 1

## Byte-Pair Encoding (BPE) Tokenizer

Machine specs are
- Model: Intel(R) Xeon(R) Platinum 8259CL @ 2.50GHz
- Cores: 8 (16 with Hyper-Threading)
- RAM: 64 GB
-
### Problem (train_bpe_tinystories): BPE Training on TinyStories

Profiling
- Reading the dataset and splitting into documents: ~4.5 seconds
- Pre-tokenizing the input into ~60k unique words: ~3 minutes 55 seconds
- Running the BPE merge operations: ~1 minute 46 seconds

Longest token is ` accomplishment`.

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText

Profiling
- Reading the dataset and preparing initial tokens: ~11 minutes
- Pre-tokenizing the input into ~6.6M unique words: completed by ~11 minutes
- Running the BPE merge operations and vocabulary growth: ~8 hours 35 minutes

Longest token written in Unicode is `ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ`.
This is expected because the dataset contains garbaged characters.

### Problem (tokenizer_experiments): Experiments with tokenizers

Vocab size:
- TinyStories: 10000
- OpenWebText: 32000

Compression ratios:
- TinyStories: 4.12 bytes/token
- OpenWebText: 4.37 bytes/token
- OpenWebText (using tokenizer trained by TinyStories): 3.17 bytes/token

Throughput:
- TinyStories: 3.57 MB/sec
- OpenWebText: 2.13 MB/sec

The reason that token dtype is uint16 is that it can represent up to 65536 unique tokens, which is sufficient for the vocab sizes used here.

## Transformer Language Model Architecture

### Problem (transformer_accounting): Transformer LM resource accounting

**Resorce accounting**

For GPT-2 XL:
- Total number of parameters: 2,127,057,600
- Parameter bytes (float32): 8114 MB

| model_name                        | params                                       | TFLOPs_attn | TFLOPs_ff | TFLOPs_output | ratio_attn | ratio_ff | ratio_output |
|-----------------------------------|----------------------------------------------|-------------|-----------|----------------|------------|----------|--------------|
| GPT-2 small                       | d_model: 768,  num_layers: 12, num_heads: 12 | 0.097       | 0.362     | 0.079          | 17.9%      | 67.3%    | 14.7%        |
| GPT-2 medium                      | d_model: 1024, num_layers: 24, num_heads: 16 | 0.309       | 0.966     | 0.105          | 22.0%      | 68.8%    | 7.5%         |
| GPT-2 large                       | d_model: 1280, num_layers: 36, num_heads: 20 | 0.676       | 1.812     | 0.132          | 25.8%      | 69.1%    | 5.0%         |
| GPT-2 XL                          | d_model: 1600, num_layers: 48, num_heads: 25 | 1.328       | 3.020     | 0.165          | 28.4%      | 64.7%    | 3.5%         |
| GPT-2 XL (context_length = 16384) | d_model: 1600, num_layers: 48, num_heads: 25 | 98.569      | 48.318    | 2.635          | 65.5%      | 32.1%    | 1.7%         |

Common parameters:
- vocab_size: 50257
- context_length: 1024
- d_ff: 6400

# Problem (adamwAccounting): Resource accounting for training with AdamW

Peak memory required by running AdamW optimizer
- parameters: 7.92 GB
- gradients: 7.92 GB
- optimizer state: 15.85 GB
- activations: 15.62 * (batch_size) GB

Max batch size fitting in 80 GB is 3.

FLOPs for one step of AdamW : 13 * #(parameters)
```
m = beta1 * m + (1 - beta1) * g  => 3
v = beta2 * v + (1 - beta2) * g^2  => 4
p = p - lr * m / (sqrt(v) + eps)  => 5
p = p * (1 - lr * weight_decay)  => 1
```

Training time estimate: 13.5 days
with conditions
- MFU: 0.5
- peak flops for A100: 19.5 TFLOP/s
- batch size: 1024
- steps: 400_000
