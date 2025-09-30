from cs336_basics.nn import Transformer


def element_sizes(batch_size, vocab_size, context_length, num_layers, d_model, num_heads):
    d_ff = 4 * d_model
    d_k = d_model // num_heads

    parameters = 0
    parameters += vocab_size * d_model  # input embedding
    parameters += num_layers * (  # transformer layers
        (d_model * d_model * 4)  # Q,K,V,O
        + (d_model * d_ff * 3)  # Feedforward, SwiGLU
        + (d_model * 2)  # RMSNorms
    )
    parameters += d_model  # final RMSNorm
    parameters += d_model * vocab_size  # output embedding

    gradients = parameters
    optimizer_state = 2 * parameters  # 1st and 2nd moment for AdamW

    activations = 0
    activations += num_layers * (
        2 * context_length * d_model  # RMSNorms
        # Multi-head Self-Attention
        + 3 * (context_length * d_k * num_heads)  # Q,K,V projection
        + context_length * context_length * num_heads  # Q^T K
        + context_length * context_length * num_heads  # softmax
        + context_length * d_k * num_heads  # weighted sum with V
        + context_length * d_k * num_heads  # output projection
        # Feedforward
        + 2 * (context_length * d_ff)  # W1, W3 projection
        + context_length * d_ff  # SiLU
        + context_length * d_model  # output projection
    )
    activations += context_length * d_model  # final RMSNorm
    activations += context_length * vocab_size  # output embedding
    activations += context_length * vocab_size  # cross entropy
    activations *= batch_size

    sizes = dict(
        parameters=parameters,
        gradients=gradients,
        optimizer_state=optimizer_state,
        activations=activations,
    )
    sizes['total'] = sum(sizes.values())
    return sizes


def element_sizes_gpt2_xl(batch_size):
    return element_sizes(
        batch_size=batch_size,
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
    )


def get_max_batch_size(memory_limit_gb):
    element_bytes = 4  # float32
    for batch_size in range(1, 1000):
        mem = {
            k: v * element_bytes / (1 << 30)
            for k, v in element_sizes_gpt2_xl(batch_size).items()
        }
        if batch_size == 1:
            print('Parameter in GB', mem, f'for {batch_size = }')
        if mem['total'] > memory_limit_gb:
            print(f'Max batch size fitting in {memory_limit_gb} GB: {batch_size - 1}')
            return
    print('Could not determine max batch size')


def adamw_flops(parameters):
    # m = beta1 * m + (1 - beta1) * g  => 3
    # v = beta2 * v + (1 - beta2) * g^2  => 4
    # p = p - lr * m / (sqrt(v) + eps)  => 5
    # p = p * (1 - lr * weight_decay)  => 1
    return parameters * 13


def training_days():
    mfu = 0.5
    peak_flops = 19.5 * (1 << 40)
    steps = 400 << 10
    batch_size = 1024
    multiple = 3  # forward + backward

    context_length = 1024
    vocab_size = 50257
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 6400

    # Flops for batch size = 1
    d_k = d_model // num_heads
    flops = multiple * 2 * (  # times 2 for mul and add
        num_layers * (
            # Multi-head Self-Attention
            2 * context_length * d_model  # RMSNorms
            + 4 * context_length * d_model * d_model  # Q,K,V projection
            + 2 * num_heads * context_length * d_k * context_length  # QK^T
            + 2 * num_heads * context_length * d_k * context_length  # Attention
            # Feedforward
            + 2 * context_length * d_model * d_ff  # W1, W3 projection
            + context_length * d_ff  # SiLU
            + context_length * d_ff * d_model  # W2 projection
            # RMSNorms
            + 2 * context_length * d_model
        )
        + context_length * d_model  # final RMSNorm
        + context_length * d_model * vocab_size  # output embedding
        + context_length * vocab_size  # cross entropy
    )
    parameters = element_sizes(
        batch_size=1,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
    )['parameters']
    flops += adamw_flops(parameters)

    total_flops = flops * batch_size * steps
    total_seconds = total_flops / (peak_flops * mfu)
    total_days = total_seconds / (60 * 60 * 24)
    print(f'Training time estimate: {total_days:.1f} days')


get_max_batch_size(memory_limit_gb=80)
training_days()
