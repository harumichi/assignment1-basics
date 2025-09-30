from cs336_basics.nn import Transformer

import yaml


def model_parameter_count():
    print('--------------------------------------')
    model = Transformer(
        vocab_size=50257,
        context_length=1024,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000.0,
    )
    num = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {num:,}')
    print(f'Parameter bytes (float32): {num * 4 >> 20} MB')


def flops():
    print('--------------------------------------')
    context_length = 1024

    common_args = dict(
        vocab_size=50257,
        context_length=context_length,
        d_ff=6400,
        rope_theta=10000.0,
    )
    args_list = {
        'GPT-2 small': dict(
            d_model=768,
            num_layers=12,
            num_heads=12,
            **common_args,
        ),
        'GPT-2 medium': dict(
            d_model=1024,
            num_layers=24,
            num_heads=16,
            **common_args,
        ),
        'GPT-2 large': dict(
            d_model=1280,
            num_layers=36,
            num_heads=20,
            **common_args,
        ),
        'GPT-2 XL': dict(
            d_model=1600,
            num_layers=48,
            num_heads=25,
            **common_args,
        ),
        'GPT-2 XL (context_length = 16384)': dict(
            d_model=1600,
            num_layers=48,
            num_heads=25,
            **common_args,
        ) | dict(
            context_length=16384,
        ),
    }
    for name, args in args_list.items():
        print(name)
        model = Transformer(**args)
        d = model.flops_count(args['context_length'])
        print(yaml.dump(d, sort_keys=False))


if __name__ == '__main__':
    model_parameter_count()
    flops()
