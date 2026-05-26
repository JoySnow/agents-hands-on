# Model Inspect - nanoGPT - local trained baby model :P

A Copy of https://github.com/JoySnow/nanoGPT/tree/master/practices/shakespeare_char/inspect_model.md
See more local try logs in https://github.com/JoySnow/nanoGPT .
Fork from source: https://github.com/karpathy/nanoGPT


### Known configs:
```
n_layer = 6
n_head = 6
n_embd = 384
```

### Layer Meanings:
```
384/6 head = 64 => 每个头分到的dims

wte: Word Token Embedding =>
    - {vocab size} = 65
    - d{model} = 384

wpe: Word Position Embedding =>
    - context window = 256

ln_1: Layer Normalization 1

attn.c_attn : Q, K, V , be concated together, 1152 / 3 = 384

attn.c_proj : W{ouptu projection}

mlp.c_fc: MLP Fully Connected 展开， 1536 = 384 * 4
    - Note: be stored in matrix 转置 [out_features, in_features] for quick dot product caculation,
            be like pick up the first row of the 转置 == load first column of matrix.
            be designed and set in nn.Linear , nn.Embedding 就不需要。 真聪明！

mlp.c_proj: Output projection

ln_f: Final Layer Normalization, before the lm_head
lm_head: Unembedding/Projection for Logits
```

### Model Layer Data - out-shakespeare-char/ckpt.pt - nanogpt
"""
$ python scripts/model_inspect.py                                     [18:11:12]
Layer Name: transformer.wte.weight                        | Shape: torch.Size([65, 384])
Layer Name: transformer.wpe.weight                        | Shape: torch.Size([256, 384])
Layer Name: transformer.h.0.ln_1.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.0.attn.c_attn.weight            | Shape: torch.Size([1152, 384])
Layer Name: transformer.h.0.attn.c_proj.weight            | Shape: torch.Size([384, 384])
Layer Name: transformer.h.0.ln_2.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.0.mlp.c_fc.weight               | Shape: torch.Size([1536, 384])
Layer Name: transformer.h.0.mlp.c_proj.weight             | Shape: torch.Size([384, 1536])
Layer Name: transformer.h.1.ln_1.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.1.attn.c_attn.weight            | Shape: torch.Size([1152, 384])
Layer Name: transformer.h.1.attn.c_proj.weight            | Shape: torch.Size([384, 384])
Layer Name: transformer.h.1.ln_2.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.1.mlp.c_fc.weight               | Shape: torch.Size([1536, 384])
Layer Name: transformer.h.1.mlp.c_proj.weight             | Shape: torch.Size([384, 1536])
Layer Name: transformer.h.2.ln_1.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.2.attn.c_attn.weight            | Shape: torch.Size([1152, 384])
Layer Name: transformer.h.2.attn.c_proj.weight            | Shape: torch.Size([384, 384])
Layer Name: transformer.h.2.ln_2.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.2.mlp.c_fc.weight               | Shape: torch.Size([1536, 384])
Layer Name: transformer.h.2.mlp.c_proj.weight             | Shape: torch.Size([384, 1536])
Layer Name: transformer.h.3.ln_1.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.3.attn.c_attn.weight            | Shape: torch.Size([1152, 384])
Layer Name: transformer.h.3.attn.c_proj.weight            | Shape: torch.Size([384, 384])
Layer Name: transformer.h.3.ln_2.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.3.mlp.c_fc.weight               | Shape: torch.Size([1536, 384])
Layer Name: transformer.h.3.mlp.c_proj.weight             | Shape: torch.Size([384, 1536])
Layer Name: transformer.h.4.ln_1.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.4.attn.c_attn.weight            | Shape: torch.Size([1152, 384])
Layer Name: transformer.h.4.attn.c_proj.weight            | Shape: torch.Size([384, 384])
Layer Name: transformer.h.4.ln_2.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.4.mlp.c_fc.weight               | Shape: torch.Size([1536, 384])
Layer Name: transformer.h.4.mlp.c_proj.weight             | Shape: torch.Size([384, 1536])
Layer Name: transformer.h.5.ln_1.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.5.attn.c_attn.weight            | Shape: torch.Size([1152, 384])
Layer Name: transformer.h.5.attn.c_proj.weight            | Shape: torch.Size([384, 384])
Layer Name: transformer.h.5.ln_2.weight                   | Shape: torch.Size([384])
Layer Name: transformer.h.5.mlp.c_fc.weight               | Shape: torch.Size([1536, 384])
Layer Name: transformer.h.5.mlp.c_proj.weight             | Shape: torch.Size([384, 1536])
Layer Name: transformer.ln_f.weight                       | Shape: torch.Size([384])
Layer Name: lm_head.weight                                | Shape: torch.Size([65, 384])
"""

"""
$ cat scripts/model_inspect.py

import torch

# 1. 把快照加载到系统内存 (CPU)
checkpoint = torch.load('out-shakespeare-char/ckpt.pt', map_location='cpu')

# 2. 提取出模型的状态字典
state_dict = checkpoint['model']

# 3. 打印它到底长什么样
for key, tensor in state_dict.items():
    # 我们只打印每一层的名字和它张量的形状 (Shape)，防止巨量的浮点数刷屏
    print(f"Layer Name: {key: <45} | Shape: {tensor.shape}")
"""