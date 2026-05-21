
#

https://learn.deeplearning.ai/courses/transformers-in-practice

## LLM behavior - as a 黑盒

Autoregressive Generation (自回归生成): 基于已经生成的历史数据，预测下一个数据.

### Token Sampling:
- Greedy
- Random

#### Sampling parameters
temperature:
top-K:
top-P:(Posiblility)

The applied order:
 原始分布-概率A -> Temperature (scale) -> 概率B -> Top-K -> Top-P -> f(Re-normalize) -> 概率C

- Temperature:
  - 0 ~ N
  - 调节概率分布的“对比度”
  - $T = 1.0$：原始分布，不加干预。
  - $T < 1.0$ (例如 0.2)：分母变小，指数项被拉伸。这会放大高概率词的优势，压制低概率词。极端情况下（$T \to 0$），就等同于贪心搜索，AI 变得极其严谨、确切。
  - $T > 1.0$ (例如 1.5)：分布被“压平”了。高概率词和低概率词的差距变小。AI 更有可能选择那些冷门的词，变得更有创造力，但也更容易胡言乱语（幻觉）。

- Top-K:
  - Order by probable LIMIT K

- Top-P (Nucleus Sampling / 核采样):
  - 0.x ~ 1
  - 基于累计概率的动态截断



## LLM Internal & attention


### 词嵌入 (Word Embedding)

LLM模型信息：
- 词汇量表 长度： $V = 100,000$
- LLM训练出的 Embedding矩阵，大小为 V * embedding_size , eg. embedding_size=4096
    - $$W_{embedding}$$

对LLM的输入：
Input: "The dog bit"
->
Token IDs: 785;5562;2699 .
->
One-hot vector:  长为V的vector，只在”2699“index上为1，其余为0.

Word Embedding：
- Transformer 在接收到用户的输入后，做的第一件事就是将 One-Hot 向量转化为 稠密向量（Dense Vector / Embedding）。
- 然后这个稠密向量才会进入 Attention 层。
- 在数学表达上，将 One-Hot 转化为 Embedding 相当于用 One-Hot 向量去乘以一个巨大的权重矩阵（Embedding Matrix）: $$E_{dense} = X_{one\_hot} \times W_{embedding}  ==>  1*embedding_size $$
- 但由于 One-Hot 只有一个位置是 1，这在工程实现上等价于一次 $O(1)$ 的数组索引查表操作。

这个压缩后的 4096 维稠密向量 表征“bit”的特征，包含了非常丰富的语义信息（比如某些维度代表“性别”，某些代表“情绪”）。
此时，“猫”和“狗”的向量点积就不会是 0，Attention 机制就能敏锐地捕捉到它们之间的上下文关联。


### 位置编码 (Positional Encoding)

注意力机制（Attention）天然是“排列不变的（Permutation Invariant）.

- 问题： 直接用 词嵌入矩阵 送入attention的话，词之间的 **位置关系信息丢失**。
- 解决方案： 我们必须给每个输入单词打上一个“Sequence ID”或“时间戳”，这就是位置编码。


传统绝对位置编码 (Absolute PE)
 - 简单粗暴：把“词的语义向量”和“位置的编码向量”直接相加。
 - 对得到$3 \times 4096$ 的词嵌入（Word Embedding）矩阵后：
    - 在Attention 计算前，Transformer 会生成一个同样大小为 $3 \times 4096$ 的位置编码矩阵（里面包含了表示绝对或相对位置的数学特征，通常通过正弦/余弦函数或可学习参数生成），
    - 然后将它与词嵌入矩阵直接相加。$$X_{\text{input}} = \text{Word\_Embedding} + \text{Positional\_Encoding}$$
 - 现在，这个 $3 \times 4096$ 的输入矩阵不仅包含了“词的意思”，还包含了“词在句子里的绝对/相对位置”。

RoPE (Rotary Position Embedding，旋转位置编码) - 相对位置：
 - RoPE 的设计哲学是：我们不显式地给特征向量做加法，而是根据它的绝对位置(Relative Position) $m$，在多维空间中对它进行“旋转” (Rotation)。 奇妙的是，当两个旋转后的向量做内积（Attention 计算）时，它们的绝对位置就会被抵消，最终的得分只与它们的相对距离有关。
 - 通过对单个 Token 注入绝对位置（旋转 $m$ 度），在 Attention 匹配时 完美获得了相对位置特征

为什么 RoPE 统治了现代大模型？(工程优势)
 - 即插即用 (只作用于 Q 和 K)：
    - RoPE 是在多头注意力（Multi-Head Attention）内部计算时，针对 $Q$ 和 $K$ 动态施加的，**不改变 Value ($V$) 向量**。
    - **不改变 Value ($V$) 向量**，这就保证了输出的数据依然是纯粹的语义信息，没有被位置信息“污染”。
 - 更好的外推性 (Extrapolation)：
    - 如果模型在训练时只见过 4096 长度的文本，但在推理时遇到了 8192 长度的文本。
    - 由于 RoPE 依赖的是相对距离，只要相对距离在训练分布内，模型就能表现出一定的适应能力（比绝对位置编码直接崩溃要好得多）


### Attention

1. QKV 的维度大小和 Embedding 维度 关系

- 我们假设模型的输入 Embedding 维度（隐层维度 Hidden Size，通常记作 $d_{model}$）是 4096。
- $W_Q, W_K, W_V$ 本质上是线性变换矩阵，所以它们的形状是 [输入维度, 输出维度]。
- 输入维度： 永远是 $d_{model}$（即 4096），因为它必须能接收上一层传来的词向量。
- 输出维度： $d_{k}$ 这就大有文章了！

2. QKV 的维度大小 $d_k$ (Dimension Size) 究竟代表什么？

- 如果我们聚焦在单个 Attention Head 的维度大小（即上文提到的 $d_k$，通常是 64 或 128），它在物理和语义上代表着什么呢？
- 你可以把它理解为“特征分辨率（Feature Resolution）”或“信息带宽（Information Bandwidth）”。
    - 如果 $d_k = 2$：你的 Query 只能包含两个维度的特征搜索，比如 [性别, 年龄]。匹配出的结果会非常粗糙。
    - 如果 $d_k = 128$：你的 Query 极其丰富，包含了 [性别, 年龄, 爱好, 性格偏好, 幽默感, 价值观...] 等 128 个维度的指标。
- 高 $d_k$ 算出的匹配度（点积）将非常精准且深刻。
- 因此，QKV 的维度大小决定了模型在一次注意力计算中，能够同时比对和传输多少个隐式的语义特征。

3. 多头注意力的QKV
- 传统架构 (Standard Multi-Head Attention)
    - 在早期的 Transformer（如 BERT 或初版 GPT）中，Q、K、V 的输出总维度通常等于 $d_{model}$。也就是说，$W_Q, W_K, W_V$ 的大小都是 $4096 \times 4096$。
    - 在内部计算时，它会被切分成多个头（比如 32 个头），每个头分到 128 维（$32 \times 128 = 4096$）。我们称这个单头的维度为 $d_k$。

- 现代大模型架构 (MQA / GQA 的革命)
    - KV Cache（显存碎片和堆积） 是推理的最大瓶颈。为了解决这个问题，现代 LLM（如 Llama 2/3, Gemini）在底层做了一个极其聪明的“阉割”：
    - 保持 $W_Q$ 的大小不变，但大幅缩小 $W_K$ 和 $W_V$ 的输出维度！
    - MQA (Multi-Query Attention):
        - 只有 1 个 K 和 1 个 V，被所有的 Q 共享。
        - $W_K$ 和 $W_V$ 的矩阵大小变成了 $4096 \times 128$。
    - GQA (Grouped-Query Attention):
        - 比如把 32 个 Q 分成 8 组，每组共享 1 个 K 和 1 个 V。
        - $W_K$ 和 $W_V$ 的大小变成了 $4096 \times (8 \times 128)$。

- 典型的多头注意力层的输出计算公式：$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W_O$$

4. 在模型训练与设计时，选择这些维度的意图是什么？

    架构师选择具体的维度大小（比如 $d_{model} = 4096$, $d_k = 128$, Heads = 32）, 主要基于以下三个工程和算法意图：
    - 意图 A：广度（多头并发）与 深度（单头维度）的博弈
        - 当前业界的 Sweet Spot（甜点区）： 经过大量实验，业界发现 $d_k = 64$ 或 $128$ 是最佳平衡点。它足够宽（能装下复杂的特征），同时允许切分出足够多的头（通常是 32 或 64 个头）来捕捉丰富的上下文关系。

    - 意图 B：对齐底层硬件架构（GPU 亲和性）
        - 在 AI 领域，Nvidia GPU 的 Tensor Core（专门做矩阵乘法的硬件单元）是按照块（Blocks）来吞吐数据的。
        - Tensor Core 最喜欢的矩阵维度是 8、16、32、64、128 的倍数,都是 $2^n$ 的衍生值。

    - 意图 C：防止点积结果爆炸 (数值稳定性)
        - 梯度消失...


### QKV

深入图解：权重矩阵到底在做什么？(线性投影)
从数学上讲，$W_Q, W_K, W_V$ 是三个线性投影矩阵（Linear Projection Matrices）。
它们的作用是提取并重组原始 Embedding 中的特征，将其映射到专门的 Q、K、V 空间。

### 计算注意力 / Attention Calculation

 Attention 公式：$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- Step 1: 计算匹配度打分 (Score Matrix)
    - 动作： 用 $Q$ 矩阵乘以 $K$ 矩阵的转置 ($K^T$)。
    - 结果： 得到一个 [3, 3] 的 **注意力得分矩阵** (Attention Score Matrix)。
    - 例子： 假设你的输入是 ["I", "love", "coding"]。这个 $3 \times 3$ 的矩阵记录了：
        Row 1: "I" 作为查询时，与 "I", "love", "coding" 的匹配分数（比如 [120, 90, 10]）。
        Row 2: "love" 作为查询时，与这三个词的匹配分数。
        Row 3: "coding" 作为查询时，与这三个词的匹配分数。

- Step 2: 缩放与归一化 (Scale & Softmax)
    - 动作： 将上一步得到的 $3 \times 3$ 分数矩阵，先除以一个常数（$\sqrt{d_k}$，为了防止数字过大导致梯度消失），然后按行应用 Softmax 函数。
    - 结果： 得到一个 [3, 3] 的 **注意力权重矩阵** (Attention Weight Matrix)。
        里面的分数变成了 0 到 1 之间的百分比概率，且每一行的和为 100%。
    - 例子： 第一行分数 [120, 90, 10] 变成了 [0.8, 0.18, 0.02]。这意味着 "I" 这个词当前要把 80% 的注意力放在自己身上，18% 放在 "love" 上。

- Step 3: 提取特征并融合 (Multiply by V)
    - 动作： 用刚才得到的 [3, 3] 的权重矩阵，去乘以 $V$ 矩阵（包含实际语义载荷的数据）。
    - 结果： 得到一个全新的矩阵，我们通常称之为 **$Z$ (Contextualized Vectors / 上下文向量)**，形状依然是 [3个词, 维度dk]。
    - 物理意义： 这本质上是一个加权求和。系统把根据权重从 $V$ 里提取出的特征打包在了一起。
        此时，矩阵 $Z$ 里的 "I" 已经不再是字典里那个干瘪的 "I" 了，而是变成了“知晓后面跟着 love coding 的 'I'”。

- Step 4: 线性投射 (Linear Projection)
    - 如果有多个 Head（多头注意力），系统会把所有的 $Z$ 矩阵拼接起来.
        - dk=128, 32个Head: concat 32 个 [3, dk] to one [3, d model(32*128)] as $Z_{concat}$ , same size as the $X$.
        - $Z_{concat}$ 的缺陷：“部门墙”与信息隔离，eg. 前 128 维全是“情感头”算出来的，接下来的 128 维全是“语法头”算出来的。
    - $Z_{concat}$ * 输出权重矩阵 $W_O$ :
        -  对 $Z_{concat}$ 这 4096 个维度做了一次全局的线性重组（全连接计算）。
        - $W_O$ (Linear Projection) [4096 * 4096] 的意义是“跨界信息融合 (Information Fusion)”
    - 结果：$Z$ [4096 * 4096]

- Step 5: 残差连接与归一化 (Add & Norm)
    - 系统不会直接把算好的 $Z$ 传给下一层，而是做一个加法：$\text{Output} = \text{LayerNorm}(X + Z)$
    - 它把最原始的输入 $X$ (the input X for Attention step) 和刚刚算出的上下文特征 $Z$ 加在了一起。
    - 你可以把 $Z$ 理解为一个“增量补丁（Delta）”。
    - 这样做的好处是，如果当前 Attention 层没学到什么有用的东西（$Z$ 全是 0），至少原始信息 $X$ 不会丢失，这在深层网络中防止了“数据退化”。


#### 注意力计算 结果 [3 * 4096]：
 - 不仅包含了原始的输入特征（由 $X$ 提供保底）。
 - 融合了多头注意力带来的全局上下文信息（由 $Z$ 提供增量补丁）。
 - 并且它的数值大小被洗得干干净净、服服帖帖（由 Norm 提供安全保障）。

此时，这 3 个极其优质的“上下文融合特征向量（Contextualized DTOs）”，下一步，它们将离开 Attention 模块的管辖范围，被正式送入我们之前聊过的 前馈神经网络 (FFN, Feed-Forward Network)，去知识库里匹配记忆、进行深度的个人升维思考了！


- Attention adds context
- 多头注意力：to learn diff things, say,


经历完这整套流程：Embedding $\rightarrow$ QKV $\rightarrow$ Attention $\rightarrow$ Add&Norm $\rightarrow$ FFN $\rightarrow$ Add&Norm，数据才算真正走完了一个 Transformer Layer。


### FFN - 前馈神经网络，Feed-Forward Network
全程：Position-wise Feed-Forward Network（逐位置前馈网络）。

**总结一下：**
**Attention 决定了“我们现在要讨论什么”（关注法国和首都）。**
**FFN 负责翻阅模型内部的参数字典，回答“对应的具体内容是什么”（吐出巴黎的特征）。**

一个FFN：有两层全连接网络（Linear Layers）加上一个非线性激活函数。
数学公式表达是这样的：$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

FFN 三个标准处理步骤：
 - Step 1: 升维投射 (Upscaling / Expand)
    - 动作： 把刚刚从 Attention 层出来的 $d_{model}$ 维度的向量（比如 4096 维），乘以一个极其巨大的权重矩阵 $W_1$。
    - 结果： 它的维度通常会被放大 4倍 或更大（比如扩展到 16384 维）。
    - 系统意义（特征解卷）： 4096 维的向量是高度压缩的 DTO。为了让后续的业务逻辑能看清楚里面的细节，系统先把它“解压缩”到一个高维空间。在这个 16384 维的空间里，词汇更细粒度的特征（比如语法特征、情感特征、实体特征）被强行拆解并暴露出来。

 - Step 2: 非线性过滤 (Non-linear Activation)
    - 动作： 上面公式里的 $\max(0, ...)$ 指的是 ReLU 激活函数（现代大模型多使用 GELU 或 SwiGLU，但原理类似）。它的作用很简单：把所有负数变成 0，正数保留。
    - 系统意义（If-Else 逻辑门）： 这是整个 Transformer 中唯一引入非线性的地方！如果没有这一步，所有的矩阵乘法最终都可以合并成一个简单的线性方程。激活函数就相当于无数个并行的 if-else：它决定了在这 16384 个维度的特征中，哪些特征是噪音需要被屏蔽（设为 0），哪些特征是关键信息需要被激活并放行。

- Step 3: 降维投射 (Downscaling / Compress)
    - 动作： 将过滤后的 16384 维向量，再乘以第二个权重矩阵 $W_2$。
    - 结果： 把它重新压缩回标准的 4096 维（$d_{model}$）。
    - 系统意义： 经过复杂的过滤和特征重组后，系统需要把结果重新打包成标准尺寸的 DTO，以便送入下一个 Transformer 层继续流转。

FFN 的物理意义：大模型的“知识库” (Key-Value Memory)
    - 这是 AI 学界近年来一个非常惊艳的发现（由 Geva 等人在 2020 年提出）：
    - FFN 本质上充当了语言模型存储常识和事实的“键值对数据库 (Key-Value Datastore)”。
    - 让我们用刚才的两层网络来类比一个后端缓存系统：
        - 第一层 ($W_1$ / 升维)：充当 Key (查询匹配)。 它在检测当前的上下文特征是否触发了某种特定的模式。
        - 第二层 ($W_2$ / 降维)：充当 Value (返回结果)。
    - 如果第一层检测到匹配，第二层就会向特征向量中注入特定的静态知识。

增加模型参数（比如从 7B 增加到 70B），很大一部分就是在扩充 FFN 里面的那个隐藏维度（让它从 16384 维变成几万甚至十几万维），从而相当于给大模型加了一块更大容量的“知识硬盘”。

### After FFN

... -> Attention -> Add & Norm -> FFN ->  Add & Norm -> ?

- 和 Attention 层一样，数据从 FFN（前馈神经网络）出来后，必须再经过一次残差连接（Add）和层归一化（Norm）。公式：
    $X_{layer\_out} = \text{LayerNorm}(X_{attention\_out} + \text{FFN}(X_{attention\_out}))$

```py
if not last layer:
    进入下一层的"Attention + FFN"
else:
    Linear Projection (逆向查表 / 反嵌入) -> 获取 Logits -> Softmax -> Sampling
```

After 整个模型的“最后一层”（最终宏观步骤）：
 - 当数据走完了所有的 Attention+FFN 循环后，我们最终得到了一个包含高级语义的 $4096$ 维向量。
 - 翻译回人类能看懂的词汇：
    - Step 1: 最终层归一化 (Final Layer Normalization)
    - Step 2: 解嵌映射层 (Unembedding Layer / LM Head)：
        - 把这个 $4096$ 维的向量乘以一个超级巨大的输出矩阵（大小是 $4096 \times \text{词表大小}$，比如 $4096 \times 128,000$）。
        - hint. 权重共享（Weight Tying）of $E_{Embedding}$ and $E_{Unembedding}$
    - Note. 有了 Logits 之后，就进入了推理引擎（如 vLLM, TensorRT-LLM）的控制领域了。
    - Step 3: 获取 Logits: 得到 $128,000$ 个得分（Logits）, for 预测词
        - Step 3.1: 惩罚机制 (Penalties) - 过滤掉“废话” - 修改 Logits 的值
        - Step 3.2: Temperature (温度缩放) - 控制“随机性”
        - Step 3.3: Softmax (转化为概率)： 用 Softmax 把 Logits 变成加起来等于 $100\%$ 的概率分布。
        - Step 3.4: 截断策略 (Truncation) - apply Top-K 与 Top-P
        - Step 3.5: Sampling (采样): 基于这概率进行带权重的随机抽签，抽出下一个字！

权重共享（Weight Tying）
- 设计初衷：共享 (Tied)
    - 早期模型（如 2017-2019 年）为了节省内存，$E_{Embedding}$ and $E_{Unembedding}$ share，互为转置矩阵。
    - 适合小模型
- 现代演进： 不共享 (Untied)
    - 现代 LLM 的绝对主流
    - 现代模型（尤其是百亿、千亿参数的大模型）越来越倾向于解绑（Untied）。
    - 因为“读（理解输入）”和“写（生成输出）”其实是两种不同的任务。 解绑可以让模型用独立参数分别优化这两个过程。
    - 且在千亿参数规模下，节省几十兆的词表参数已经微不足道。


### 模型的层数 of "Attention + FFN"

每层的架构一模一样（都是 Attention + FFN），但是它们内部的权重矩阵（$W_Q, W_K, W_V, W_1, W_2$）是完全独立、互不相同的！

第 0 层负责提取浅层的语法特征（主谓宾），到了第 31 层，矩阵里装的已经是深层的逻辑推理和宏观概念了。

主流模型的真实设定参数（例子）：
- 轻量级模型 (如 BERT-base): 循环 12 次。
- 百亿级模型 (如 Llama-3-8B): 循环 32 次。
- 千亿级巨兽 (如 Llama-3-70B, GPT-3): 循环 80 到 96 次 不等。



### 模型参数(Model Parameters)

这些参数分布在 Transformer 的哪里？（参数资产盘点）
- A. 词嵌入层 (Embedding Layer) - 约占 5% 到 10%
    - 它是什么： 那个巨大的字典查询表。
    - 参数量： 词表大小 × 隐藏层维度, $W_{embedding}$
    - 例子： 假设词表有 100,000 个词，每个词被映射为 4096 维的向量。这里的参数量就是 $100000 \times 4096 \approx 4$ 亿个浮点数。

- B. 注意力层 (Attention Layer) - 约占 25% 到 30%
    - 它是什么： 我们聊过的 DTO 转换接口。
    - 参数量： 主要是 $W_Q, W_K, W_V$ 这三个负责把输入映射为 QKV 的矩阵，以及最后把多头结果拼接后重新映射回来的 $W_O$ 输出矩阵。
    - 例子： 在一个标准的 Transformer 层里，如果维度是 4096，这四个矩阵的大小通常都是 $4096 \times 4096$。由于模型往往有 32 层，这部分参数会随着层数成倍叠加。

- C. 前馈网络层 (FFN Layer) - 参数量的“绝对大头”，占 60% 以上
    - 它是什么： 存储常识和事实的“键值对知识库”。
    - 参数量： 包含升维矩阵 $W_1$ 和降维矩阵 $W_2$。
    - 例子： 像我们之前讨论的，为了存储海量知识，FFN 内部的维度通常会被拉伸 4 倍甚至 8 倍（比如从 4096 维拉伸到 16384 维）。因此，$W_1$ 的大小是 $4096 \times 16384$。这极其庞大的矩阵占据了整个模型绝大部分的参数量。

 - D. 归一化层 (LayerNorm) - 占比极小（可忽略不计）
    - 它是什么： 用于防止数值爆炸的缩放和平移系数（$\gamma$ 和 $\beta$）。
    - 参数量： 通常只有几千个数字，相较于总规模连九牛一毛都算不上。


3. "7B", "70B" 到底意味着什么系统开销？当我们讨论 Llama-3-8B 或者 Qwen-72B 时，这个 B (Billion, 十亿) 就是指模型参数的总数量。作为后端，你需要立刻将这个数字转化为系统资源（特别是 GPU 显存 VRAM）的开销。一个非常实用的工程粗算公式：现代模型通常使用半精度浮点数（FP16 或 BF16）来存储参数。在计算机里，1 个 16 位浮点数占用 2 Bytes（字节） 的空间。计算 Llama-2-7B (70亿参数) 的静态显存：参数总数：7,000,000,000占用空间：$7,000,000,000 \times 2 \text{ Bytes} \approx 14 \text{ GB}$这意味着什么？即便你什么用户请求都不处理（没有我们之前聊过的 KV Cache，没有 Batch 堆积），仅仅是把 Llama-2-7B 的这套“配置规则”原封不动地加载到显卡上，它就会立刻吞噬掉整整 14 GB 的显存！如果你要跑 70B 的超大模型，光是参数加载就需要 $70 \times 2 = 140 \text{ GB}$ 的显存（通常需要用到两张极其昂贵的 80GB A100 显卡）。


### Quantization 量化

以 CPU 换 Mem IO 的交易 (Trade Compute for Memory Bandwidth)

For a 7B model, 16bit parameter, cost 14 GB 的显存: 7,000,000,000 * 2 Byte = 14GB

量化 - “数据压缩”手段, 不减少参数，14GB(16-bit) -> 4GB(4-bit) .

量化过程：
- A. 静态映射：如何把 FP16 塞进 INT4？(量化过程)
    - 假设我们有一组原始的 Float16 权重：[2.15, -1.82, 0.44, 1.11]。
    - 如线形量化： $W_{int4} = \text{round}\left(\frac{W_{fp16}}{\text{scale}}\right) + \text{zero\_point}$
    - =》 4-bit 整数：[7, -8, 1, 4] , 额外 + scale 值（FP16）, zero_point。

- B. 动态计算：运行时的“反量化” (Dequantization)
    - 显存读取（省带宽）： 系统从 GPU 显存（VRAM）中以极快的速度读取 4-bit 的参数。这是量化最核心的收益！极大地缓解了Memory-Bound（内存带宽瓶颈）。
    - 寄存器反量化（耗算力）： 在数据到达 GPU 计算单元（SRAM/寄存器）的那一瞬间，系统会利用刚才保存的 scale 将其动态还原（反推）回Float16.

量化 版本/级别:
- 16-bit, 如 BF16 / FP16 : 原版
- 8-bit，如 INT8 或 FP8 : Sweet Spot
- 4-bit，如 INT4, AWQ, GPTQ, GGUF 格式: 边缘计算版, like laptop.
- 2-bit / 3-bit ： 极限压缩版，按需提供，like 嵌入式硬件限制.

在 AI 领域，量化主要分为两派，主流采用的是PTQ：
- 1. PTQ (Post-Training Quantization / 训练后量化) —— 绝对的主流，超快，无需训练。
- 2. QAT (Quantization-Aware Training / 量化感知训练) —— 高端但昂贵，需要训练

量化技术：
- 线形量化
- GPTQ (Generative Pre-trained Transformer Quantization)
    - 误差补偿（Error Compensation）, 偏移
- 激活感知权重量化 - AWQ (Activation-aware Weight Quantization)
    - 它保留这 1% 的权重在 FP16 高精度，只把剩下的 99% 压缩成 INT4。
- GGUF (GPT-Generated Unified Format):
    - 核心定位： 打破 GPU 垄断的“全能打包格式”，C++ 生态的救星。
    - llama.cpp , ollama

---

### Cross-Attention & Self-Attention

- Self-Attention 解决的是“高内聚”问题（处理单一数据源的内部逻辑）
- Cross-Attention 解决的则是“低耦合环境下的跨端通信”问题（不同模态、不同系统间的数据对齐）。

在经典的 Transformer 诞生时（2017年），它包含两块：
 - Encoder (编码器)： 负责阅读源文本（全向 Self-Attention）。
    - 计算出最终的 K，V， for Decoder 使用。
 - Decoder (解码器)： 负责生成目标文本（单向 Self-Attention + Cross-Attention）。
    - 在 Cross-Attention阶段，用自己的 $Q_{cross}$ 去 encoder部分做查询，获取信息。

### modern LLM are all Decoder-Only ?

Yes! 在主流的现代 LLM 中，Encoder 被砍掉了，Cross-Attention 也被彻底抛弃了！

OpenAI 在做 GPT 系列时，做了一个极其激进的系统架构简化：
 - 直接把 Encoder 和 Cross-Attention 删掉，只保留 Decoder 中的单向 Self-Attention 和 FFN。
 - 系统不再区分“源数据”和“目标数据”。
    - 以前 (Encoder-Decoder)： 把问题喂给 Encoder，让 Decoder 生成答案。
    - 现在 (Decoder-Only)： 把问题和答案强行拼成一条完整的长字符串，直接送进 Decoder！

既然只有 Decoder，多模态数据怎么处理？
 - LLM 只认识文本 Token，把 图片“伪装”成文本 Token 塞进去！

这个过程分为极其优雅的三步：
- Step 1: 外部组件提炼特征 (External Vision Encoder)
我们依然需要一个专门看图的组件（比如 ViT, 视觉 Transformer）。它看完图片后，输出了 256 个图像特征向量。
注意：这个组件独立于 LLM 之外，就像一个外部微服务。
- Step 2: 降维与协议转换 (The Projector / 适配器)
这 256 个图像向量的维度，跟 LLM 底层的文本向量维度是对不上的。于是，工程师在中间加了一个非常简单的线性投影层（Linear Projector，通常就是一个小型的 MLP 网络）。
它的作用是把图像特征在数学空间上“扭曲、平移”，强制对齐到 LLM 的文本特征空间里。
- Step 3: 暴力拼接输入流 (Sequence Concatenation)
这是最神奇的一步！系统把这 256 个经过伪装的“图像 Token”，和用户的“文本 Token”，直接在物理层面拼接成了一个一维数组。
你的输入 Payload 变成了类似这样：
[Image_Token_1, Image_Token_2, ..., Image_Token_256, 文本_Token_"这", 文本_Token_"是", 文本_Token_"什么"]

- Step 4: 全局 Self-Attention (统一计算)
这个混合了图片和文字的一维数组，被直接送入 Decoder-only 的 LLM 中。
LLM 的底层根本不知道前面那 256 个 Token 是图片！它只是觉得：“哦，这是 256 个我不认识的火星文单词，但没关系，我用一视同仁的 Self-Attention 去计算它们和后面汉字的关联度（QKV 匹配）。”


目前 AI 工业界在**多模态**架构上的两条不同路线（或者说“专才”与“通才”的区别）：
- 路线 A：保留 Cross-Attention 的“专家系统” (如 Whisper, Midjourney)
    - 特点： 依然采用 Encoder-Decoder。视觉/听觉和文本是物理隔离的。
    - 优势： 在特定的单项任务（如精准的语音识别、高并发的 AI 绘画）上性能极致，因为跨模态的特征提取非常清晰。
- 路线 B：抛弃 Cross-Attention 的“通用大一统大模型” (如 GPT-4V, Llama-3-Vision, Qwen-VL)
    - 特点： Decoder-Only 架构。利用 Projector 将万物（声音、图片、视频帧）统统转化为 Token 序列，强行塞进同一个 Self-Attention 引擎里。
    - 优势： 极度的通用性！ 你可以在同一个对话框里，传一张图，传一段音频，再加一段文字。模型底层的架构不需要做任何修改，只要显存塞得下这个超长的混合 Token 数组，它就能通过 Self-Attention 算出它们之间所有的隐藏逻辑关系。

## LLM deployment

### Prefill & Decode 阶段

LLM 运行 on a 1000 token prompt input:
- Prefill 阶段 - (Compute-Bound)
    - 推理引擎会把这 1000 个 Token 一次性、并行地输入到模型中。
    - 核心目的（为什么叫预填充？）： 为这 1000 个输入 Token 计算出它们在每一层的 $K$ (Key) 和 $V$ (Value) 向量，并将这些向量“填充（Fill）”到 GPU 的显存中（即建立 KV Cache）。
    - 输出： 跑完这一次重型的并行计算后，模型吐出第 1001 个词（即 AI 回答的第一个字），并且缓存（Cache）已经“预热”完毕。
- Decode 阶段 - (Memory-Bound)
    - 极其低效的单步循环： 此时，模型每次只接收 1 个 Token（即刚刚生成的那个字）。
    - 复用缓存： 这个新的 Token 在穿过网络时，需要用到之前的上下文信息，但它不需要把前面 1000 个字重新算一遍。它只需要去显存里读取 Prefill 阶段建好的 KV Cache，和当前的 Token 做 Attention 计算即可。
    - 输出： 吐出下一个新词，并把这个新词的 $K, V$ 追加到 KV Cache 中。然后重复这个循环，直到生成结束。

#### **痛点**：长 Prompt 造成的“世界停顿” (Stop-The-World)
- in 传统单节点引擎
- 如果此时有 10 个用户正在愉快的进行一问一答（处于 Decode 阶段，生成极快），突然第 11 个用户发来了一份长达 10 万字的 PDF 让模型总结。
- 模型为了处理这个庞大的 Prefill 请求，会把 GPU 的算力全部霸占。
- 这会导致什么？那 10 个正在接收文字的用户会突然卡住（延迟毛刺，Latency Spikes），屏幕上的字不输出了，直到那 10 万字处理完毕。

#### Chunked Prefill 的运行机制

Chunked Prefill 是目前解决“算力与访存抢占”最优雅的**纯软件级调度**方案。

引入操作系统中最经典的调度概念：时间片轮转（Time-Slicing） 与 任务拆分:
 - 切片 (Chunking)：
    - 引擎不再要求一次性吞下那 10 万字的 Prompt。它设定一个全局的“最大计算块大小（Chunk Size）”，比如 512 个 Token。那 10 万字的长请求，会被切分成大约 200 个大小为 512 的小块。
 - 时间片调度 (Scheduling)：
    - 在 GPU 的下一个计算周期（Tick）里，调度器会这样分配工作：
        - 拿出一个 Prefill 的切片（512 个 Token）。
        - 把那 10 个老用户的 Decode 请求（共 10 个 Token）也拉过来。
        - 混合批处理 (Mixed Batching)： 将这 512 + 10 = 522 个 Token 拼成一个 Batch，一起送给 GPU 计算。
 - 状态保留与让出 (Yield)：
    - 算完这 522 个 Token 后，那 10 个老用户顺利拿到了他们的新字，延迟完全不受影响。
    - 而那个长请求，虽然只处理了第一块（512 个字），但引擎会把这部分算好的 KV Cache 先存起来，然后主动让出（Yield）执行权，等待下一个周期继续处理第二块。

#### Prefill-Decode 分离架构（Disaggregation）

Prefill-Decode 分离架构本质上是用“极高的网络带宽”来换取“GPU 算力和显存的完美隔离与极致利用率”。

解决方案：Prefill-Decode 分离架构

工程师们将 GPU 集群按职责划分成了 **两个物理上隔离的资源池（Pools）**。
1. Prefill 节点池 (计算特化集群)
 - 硬件画像： 采用算力极强（如 H100）、但不需要配备超大显存的 GPU。
 - 职责： 专门负责接收新用户的长 Prompt，进行高并发的矩阵乘法，快速计算出这批 Prompt 的状态数据（KV Cache）。
 - 运行状态： 批处理模式，算力利用率极高，算完立刻把结果抛出。

2. Decode 节点池 (存储/访存特化集群)
 - 硬件画像： 采用显存容量极大、带宽极高（如多张 A100 80G 组网，甚至未来的特殊大显存芯片），算力普通即可。
 - 职责： 专门维护海量并发用户的上下文状态（KV Cache），在一个相对纯粹、没有算力抢占的环境中，以极低延迟逐字生成 Token。
 - 运行状态： 类似 Redis 集群，吃满内存和带宽，但响应极快。

Prefill-Decode 分离架构最大的工程挑战就在于：状态迁移（State Transfer）- 网络传输瓶颈 (The Network Toll)
- 在分离架构中，KV Cache 数据 流转过程：
    - Prefill 节点算出了几千个词的 KV Cache。
    - 必须把这几百兆甚至几个 G 的 KV Cache 张量数据，通过网络传输给 Decode 节点。
    - Decode 节点接收完毕后，才能开始生成第一个字（Time To First Token, TTFT）。
- 博弈
    - 超长上下文(eg. 100K Token)，其 KV Cache 可能高达数十 GB。
    - 即使是使用最顶级的 InfiniBand (IB) 网络 或 NVLink 做 RDMA 直通，通过网络传输这么庞大的数据，所消耗的时间也可能超过了分离架构带来的收益。

### Prompt Caching

为什么要引入 Prompt Caching？（痛点场景）
- 我们先看几个真实业务中极其浪费算力的场景：
    - System Prompt（系统提示词）冗余： 你的 AI 应用可能在后台悄悄给每个用户的请求前面都拼了一段长达 1000 字的系统指令（比如：“你是一个资深架构师，请遵循以下 50 条代码规范...”）。如果有 1 万个并发用户，这 1000 个词的 Prefill 计算会被重复执行 1 万次。
    - 多轮对话的长尾效应： 在一个 10 轮的对话中，当计算第 10 轮的回复时，前 9 轮的历史记录又会被作为 Prompt 发给模型重新 Prefill 一遍。
    - RAG（检索增强生成）与长文档问答： 用户上传了一份 5 万字的财报 PDF，然后连续问了 5 个不同的问题。如果不做缓存，每次提问都要把这 5 万字重新跑一遍矩阵乘法。
- 核心结论：
    - 这些场景中，Prompt 的前缀（Prefix）是完全相同的。
    - 如果能把这些相同前缀已经算好的 KV Cache 保留下来，当新请求到来时直接复用，就能把 Compute-Bound（极耗算力的 Prefill）瞬间降维成 Memory-Bound（只需简单读取的 Decode）。

底层数据结构：如何高效匹配和管理缓存？
- 不能使用简单的哈希表，
- 使用了 基数树（Radix Tree / Prefix Tree）

对新输入，做“最长前缀匹配”:
 - 匹配部分，复用，对应 物理显存块的引用计数+1；
 - 其余部分，做 Prefill 计算

### KV Cache 量化

**结论**：权重占用的显存是固定的（静态开销），而 KV Cache 占用的显存随“用户数 × 文本长度”呈线性爆炸（动态开销）。

**总结**：
通过 **KV Cache 量化 (数据压缩) + KV Cache 淘汰 (冷数据清理) + PagedAttention (内存碎片整理)** 这三套后端系统经典的组合拳，
AI 工程师成功将大模型支持的上下文长度从早期的 2K、4K，硬生生撑到了现在的百万级别（如 Gemini 1.5 Pro 的 1M~2M Context Window）。


1. 痛点：为什么必须量化 KV Cache？（显存爆炸算一笔账）
陷阱在于并发（Concurrency）和长文本（Long Context）。

以 Llama-3-8B 为例，采用 GQA 架构，计算一个 Token 的 KV Cache 占用大小公式是：
**2 (K和V) × 32 (层数) × 8 (KV头数) × 128 (单头维度) × 2 (FP16字节数) = 131 KB / Token**

==> 100K 文本可能需要 13GB 的 KV Cache.


2. 工程挑战：KV Cache 量化难在哪里？

压缩高并发的实时数据流（Online 实时压缩）

3. 工业界的主流解法：怎么做 KV 量化？
为了平衡“实时压缩耗时”和“显存节省比例”，目前工业界（如 vLLM, TensorRT-LLM 框架）采取了以下几种主流方案：

- 方案 A：FP8 KV Cache (当下的最佳甜点)
    - 原理： 将 16-bit 浮点数降级为 8-bit 浮点数（FP8）。
    - 优势： 现代 Nvidia GPU（如 Hopper 架构的 H100）在**硬件底层原生支持** FP8 计算！这意味着不需要复杂的代码来回转换，显卡自己就能极速完成读写。它能将 KV Cache 显存完美减半（13GB 变成 6.5GB），且精度几乎零损失。
    - 现状： 这是目前大厂生产环境中长文本模型最常用的方案。

- 方案 B：INT8 KV Cache (广泛兼容)
    - 原理： 压成 8 位整数。对于不支持 FP8 的老显卡（如 A100 或 RTX 30/40 系），使用 INT8 配合动态 Scale。
    - 特点： 同样节省 50% 显存。因为 K 和 V 的数值分布特征不同，通常采用通道级量化（Per-Channel Quantization），即为每一个特征维度单独算一个 Scale，以保证精度。

 - 方案 C：KIVI / INT4 KV Cache (激进的极限压缩)
    - 痛点： 把动态数据压到 4-bit 极度容易崩溃，因为前文提到的“离群值（Outliers）”在动态激活数据中表现得非常狂野。
    - KIVI 算法黑科技： 学术界和工业界发现，Token 的特征数据如果按 Token 划分（纵向），差距极大；但如果按维度划分（横向），却非常平滑。因此，类似 KIVI 这样的算法采用了二维的非对称量化，并在显存中保留极少部分的高精度数据，成功将 KV Cache 压到了 4-bit 甚至 2-bit，依然能保持不错的上下文理解能力。

4. 终极系统优化：不仅量化，还要“淘汰” (KV Cache Eviction)
LRU / LFU 缓存淘汰策略。
H2O (Heavy Hitter Oracle) / StreamingLLM 等算法：
    在系统运行中，实时监控哪些 Token 被 Attention Q 查询的次数最少（即“冷数据”），直接把它们的 KV Cache 从 GPU 显存中永久删除（Evict）！只保留“最新生成的 Token”和“被高频访问的关键 Token（Heavy Hitters）”。

### 1M context window model and API

当下的大语言模型市场中，各个厂商API context window:
- 第一梯队：商业 API 的“黄金标准” (128K - 200K)
    - Sweet Spot
- 第二梯队：极致长文本的“炫技与护城河” (1M - 2M+)
    - 代表产品： Google 的 Gemini 1.5 Pro (原生支持 1M 到 2M)、月之暗面 Kimi (支持 2M 甚至尝试更高)、阿里 Qwen 2.5 (部分版本支持 1M)。
- 第三梯队：开源部署与企业私有化的“务实之选” (8K - 32K)
    - eg. on my local ollama, default to 32K.


工程解法 (Context Parallelism / 上下文并行) for 极致长文本API:
- 为了在 API 层面真正向你提供 1M 的并发调用能力，API 提供商的底层平台必须引入极其复杂的分布式架构，
- 比如 Ring Attention (环形注意力)。系统会把你的 1M 请求切成 4 块，分别派发给 4 张跨节点的 GPU。这 4 张 GPU 在计算 Attention 时，会像一个环形网络一样，通过极其高速的 NVLink 带宽，一边计算一边互相传递（点对点通信）自己切片内的 KV 块。

### 投机解码 (Speculative Decoding) 或 推测采样 (Speculative Sampling)

解决的问题：GPU 计算单元 与 显存 之前的 I/O瓶颈 - Memory Bandwidth
    - 基础模式下，每计算一个token，做 搬运 1 次权重的耗时

Target + Draft model 模式：
 - have a input prompt;
 - Draft model 做推理，预测出一批token， 比如5个；
 - Target model：
    - 接收 “prompt + 5预测token”，正常运算；
    - 依据target运算结果，判断draft预测的5个token是否准确；
    - 准确的就采纳 =》 target一次性做5个token；
    - 不准确，如从第四个开始不准确，就采纳前三个，用自己的第四个 =》 target一次性做4个token；
 - Target的计算量反而增大了。但是整体速度提升了，一次权重搬运，做出了多个token。

主流大模型现在是怎么用它的？（真实工业界案例）
 - 投机解码已经是目前顶尖推理引擎（如 vLLM, TensorRT-LLM, TGI）的标配，并在不断进化。
 - A. 经典双模型架构 (Target + Draft)
 - B. 进化版：单模型挂载“美杜莎多头” (Medusa Heads / EAGLE)


### TBC. 推理引擎（Inference Engine)

推理引擎包括开源的 vLLM、HuggingFace 的 TGI (Text Generation Inference)，以及 Nvidia 闭源的 TensorRT-LLM。


## Future - Mamba

Mamba 的全称是 Selective State Space Model（选择性状态空间模型），由 Albert Gu 和 Tri Dao 在 2023 年底提出。
它一出世就震惊了 AI 界，因为它不仅在效果上媲美 Transformer，而且真的在理论和工程上实现了线性复杂度 $O(N)$！


- Mamba的前身 SSM（State Space Model）
    - 核心像RNN，有 hidden State，所以O(N)

既然和 RNN 这么像，为什么 RNN 失败了，Mamba 却成功了？
- RNN 的两大死穴：1. 记忆遗忘（信息被无差别冲刷）；2. 串行计算导致训练极慢。

Mamba 两项革命性的创新
1. 解决遗忘：Selective Mechanism（选择性机制）
    - 早期的 SSM 和 RNN 有一个致命弱点：它们的矩阵 $A, B, C$ 是静态的、与输入数据无关的（Input-independent）。不管输入的词是重要的动词，还是毫无意义的标点符号，系统都会用同样的力度去更新那个固定大小的记忆缓存 $h_t$。这导致有效信息很快被噪音淹没。Mamba 的破局点在于加入了“选择性（Selective）”：Mamba 让 $B, C$ 以及控制步长的参数 $\Delta$ 变成了当前输入 $x_t$ 的函数。后端类比（智能日志路由器）：你可以把 Mamba 想象成一个带有了动态 if-else 判断的日志缓存系统。当输入的词是一个毫无意义的语气词（比如 "um", "ah"）时，系统算出的 $\Delta$ 会非常小，Mamba 会直接忽略这个词，保持旧的状态 $h_{t-1}$ 不变。当输入的词是一个极其关键的主语或动词时，Mamba 会立刻放大更新力度，将关键信息强行写入状态 $h_t$ 中。
    - 结果： Mamba 能够像人类一样，自动过滤掉长文本中的废话，精准记住 10 万字以前发生的核心关键点。它用 $O(N)$ 的复杂度，实现了媲美 Transformer $O(N^2)$ 的长程上下文提取能力！
2. 解决训练慢：Hardware-aware Parallel Scan（硬件感知的并行扫描）
    - RNN 最大的痛点是训练时必须一步步算，无法利用 GPU 的高并发。而 Mamba 虽然在推理时是 RNN，但在训练时，它可以化身为完全并行的形态！数学魔法（Associative Scan）： 因为 Mamba 去掉了所有非线性激活函数，它的状态更新是纯线性的。在数学上，连续的线性矩阵乘法满足结合律。这意味着系统可以通过类似“并行前缀和（Parallel Prefix Sum）”的算法，在 GPU 上同时计算出所有时刻的隐藏状态，彻底打破了时间序列的阻塞。

Mamba 最硬核的底层优化：SRAM 显存刺客
 - 把极致的 GPU 硬件级优化带入了 Mamba。
 - 痛点： Mamba 既然要动态计算每个时刻的矩阵 $B, C, \Delta$，意味着要产生大量的中间变量。如果把这些变量频繁地在 GPU 的 HBM（主显存，容量大但极慢）和 计算单元之间搬来搬去，巨大的 I/O 延迟会彻底拖垮速度。Mamba 的工程解法（Kernel Fusion）：Tri Dao 亲自写了极其硬核的 CUDA 底层代码。Mamba 会把那个巨大的隐藏状态 $h$ 死死地锁在 GPU 的 SRAM（超高速缓存，极小但极快，类似于 CPU 的 L1/L2 Cache）里，绝不把它写回主显存！在 SRAM 内部，系统瞬间完成所有的动态权重生成、状态更新和输出，最后只把算好的最终结果 $y_t$ 写回 HBM。
 - 这种极致的“避免显存读写（Memory IO-Bound Optimization）”，让 Mamba 的运行速度比同级别的 Transformer 快了惊人的 3 到 5 倍。

Mamba 代表了未来最美的工程愿景——没有 KV Cache 的无限上下文生成。目前它在 7B 及以下级别的小参数模型上（如 Jamba, Mamba-Codestral）已经展现出了打平甚至超越 Transformer 的实力。