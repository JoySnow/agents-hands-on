# AI 基础知识学习

- [AI 基础知识学习](#ai-基础知识学习)
  - [理解AI发展史](#理解ai发展史)
    - [从规则学习 -\> 机器学习 -\> 深度学习(CNN,RNN)](#从规则学习---机器学习---深度学习cnnrnn)
    - [CNN -\> RNN -\> Transformer](#cnn---rnn---transformer)
    - [LLM增强技术： 从Prompt Engineering 到 Agent](#llm增强技术-从prompt-engineering-到-agent)
    - [区分：量化 vs 蒸馏 vs 训练](#区分量化-vs-蒸馏-vs-训练)
    - [模型蒸馏：学习“学霸”的思考过程.](#模型蒸馏学习学霸的思考过程)
    - [量化 (Quantization)](#量化-quantization)
    - [Llama2 7B vs 70B: 独立训练](#llama2-7b-vs-70b-独立训练)
    - [模型里面有什么？](#模型里面有什么)
  - [LLM 推理引擎(serving engine), eg. vLLM](#llm-推理引擎serving-engine-eg-vllm)
    - [vLLM](#vllm)

## 理解AI发展史

### 从规则学习 -> 机器学习 -> 深度学习(CNN,RNN)

- 规则学习: eg. 定义规则来做邮件分类
- 机器学习: 有监督学习，eg.图片小猫小狗判断；无监督学习，eg. 聚类分析。
  - eg.在聚类分析里，单层感知机对于非线性可分的问题，无法给出一个明确的划分规则。引入隐藏层，来做拉点的操作，做成多层感知机。
  - 多层网络，调整权重，来实现分类/预测。
  - BP网络（Back-Propagation Network）算法来优化/加速 调整权重。
  - 随着网络层数增多，可能造成梯度消失或梯度爆炸。
- 深度学习：初为 深度信念网络，新模式：预训练+微调。
  - 逐层来做预训练，最后整体微调。note。此前是直接做整体的权重调整，没有预训练。
  - CNN，应运而生，AlexNet 图像识别。

https://www.bilibili.com/video/BV1vZHmzvE6D

### CNN -> RNN -> Transformer

- CNN卷积神经网络：
  - 卷积核，探测器，匹配局部特征；池化，压缩特征。
  - 缺点：CNN is for 图片。not work for 序列。
- RNN循环神经网络：
  - 依序列，逐字计算 隐藏状态 。早期NLP
  - 缺点：速度慢，无法并行；记性差，长距离依赖问题，梯度爆炸/梯度消失。
- Transformer： Attention is all you need.
  - 注意力机制，抓重点。支持并行计算。 +位置编码。
  - Encoder+Decoder.

- Bert: on Encoder：双向理解，擅长 文本分类，关键词提取，语义匹配；eg. 论文查重；
- GPT：on Decoder：单向注意力。预训练+微调。

### LLM增强技术： 从Prompt Engineering 到 Agent

- Prompt Engineering:
  - 标准提示：背景/角色/指令/要求
  - Few Shot： 给几个例子
  - CoT(Chain of Thought) / ToT(Tree of Thought)
- RAG：
  - 朴素RAG：索引，检索，to添加相关上下文给Prompt
  - 进阶RAG： 文档增强/查询转换,  重排序检索结果，长文档总结提炼。
  - 模块化RAG： 按需组装。
- Fine-tune：
  - 给通用大模型做“专业”向微调
  - LoRA （Low-Rank Adaptation）
  - QLoRA （Quantazation + LoRA）
- Agent:
  - 自主规划
  - 调用工具
  - 记忆fank
  - 多步执行

https://www.bilibili.com/video/BV1NHmsBwEbT


### 区分：量化 vs 蒸馏 vs 训练
操作	改变参数量？	改变架构？	核心逻辑
从头训练	✅ 是	✅ 是	白纸一张，喂数据，学规律
模型蒸馏	✅ 是	✅ 是	小模型学习大模型的思维过程
量化	❌ 否	❌ 否	仅压缩数字精度，参数总数不变

### 模型蒸馏：学习“学霸”的思考过程.

模型蒸馏步骤：
- 选定“教师”与“学生”：选教师，eg. DeepSeek-R1 ; 选定一个或多个体积更小的开源模型作为“学生”，eg.Qwen 或 Llama 系列的不同参数版本（1.5B, 7B, 8B, 14B等).
- 生成“教材”： “教师”模型针对大量的输入问题，不仅给出最终答案，更重要的是，它会生成详细的推理过程数据
- “学生”学习：训练或微调。这个过程的目标是让“学生”模型不仅学习预测正确答案，更重要的是模仿“教师”的推理逻辑。

举例：
- deepseek-r1:8b 和 deepseek-r1:1.5b 都是通过模型蒸馏（Knowledge Distillation） 技术创造的 “学生模型”， 它们的“老师”是 DeepSeek 研发的、拥有 6710 亿参数的完整版 DeepSeek-R1 模型。
- 8B/1.5B 等“学生模型”学习完整版模型（671B）“老师”的**推理过程**，而非直接复制其参数。


### 量化 (Quantization)

量化是一种对模型做“物理压缩”的技术，它不改变模型的结构（层数、参数量），只改变每个参数的存储精度。

一个极简例子：从 FP16 到 INT4
假设模型里有 3 个权重参数，它们的原始数值（FP16 格式）是：[ 0.52, -1.34, 2.78 ]

- 第一步：找到数值范围
找出最大值和最小值： 最大值 = 2.78 ； 最小值 = -1.34

- 第二步：映射到 4-bit 能表示的整数范围
4-bit 可以表示 2^4 = 16 个整数，通常是 [0, 15] 或 [-8, 7]。我们选用 [0, 15]。
我们需要把 [-1.34, 2.78] 这个区间，均匀地划分成 15 个间隔，映射到 [0, 15] 这 16 个刻度上。
计算每个刻度代表的步长（Scale）：
区间宽度 = 2.78 - (-1.34) = 4.12
步长 = 4.12 / 15 ≈ 0.2747

- 第三步：将原始数值转化为整数代号（Quantize）
公式：量化值 = round( (原始值 - 最小值) / 步长 )
0.52 → round( (0.52 - (-1.34)) / 0.2747 ) = round( 6.77 ) → 7
至此，三个高精度的浮点数 [0.52, -1.34, 2.78] 被压缩成了三个 4-bit 整数：[7, 0, 15]。

存储空间对比：
原始 FP16：3 × 16 bits = 48 bits
量化 INT4：3 × 4 bits = 12 bits

压缩率高达 75%，这就是 4-bit 量化能将 16GB 模型变成 4GB 的根本原因。

- 第四步：使用时还原（Dequantize）
当你运行模型时，程序会悄悄把整数代号还原成近似的浮点数进行计算：
还原值 ≈ 最小值 + 量化值 × 步长
7 → -1.34 + 7 × 0.2747 ≈ 0.58（原值 0.52，误差 +0.06）
可以看到，还原后的数值非常接近原值。这种微小的精度损失在数十亿参数协同工作时，对最终生成文本的影响微乎其微，却换来了能跑在 8GB 显存上的巨大红利。

Note. 现实中的量化会稍微复杂一点
真实的模型量化不会只用一个全局步长，通常会采用分组量化（Block-wise Quantization）：把每 64 个或 128 个参数分为一组，每组单独计算自己的步长。这样能让精度保留得更细腻。

### Llama2 7B vs 70B: 独立训练

Meta的 Llama 2 7B 并非由 70B 模型量化而来。它们是参数规模不同，但各自从零开始、独立训练的模型。二者的核心区别如下：

- 参数规模：7B为70亿参数，70B为700亿参数，相差10倍。
- 训练资源：独立训练，7B消耗约18.4万GPU小时，70B消耗约172万GPU小时。
- 架构：70B使用了更高效的GQA（分组查询注意力）技术来提升推理速度，7B则未使用


### 模型里面有什么？
~~~
$ ollama show deepseek-r1:8b
  Model
    architecture        qwen3
    parameters          8.2B
    context length      131072
    embedding length    4096
    quantization        Q4_K_M

  Capabilities
    completion
    thinking

  Parameters
    stop           "<｜begin▁of▁sentence｜>"
    stop           "<｜end▁of▁sentence｜>"
    stop           "<｜User｜>"
    stop           "<｜Assistant｜>"
    temperature    0.6
    top_p          0.95

  License
    MIT License
    Copyright (c) 2023 DeepSeek
    ...
~~~



## LLM 推理引擎(serving engine), eg. vLLM

国内外AI厂商通常围绕自有模型构建技术栈，其推理引擎策略分为两类：一是基于开源引擎二次开发，二是全自研。

### vLLM

1. vLLM 是什么？
一句话定义：vLLM 是一个高吞吐量、低内存占用的 LLM 推理和服务引擎，由 UC Berkeley 开发。

把它类比成 Web 服务器就懂了：
- Flask / Express：处理 HTTP 请求，返回 HTML。
- vLLM：处理 Prompt 请求，返回 Token 流。

它最革命性的东西是 PagedAttention（分页注意力），这解决了之前大家用 PyTorch/HuggingFace 跑模型时显存浪费严重、并发能力差的问题。

2. vLLM 解决了什么具体痛点?
“Transformers 原生推理的经典内存陷阱” in 生产环境中一定会遇到的.
这个问题本质上是 静态内存预分配策略 与 动态生成长度不确定性 之间的根本矛盾。

传统方式 (HuggingFace + FastAPI)

这就是为什么你现在看到几乎所有的大模型云服务（AWS、阿里云、腾讯云），底层推理引擎都从原生 PyTorch 迁移到了 vLLM 或 SGLang。

3. vLLM 是怎么工作的？（核心机制 + 简单示例）
vLLM 的秘密武器是 PagedAttention，灵感来源于操作系统的虚拟内存分页。

传统方法的 KV Cache：像 C 语言的 Malloc
- 预分配大块连续内存：每个请求来了，显存里划一块长条形的区域存 KV Cache。
- 内部碎片：对话很短，后面的空间全空着不能用。
- 外部碎片：多来几个请求，剩下的显存东一块西一块，没法塞进大请求。

vLLM 的 PagedAttention：像操作系统的虚拟内存页表
- 把 KV Cache 切分成固定大小的 Block（比如每块 16 个 Token）。
- 请求来了，需要几个 Block 就给几个，不够了再动态申请。

结果：显存利用率从 30% 飙升到 >90%。

4. 总结
vLLM 是什么：LLM 领域的高性能 Nginx / CDN。

它解决了什么：显存浪费和并发能力弱鸡的问题。

它是怎么工作的：像操作系统管理内存一样管理 KV Cache（PagedAttention），配合 Continuous Batching（连续批处理）把 GPU 的矩阵计算单元塞满。

如果你现在要上线一个 LLM 应用，不用 vLLM（或同类引擎如 TensorRT-LLM），成本会贵一个数量级。

Ollama：装好轮子的滑板车（个人玩具级部署）。
vLLM：给轮子工厂配的全自动流水线（生产级高并发服务）
