
## Train on Single GPU V.S. DDP (Distributed Data Parallel)


### 单卡训练 (Single GPU Training)
慢。

受限于单张显卡的算力（训练慢）和显存（无法训练超大模型）。

这是最基础的训练模式。在这个模式下，模型 (Model)、数据 (Data) 和 优化器状态 (Optimizer) 全部驻留在一个 GPU 的显存中。

### DDP: 分布式数据并行 (Distributed Data Parallel)

DDP 是 PyTorch 中最流行的大规模训练方式之一。

在 DDP 中，“数据是分布式的，但模型是复制的”。

工作流程：
- 复制 (Replicate)：系统将你完整的模型复制到每一张 GPU 上。
- 分发数据 (Scatter Data)：每一批全量数据（Global Batch）被切分成多个子批次（Mini-batches），分发给不同的 GPU。
- 独立计算 (Local Compute)：每张 GPU 独立地对自己分到的那一小批数据进行前向和反向传播，计算出局部的梯度。
- 同步与平均 (All-Reduce)：这是 DDP 的关键！所有 GPU 通过高速网络（如 NVLink 或 PCIe）相互通信，把各自算出的梯度加在一起求平均值。
- 同步更新 (Synchronized Update)：每张 GPU 使用相同的平均梯度去更新自己手里的模型参数。这样一轮下来，所有 GPU 上的模型依然保持完全一致。

瓶颈：
- on All-Reduce
- 当 GPU 的计算时间小于梯度的传输和同步时间时，GPU 就会处于“饥饿”或“闲置”状态，干等着数据传完——这就是典型的 I/O 瓶颈。
- AI 工程师们通常会使用像 Ring All-Reduce 这样的算法来优化网络拓扑，或者使用更高带宽的硬件（如 NVLink 甚至 InfiniBand 网络）。

