import torch
import time

# ==========================================
# 阶段 1：环境检测与设备指派
# ==========================================
print("--- 阶段 1：环境检测 ---")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 成功检测到 Apple Silicon MPS (Metal) 后端！任务将交给 M3 Pro 的 GPU 核心。")
else:
    device = torch.device("cpu")
    print("❌ 未检测到 MPS，将回退到 CPU。")

# ==========================================
# 阶段 2：数据分配与内存观察
# ==========================================
print("\n--- 阶段 2：数据分配与内存观察 ---")
# 我们把矩阵稍微设大一点，比如 4096 x 4096，以便更明显地观察耗时
N = 4096

# 清理历史缓存，获取初始内存状态 (仅针对 MPS 测量的 API)
torch.mps.empty_cache()
mem_before = torch.mps.current_allocated_memory() / (1024 ** 2)

# 在 MPS 设备上初始化数据
a = torch.randn(N, N, device=device)
b = torch.randn(N, N, device=device)

mem_after = torch.mps.current_allocated_memory() / (1024 ** 2)
print(f"创建两个 {N}x{N} 的矩阵后，MPS 视角下的已分配内存增加了约: {mem_after - mem_before:.2f} MB")
print("💡 提示：在 M3 Pro 上，由于是统一内存，这其实并没有发生传统的 CPU 到 GPU 的物理搬运，只是把内存的控制权交给了 Metal 接口。")

# ==========================================
# 阶段 3：底层预热 (Warm-up)
# ==========================================
print("\n--- 阶段 3：硬件预热 ---")
# ⚠️ 极其重要：第一次调用 GPU/MPS 时，系统需要花费额外时间去编译 Kernel（图纸）、初始化上下文。
# 如果直接计时，会把这些准备时间也算进去，导致结果不准。我们需要先让它“空跑”几次。
for i in range(5):
    _ = a @ b

# 确保预热任务真正执行完毕
torch.mps.synchronize()
print("✅ 预热完毕，GPU 核心已进入战斗状态。")

# ==========================================
# 阶段 4：正式执行与异步计时
# ==========================================
print("\n--- 阶段 4：正式执行与计时 ---")
num_runs = 10  # 跑 10 次取平均值

start_time = time.perf_counter()

for _ in range(num_runs):
    c = a @ b

# ⚠️ 极其重要：异步执行的同步锁
# PyTorch 的 GPU 执行是异步的。如果不加下面这行代码，Python (老板) 下达完 10 次指令后就会立刻停止计时，
# 此时 GPU (工人) 实际上还在后台拼命算。
# synchronize() 的作用是强迫 Python 老板停下来，直到 GPU 报告“全部算完了”再继续往下走。
torch.mps.synchronize()

end_time = time.perf_counter()

avg_time_ms = (end_time - start_time) / num_runs * 1000
print(f"🎉 执行完毕！")
print(f"👉 矩阵 c 的形状: {c.shape}")
print(f"👉 M3 Pro 平均每次计算耗时: {avg_time_ms:.2f} 毫秒")

"""
$ python math/matrix_multiplitaion_on_m3.py
--- 阶段 1：环境检测 ---
✅ 成功检测到 Apple Silicon MPS (Metal) 后端！任务将交给 M3 Pro 的 GPU 核心。

--- 阶段 2：数据分配与内存观察 ---
创建两个 4096x4096 的矩阵后，MPS 视角下的已分配内存增加了约: 128.00 MB
💡 提示：在 M3 Pro 上，由于是统一内存，这其实并没有发生传统的 CPU 到 GPU 的物理搬运，只是把内存的控制权交给了 Metal 接口。

--- 阶段 3：硬件预热 ---
✅ 预热完毕，GPU 核心已进入战斗状态。

--- 阶段 4：正式执行与计时 ---
🎉 执行完毕！
👉 矩阵 c 的形状: torch.Size([4096, 4096])
👉 M3 Pro 平均每次计算耗时: 26.49 毫秒
"""