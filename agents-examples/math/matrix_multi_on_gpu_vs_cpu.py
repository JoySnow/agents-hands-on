import torch
import time

def benchmark_matmul(device_name, size, num_runs=10):
    print(f"\n[{device_name.upper()} 测试启动] ==========================")
    device = torch.device(device_name)

    # 1. 初始化张量
    # 注意：这里我们明确指定 device，把数据分配给 CPU 或 GPU
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    print(f"✅ 成功在 {device_name.upper()} 上分配 {size}x{size} 的矩阵")

    # 2. 预热 (Warm-up)
    # CPU 也需要预热来填满高速缓存(Cache)，让操作系统把频率拉到最高
    print("🔥 正在预热硬件...")
    for _ in range(5):
        _ = a @ b

    if device_name == "mps":
        torch.mps.synchronize()

    # 3. 正式测速
    print("⏱️ 开始正式计时...")
    start_time = time.perf_counter()

    for _ in range(num_runs):
        c = a @ b

    # 同步操作：如果是 GPU，必须等待它算完
    if device_name == "mps":
        torch.mps.synchronize()

    end_time = time.perf_counter()

    # 4. 计算结果
    avg_time = (end_time - start_time) / num_runs * 1000
    print(f"🏆 {device_name.upper()} 平均单次耗时: {avg_time:.2f} 毫秒")
    return avg_time

# ==========================================
# 主程序运行区
# ==========================================
if __name__ == "__main__":
    # 定义矩阵大小，4096 已经足够让差距显现出来
    MATRIX_SIZE = 8192

    # 测试 CPU
    cpu_time = benchmark_matmul("cpu", MATRIX_SIZE)

    # 测试 GPU (如果支持 MPS)
    if torch.backends.mps.is_available():
        gpu_time = benchmark_matmul("mps", MATRIX_SIZE)

        # 终极对比
        print("\n=============================================")
        print(f"📊 战报统计 (矩阵大小: {MATRIX_SIZE}x{MATRIX_SIZE})")
        print(f"-> CPU 耗时: {cpu_time:.2f} 毫秒")
        print(f"-> GPU 耗时: {gpu_time:.2f} 毫秒")
        print(f"🚀 性能差距: GPU 比 CPU 快了 {cpu_time / gpu_time:.2f} 倍！")
        print("=============================================")
    else:
        print("\n❌ 抱歉，未检测到 MPS (Apple Silicon GPU) 环境。")


"""
$ python math/matrix_multi_on_gpu_vs_cpu.py

[CPU 测试启动] ==========================
✅ 成功在 CPU 上分配 8192x8192 的矩阵
🔥 正在预热硬件...
⏱️ 开始正式计时...
🏆 CPU 平均单次耗时: 778.85 毫秒

[MPS 测试启动] ==========================
✅ 成功在 MPS 上分配 8192x8192 的矩阵
🔥 正在预热硬件...
⏱️ 开始正式计时...
🏆 MPS 平均单次耗时: 235.93 毫秒

=============================================
📊 战报统计 (矩阵大小: 8192x8192)
-> CPU 耗时: 778.85 毫秒
-> GPU 耗时: 235.93 毫秒
🚀 性能差距: GPU 比 CPU 快了 3.30 倍！
=============================================
"""