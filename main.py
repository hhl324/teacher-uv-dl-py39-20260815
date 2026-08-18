"""
PyTorch 安装验证脚本（精简版）
检查 PyTorch 安装和 CUDA 配置

PyTorch 1.8.1 + CUDA 10.2 + NumPy 1.26.4

"""
import torch
import time


def check_pytorch():
    """精简版 PyTorch 环境检查"""

    print("=" * 50)
    print("PyTorch 环境检查")
    print("=" * 50)



    # 2. PyTorch 信息
    print(f"\n📦 PyTorch: {torch.__version__}")

    # 3. CUDA 检查
    cuda_available = torch.cuda.is_available()
    print(f"\n🖥️  CUDA 可用: {'✅ 是' if cuda_available else '❌ 否'}")

    if cuda_available:
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1024 ** 3:.1f} GB)")

    # 4. 简单运算测试
    print("\n🧪 运算测试:")
    try:
        # CPU 测试
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        print(f"   ✅ CPU: {x.numpy().tolist()} -> {y.numpy().tolist()}")

        # GPU 测试
        if cuda_available:
            x_gpu = x.cuda()
            y_gpu = x_gpu * 2
            print(f"   ✅ GPU: {x_gpu.cpu().numpy().tolist()} -> {y_gpu.cpu().numpy().tolist()}")
    except Exception as e:
        print(f"   ❌ 运算失败: {e}")

    # 5. 简单性能测试
    print("\n⚡ 性能测试 (1000x1000 矩阵乘法):")
    size = 1000

    # CPU
    start = time.time()
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    c = torch.mm(a, b)
    cpu_time = time.time() - start
    print(f"   CPU: {cpu_time:.3f}s")

    # GPU
    if cuda_available:
        torch.cuda.synchronize()
        start = time.time()
        a_gpu = torch.randn(size, size).cuda()
        b_gpu = torch.randn(size, size).cuda()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"   GPU: {gpu_time:.3f}s")
        print(f"   加速比: {cpu_time / gpu_time:.1f}x")

    # 6. 神经网络测试
    print("\n🧠 神经网络测试:")
    try:
        import torch.nn as nn
        model = nn.Linear(10, 5)
        if cuda_available:
            model = model.cuda()
        x = torch.randn(3, 10)
        if cuda_available:
            x = x.cuda()
        out = model(x)
        print(f"   ✅ 前向传播成功: {x.shape} -> {out.shape}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")

    # 7. 总结
    print("\n" + "=" * 50)
    if cuda_available:
        print("✅ 环境正常，GPU 可用，可以开始深度学习！")
    else:
        print("⚠️  GPU 不可用，将使用 CPU 进行运算")
    print("=" * 50)


if __name__ == "__main__":
    check_pytorch()