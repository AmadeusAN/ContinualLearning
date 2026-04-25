"""
Author: cjh 2514642620@qq.com
Date: 2025-09-18 09:53:55
LastEditors: cjh 2514642620@qq.com
LastEditTime: 2026-03-30 14:41:26
Description: GPU 显存占用 + 利用率模拟
"""

import torch
import time
import os
import random
import threading


def gpu_busy_worker(gpu_id, stop_event, target_util=0.5):
    """
    后台线程：在指定 GPU 上跑矩阵运算，模拟真实负载。

    Args:
        gpu_id (int): GPU ID
        stop_event (threading.Event): 停止信号
        target_util (float): 目标利用率 0.0~1.0
                             通过调整 计算/睡眠 的比例来近似控制
    """
    device = f"cuda:{gpu_id}"
    # 矩阵大小影响单次计算的耗时，可调节
    size = 2048

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # 计算时间和睡眠时间的比例来模拟不同利用率
    # target_util=0.5 → 算一会儿歇一会儿，看起来 ~50%
    compute_time = 0.05  # 每轮计算持续时间(秒)
    if target_util >= 0.99:
        sleep_time = 0.0
    elif target_util <= 0.01:
        sleep_time = 1.0
    else:
        sleep_time = compute_time * (1.0 - target_util) / target_util

    print(f"GPU {gpu_id}: 后台计算线程启动 (目标利用率 ~{int(target_util * 100)}%)")

    while not stop_event.is_set():
        # --- 计算阶段 ---
        end_time = time.time() + compute_time
        while time.time() < end_time and not stop_event.is_set():
            # 矩阵乘法，真正吃 GPU 算力
            c = torch.mm(a, b)
            # 偶尔加点别的操作，让 profile 看起来更真实
            c = torch.relu(c)
            c = c + a

        # --- 休眠阶段（控制利用率不要100%顶满）---
        if sleep_time > 0:
            time.sleep(sleep_time)

    # 清理
    del a, b
    torch.cuda.empty_cache()
    print(f"GPU {gpu_id}: 后台计算线程已停止")


# ============ 全局状态 ============
occupied_memory = {}
worker_threads = {}  # gpu_id -> (thread, stop_event)


def occupy_gpu_memory(gpu_id, memory_ratio, target_util=0.5):
    """
    占用指定 GPU 显存 + 启动后台计算线程模拟利用率
    """
    global occupied_memory, worker_threads

    if gpu_id in occupied_memory:
        print(f"GPU {gpu_id}: 已占用显存。")
        return

    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    memory_to_occupy = int(total_memory * memory_ratio)

    # 占显存
    occupied_tensor = torch.empty(
        (memory_to_occupy // 4), dtype=torch.float, device=f"cuda:{gpu_id}"
    )
    occupied_memory[gpu_id] = occupied_tensor
    print(f"GPU {gpu_id}: 占用 {memory_to_occupy // (1024**2)} MB 显存。")

    # 启动后台计算线程
    stop_event = threading.Event()
    t = threading.Thread(
        target=gpu_busy_worker, args=(gpu_id, stop_event, target_util), daemon=True
    )
    t.start()
    worker_threads[gpu_id] = (t, stop_event)


def release_gpu_memory():
    """释放所有已占用的 GPU 显存并停止计算线程"""
    global occupied_memory, worker_threads

    # 先停计算线程
    for gpu_id in list(worker_threads.keys()):
        t, stop_event = worker_threads[gpu_id]
        stop_event.set()
        t.join(timeout=3)
        del worker_threads[gpu_id]

    # 再释放显存
    for gpu_id in list(occupied_memory.keys()):
        del occupied_memory[gpu_id]
        torch.cuda.empty_cache()
        print(f"GPU {gpu_id}: 显存已释放。")
    occupied_memory = {}


if __name__ == "__main__":
    print("=" * 60)
    print("GPU 占用工具（显存 + 利用率模拟）")
    print("=" * 60)
    print("命令:")
    print("  <gpu_id>           - 占用指定 GPU (随机显存 + 随机利用率)")
    print("  <gpu_id> <util%>   - 占用指定 GPU，指定目标利用率")
    print("                       例如: 3 70  → GPU3, ~70% 利用率")
    print("  q                  - 释放所有 GPU")
    print("  Ctrl+C             - 退出")
    print("=" * 60)

    try:
        while True:
            user_input = input("\n请输入: ").strip()

            if user_input.lower() == "q":
                release_gpu_memory()
                print("所有显存已释放，可以重新占用。")
                continue

            parts = user_input.split()
            try:
                gpu_id = int(parts[0])

                # 解析目标利用率
                if len(parts) >= 2:
                    target_util = float(parts[1]) / 100.0
                    target_util = max(0.05, min(0.99, target_util))
                else:
                    target_util = random.uniform(0.3, 0.8)

                memory_ratio = random.uniform(0.7, 0.9)
                occupy_gpu_memory(gpu_id, memory_ratio, target_util)

            except (ValueError, IndexError):
                print("无效输入。示例: '3' 或 '3 70' 或 'q'")
            except Exception as e:
                print(f"错误: {e}")

    except KeyboardInterrupt:
        print("\n程序中断，释放显存...")
        release_gpu_memory()
        print("完成。")
