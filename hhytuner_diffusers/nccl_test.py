import torch
import time
import os
import socket
import torch.distributed as dist
from accelerate import Accelerator

def create_data(size_gb=4):
    num_elements = int(size_gb * (1024**3) / 4)
    return torch.randn(num_elements, dtype=torch.float32)

def test_nccl_transfer(accelerator, size_gb=4, num_iterations=5):
    if accelerator.is_main_process:
        print(f"正在创建 {size_gb}GB 的数据...")
    
    data = create_data(size_gb)
    data = data.to(accelerator.device)
    
    # 获取正确的进程信息
    local_rank = accelerator.local_process_index
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 获取机器信息
    machine_rank = int(os.environ.get("MACHINE_RANK", "0"))
    num_machines = int(os.environ.get("NUM_MACHINES", "1"))
    processes_per_machine = world_size // num_machines
    
    # 使用当前设备进行barrier
    current_device = torch.cuda.current_device()
    dist.barrier(device_ids=[current_device])
    
    # 计算实际数据大小（字节）
    data_bytes = data.element_size() * data.nelement()
    data_gb = data_bytes / (1024**3)
    
    if accelerator.is_main_process:
        print("\n开始 NCCL 传输测试...")
        print(f"GPU 总数: {world_size}")
        print(f"机器数量: {num_machines}")
        print(f"每台机器GPU数: {processes_per_machine}")
        print(f"数据大小: {data_gb:.2f}GB ({data_bytes} 字节)")
        print(f"迭代次数: {num_iterations}")
        print("-" * 50)

    transfer_times = []
    
    for i in range(num_iterations):
        if accelerator.is_main_process:
            print(f"\n第 {i+1}/{num_iterations} 次迭代")
        
        # 使用当前设备进行barrier
        dist.barrier(device_ids=[current_device])
        torch.cuda.synchronize()
        
        start_time = time.time()
        
        # 执行all_reduce操作
        dist.all_reduce(data)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        transfer_time = end_time - start_time
        transfer_times.append(transfer_time)
        
        # 计算单次操作的实际数据传输量
        # 在all_reduce中，每个GPU需要发送和接收数据
        # 单个GPU的数据传输量
        single_gpu_bytes = data_bytes
        
        # 单次操作的总数据传输量 (所有GPU)
        # 在环形算法中，每个GPU发送和接收约2*(n-1)/n倍的数据
        total_bytes_ring = 2 * single_gpu_bytes * (world_size - 1)
        
        # 单次操作的有效带宽 (数据大小/时间)
        effective_bandwidth = data_gb / transfer_time
        
        # 总带宽 (所有GPU的总传输量/时间)
        total_bandwidth_ring = total_bytes_ring / (1024**3) / transfer_time
        
        # 机器间带宽 (只考虑跨机器通信)
        if num_machines > 1:
            # 每台机器需要发送的数据量
            inter_node_bytes = single_gpu_bytes * processes_per_machine * (num_machines - 1) / num_machines
            inter_node_bandwidth = inter_node_bytes / (1024**3) / transfer_time
        else:
            inter_node_bandwidth = 0
        
        if accelerator.is_main_process:
            print(f"传输时间: {transfer_time:.3f} 秒")
            print(f"单GPU有效带宽: {effective_bandwidth:.2f} GB/秒 (单个数据块传输速度)")
            print(f"总聚合带宽: {total_bandwidth_ring:.2f} GB/秒 (所有GPU总传输量/时间)")
            
            if num_machines > 1:
                print(f"机器间带宽: {inter_node_bandwidth:.2f} GB/秒")
    
    if accelerator.is_main_process and transfer_times:
        avg_time = sum(transfer_times)/len(transfer_times)
        
        # 计算平均带宽
        avg_effective_bandwidth = data_gb / avg_time
        avg_total_bandwidth = total_bytes_ring / (1024**3) / avg_time
        
        print("\n" + "=" * 50)
        print("最终结果:")
        print(f"平均传输时间: {avg_time:.3f} 秒")
        print(f"平均单GPU有效带宽: {avg_effective_bandwidth:.2f} GB/秒")
        print(f"平均总聚合带宽: {avg_total_bandwidth:.2f} GB/秒")
        
        if num_machines > 1:
            avg_inter_node_bandwidth = inter_node_bytes / (1024**3) / avg_time
            print(f"平均机器间带宽: {avg_inter_node_bandwidth:.2f} GB/秒")
        
        print("=" * 50)

def main():
    # 初始化accelerator
    accelerator = Accelerator()
    
    # 打印更详细的分布式信息以便调试
    hostname = socket.gethostname()
    
    # 计算每台机器上的进程数
    machine_rank = int(os.environ.get("MACHINE_RANK", "0"))
    num_machines = int(os.environ.get("NUM_MACHINES", "1"))
    world_size = dist.get_world_size()
    processes_per_machine = world_size // num_machines
    
    # 确保所有进程都打印自己的信息
    for i in range(dist.get_world_size()):
        if dist.get_rank() == i:
            print(f"[Rank {i} on {hostname}] Local rank: {accelerator.local_process_index}, Device: {torch.cuda.current_device()}")
            if i == 0:  # 只在主进程打印详细信息
                print(f"分布式信息:")
                print(f"- 主机名: {hostname}")
                print(f"- 机器总数: {num_machines}")
                print(f"- 当前机器rank: {machine_rank}")
                print(f"- 每台机器进程数: {processes_per_machine}")
                print(f"- 全局进程总数: {world_size}")
                print(f"- 当前进程全局rank: {dist.get_rank()}")
                print(f"- 当前进程本地rank: {accelerator.local_process_index}")
                print(f"- 当前设备: {torch.cuda.current_device()}")
                print(f"- 环境变量:")
                for var in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK']:
                    print(f"  {var}: {os.environ.get(var, 'Not set')}")
        
        # 使用当前设备进行barrier
        current_device = torch.cuda.current_device()
        dist.barrier(device_ids=[current_device])
    
    size_gb = 10 
    num_iterations = 4  
    
    try:
        test_nccl_transfer(accelerator, size_gb=size_gb, num_iterations=num_iterations)
    finally:
        # 使用当前设备进行barrier
        if dist.is_initialized():
            current_device = torch.cuda.current_device()
            dist.barrier(device_ids=[current_device])

if __name__ == "__main__":
    main() 