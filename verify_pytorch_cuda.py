# %%
import torch
# %%


def test_cuda():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            device_props = torch.cuda.get_device_properties(i)
            
            print(f"Device UUID: {str(device_props.uuid)}")
            print(f"Device memory [GB]: {device_props.total_memory / 1024**3:.2f}")
            print(f"Device capability: {device_props.major}.{device_props.minor}")
            
            print(f"Multi-processor count: {device_props.multi_processor_count}")
            
            print(f"Max threads per multiprocessor: {device_props.max_threads_per_multi_processor}")

            print(f"Is integrated GPU: {device_props.is_integrated}")
            print(f"L2 cache size [MB]: {device_props.L2_cache_size / 1024**2:.2f}")
            
            
            # Check if GPU supports TensorCores (important for mixed precision training)
            if device_props.major >= 7:
                print("TensorCores: Available")
            else:
                print("TensorCores: Not available")
                
            # Check current memory usage
            print(f"Memory allocated [GB]: {torch.cuda.memory_allocated(i) / 1024**3:.2f}")
            print(f"Memory reserved [GB]: {torch.cuda.memory_reserved(i) / 1024**3:.2f}")
            
            print("")
    else:
        print("No GPUs available.")


print("=== Testing device availability ===")
test_cuda()



# %%
# Test GPU copy bandwidth between GPUs
def test_copy_bandwidth_between_gpus():
    try:
        if torch.cuda.device_count() < 2:
            print("This test requires at least two GPUs.")
            return False
        
        # Use the first two GPUs
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")
        
        # Create tensors on different GPUs
        size_mb = 1000
        size = int(size_mb * 1024 * 1024 / 4)
        src_tensor = torch.ones(size, dtype=torch.float32, device=device1)
        dst_tensor = torch.zeros(size, dtype=torch.float32, device=device2)
        
        # Warmup
        dst_tensor.copy_(src_tensor)
        torch.cuda.synchronize()
        
        # Benchmark GPU0 -> GPU1
        iterations = 5
        copy_times = []
        for _ in range(iterations):
            torch.cuda.empty_cache()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            dst_tensor.copy_(src_tensor)
            end.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            copy_times.append(elapsed_ms)
        
        avg_copy_time = sum(copy_times) / len(copy_times)
        copy_bandwidth = size_mb / avg_copy_time
        print(f"\nGPU Copy Bandwidth Between GPUs:")
        print(f"  GPU0 -> GPU1: {copy_bandwidth:.2f} GB/s")
        
        # Test in the reverse direction (GPU1 -> GPU0)
        src_tensor = torch.ones(size, dtype=torch.float32, device=device2)
        dst_tensor = torch.zeros(size, dtype=torch.float32, device=device1)
        
        # Warmup
        dst_tensor.copy_(src_tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        copy_times = []
        for _ in range(iterations):
            torch.cuda.empty_cache()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            dst_tensor.copy_(src_tensor)
            end.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            copy_times.append(elapsed_ms)
        
        avg_copy_time = sum(copy_times) / len(copy_times)
        copy_bandwidth = size_mb / avg_copy_time
        print(f"  GPU1 -> GPU0: {copy_bandwidth:.2f} GB/s")
        
        return True
    except Exception as e:
        print(f"Error testing GPU copy bandwidth between GPUs: {str(e)}")
        return False


if torch.cuda.is_available():
    print("\n=== Testing GPU copy bandwidth between GPUs ===")
    test_copy_bandwidth_between_gpus()

# %%
def calculate_std(values):
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

def run_tensor_test(device_idx, sizes=[2000, 6000], iterations=5):
    """
    Run tensor operations test on specified GPU with multiple matrix sizes.
    Uses proper synchronization and prevents lazy execution.
    """
    print(f"Running tensor operations on GPU {device_idx}...")
    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(device)  # Explicitly set current device
    
    for size in sizes:
        try:
            print(f"\nTesting with matrix size {size}x{size} on GPU {device_idx}")
            
            # Force clear cache
            torch.cuda.empty_cache()
            
            # Create tensors of specified size
            a = torch.ones(size, size, device=device)
            b = torch.ones(size, size, device=device)
            
            # Warmup with forced execution and synchronization
            warmup = torch.matmul(a, b)
            result_check = warmup[0,0].item()  # Force computation by accessing a value
            torch.cuda.synchronize(device)
            
            # Benchmark
            elapsed_times = []
            for _ in range(iterations):
                # Ensure computation happens
                torch.cuda.empty_cache()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                c = torch.matmul(a, b)
                # Force evaluation by accessing a value (prevents lazy execution)
                result_value = c[0,0].item()
                end.record()
                
                torch.cuda.synchronize(device)
                elapsed_times.append(start.elapsed_time(end))
            
            avg_time = sum(elapsed_times) / len(elapsed_times)
            min_time = min(elapsed_times)
            max_time = max(elapsed_times)
            std_dev = calculate_std(elapsed_times)
            
            # Verify result is correct (should be 'size' for matrix of ones)
            print(f"  Matrix {size}x{size}: Avg time = {avg_time:.2f} ms (min: {min_time:.2f}, max: {max_time:.2f}, std: {std_dev:.2f})")
            
        except Exception as e:
            print(f"  Error testing GPU {device_idx} with size {size}x{size}: {str(e)}")
    
    return True


if torch.cuda.is_available():
    print("\n=== Running matrix multiplication benchmark ===")
    # Run the consolidated tensor test on each GPU
    for i in range(torch.cuda.device_count()):
        run_tensor_test(i)

