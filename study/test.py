import torch.multiprocessing as mp

mp.set_start_method('spawn')

ctx = mp.get_context("gpu")
device_id = 0
device = ctx._devices[device_id]
print(f"Total memory on device {device_id}: {device.total_memory}")
print(f"Allocated memory on device {device_id}: {device.memory_allocated}")
print(f"Reserved memory on device {device_id}: {device.memory_reserved}")