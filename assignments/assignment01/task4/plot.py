import matplotlib.pyplot as plt

# Sizes
sizes = [100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000]

# Times in µs
cpu_times = [47, 410, 1645, 6807, 14520, 25886, 42974, 139142, 210516]
gpu_times = [553.5, 1812.2, 4879.6, 18032.3, 45297.9, 69967.8, 107977.2, 275755.5, 576165.6]

plt.figure(figsize=(8, 6))
plt.plot(sizes, cpu_times, label='CPU', marker='o')
plt.plot(sizes, gpu_times, label='GPU', marker='s')
plt.xlabel('Matrix Size')
plt.ylabel('Time (µs)')
plt.title('CPU vs GPU Matrix Addition')
plt.legend()
plt.grid(True)
plt.savefig("cpu-vs-gpu.png")