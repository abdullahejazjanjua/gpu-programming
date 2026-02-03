import matplotlib.pyplot as plt

sizes = [100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000]
cpu_times = [47, 410, 1645, 6807, 14520, 25886, 42974, 139142, 210516]
gpu_total_times = [553.5, 1812.2, 4879.6, 18032.3, 45297.9, 69967.8, 107977.2, 275755.5, 576165.6]
gpu_kernel_times = [102.75, 59.04, 104.77, 294.37, 593.73, 848.06, 1316.42, 3218.66, 4988.93]


plt.figure(figsize=(10, 7))


plt.plot(sizes, cpu_times, label='CPU Total', marker='o', color='blue')
plt.plot(sizes, gpu_total_times, label='GPU Total (Transfer + Compute)', marker='s', color='red')
plt.plot(sizes, gpu_kernel_times, label='GPU Compute Only', marker='^', color='green', linestyle='--')


plt.xlabel('Matrix Size (N)')
plt.ylabel('Time (microseconds)')
plt.title('Performance Analysis: CPU vs GPU (Total vs Compute)')
plt.legend()
plt.grid(True)

plt.yscale('log') 

plt.savefig("cpu-vs-gpu.png")
print("Graph saved as cpu-vs-gpu.png")