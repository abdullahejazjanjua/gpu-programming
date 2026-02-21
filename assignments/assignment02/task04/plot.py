import matplotlib.pyplot as plt
import numpy as np

sizes = ['128x128', '256x256', '512x512', '1024x1024', '128x512', '512x256', '256x1024', '1024x256', '100x200', '500x1000', '200x500', '2048x512', '4096x1024', '300x300', '600x900']
cpu_times = [1685, 14647, 158732, 2852273, 14798, 13857, 356954, 111118, 8170, 87230, 87989, 2822630, 1283149, 45033, 144291]
gpu_naive_times = [149.5, 427.5, 1832.5, 8070.8, 408.7, 307.5, 1301.5, 1315.3, 223.0, 1186.4, 1069.3, 9317.4, 7546.1, 642.5, 1531.8]
gpu_tiled_times = [132.5, 418.8, 1656.7, 7028.2, 308.5, 289.0, 1339.7, 1526.8, 237.6, 1053.4, 1012.6, 8512.3, 7312.3, 619.0, 1411.9]

plt.figure(figsize=(10,6))
plt.plot(sizes, np.log(cpu_times), label='CPU', marker='o')
plt.plot(sizes, np.log(gpu_naive_times), label='GPU Naive', marker='s')
plt.plot(sizes, np.log(gpu_tiled_times), label='GPU Tiled', marker='^')

plt.xlabel('Matrix Size')
plt.ylabel('Time (microseconds)')
plt.title('CPU vs GPU Naive vs GPU Tiled')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("comp.png")
