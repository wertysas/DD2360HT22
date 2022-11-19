
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def size_bw_from_index(lines, idx):
    transfer_sizes = []
    bandwidths = []
    print(lines[idx+1])
    print(lines[idx+2])
    print(lines[idx+3])
    for line in lines[idx+4:]:
        if line.strip() == "":
            break
        size, bandwidth = line.split()
        size = float(size)
        bandwidth = float(bandwidth)
        transfer_sizes.append(size)
        bandwidths.append(bandwidth)
    return np.array(transfer_sizes), np.array(bandwidths)


with open('data/shmoo.txt') as f:
    lines = f.readlines()


idxs = []
for i, line in enumerate(lines):
    if "."*20 in line:
        idxs.append(i)

bw_data = {}


# for idx in idxs:
#     transfer_sizes = []
#     bandwidths = []
#     print(lines[idx+1])
#     print(lines[idx+2])
#     print(lines[idx+3])
#     for line in lines[idx+4:]:
#         if line.strip() == "":
#             break
#         size, bandwidth = line.split()
#         size = int(size)
#         bandwidth = float(bandwidth)
#         transfer_sizes.append(size)
#         bandwidths.append(bandwidth)


rcParams.update({'font.size': 20})

# Host to Device
label = "Host to Device"
transfer_sizes, bandwidths = size_bw_from_index(lines, idxs[0])
transfer_sizes *= 1e-6
fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(transfer_sizes,
        bandwidths,
        marker="o",
        linestyle=":",
        linewidth=2.0,
        label=label)
plt.legend()
plt.xlabel("Transfer size (MB)")
plt.ylabel("Bandwidths (GB/s)")
fig.suptitle(label)
fig.tight_layout()
fig.savefig("figs/"+label+".pdf", format="pdf")
plt.show()

# Host to Device
label = "Device to Host"
transfer_sizes, bandwidths = size_bw_from_index(lines, idxs[1])
transfer_sizes *= 1e-6
fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(transfer_sizes,
        bandwidths,
        marker="o",
        linestyle=":",
        linewidth=2.0,
        label=label)
plt.legend()
plt.xlabel("Transfer size (MB)")
plt.ylabel("Bandwidths (GB/s)")
fig.suptitle(label)
fig.tight_layout()
fig.savefig("figs/"+label+".pdf", format="pdf")
plt.show()

# Device to Device
label = "Device to Device"
transfer_sizes, bandwidths = size_bw_from_index(lines, idxs[2])
transfer_sizes *= 1e-6
fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(transfer_sizes,
        bandwidths,
        marker="o",
        linestyle=":",
        linewidth=2.0,
        label=label)
plt.legend()
plt.xlabel("Transfer size (MB)")
plt.ylabel("Bandwidths (GB/s)")
fig.suptitle(label)
fig.tight_layout()
fig.savefig("figs/"+label+".pdf", format="pdf")
plt.show()
