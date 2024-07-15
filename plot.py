from matplotlib import pyplot as plt
import torch


def plot(net,title=None):
  # Create a 2x2 grid of subplots
  fig, axs = plt.subplots(2, 2, figsize=(12, 8))
  fig.suptitle(title or "Plot", fontsize=16)

  # Plot 1 - Voltage and Current
  axs[0, 0].plot(net["u", 0].cpu(),label='Voltage')
  axs[0, 0].plot(net["I", 0].cpu() -torch.max(net["I", 0]) - 80, linestyle='--', label='Current')  # Adjust linestyle for current
  axs[0, 0].set_xlabel("time")
  axs[0, 0].legend(loc='lower right')


  # Plot 2 - Current
  axs[0, 1].plot(net["I", 0].cpu())
  axs[0, 1].set_ylabel("Current")
  axs[0, 1].set_xlabel("time")

  # Plot 3 - W
  axs[1, 0].plot(net["w", 0].cpu())
  axs[1, 0].set_ylabel("W")
  axs[1, 0].set_xlabel("time")

  # Plot 4 - Spike
  axs[1, 1].scatter(net["spike", 0][:, 0].cpu(), net["spike", 0][:, 1].cpu())
  axs[1, 1].set_ylabel("spike")
  axs[1, 1].set_xlabel("time")

  # Adjust layout to prevent overlap
  plt.tight_layout()

  # Show the plot
  plt.show()
    