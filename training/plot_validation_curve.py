import matplotlib.pyplot as plt
import re
import os

# Function to parse eval losses from a file
def parse_eval_losses(file_path):
    eval_losses = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r"Eval Loss: ([\d.]+)", line)
            if match:
                eval_loss = float(match.group(1))
                eval_losses.append(eval_loss)
    return eval_losses[:10]

# Function to plot eval losses
def plot_eval_losses(eval_losses):
    steps = [i * 500 for i in range(len(eval_losses))]  # Assuming each eval loss indicates 500 batches processed
    plt.plot(steps, eval_losses, linestyle='-', color='b')
    plt.xlabel('Steps')
    plt.ylabel('Eval Loss')
    plt.title('Evaluation Loss vs. Steps')
    plt.grid(True)
    os.makedirs("validation", exist_ok=True)
    plt.savefig(os.path.join('validation', 'eval_losses_plot_FLAVA_slurm-55590092.png'))


# Assuming your file is named 'your_file.txt'
file_path = "/home/jmonas/acceltran/training/FLAVA/slurm-55590092.out"
eval_losses = parse_eval_losses(file_path)
plot_eval_losses(eval_losses)
