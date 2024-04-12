import matplotlib.pyplot as plt
import re
import os

# # Function to parse eval losses from a file
# def parse_eval_losses(file_path):
#     eval_losses = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             match = re.search(r"Eval Loss: ([\d.]+)", line)
#             if match:
#                 eval_loss = float(match.group(1))
#                 eval_losses.append(eval_loss)
#     return eval_losses[:10]

# # Function to plot eval losses
# def plot_eval_losses(eval_losses):
#     steps = [i * 500 for i in range(len(eval_losses))]  # Assuming each eval loss indicates 500 batches processed
#     plt.plot(steps, eval_losses, linestyle='-', color='b')
#     plt.xlabel('Steps')
#     plt.ylabel('Eval Loss')
#     plt.title('Evaluation Loss vs. Steps')
#     plt.grid(True)
#     os.makedirs("validation", exist_ok=True)
#     plt.savefig(os.path.join('validation', 'eval_losses_plot_FLAVA_slurm-55590092.png'))


# # Assuming your file is named 'your_file.txt'
# file_path = "/home/jmonas/acceltran/training/FLAVA/slurm-55590092.out"
# eval_losses = parse_eval_losses(file_path)
# plot_eval_losses(eval_losses)



# Function to specifically parse training losses that start with a step number from a file
def parse_step_prefixed_training_losses(file_path):
    step_prefixed_losses = []
    steps = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'^(\d+), Loss: ([\d.]+)', line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                steps.append(step)
                step_prefixed_losses.append(loss)
    return steps, step_prefixed_losses

# Function to plot and save step-prefixed training losses without markers
def plot_step_prefixed_training_losses(steps, losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linestyle='-', color='b')  # No markers specified
    plt.xlabel('Steps')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Steps')
    plt.grid(True)
    
    # Ensure the folder exists
    os.makedirs(save_path, exist_ok=True)
    
    # Save the figure to the specified path
    plt.savefig(os.path.join(save_path, 'FLAVA_slurm-55590092_training_losses_plot.png'))
    
    # Optionally display the plot
    plt.show()

# Specify the path to your file and the folder where you want to save the plot
file_path = "/home/jmonas/acceltran/training/FLAVA/slurm-55590092.out"
save_path = 'training_plots'  # Folder where the plot will be saved

steps, step_prefixed_losses = parse_step_prefixed_training_losses(file_path)
plot_step_prefixed_training_losses(steps, step_prefixed_losses, save_path)
