import wandb
import numpy as np
import matplotlib.pyplot as plt


def plot_actions(
    pred_actions,
    true_actions,
    file_name,
    act_dim_labels=["x", "y", "z", "yaw", "pitch", "roll", "grasp"],
):
    """
    Plots predicted vs. ground truth actions (7-dim) along with a corresponding image strip.
    Logs the plot to WandB.
    """
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplot_mosaic([act_dim_labels])
    fig.set_size_inches([40, 5])

    # Ensure proper input formatting for actions
    pred_actions = np.array(pred_actions).squeeze()  # Bx7
    true_actions = np.array(true_actions).squeeze()  # Bx7

    # Plot actions for each dimension
    for action_dim, action_label in enumerate(act_dim_labels):
        axs[action_label].plot(pred_actions[:, action_dim], label="Predicted")
        axs[action_label].plot(true_actions[:, action_dim], label="Ground Truth")
        axs[action_label].set_title(action_label)
        axs[action_label].set_xlabel("Time (steps)")
        axs[action_label].legend()

    plt.tight_layout()
    # wandb.log({wandb_title: wandb.Image(fig)})
    plt.savefig(f"{file_name}.png")
    # plt.close(fig)
    
def plot_actions_and_log_wandb(pred_actions, true_actions, wandb_title, epoch):
    """
    Plots predicted vs. ground truth actions (7-dim) along with a corresponding image strip.
    Logs the plot to WandB.
    """

    ACTION_DIM_LABELS = [str(i) for i in range(pred_actions.shape[-1])]

    figure_layout = [ACTION_DIM_LABELS]
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([40, 5])

    # Ensure proper input formatting for actions
    pred_actions = np.array(pred_actions).squeeze()  # Bx7
    true_actions = np.array(true_actions).squeeze()  # Bx7

    # Plot actions for each dimension
    for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
        axs[action_label].plot(pred_actions[:, action_dim], label="Predicted")
        axs[action_label].plot(true_actions[:, action_dim], label="Ground Truth")
        axs[action_label].set_title(action_label)
        axs[action_label].set_xlabel("Time (steps)")
        axs[action_label].legend()

    plt.tight_layout()
    wandb.log({"epoch": epoch, wandb_title: wandb.Image(fig)})
    plt.close(fig)