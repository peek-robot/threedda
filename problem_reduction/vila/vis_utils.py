import numpy as np
import matplotlib.pyplot as plt

def plot_model_results_per_task(model_results, distance_key, out_path=None):
    """
    Generalized function to plot results from multiple models side by side.
    
    :param model_results: Dictionary: {model_name: {task_name: {distance_key: distance_value}}} containing results.
    :param distance_key: The metric name (e.g., "hausdorff" or "dtw_euclidian") to extract from each task.
    :param out_path (str): path to save the plot

    """
    keys = list(model_results.keys())
    
    # Sort tasks by the first model's values using the specified distance key
    first_model = keys[0]
    sorted_tasks = dict(sorted(model_results[first_model].items(), key=lambda item: item[1][distance_key]))
    task_labels = list(sorted_tasks.keys())

    # Extract values for all models in the sorted order of task_labels
    values = {key: [model_results[key][task][distance_key] for task in task_labels] for key in keys}
    
    # Set up bar width and x-axis positions
    x = np.arange(len(task_labels))
    bar_width = 0.8 / len(keys)  # Adjust bar width dynamically based on number of models

    plt.figure(figsize=(12, 6))
    
    # Plot each model's bars
    for i, key in enumerate(keys):
        plt.bar(x + (i - len(keys)/2) * bar_width, values[key], width=bar_width, label=key)

    # Formatting
    plt.xticks(x, task_labels, rotation=60, ha='right')
    plt.title(distance_key)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()

def plot_keys_per_task(model_results, distance_keys, model_name, normalize=False, out_path=None):
    """
    Generalized function to plot multiple distance metrics per model, with optional normalization.
    
    :param model_results: Dictionary: {model_name: {task_name: {distance_key: distance_value}}} containing results.
    :param distance_keys: List of metric names (e.g., ["hausdorff", "dtw_euclidian"]) to plot.
    :param model_name: The model name to extract results from.
    :param normalize: Boolean flag to normalize distances to [0,1] across tasks.
    :param out_path (str): Path to save the plot.
    """
    if model_name not in model_results:
        raise ValueError(f"Model '{model_name}' not found in model_results.")

    model_data = model_results[model_name]

    # Extract task names and sort them by the first distance key
    sorted_tasks = dict(sorted(model_data.items(), key=lambda item: item[1][distance_keys[0]]))
    task_labels = list(sorted_tasks.keys())

    # Extract values for all distance metrics in the sorted order of task_labels
    values = {key: np.array([model_data[task][key] for task in task_labels]) for key in distance_keys}

    # Normalize values if flag is set
    if normalize:
        for key in distance_keys:
            min_val, max_val = np.min(values[key]), np.max(values[key])
            if max_val > min_val:  # Avoid division by zero
                values[key] = (values[key] - min_val) / (max_val - min_val)

    # Set up bar width and x-axis positions
    x = np.arange(len(task_labels))
    bar_width = 0.8 / len(distance_keys)  # Adjust bar width dynamically based on number of metrics

    plt.figure(figsize=(12, 6))
    
    # Plot each distance metric's bars
    for i, key in enumerate(distance_keys):
        plt.bar(x + (i - len(distance_keys)/2) * bar_width, values[key], width=bar_width, label=key)

    # Formatting
    plt.xticks(x, task_labels, rotation=60, ha='right')
    plt.title(f"Distance Metrics for {model_name}" + (" (Normalized)" if normalize else ""))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save or show the plot
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()

def plot_path_len(paths_1, paths_2, labels=["gt", "pred"], out_path=None):
    """
    Plot histogram of path lengths for two sets of paths.
    - paths_1 (list): list of paths
    - paths_2 (list): list of paths
    - out_path (str): path to save the plot
    """

    path_lens_1 = [len(p) for p in paths_1]
    path_lens_2 = [len(p) for p in paths_2]

    # Compute histogram frequencies manually
    bins = np.arange(min(path_lens_1 + path_lens_2), max(path_lens_1 + path_lens_2) + 1, 1)

    gt_counts, _ = np.histogram(path_lens_1, bins=bins)
    pred_counts, _ = np.histogram(path_lens_2, bins=bins)

    # Set up bar width and x-axis positions
    x = bins[:-1]  # Use bin edges except the last one
    bar_width = 0.3  # Adjusted width for side-by-side display

    plt.figure(figsize=(10, 5))

    # Plot histograms side by side using plt.bar
    plt.bar(x - bar_width/2, gt_counts, width=bar_width, label=labels[0], color="tab:blue")
    plt.bar(x + bar_width/2, pred_counts, width=bar_width, label=labels[1], color="tab:orange")

    # extend bins by -/+ 1 and control ticks
    bins = np.concatenate(([bins[0] - 1], bins, [bins[-1] + 1]))
    plt.xticks(bins[:-1])

    plt.xlabel("len(path)")
    plt.ylabel("frequency")
    plt.title(f"len(path) distribution")
    plt.legend()

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()
