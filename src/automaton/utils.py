"""
General utility functions for data processing and visualization.
These are not specific to any automaton type.
"""
import matplotlib.pyplot as plt


def add_position_to_sample(samples):
    """
    Add position information to each sample feature.
    Converts each element v to string to ensure it's treated as a complete symbol, not split into characters.
    
    Example:
      Input: [['hello', 'world'], ['good', 'day']]
      Output: [['p0_(hello)', 'p1_(world)'], ['p0_(good)', 'p1_(day)']]
    """
    return [[f'p{i}_({str(v)})' for i, v in enumerate(sample)] for sample in samples]


def tokenize_sentence(samples):
    """
    Convert sentence strings into tokenized format with length.
    Example: 'hello world test' → ['hello', 'world', 'test']
    
    : param samples: List of sentence strings
    : return: List of tokenized samples (each with words + length)
    """
    tokenized = []
    for sentence in samples:
        if isinstance(sentence, str):
            words = sentence.split()
            tokenized.append(words)
        else:
            tokenized.append(sentence)
    return tokenized


def plot_beam_stats(iteration_stats, beam_size, output_dir="test_result/explain", show=False):
    """
    Plot beam search stats and save figures to disk.
    Supports both Tabular and Text data types.

    Parameters
    ----------
    iteration_stats : list
        Stats per iteration produced by the explainer.
    beam_size : int
        Number of candidates tracked per iteration.
    output_dir : str
        Directory to save the generated plots.
    show : bool
        Whether to display plots interactively. Defaults to False.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    plots_config = [
        ("Accuracy (Train vs Validation)", "Accuracy", ["training_accuracies", "validation_accuracies"], ["-", "--"], ["Training (Best)", "Validation (Best)"]),
        ("States", "States", ["states"], ["-"], ["Min States"])
    ]

    for title, ylabel, data_keys, styles, labels in plots_config:
        fig = plt.figure(figsize=(10, 6))
        
        for key, style, label in zip(data_keys, styles, labels):
            iterations = []
            values_best = []
            values_avg = []
            values_worst = []
            
            for step in iteration_stats:
                if key not in step:
                    continue
                    
                # Get all values for this key (across all beam candidates)
                all_vals = []
                for i in range(len(step[key])):
                    val = step[key][i]
                    # Flatten if it's an array/list (handle extra dimensions)
                    if hasattr(val, '__len__') and not isinstance(val, str):
                        # Extract scalar from nested array
                        while hasattr(val, '__len__') and not isinstance(val, str) and len(val) > 0:
                            val = val[0]
                    all_vals.append(val)
                
                if not all_vals:
                    continue
                
                # For accuracy: show best, average, and worst performers
                # For states: take min (simplest automaton)
                if 'accuracy' in key or 'accuracies' in key:
                    best_val = max(all_vals)
                    avg_val = sum(all_vals) / len(all_vals)
                    worst_val = min(all_vals)
                    values_best.append(best_val)
                    values_avg.append(avg_val)
                    values_worst.append(worst_val)
                else:  # states - only show minimum
                    best_val = min(all_vals)
                    values_best.append(best_val)
                
                iterations.append(step['iteration'])
            
            if not values_best:
                continue
            
            marker = 'o' if style == '-' else 'x'
            # Plot best (solid line)
            plt.plot(iterations, values_best, marker=marker, linestyle=style, label=label, linewidth=2)
            
            # For accuracy: also plot average and worst to show diversity
            if 'accuracy' in key or 'accuracies' in key:
                plt.plot(iterations, values_avg, marker=marker, linestyle=":", label=label.replace("Best", "Avg"), linewidth=1.5, alpha=0.6)
                plt.plot(iterations, values_worst, marker=marker, linestyle="-.", label=label.replace("Best", "Worst"), linewidth=1, alpha=0.4)

        plt.title(f"{title} over iterations")
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        # Save figure to disk
        safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        file_path = os.path.join(output_dir, f"{safe_title}.png")
        fig.savefig(file_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)


# Backward compatibility alias
plot_dfa_beam_stats = plot_beam_stats
