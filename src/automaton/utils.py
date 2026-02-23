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
        ("Accuracy (Train vs Test)", "Accuracy", ["training_accuracies", "testing_accuracies"], ["-", "--"], ["Training", "Testing"]),
        ("States", "States", ["states"], ["-"], ["Min States"])
    ]

    for title, ylabel, data_keys, styles, labels in plots_config:
        fig = plt.figure(figsize=(10, 6))
        
        for key, style, label in zip(data_keys, styles, labels):
            iterations = []
            values = []
            
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
                
                # For accuracy: take max (best performer)
                # For states: take min (simplest automaton)
                if 'accuracy' in key or 'accuracies' in key:
                    best_val = max(all_vals)
                else:  # states
                    best_val = min(all_vals)
                
                iterations.append(step['iteration'])
                values.append(best_val)
            
            if not values:
                continue
            
            marker = 'o' if style == '-' else 'x'
            plt.plot(iterations, values, marker=marker, linestyle=style, label=label, linewidth=2)

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
