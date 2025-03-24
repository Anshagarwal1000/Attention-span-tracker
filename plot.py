import matplotlib.pyplot as plt

def plot_focus(timestamps, attentiveness):
    """Plot the focus analysis over time."""
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(timestamps, attentiveness, color='dodgerblue', linewidth=2, linestyle='-', 
             marker='o', markersize=4)
    plt.fill_between(timestamps, attentiveness, color='lightgreen', alpha=0.2)
    plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
    plt.ylabel('Attentiveness', fontsize=14, fontweight='bold')
    plt.title('Focus Analysis Over Time', fontsize=16, fontweight='bold')
    plt.yticks([0, 1], ['Distracted', 'Focused'], fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "_main_":

    sample_timestamps = [0, 1, 2, 3, 4]
    sample_attentiveness = [0, 1, 1, 0, 1]
    plot_focus(sample_timestamps, sample_attentiveness)