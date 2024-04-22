import re
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# Setting a style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
def parse_log(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'loss:' in line and 'accuracy:' in line:
                train_match = re.search(r'\[(\d+/\d+)\].*loss: ([\d\.]+).*accuracy: ([\d\.]+)%', line)
                if train_match:
                    epoch, loss, accuracy = train_match.groups()
                    data.append({
                        'epoch': int(epoch.split('/')[0]),
                        'loss': float(loss),
                        'accuracy': float(accuracy)
                    })
            if 'val_acc:' in line and 'val_loss:' in line:
                val_match = re.search(r'val_acc: ([\d\.]+)%, val_loss: ([\d\.]+)', line)
                if val_match:
                    val_acc, val_loss = val_match.groups()
                    if data:
                        data[-1].update({
                            'val_accuracy': float(val_acc),
                            'val_loss': float(val_loss)
                        })
    return data


def plot_metrics(datasets, file_names, save_path=None):
    fig, axs = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey='row')
    
    colors = ['green', 'blue', 'orange', 'red', 'purple', 'brown']
    
    for idx, data in enumerate(datasets):
        color = colors[idx % len(colors)]
        epochs = [entry['epoch'] for entry in data]
        # Get simple model names for legend entries
        model_name = os.path.splitext(file_names[idx])[0]

        # Plot the metrics with appropriate labels and colors
        axs[0, 0].plot(epochs, [entry['accuracy'] for entry in data], '.-', color=color, label=model_name)
        axs[0, 1].plot(epochs, [entry['val_accuracy'] for entry in data], '.-', color=color, label=model_name)
        axs[1, 0].plot(epochs, [entry['loss'] for entry in data], '.-', color=color, label=model_name)
        axs[1, 1].plot(epochs, [entry['val_loss'] for entry in data], '.-', color=color, label=model_name)

    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 1].set_xlabel('Epoch')
    axs[0, 0].set_title('Training Accuracy')
    axs[0, 1].set_title('Validation Accuracy')
    axs[1, 0].set_title('Training Loss')
    axs[1, 1].set_title('Validation Loss')
    
    for ax in axs.flat:
        ax.legend()
        break

    for ax in axs.flat:
        ax.xaxis.set_major_locator(MultipleLocator(2))

    plt.suptitle('Comparison of Training Logs')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare multiple training log files.')
    parser.add_argument('file_paths', nargs='+', type=str, help='The paths to the log files.')
    parser.add_argument('--save', type=str, help='Path to save the plot as a PNG file.', default=None)
    args = parser.parse_args()
    
    datasets = []
    file_names = []
    for file_path in args.file_paths:
        datasets.append(parse_log(file_path))
        file_names.append(os.path.basename(file_path))

    plot_metrics(datasets, file_names, args.save)

if __name__ == '__main__':
    main()