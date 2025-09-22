import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_losses():
    # Use absolute path to ensure we find the files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, 'toy_exp')
    
    dataset = 'ETTh2'
    loss_type = 'valid'
    id = '512_96'
    r_ema = [0.0, 0.99, 0.996, 0.999]
    
    print(f"Looking for files in: {path}")
    
    # Create figure with professional styling
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    
    # Define colors and styles similar to the reference image
    colors = ['#2986cc', '#a64d79', '#ce7e00', '#9f2f2f']  # Blue, orange, green, red
    markers = ['o', 's', '^', 'D']  # Circle, square, triangle, diamond
    linestyles = ['-', '-', '-', '-']
    
    for i, r in enumerate(r_ema):
        filename = f'{path}/{loss_type}_loss_{dataset}_{id}_{r}.npy'
        print(f"Checking: {filename}")
        
        if os.path.exists(filename):
            print(f"  Found! Loading data...")
            data = np.load(filename)
            epochs = range(1, len(data) + 1)
            
            plt.plot(
                    # epochs[-4:], data[-4:],
                    epochs[:5], data[:5],  
                    label=f'r_ema={r}', 
                    color=colors[i], 
                    marker=markers[i], 
                    linestyle=linestyles[i],
                    linewidth=3, 
                    markersize=8,
                    markerfacecolor=colors[i],
                    markeredgecolor='white',
                    markeredgewidth=1)
        else:
            print(f"  Not found!")
    
    # Professional styling
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel(f'{loss_type} Loss', fontsize=16, fontweight='bold')
    plt.title(f'{dataset} {loss_type} Loss Comparison', fontsize=18, fontweight='bold', pad=20)
    
    # Grid styling
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend styling
    plt.legend(fontsize=12, 
              frameon=True, 
              fancybox=True, 
              shadow=True,
              framealpha=0.9,
              loc='upper right')
    
    # Axis styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    # Tick styling
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().tick_params(width=1.5, length=6)
    
    # Set y-axis to start from 0 for better comparison
    # plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    output_file = f'{path}/{dataset}_{loss_type}_{id}_comparison.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_losses()