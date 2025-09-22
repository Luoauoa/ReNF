import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_performances():
    dataset = 'Traffic'
    loss_type = 'Test'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, 'toy_exp')
    # The following data is obatined from 336-720 forecast with varing d_layers, check it using ETTh1.sh
    # x = ['1', '2', '3', '4', '5', '6', '8']
    x = ['1', '2', '3', '4', '6', '8']
    y1 = [0.359, 0.347, 0.343, 0.342, 0.342, 0.343] # wo BDO
    y2 = [0.359, 0.345, 0.339, 0.336, 0.335, 0.334] # w BDO
    
    # Create grouped bar chart
    x_pos = np.arange(len(x))
    width = 0.4  # Width of bars
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars2 = ax.bar(x_pos - width/2, y1, width, label='Without BDO', 
                   color='lightblue', edgecolor='black', linewidth=2, alpha=0.7)
    bars1 = ax.bar(x_pos + width/2, y2, width, label='With BDO', 
                   color='lightcoral', edgecolor='black', linewidth=2, alpha=0.7)
    # bars3 = ax.bar(x_pos + width/2, y3, width, label='Without BDO', 
    #                color='lightcoral', edgecolor='black', linewidth=2, alpha=0.7)
    
    # Add value annotations on bars
    y_min = min(y2) - 0.01  # Bottom of y-axis
    y_max = max(y1) + 0.005  # Top of y-axis
    
    for bar, value in zip(bars1, y2):
        visual_center = y_min + (value - y_min) / 2
        ax.text(bar.get_x() + bar.get_width()/2, visual_center, 
                f'{value:.3f}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='black')
    
    for bar, value in zip(bars2, y1):
        visual_center = y_min + (value - y_min) / 2
        ax.text(bar.get_x() + bar.get_width()/2, visual_center, 
                f'{value:.3f}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='black')

    ax.set_xlabel('Number of sub-forecasts', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'{loss_type} Loss', fontsize=16, fontweight='bold')
    ax.set_title(f'{dataset} Performance Comparison', fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend styling
    ax.legend(fontsize=12, 
              frameon=True, 
              fancybox=True, 
              shadow=True,
              framealpha=0.9,
              loc='upper right')
    
    # Axis styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    
    # Set y-axis limits
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    
    output_file = f'{path}/{dataset}_{loss_type}_layer_comparison.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_performances()