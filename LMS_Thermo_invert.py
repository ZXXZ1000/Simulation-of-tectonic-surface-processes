import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from matplotlib.gridspec import GridSpec

# Load data from Excel file
file_path = r'D:\OneDrive - zju.edu.cn\文档\MATLAB 代码\河流反演\test1\Profile_Age_project\Themodata_point_with_elevation.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
print(df.head())

# Constants
surface_temperature = 20  # Surface temperature in °C
geothermal_gradient = 35  # Geothermal gradient in °C/km

# Closure temperatures and depths for different methods
closure_temperatures = {"AHe": 70, "ZHe": 190, "ZFT": 240, "AFT": 110}
depths = {"AHe": (70) / geothermal_gradient,
          "ZHe": (190) / geothermal_gradient,
          "ZFT": (240) / geothermal_gradient,
          "AFT": (110) / geothermal_gradient}

# Exclude specific samples
exclude_samples = ['LME-14', 'LME-15', 'LME-09']

# Define color palettes for each reference
reference_colors = {
    '(E. Wang et al., 2012)': 'Reds',
    '(Godard et al., 2009)': 'gist_gray_r',
    '(Tan et al., 2017b)': 'Blues'
}

# Create the main plot and inset plot with envelopes for error bars, excluding specified samples
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(1, 2, width_ratios=[2, 1])

main_ax = fig.add_subplot(gs[0])
inset_ax = fig.add_subplot(gs[1])

# Plot each sample with different methods and connect their points, adding error bars and envelopes
references = df['Reference'].unique()

for reference in references:
    base_color = reference_colors.get(reference, 'gray')  # 使用预定义的颜色，如果没有定义则使用灰色
    samples = df[df['Reference'] == reference]['Sample'].unique()
    color_palette = sns.color_palette(f"{base_color}", n_colors=len(samples))
    
    for i, sample_id in enumerate(samples):
        if sample_id not in exclude_samples:
            color = color_palette[i]
            group = df[(df['Reference'] == reference) & (df['Sample'] == sample_id)]
            
            # Extract data for the current sample
            ages = []
            sample_depths = []
            age_errors = []
            for _, row in group.iterrows():
                age = row['Age']
                method = row['Method']
                depth = (closure_temperatures[method]) / geothermal_gradient
                ages.append(age)
                sample_depths.append(depth)
                age_errors.append(row['Std_Dev'])

            # Sort ages and depths for plotting
            sorted_indices = sorted(range(len(ages)), key=lambda k: ages[k])
            ages = [ages[i] for i in sorted_indices]
            sample_depths = [sample_depths[i] for i in sorted_indices]
            age_errors = [age_errors[i] for i in sorted_indices]

            # Plot on main and inset axes
            for ax in [main_ax, inset_ax]:
                ax.errorbar([0] + ages, [0] + sample_depths, xerr=[0] + age_errors, fmt='o-', label=f'{reference} - {sample_id}', color=color)
                ax.fill_betweenx([0] + sample_depths, [0] + [a - e for a, e in zip(ages, age_errors)], 
                                 [0] + [a + e for a, e in zip(ages, age_errors)], alpha=0.2, color=color)


# Add labels and legend to the main plot
main_ax.set_xlabel('Age (Ma)')
main_ax.set_ylabel('Depth (km)')
main_ax.legend(loc='upper left',fontsize=8)
main_ax.set_xlim(0, 400)
main_ax.set_ylim(0, 10)
main_ax.invert_xaxis()
main_ax.invert_yaxis()
main_ax.xaxis.tick_top()
main_ax.xaxis.set_label_position('top')
main_ax.yaxis.tick_right()
main_ax.yaxis.set_label_position('right')

#标签：地温梯度、地表温度、方法及closure_temperatures
main_ax.text(390, 9.5, f'Geothermal Gradient: {geothermal_gradient} °C/km\nSurface Temperature: {surface_temperature} °C')
for method, depth in depths.items():
    main_ax.text(170, depth, f'{method} Closure Temperature: {closure_temperatures[method]} °C', color='gray', fontsize=8,alpha=0.7)

# Add the gray shaded area to the main plot
main_ax.axvspan(0, 15, color='gray', alpha=0.2, zorder=1)
main_ax.text(15, 0.15, '(b)', color='black', fontsize=8, alpha=1)

# Add labels and legend to the inset plot
inset_ax.set_xlabel('Age (Ma)')
inset_ax.set_ylabel('Depth (km)')
inset_ax.invert_xaxis()
inset_ax.invert_yaxis()
inset_ax.xaxis.tick_top()
inset_ax.xaxis.set_label_position('top')
inset_ax.yaxis.tick_right()
inset_ax.yaxis.set_label_position('right')
inset_ax.set_xlim(15, 0)
inset_ax.set_ylim(10, 0)

#图框左下角标记图(a)、(b)，并在图(a)的灰色阴影区域标记为图(b)
main_ax.text(0.01, 0.01, '(a)', fontsize=12, transform=main_ax.transAxes)
inset_ax.text(0.01, 0.01, '(b)', fontsize=12, transform=inset_ax.transAxes)

# 图b中插入趋势线，用于标记抬升速率
inset_ax.plot([15, 0], [9, 0], color='black', linestyle='--', alpha=1)
inset_ax.text(12.5, 8, '~0.6-0.8 km/ma', fontsize=10, alpha=0.9, color='black')

#inset_ax.axvspan(0, 10, color='gray', alpha=0.2, zorder=1)

# Make sure the plot frame is closed
for ax in [main_ax, inset_ax]:
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.grid(True, linestyle='--', alpha=0.2)

# Save the figure as an SVG file
plt.tight_layout()
plt.savefig('thermal_history.svg', format='svg')
plt.show()
