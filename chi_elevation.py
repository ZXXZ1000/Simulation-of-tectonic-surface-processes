import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyproj import Proj, transform

# Load the Excel file
file_path = r'D:\OneDrive - zju.edu.cn\文档\MATLAB 代码\河流反演\test1\Profile_Age_project\min_elbow_chi_profile.xlsx'  # 预留Excel读取位置
data = pd.read_excel(file_path)

# Define the columns for x and y coordinates
x_columns = [col for col in data.columns if col.startswith('x-')]
y_columns = [col for col in data.columns if col.startswith('y-')]
chi_columns = [col for col in data.columns if col.startswith('chi-')]
ele_min = data['elevation-min']
chi_min = data['chi-min']
ele_fu = data['elevation-fu']
chi_fu = data['chi-fu']
#ele_paloemin = data['elevation-paloemin']
#chi_paloemin = data['chi-paloemin']

# Ensure that the number of x and y columns are equal
#assert len(x_columns) == len(y_columns) == len(chi_columns)

# Create a color palette
palette = plt.cm.rainbow(np.linspace(0, 1, len(x_columns)))

# Define the UTM 48N projection and WGS84 (geographic) projection
utm_proj = Proj(proj='utm', zone=48, ellps='WGS84')
wgs84_proj = Proj(proj='latlong', datum='WGS84')

# Function to convert UTM to geographic coordinates
def utm_to_geo(x, y):
    lon, lat = transform(utm_proj, wgs84_proj, x, y)
    return lon, lat


def plot_chimap():
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each curve with color mapping based on 'chi-min' values
    scatter = None
    for idx, (x_col, y_col, chi_col) in enumerate(zip(x_columns, y_columns, chi_columns)):
        scatter = ax.scatter(data[x_col], data[y_col], c=data[chi_col], cmap='Spectral_r', alpha=1, label=f'{x_col} vs {y_col}')


    # Adding a color bar to show the mapping of 'chi' values to colors
    colorbar = plt.colorbar(scatter,ax = ax, shrink=0.1, aspect=3, pad=0.05, location='right')
    colorbar.set_label('χ (km)', fontsize=15, labelpad=10)
    colorbar.ax.tick_params(labelsize=12)

    # Convert axis labels from UTM to geographic coordinates
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    xticks_geo = [utm_to_geo(x, yticks[0])[0] for x in xticks]
    yticks_geo = [utm_to_geo(xticks[0], y)[1] for y in yticks]
    ax.set_xticklabels([f'{xtick:.2f}' for xtick in xticks_geo])
    ax.set_yticklabels([f'{ytick:.2f}' for ytick in yticks_geo])


    # Adding labels and title
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        
    ax.grid(True, color='gray', linestyle='--', alpha=0.1)

    # Adjusting the axes for UTM 48N projection coordinates
    ax.set_aspect('equal', adjustable='box')
plot_chimap()

def plot_chi_elevation():
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(6, 7))
    #ax.plot(chi_min, ele_min, label='Min River', color='#54ADD8', linewidth=2)
    ax.plot(chi_fu, ele_fu, label='Fu River', color='#E2382C', linewidth=2)
    ax.plot(chi_paloemin, ele_paloemin, label='Paloemin River', color='#54ADD8', linewidth=2)

    # Adding labels and title
    ax.set_xlabel('χ (km)', fontsize=15)
    ax.set_ylabel('Elevation (m)', fontsize=15)
    ax.set_xlim(0, 4)
    ax.set_ylim(1000,2500)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    # 增加刻度
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Adding grid lines
    ax.grid(False)

#plot_chi_elevation()

# Show plot
plt.show()
