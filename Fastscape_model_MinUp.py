import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from fastscape.models import basic_model
import rasterio
from rasterio.enums import Resampling
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter
import xsimlab as xs
from matplotlib.colors import LightSource
import richdem as rd
from matplotlib.animation import FuncAnimation, PillowWriter
from fastscape.processes import SingleFlowRouter
from landlab.components import FlowAccumulator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fastscape

#读取netcdf文件
netcdf_path = r"C:\Users\Administrator\Videos\fastscape_test\test3\re-draw_MinUp_step1\out_Minup_step.nc"
ds = xr.open_dataset(netcdf_path)
#读取最终高程数据
dem = ds['topography__elevation'].isel(out=-1)  #-1 or int(len(ds['topography__elevation']) * 0.5)
#初始化模型
#==================================================================分割线==================================================================
# Initialize uplift rate matrix
file_path = r"D:\OneDrive - zju.edu.cn\MR.Z  所有资料\Phd Period\大地测量数据\MOHO_depth_heatmap_clipped.tif"
save_path = r"C:\Users\Administrator\Videos\fastscape_test\test3\re-draw_MinUp_step2"
# Function to create the initial elevation field
def create_initial_elevation_field(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        data_resampled = src.read(out_shape=(1, 300, 300), resampling=Resampling.bilinear)[0]

    x, y = np.indices(data_resampled.shape)
    valid_mask = ~np.isnan(data_resampled)
    coords = np.array([x[valid_mask], y[valid_mask]]).T
    values = data_resampled[valid_mask]
    interpolator = NearestNDInterpolator(coords, values)
    filled_data = interpolator(x, y)
    data_smoothed = gaussian_filter(filled_data, sigma=1)

    data_min, data_max = np.nanmin(data_smoothed), np.nanmax(data_smoothed)
    normalized_data = (data_smoothed - data_min) / (data_max - data_min)
    uplift_rate_matrix = normalized_data * (0.001 - 0) + 0

    return uplift_rate_matrix
uplift_rate_matrix = create_initial_elevation_field(file_path)

# Adding gradient uplift rate
x_gradient = np.linspace(0.01, 0.05, 300)
y_gradient = np.linspace(0.01, 0.01, 300)
uplift_rate_matrix += np.outer(y_gradient, x_gradient)

# Function to apply LMS uplift
def gaussian_along_LMS(x, y, x1, y1, x2, y2, width, height):
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    distance = np.abs(dy * x - dx * y + x2 * y1 - y2 * x1) / length
    projection = (dx * (x - x1) + dy * (y - y1)) / length
    mask = (projection >= 0) & (projection <= length)
    gaussian = height * np.exp(-(distance**2) / (2 * (width**2)))
    sinusoid = np.sin(np.pi * projection / length)
    uplift = np.zeros_like(x, dtype=float)
    uplift[mask] = gaussian[mask] * sinusoid[mask]
    return uplift

def apply_LMS_proportion(uplift_rate_matrix):
    X, Y = np.meshgrid(np.arange(300), np.arange(300))
    x1, y1 = 38*3, 10*3
    x2, y2 = 68*3, 45*3
    width = 5*3
    height = 0.0008
    uplift_rate_matrix += gaussian_along_LMS(X, Y, x1, y1, x2, y2, width, height)
    return uplift_rate_matrix

uplift_rate_matrix = apply_LMS_proportion(uplift_rate_matrix)

def make_fault(uplift_rate_matrix):
    X, Y = np.meshgrid(np.arange(300), np.arange(300))
    x1, y1 = 32*3, 14*3
    x2, y2 = 60*3, 46*3
    width = 1*3
    height = 0.0002
    uplift_rate_matrix -= gaussian_along_LMS(X, Y, x1, y1, x2, y2, width, height)
    return uplift_rate_matrix

uplift_rate_matrix = make_fault(uplift_rate_matrix)

def make_lms_fault2(uplift_rate_matrix):
    X, Y = np.meshgrid(np.arange(300), np.arange(300))
    x1, y1 = 114, 36
    x2, y2 = 120, 81
    width = 3
    height = 0.0004
    uplift_rate_matrix -= gaussian_along_LMS(X, Y, x1, y1, x2, y2, width, height)
    return uplift_rate_matrix

#uplift_rate_matrix = make_lms_fault2(uplift_rate_matrix)

# Function to apply Min uplift
def gaussian_along_min(x, y, x1, y1, x2, y2, width, height):
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    distance = np.abs(dy * x - dx * y + x2 * y1 - y2 * x1) / length
    projection = (dx * (x - x1) + dy * (y - y1)) / length
    mask = (projection >= 0) & (projection <= length)
    gaussian = height * np.exp(-(distance**2) / (2 * (width**2)))
    sinusoid = np.sin(0.99*np.pi * projection / length)
    uplift = np.zeros_like(x, dtype=float)
    uplift[mask] = gaussian[mask] * sinusoid[mask]
    return uplift

def apply_min_proportion(uplift_rate_matrix):
    X, Y = np.meshgrid(np.arange(300), np.arange(300))
    x1, y1 = 65*3, 120*3
    x2, y2 = 65*3, 35*3
    width = 5*3
    height = 0.003
    uplift_rate_matrix += gaussian_along_min(X, Y, x1, y1, x2, y2, width, height)
    return uplift_rate_matrix

uplift_rate_matrix = apply_min_proportion(uplift_rate_matrix)

def make_fault(uplift_rate_matrix):
    X, Y = np.meshgrid(np.arange(300), np.arange(300))
    x1, y1 = 55*3, 100*3
    x2, y2 = 55*3, 40*3
    width = 1*3
    height = 0.0002
    uplift_rate_matrix -= gaussian_along_min(X, Y, x1, y1, x2, y2, width, height)
    return uplift_rate_matrix

uplift_rate_matrix = make_fault(uplift_rate_matrix)

#Function to apply Tudiling uplift
def apply_TDL_proportion(uplift_rate_matrix):
    X, Y = np.meshgrid(np.arange(300), np.arange(300))
    x1, y1 = 65*3, 58*3
    x2, y2 = 65*3, 40*3
    width = 1*3
    height = 0.002
    uplift_rate_matrix += gaussian_along_min(X, Y, x1, y1, x2, y2, width, height)
    return uplift_rate_matrix

#uplift_rate_matrix = apply_TDL_proportion(uplift_rate_matrix)

# Define Ksp field
Ksp = np.ones((300, 300)) * 4.3e-6
border_width = 3
# 将边界区域的侵蚀系数设置为 0
Ksp[:border_width, :] = 0  # 下边界
Ksp[-border_width:, :] = 0  # 上边界
Ksp[:, :border_width] = 0  # 左边界
Ksp[:, -border_width:] = 0  # 右边界
def apply_LMS_Ksp(Ksp):
    X, Y = np.meshgrid(np.arange(300), np.arange(300))
    x1, y1 = 29.4*3, 0
    x2, y2 = 100*3, 82*3
    width = 5*3
    height = 1.3e-6
    Ksp -= gaussian_along_LMS(X, Y, x1, y1, x2, y2, width, height)
    return Ksp

Ksp = apply_LMS_Ksp(Ksp)

# Flatten the 2D arrays for Fastscape
uplift_rate = uplift_rate_matrix.flatten()
uplift_rate_matrix = uplift_rate_matrix*1
Ksp = Ksp.flatten()

# Plot uplift rate and Ksp fields
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Plot Uplift Rate Field
im1 = ax[0].imshow(uplift_rate_matrix * 1000, cmap='RdBu_r', origin='lower')
ax[0].set_title('Uplift Rate Field (mm/yr)')
divider1 = make_axes_locatable(ax[0])
cax1 = divider1.append_axes("right", size="5%", pad=0.2)
fig.colorbar(im1, cax=cax1, label='Uplift rate (mm/yr)')
# Add contour lines
contours = ax[0].contour(uplift_rate_matrix * 1000, levels=20, colors='black', linewidths=0.5)
ax[0].clabel(contours, inline=True, fontsize=8)

# Plot Ksp Field
im2 = ax[1].imshow(Ksp.reshape(300, 300), cmap='viridis_r', origin='lower')
ax[1].set_title('Ksp Field')
divider2 = make_axes_locatable(ax[1])
cax2 = divider2.append_axes("right", size="5%", pad=0.2)
fig.colorbar(im2, cax=cax2, label='Ksp')
# Adjust layout and save the figure
plt.tight_layout()
fig.savefig(save_path + '/uplift_rate_Ksp_fields.svg', dpi=300)
plt.show()

#==================================================================分割线==================================================================

#选择是否继续模拟过程
user_input = input("是否继续模拟？(y/n)")
if user_input == 'n':
    exit(0)
elif user_input == 'y':
    pass

#模拟模块
#==================================================================分割线==================================================================
# Create DataArray for uplift rate and Ksp
x_coords = np.arange(300)
y_coords = np.arange(300)
uplift_rate_da = xr.DataArray(uplift_rate_matrix, coords={'y': y_coords, 'x': x_coords}, dims=('y', 'x'))
Ksp_da = xr.DataArray(Ksp.reshape(300, 300), coords={'y': y_coords, 'x': x_coords}, dims=('y', 'x'))

# Initialize Fastscape model
basic_model = basic_model.drop_processes('init_topography')
time=2e6
steps = 101
interval = int(time / (steps-1))
# Create the input dataset
in_ds = xs.create_setup(
    model=basic_model,
    clocks={
        'time': np.linspace(0., time, 500+1),
        'out': np.linspace(0., time, steps)
    },
    master_clock='time',
    input_vars={
        'grid__shape': [300, 300],
        'grid__length': [3e5, 3e5],
        "topography__elevation": dem,
        'boundary__status': ['core', 'core', 'fixed_value', 'core'],
        'uplift__rate': uplift_rate_da,
        'spl': {'k_coef': Ksp_da, 'area_exp': 0.42, 'slope_exp': 1.},
        'diffusion__diffusivity': 6e-2,
    },
    output_vars={
        'topography__elevation': 'out',
        'drainage__area': 'out',
        'flow__basin': 'out',
        'spl__chi': 'out',
        'spl__erosion': 'out',
        'erosion__rate': 'out',
        'terrain__slope': 'out',

    }
)
print("in_ds created successfully")

# Run the model
with xs.monitoring.ProgressBar():
    out_ds = in_ds.xsimlab.run(model=basic_model)
print("Model run successfully")
out_ds.to_netcdf(save_path + '/out_Minup_step.nc')
print("out_ds saved successfully")

#绘图模块
#==================================================================分割线==================================================================

def plot_key_steps(out_ds, save_path):
    # Define the key steps for plotting
    key_steps = [int(len(out_ds['topography__elevation']) * 0.25), 
                 int(len(out_ds['topography__elevation']) * 0.5), 
                 int(len(out_ds['topography__elevation']) * 0.75), 
                 int(len(out_ds['topography__elevation']) * 1.00)-1]
    
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs = axs.ravel()  # Flatten the 2D array of axes for easy indexing
    
    # Get the overall min and max elevation for consistent colormap
    elevation_all = out_ds['topography__elevation'].values
    vmin, vmax = np.nanmin(elevation_all), np.nanmax(elevation_all)
    
    # Plot the key steps
    for i, step in enumerate(key_steps):  
        elevation_data = out_ds['topography__elevation'].values[step].reshape((300, 300))
        drainage_data = out_ds['drainage__area'].values[step].reshape((300, 300))
        
        mask = drainage_data >= 5e8
        elevation_data_masked = np.where(mask, 0, elevation_data)  # Use NaN instead of 0 for better visualization
        
        im = axs[i].imshow(elevation_data_masked, cmap='terrain', vmin=vmin, vmax=vmax, origin='lower')
        axs[i].set_title(f'Time = {step * interval} years')
        axs[i].set_xlabel('X coordinate')
        axs[i].set_ylabel('Y coordinate')
    
    # Add a colorbar
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal',label='Elevation')
    plt.savefig(save_path + '/key_steps_terrain.svg', dpi=300)
    plt.show()
plot_key_steps(out_ds, save_path)
print("Key steps saved successfully")


#动画模块
#==================================================================分割线==================================================================

def create_and_save_animation(out_ds, save_path):
    #绘制动图
    from landlab import RasterModelGrid
    from landlab.components import FlowAccumulator
    # Extracting the topographic data
    topo_data = out_ds['topography__elevation'].values
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Define the update function for the animation
    def update(frame):
        ax.clear()  # Clear previous frame
        elevation_data = topo_data[frame].reshape((300, 300))  # Select elevation data for current frame and reshape
        drainage_data = out_ds['drainage__area'].values[frame].reshape((300, 300))  # Select drainage area data and reshape
        # Create a mask where drainage_area >= 1e7
        mask = drainage_data >= 5e8
        # Apply the mask to elevation_data
        elevation_data_masked = np.where(mask, 0, elevation_data)
        #选取最后一帧的数据
        elevation_last = topo_data[-1].reshape((300, 300))
        # Plot the elevation data with masked drainage areas
        im = ax.imshow(elevation_data_masked, cmap='terrain', vmin=np.nanmin(elevation_data), vmax=np.nanmax(elevation_last), origin='lower')
        ax.set_title(f'Time = {frame * interval} years')  # Adjust time interval as needed
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(topo_data), blit=False, interval=1)
    # Save the animation as a GIF
    gif_path = save_path + '/topography_animation.gif'
    ani.save(gif_path, writer=PillowWriter(fps=25), dpi=500)
    print(f"Animation saved to {gif_path}")

    plt.tight_layout()
    plt.show()

create_and_save_animation(out_ds, save_path)
