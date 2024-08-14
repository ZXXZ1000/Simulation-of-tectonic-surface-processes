import time
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rasterio
from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    ChiFinder,
    SteepnessFinder,
    DepressionFinderAndRouter
)

# 读取接口
nc_file = r"C:\Users\Administrator\Videos\fastscape_test\test4\MinUp_step2_1ma\out_Minup_step.nc"
dem_file = r"D:\OneDrive - zju.edu.cn\文档\MATLAB 代码\模拟用DEM\dem_min_fastscape.tif"

def load_data(nc_file, dem_file):
    start_time = time.time()
    # 读取 .nc 格式的模拟地形
    dataset = xr.open_dataset(nc_file)
    elevation_sim = dataset['topography__elevation'].isel(out=int(len(dataset['topography__elevation']) * 0.5)).values.astype(np.float64)
    print(f"Loaded simulated elevation in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    # 读取 .tif 格式的实际地形
    with rasterio.open(dem_file) as src:
        elevation_real = src.read(1).astype(np.float64)
        transform = src.transform
    print(f"Loaded real elevation in {time.time() - start_time:.2f} seconds")

    return elevation_sim, elevation_real, transform

def create_landlab_grid(elevation, dx, dy):
    start_time = time.time()
    ny, nx = elevation.shape
    print(f"Creating Landlab grid with shape: {ny}x{nx}")

    grid = RasterModelGrid((ny, nx), xy_spacing=(dx, dy))
    grid.add_field("topographic__elevation", elevation.astype(np.float64), at="node", clobber=True)
    
    # 检查 elevation 的基本统计信息
    print(f"Elevation stats - min: {np.min(elevation)}, max: {np.max(elevation)}, mean: {np.mean(elevation)}")  
    return grid


def calculate_index(grid):
    start_time = time.time()
    
    # 计算坡度
    slope = grid.calc_slope_at_node()
    print(f"Calculated slope in {time.time() - start_time:.2f} seconds")
    
    # 重置计时器
    start_time = time.time()
    
    # 计算流向和汇流区
    flow_accumulator = FlowAccumulator(grid, 'topographic__elevation')
    flow_accumulator.run_one_step()
    print(f"Calculated flow accumulation in {time.time() - start_time:.2f} seconds")
    
    # 重置计时器
    start_time = time.time()
    
    # 计算chi
    chi_finder = ChiFinder(grid, min_drainage_area=1e6, reference_concavity=0.42)
    chi_finder.calculate_chi()
    print(f"Calculated chi in {time.time() - start_time:.2f} seconds")
    
    # 重置计时器
    start_time = time.time()
    
    # 计算ksn
    steepness_finder = SteepnessFinder(grid, reference_concavity=0.42)
    steepness_finder.calculate_steepnesses()

    # 打印诊断信息
    print(f"Chi range: {grid.at_node['channel__chi_index'].min()} to {grid.at_node['channel__chi_index'].max()}")
    print(f"Steepness range: {grid.at_node['channel__steepness_index'].min()} to {grid.at_node['channel__steepness_index'].max()}")
    print(f"Calculated steepness in {time.time() - start_time:.2f} seconds")
    
    # 重置计时器
    start_time = time.time()
    
    # 计算局部起伏度 (使用简单的高斯滤波方法)
    from scipy.ndimage import gaussian_filter
    elevation = grid.at_node['topographic__elevation'].reshape(grid.shape)
    smoothed_elevation = gaussian_filter(elevation, sigma=5)
    relief = smoothed_elevation - elevation
    print(f"Calculated local relief in {time.time() - start_time:.2f} seconds")
    
    # 使用正确的字段名称
    return (
        slope.reshape(grid.shape),
        grid.at_node['channel__chi_index'].reshape(grid.shape),
        grid.at_node['channel__steepness_index'].reshape(grid.shape),
        relief
    )


def plot_terrain(elevation_sim, elevation_real, indices_sim, indices_real):
    titles = ['Elevation', 'Slope', 'Chi', 'Ksn', 'Relief']
    cmaps = ['terrain', 'viridis', 'plasma', 'magma', 'inferno']
    
    for i, (title, cmap) in enumerate(zip(titles, cmaps)):
        start_time = time.time()
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        im_sim = axes[0].imshow(indices_sim[i], cmap=cmap, origin='lower',vmin = np.min(indices_sim[i]), vmax = np.max(indices_sim[i]))
        im_real = axes[1].imshow(indices_real[i], cmap=cmap, vmin = np.min(indices_real[i]), vmax = np.max(indices_real[i]))
        axes[0].set_title(f'Simulated {title}')
        axes[1].set_title(f'Actual {title}')
        
        fig.colorbar(im_sim, ax=axes[0])
        fig.colorbar(im_real, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
        print(f"Plotted {title} in {time.time() - start_time:.2f} seconds")


# 主程序
if __name__ == "__main__":
    try:
        # 加载数据
        elevation_sim, elevation_real, transform = load_data(nc_file, dem_file)
        
        # 创建Landlab网格
        dx, dy = transform[0], -transform[4]  # 假设栅格是正方形的
        grid_sim = create_landlab_grid(elevation_sim, dx, dy)
        grid_real = create_landlab_grid(elevation_real, dx, dy)
        
        # 计算指标
        indices_sim = [elevation_sim] + list(calculate_index(grid_sim))
        indices_real = [elevation_real] + list(calculate_index(grid_real))
        
        # 绘图
        plot_terrain(elevation_sim, elevation_real, indices_sim, indices_real)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Detailed traceback:")
        import traceback
        traceback.print_exc()
