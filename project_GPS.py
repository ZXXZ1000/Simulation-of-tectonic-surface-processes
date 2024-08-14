import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy import stats

# 读取CSV文件
data = pd.read_csv(r"D:\OneDrive - zju.edu.cn\MR.Z  所有资料\Phd Period\大地测量数据\大地变形测量数据_含隆升分量\points_upliftmin_projection.csv")

# 检查数据
print(data.head())

# 创建LMS profile的图表
fig, ax1 = plt.subplots(figsize=(8, 6))

# 绘制LMS profile (假设x是Distance, y是Elevation)
ax1.plot(data['Distance(m)'], data['min'], label='min', color='gray')
ax1.plot(data['Distance(m)'], data['mean'], label='mean', color='gray')
ax1.plot(data['Distance(m)'], data['max'], label='max', color='black')
ax1.fill_between(data['Distance(m)'], 0, data['max'], color='gray', alpha=0.2)

# 设置坐标轴标签和刻度格式
ax1.set_xlabel('Distance (m)', fontsize=14, fontstyle='italic')
ax1.set_ylabel('Elevation (m)', fontsize=14, fontstyle='italic')
ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:g}'))
ax1.grid(linestyle='--', alpha=0.2)

# 给定的 x 值，将数据分为两部分
split_x = 200000  # 假设给定的 x 值为 150000

# 绘制散点图和拟合线
if 'VU_TRUE' in data.columns:
    ax2 = ax1.twinx()
    
    # 筛选有效数据点
    valid_data = data.dropna(subset=['VU_TRUE'])
    
    # 分成两部分
    part1 = valid_data[valid_data['x'] <= split_x]
    part2 = valid_data[valid_data['x'] > split_x]
    
    # Convert 'sigma_vU' column to numeric
    part1['sigma_vU'] = pd.to_numeric(part1['sigma_vU'], errors='coerce')
    part2['sigma_vU'] = pd.to_numeric(part2['sigma_vU'], errors='coerce')

    # 绘制散点图
    ax2.errorbar(part1['x'], part1['VU_TRUE'], yerr=part1['sigma_vU'], fmt='o', color='#CE7698', label=f'Minshan Thrust Belt', alpha=0.8)
    ax2.errorbar(part2['x'], part2['VU_TRUE'], yerr=part2['sigma_vU'], fmt='o', color='#7698CE',  label=f'Fujiang Basin', alpha=0.8)
    ax2.set_ylabel('Uplift Rate (mm/y)', fontsize=14, fontstyle='italic')
    #ax2.set_yscale('log')
    
    # 设置ax2的y轴刻度格式为非科学计数法
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.yaxis.get_major_formatter().set_scientific(False)
    ax2.yaxis.set_minor_formatter(ScalarFormatter())
    ax2.yaxis.get_minor_formatter().set_scientific(False)
    ax2.ticklabel_format(style='plain', axis='y')
    
    # 计算部分1的线性拟合，并标注斜率
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(part1['x'], part1['VU_TRUE'])
    line1 = slope1 * part1['x'] + intercept1
    ax2.plot(part1['x'], line1, color='#CE7698', linestyle='--', linewidth=1.5 )
    
    
    # 计算部分2的线性拟合，并标注斜率
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(part2['x'], part2['VU_TRUE'])
    line2 = slope2 * part2['x'] + intercept2
    ax2.plot(part2['x'], line2, color='#7698CE', linestyle='--', linewidth=1.5)
    
    #绘制part1和part2标准差的包络线
    ax2.fill_between(part1['x'], part1['VU_TRUE'] - part1['sigma_vU'], part1['VU_TRUE'] + part1['sigma_vU'], color='#CE7698', alpha=0.2)
    ax2.fill_between(part2['x'], part2['VU_TRUE'] - part2['sigma_vU'], part2['VU_TRUE'] + part2['sigma_vU'], color='#7698CE', alpha=0.2)
        
    # 添加斜率标注
    ax2.text(50000, 12, f'Average shortening ratio: {slope1/1000:.0e}/y', fontsize=12, color='#CE7698', fontstyle='italic')
    ax2.text(190000, 10, f'Average shortening ratio: {slope2/1000:.0e}/y', fontsize=12, color='#7698CE', fontstyle='italic')
    # Calculate the mean value of y2
    mean_y1 = np.mean(part1['VU_TRUE'])
    mean_y2 = np.mean(part2['VU_TRUE'])
    ax2.text(250000,-1.2, f'Mean value: {mean_y2:.2f}', fontsize=12, color='#7698CE', fontstyle='italic')
    
  
    #绘制part1最后一个点和part2第一个点的连线，并标记Velocity magnitud差值
    ax2.plot([split_x, split_x], [line1.iloc[-1], line2.iloc[0]], color='orange', linestyle='-', linewidth=3 , alpha=0.7,label='Velocity fall')
    ax2.text(160000, 0, f'{mean_y1- mean_y2:.2f}mm/y', fontsize=12, color='orange', fontstyle='italic')
    

# 添加图例
handles1, labels1 = ax1.get_legend_handles_labels()
if 'VU_TRUE' in data.columns:
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
else:
    ax1.legend(loc='upper right')

# 设置轴范围
ax1.set_ylim(0, 20000)
if 'VU_TRUE' in data.columns:
    ax2.set_ylim(-4, 6)

# 显示图表
plt.title('Minshan Measured Uplift', fontsize=16, fontstyle='italic')
plt.show()
