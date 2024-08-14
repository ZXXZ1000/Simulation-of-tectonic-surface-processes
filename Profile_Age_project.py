import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy import stats

# 预留读取Excel文件的位置
excel_path = r"D:\OneDrive - zju.edu.cn\文档\MATLAB 代码\河流反演\test1\Profile_Age_project\swath data_crossLMS.xlsx"
data = pd.read_excel(excel_path)

# Define a color palette and marker style for each method
method_palette = sns.color_palette("husl", len(data['Method'].unique()))
method_markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'H', 'X']

# Create a dictionary for mapping methods to colors and markers
method_styles = {method: (color, marker) for method, color, marker in zip(data['Method'].unique(), method_palette, method_markers)}

fig, ax1 = plt.subplots(figsize=(8, 6))

def plot_profile_Min(data):
     # Plot the first three columns: Distance(m), min, mean, max
    ax1.plot(data['Distance(m)']/1000, data['min']/1000, label='Min', color='gray')
    ax1.plot(data['Distance(m)']/1000, data['mean']/1000, label='Mean', color='gray')
    ax1.plot(data['Distance(m)']/1000, data['max']/1000, label='Max', color='Black')
    ax1.fill_between(data['Distance(m)']/1000, 0, data['max']/1000, color='gray', alpha=0.2)
    ax1.set_xlabel('Distance (km)',fontsize=14, fontstyle='italic')
    ax1.set_xlim(0, max(data['Distance(m)']) / 1000)
    ax1.set_ylabel('Elevation(km)', fontsize=14, fontstyle='italic')
    ax1.set_ylim(0, 20)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:g}'))
    #ax1.legend(loc='upper left')
    ax1.grid(linestyle='--', alpha=0.2)
    ax1.text(5, 6,'Dadu River', fontsize=10, color='black', fontstyle='italic')
    ax1.text(40, 5.8,'|', fontsize=20, color='black')
    ax1.text(31, 4,'Qionglai\n  shan', fontsize=8, fontstyle='italic', weight='bold', color='black')
    ax1.text(100, 6,'Min River', fontsize=10, color='black', fontstyle='italic')
    ax1.text(180, 5.8,'|', fontsize=20, color='black')
    ax1.text(190, 5,'Minshan', fontsize=8, fontstyle='italic', weight='bold', color='black')
    ax1.text(200, 6,'Fu River', fontsize=10, color='black', fontstyle='italic')
#plot_profile_Min(data)

def plot_profile_LMS(data):
     # Plot the first three columns: Distance(m), min, mean, max
    ax1.plot(data['Distance(m)']/1000, data['min']/1000, label='Min', color='gray')
    ax1.plot(data['Distance(m)']/1000, data['mean']/1000, label='Mean', color='gray')
    ax1.plot(data['Distance(m)']/1000, data['max']/1000, label='Max', color='Black')
    ax1.fill_between(data['Distance(m)']/1000, 0, data['max']/1000, color='gray', alpha=0.2)
    ax1.set_xlabel('Distance (km)',fontsize=14, fontstyle='italic')
    ax1.set_xlim(0, max(data['Distance(m)']) / 1000)
    ax1.set_ylabel('Elevation(km)', fontsize=14, fontstyle='italic')
    ax1.set_ylim(0, 20)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:g}'))
    #ax1.legend(loc='upper left')
    ax1.grid(linestyle='--', alpha=0.2)
    ax1.text(3, 6,'Eastern Tibetan plateau', fontsize=10, color='black', fontstyle='italic')
    ax1.text(84, 5.8,'|', fontsize=20, color='black')
    ax1.text(75, 5.2,'Longriba', fontsize=8, fontstyle='italic', weight='bold', color='black')
    ax1.text(105, 6,'Longmenshan Thrust Belt', fontsize=10, color='black', fontstyle='italic')
    ax1.text(210, 5.8,'|', fontsize=20, color='black')
    ax1.text(220, 4.5,'Pengguan\n Massif', fontsize=8, fontstyle='italic', weight='bold', color='black')
    ax1.text(225, 6,'SIchuan Basin', fontsize=10, color='black', fontstyle='italic')
plot_profile_LMS(data)

def plot_thermochronology(data):
    # Create a second y-axis for the scatter plot
    ax2 = ax1.twinx()
    # 筛选只显示方法为AHE和AFT的数据，并将scatter方法换成plot方法，同时标记errorbar
    for method, (color, marker) in method_styles.items():
        if method in ['AHe', 'AFT','ZHe']:
            method_data = data[data['Method'] == method]
            ax2.scatter(method_data['x'] / 1000, method_data['Age'], marker=marker, color=color, linestyle='-')
            ax2.errorbar(method_data['x'] / 1000, method_data['Age'], yerr=method_data['Std_Dev'], fmt='none', ecolor=color)
            # 添加数字标签
            #for i, txt in enumerate(method_data['Age']):
                #ax2.text(method_data['x'].iloc[i] / 1000-2, method_data['Age'].iloc[i]+0.5, txt, fontsize=8, color=color)

    #计算置信区间并绘制包络图
    for method in ['AHe', 'AFT', 'ZHe']:
        method_data = data[data['Method'] == method]
        ax2.plot(method_data['x'] / 1000, method_data['Age'], marker=method_styles[method][1], linestyle='none', color=method_styles[method][0], label=method,)
        ax2.fill_between(method_data['x'] / 1000, method_data['Age'] - method_data['Std_Dev'], method_data['Age'] + method_data['Std_Dev'], color=method_styles[method][0], alpha=0.2)

    # 设置第二个y轴的标签
    ax2.set_ylabel('Thermochronology Age (Ma)',fontsize=14, fontstyle='italic')
    ax2.set_yscale('log')
    ax2.set_ylim(0.01, 1000)

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
#plot_thermochronology(data)

def plot_total_denudation(data):
    # Create a second y-axis for the scatter plot
    ax2 = ax1.twinx()
    # 筛选只显示方法为AHE和AFT的数据，并将scatter方法换成plot方法，同时标记errorbar
    for method, (color, marker) in method_styles.items():
        if method in ['AHe', 'AFT','ZHe']:
            method_data = data[data['Method'] == method]
            ax2.scatter(method_data['x'] / 1000, method_data['Total Denudation'], marker=marker, color=color, linestyle='-')
            ax2.errorbar(method_data['x'] / 1000, method_data['Total Denudation'], yerr=method_data['Std_Dev_Denu'], fmt='none', ecolor=color)
            # 添加数字标签
            #for i, txt in enumerate(method_data['Age']):
                #ax2.text(method_data['x'].iloc[i] / 1000-2, method_data['Age'].iloc[i]+0.5, txt, fontsize=8, color=color)

    #计算置信区间并绘制包络图
    for method in ['AHe', 'AFT', 'ZHe']:
        method_data = data[data['Method'] == method]
        ax2.plot(method_data['x'] / 1000, method_data['Total Denudation'], marker=method_styles[method][1], linestyle='none', color=method_styles[method][0], label=method,)
        ax2.fill_between(method_data['x'] / 1000, method_data['Total Denudation'] - method_data['Std_Dev_Denu'], method_data['Total Denudation'] + method_data['Std_Dev_Denu'], color=method_styles[method][0], alpha=0.2)

    # 设置第二个y轴的标签
    ax2.set_ylabel('Denudation Rates (km/ma)',fontsize=14, fontstyle='italic')
    ax2.set_yscale('log')
    ax2.set_ylim(0.001, 10)

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
plot_total_denudation(data)

#plt.title('Thermochronology Data', fontsize=16,fontstyle='italic')
plt.show()

