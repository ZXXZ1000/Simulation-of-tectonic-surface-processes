import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
from scipy.stats import stats
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
from scipy.stats import t

#实测值数据路径
data_path = r'D:\OneDrive - zju.edu.cn\文档\MATLAB 代码\河流反演\多元回归分析.xlsx' 
data = pd.read_excel(data_path) 
#输入Savg值数据路径
dataSavg = pd.read_excel(r'D:\OneDrive - zju.edu.cn\文档\MATLAB 代码\河流反演\test4\basin_config.xlsx', sheet_name=0)
Savg = dataSavg['mean_gradie']
#输出预测E值数据路径
output_path = r'D:\OneDrive - zju.edu.cn\文档\MATLAB 代码\河流反演\test4\Erosion_rate_refit.xlsx'

# Constants
Sc = 0.76
D = 0.003
beta = 2
K = 1e-6  # Assume some value for K
n = 1

# Function to calculate E from Savg
def NonlinearFit(Savg, a=3.91e-5, b=4.01):
    return a * np.exp(b * Savg)

# Function to find L_H from E
def solve_LH(E, initial_guess=1e-4):
    def equation_for_LH(L_H):
        left_side = (E / (2 * K * L_H**2))
        right_side = (D * Sc**2 / (beta * E * abs(L_H))) * (np.sqrt(1 + (beta * E * L_H / (D * Sc))**2) - 1)
        return left_side - right_side
    return fsolve(equation_for_LH, initial_guess)[0]

# Function for z(x)
def z(x, E, L_H):
    factor = (-Sc**2 / (2 * beta * E))**(1/n)
    sqrt_term = np.sqrt(D**2 + (2 * beta * E * x / Sc)**2)
    ln_term = np.log((sqrt_term + D) / (2 * beta * E / Sc))
    return factor * (sqrt_term - D * ln_term)

# Function to calculate Savg
def compute_Savg(E):
    L_H = solve_LH(E)  # Get L_H from the E-L_H relationship
    z0 = z(0, E, L_H)
    zLH = z(L_H, E, L_H)
    return (z0 - zLH) / L_H

#用Savg计算E
def compute_E(Savg):
    def equation_for_E(E):
        return compute_Savg(E) - Savg
    return fsolve(equation_for_E, 1e-4)[0]

# Generate a range of E values
E_values = np.linspace(1e-5, 1e-3, 1000)
# Calculate Savg values for a range of E values
Savg_values = [compute_Savg(E) for E in E_values]
# Plot the relationship between Savg and E
plt.figure(figsize=(8, 6))
plt.plot(Savg_values,E_values, color='gray', label='HillslopeDiffusion Model\n (Roering et al.2007)',linewidth=2)


#get proved E and S_avg
Gradient = data['Mean Gradient']
Erosion = data['E(m/y)']
Erosion_Stdev = data['E_Std']
sorted_indices = np.argsort(Gradient)
Gradient_sorted = Gradient[sorted_indices]


#plot the points of NonlinearFit(Savg)
Savg_values1 = np.linspace(0, 1, 1000)
E_values_linearfit = [NonlinearFit(savg) for savg in Savg_values1]
plt.plot(Savg_values1,E_values_linearfit, color='black', label='Linear Fit Model(This Study)\n' + r'$E = 3.91 \times 10^{-5} \times e^{4.01 \times S_{mean}}$',linewidth=2)
r2_value1 = r2_score(Erosion, [NonlinearFit(savg) for savg in Gradient])
plt.text(0.9, 0.8, f'R² = {r2_value1:.4f}', transform=plt.gca().transAxes, fontsize=12, weight='bold', fontstyle='italic', color='black')

#plot the points of proved E and S_avg
plt.scatter(Gradient, Erosion, color='#B32626', label='Measured Erosion and Slope\n(Ansberque et al., 2015)',marker='o',s=80,alpha=0.8, edgecolors='black')
#计算Gradient-Erosion散点与E-S_avg曲线在散点范围内的R^2值
# Calculate R^2 value
r2_value = r2_score(Erosion, [compute_E(Savg) for Savg in Gradient])
# 在图中标注R^2的值
plt.text(0.6, 0.8, f'R² = {r2_value:.4f}', transform=plt.gca().transAxes, fontsize=12, weight='bold', fontstyle='italic', color='gray')

# Fit the nonlinear model to the data to estimate parameters
params, cov = curve_fit(NonlinearFit, Gradient_sorted, Erosion)
# Function to predict E values and compute standard error of the prediction
def predict_with_confidence(x, params, cov):
    # Predicted values using the fitted model
    pred = [NonlinearFit(savg) for savg in x]
    # Standard error calculation
    x = np.array(x)
    # Jacobian of the NonlinearFit function with respect to parameters 'a' and 'b'
    J = np.vstack([np.exp(params[1] * x), params[0] * x * np.exp(params[1] * x)]).T
    # Variance in predictions
    var_pred = np.sum(J @ cov * J, axis=1)
    sigma = np.sqrt(var_pred)
    return pred, sigma
# Generate predictions and confidence intervals
predicted_E, sigma_E = predict_with_confidence(Gradient_sorted, params, cov)
#Predicted_E_sorted = predicted_E[sorted_indices]


# Confidence interval calculation
alpha = 0.05  # significance level
t_value = t.ppf(1 - alpha / 2, len(Gradient_sorted) - 1)  # t-value for confidence interval
predicted_E_upper = predicted_E + t_value * sigma_E
predicted_E_lower = predicted_E - t_value * sigma_E
# Plotting predicted values and confidence intervals
plt.fill_between(Gradient_sorted, predicted_E_lower, predicted_E_upper, color='black', alpha=0.2, label='95% Confidence Interval of\n Linear Fit Model')
#拟合predicted_E_upper和predicted_E_lower延长线



#y轴取对数
plt.xlabel('Mean Gradient', fontsize=14, weight='bold', fontstyle='italic')
plt.xlim(0, 0.8)
plt.ylabel('Erosion Rate(m/y)', fontsize=14, weight='bold')  # 使用LaTeX格式和加粗选项
#plt.yscale('log')
plt.ylim(2e-5, 0.8e-3)
# 设置y轴为科学计数法表示
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.title('Mean Gradient vs Erosion Rate', fontsize=16, weight='bold', fontstyle='italic')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)


# 代入已知的S_avg值，计算E值
E_compute_E = []
E_NonlinearFit = []
for i in range(len(Savg)):
    E_compute_E.append(compute_E(Savg[i]))
    E_NonlinearFit.append(NonlinearFit(Savg[i]))


    
#初始化表格，第一列为S_avg，第二列为计算得到的E
df = pd.DataFrame({'Mean Slope': Savg, 'Erosion Rate by HillslopeDiffusion Model': E_compute_E, 'Erosion Rate by NonlinearFit Model': E_NonlinearFit})
#将表格写入文件
df.to_excel(output_path, index=False)

plt.show()
