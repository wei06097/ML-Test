import pandas as pd

data = pd.read_csv("./Salary_Data.csv") # 年資對應薪水
x = data["YearsExperience"]
y = data["Salary"]

def compute_gradient(x, y, w, b):
    w_gradient = (2*x*(w*x+b-y)).mean()
    b_gradient = (2*(w*x+b-y)).mean()
    return w_gradient, b_gradient

def compute_cost(x, y, w, b):
    y_pred = w * x + b
    cost = (y_pred - y) ** 2
    cost = cost.mean()
    return cost

def gradient_descent(x, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter=0):
    w = w_init
    b = b_init
    cost = compute_cost(x, y, w, b)
    w_hist = [w]
    b_hist = [b]
    c_hist = [cost]
    for i in range(run_iter):
        w_gradient, b_gradient = gradient_function(x, y, w, b)
        w = w - w_gradient * learning_rate
        b = b - b_gradient * learning_rate
        cost = cost_function(x, y, w, b)
        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)
        if (p_iter == 0): continue
        if ((i+1) % p_iter != 0): continue
        print(f"cost{i+1:5}: {cost: .4e}", f"w: {w: .2e}", f"b: {b: .2e}", f"w_gradient: {w_gradient: .2e}", f"b_gradient: {b_gradient: .2e}")
    return w, b, w_hist, b_hist, c_hist

# ======================================== #
w_init = -100
b_init = -50
learning_rate = 0.001
run_iter = 5000
w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x, y, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter)


YearsExperience = 3.5
Salary = w_final * YearsExperience + b_final
print(f"年資: {YearsExperience}  預測薪水: {Salary:.2f}k")

# ======================================== #
import numpy as np
import matplotlib.pyplot as plt

# 畫出 cost 變化
# plt.plot(np.arange(0, len(c_hist)), c_hist)
# plt.xlabel("iteration")
# plt.ylabel("cost")
# plt.show()

# ======================================== #
# 設定顯示範圍
ws = np.arange(-100, 101)
bs = np.arange(-100, 101)
costs = np.zeros((201, 201))

i = 0
for w in ws: 
  j = 0
  for b in bs: 
    cost = compute_cost(x, y, w, b)
    costs[i,j] = cost
    j += 1
  i += 1

# ======================================== #
# 3D 繪圖設定
plt.figure(figsize=(7, 7))
ax = plt.axes(projection="3d")
ax.view_init(20, -65)
b_grid, w_grid = np.meshgrid(bs, ws)
# 畫出曲面
ax.plot_surface(w_grid, b_grid, costs, alpha=0.5)
ax.set_title("cost with w & b")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("cost")
# 畫出起始點 最低點 歷史紀錄
w_index, b_index = np.where(costs == np.min(costs))
ax.scatter(ws[w_index], bs[b_index], costs[w_index, b_index], color="red", s=40)
ax.scatter(w_hist[0], b_hist[0], c_hist[0], color="green", s=40)
ax.plot(w_hist, b_hist, c_hist)
plt.show()
