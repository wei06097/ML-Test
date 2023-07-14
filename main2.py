import pandas as pd
data = pd.read_csv("./Salary_Data2.csv")
# 資料預處理 Label Encoding
data["EducationLevel"] = data["EducationLevel"].map({
        "高中以下" : 0,
        "大學" : 1,
        "碩士以上" : 2
    })
# 資料預處理 One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(data[["City"]])
data[["CityA", "CityB", "CityC"]] = onehot_encoder.transform(data[["City"]]).toarray()
data = data.drop(["City", "CityC"], axis=1)
# 參數
x = data[["YearsExperience", "EducationLevel", "CityA", "CityB"]]
y = data["Salary"]
# 資料預處理 訓練集、測試集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

# ======================================== #
# 特徵縮放 => 梯度下降會比較快
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) # 只能放訓練的
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# ======================================== #
import numpy as np

def compute_cost(x, y, w, b):
    y_pred = (w * x).sum(axis=1) + b
    cost = ((y - y_pred) ** 2).mean()
    return cost

def compute_gradient(x, y, w, b):
    y_pred = (w * x).sum(axis=1) + b
    gradient_w = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        gradient_w[i] = (-2*x[:, i]*(y - y_pred)).mean()
    gradient_b = (-2*(y - y_pred)).mean()
    return gradient_w, gradient_b

def gradient_descent(x, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter):
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
    return w, b, w_hist, b_hist, c_hist

# ======================================== #
w_init = np.array([1, 2, 2, 4])
b_init = 0
learning_rate = 0.01
run_iter = 5000
w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x_train, y_train, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter)

# 比較輸出
y_pred = (w_final*x_test).sum(axis=1) + b_final
result = pd.DataFrame({
    "y_pred" : y_pred,
    "y_test" : y_test
})
print(result)
# 比較cost
cost_test = compute_cost(x_test, y_test, w_final, b_final)
print(f"\n測試集 cost: {cost_test}", f"\n訓練集 cost: {c_hist[len(c_hist)-1]}\n")

# ======================================== #
# [5.3 碩士以上 城市A] [7.2 高中以下 城市B]
x_real = np.array([[5.3, 2, 1, 0], [7.2, 0, 0, 1]])
x_real = scaler.transform(x_real)
y_real = (w_final*x_real).sum(axis=1) + b_final
print(y_real)
