from scipy.optimize import fsolve

# 定义需求函数
def demand(p):
    q0d = 100 - 10 * p[0] - 5 * p[1]
    q1d = 50 - p[0] - 10 * p[1]
    return [q0d, q1d]


# 定义供应函数
def supply(p):
    q0s = 10 * p[0] + 5 * p[1]
    q1s = 5 * p[0] + 10 * p[1]
    return [q0s, q1s]

# 定义均衡条件函数
def equilibrium_condition(p):
    return [demand(p)[0] - supply(p)[0], demand(p)[1] - supply(p)[1]]

# 初始价格猜测值
initial_guess = [1, 1]

# 求解均衡价格和数量
equilibrium_prices = fsolve(equilibrium_condition, initial_guess)

# 输出均衡价格和数量
print("均衡价格：", equilibrium_prices)
print("均衡数量：", demand(equilibrium_prices))
