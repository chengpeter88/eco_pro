import numpy as np

def ces_constructor(alpha, sigma):
    def ces_utility(X):
        if not isinstance(alpha, (float, int, np.ndarray)):
            raise ValueError("Alpha must be a float, int, or numpy array.")
        
        if isinstance(alpha, (float, int)):
            alpha_array = np.full(X.shape, alpha)
        else:
            alpha_array = alpha
        
        return np.sum(alpha_array * X**(-sigma))**(-1/sigma)
    
    return ces_utility

# Create a CES utility function with alpha=0.7 and sigma=0.5
ces = ces_constructor(0.7, 0.5)

# Test the CES utility function with X as a numpy array
X = np.array([1.0, 2.0, 3.0])
result = ces(X)
print("CES Utility Function Result:", result)



import numpy as np
def ces_constructor(alpha, sigma):
    def ces_utility(X):
        if isinstance(alpha, (float, int)):
            alpha_array = np.full(X.shape, alpha)
        else:
            alpha_array = alpha
        return np.sum((X**(-sigma)) * alpha_array)**(-1/sigma)
    return ces_utility


ces = ces_constructor(0.7, 0.5)
X = np.array([1, 2, 3])
ces_value = ces(X)
print("CES Utility Value:", ces_value)





import numpy as np
def ces_ge_par(alpha, sigma):
    def ces_ge(x):
        X = np.array(x)  # 将输入参数转化为NumPy数组
        return sum(X**(-sigma)*alpha)**(-1/sigma)  # 使用 ** 进行幂次计算
    return ces_ge

ces_ge = ces_ge_par([0.7, 0.2, 0.1], 0.5)  # 修复了 sigma 参数为标量值
result = ces_ge([1, 2, 3])  # 调用 ces_ge 函数并传递参数
print(result)

import olg2 as olg 

ce = olg.consumer_example() 
print(ce)


def Constructor_Cobb_Do(u, alpha):
    def Cobb_Do(x, y):
        return u(x, y, alpha)
    return Cobb_Do

Constructor_Cobb_Do(lambda x, y, alpha: x**alpha*y**(1-alpha), 0.3)(1, 2)


def Constructor_Cobb_Douglas(alpha):
    def cobb(x,y):
        return x**alpha*y**(1-alpha)
    return cobb
cobb = Constructor_Cobb_Douglas(0.5)
cobb(1, 2)
cobb(1, 3)

class Consumer:
    def __init__(self, beta, u):
        # Initialize the discount factor and utility function for the consumer
        self.beta = beta
        self.u = u
    
    def saving(self, w, R):
        # Calculate the optimal amount of savings for the consumer given their wage and the interest rate
        return (self.beta/(1+self.beta))*w

# Create a new Consumer object with a discount factor of 0.9 and a logarithmic utility function
c = Consumer(0.9, lambda x: np.log(x))

# Calculate the optimal amount of savings for a consumer with a wage of 100 and an interest rate of 0.05
savings = c.saving(100, 0.05)

# Print the result
print("Optimal savings:", savings)
#########
def derivative(f,x, h=1e-5):
    return (f(x+h)-f(x))/h

def f(x):
    return x**2

derivative(f,1, h=1e-6)
#########
def Derivative(f):
    def df(x,h=1e-5):
        return (f(x+h)-f(x))/h
    return df
df=Derivative(f)
df(1,h=1e-6)

def f(x):
    return x[0]**2+x[1]
f([1,2])

def gradinet(f,x,h=1e-5):
    grad=[]
    for i in range(len(x)):
        x1=x.copy()
        x1[i]=x1[i]+h
        grad.append((f(x1)-f(x))/h)
    return grad
gradinet(f,[1,2,3,4,5,6])

def gradient(f, x, h=0.00001):
    x0 = x.copy()
    x0[0] += h # += is equivalent to x0[0] = x0[0] + h but more efficient
    x1 = x.copy()
    x1[1] += h
    return np.array([(f(x0)-f(x))/h, (f(x1)-f(x))/h])

gradient(f, [1, 2], 1e-6), gradient(f, [8, 2])


def gradient(x_0, x_1, h=1e-5):
    def f(x_0,x_1):
        return x_0**2*x_1
    import numpy as np
    return np.array([(f(x_0+h,x_1)-f(x_0,x_1))/h,(f(x_0,x_1+h)-f(x_0,x_1))/h])
gradient(1,2)

########################

# k_{t+1} = [(s z k^α_t) + (1 - δ)k_t] /(1 + n)
class Solow:
    def __init__(self,n=0.05,s=0.1,δ=0.1, α=0.3,k=1.0,z=2.0) :
        self.n=n
        self.s=s    
        self.δ=δ    
        self.α=α    
        self.k=k 
        self.z=z   
    def h(self):
        n=self.n
        s=self.s    
        δ=self.δ
        α=self.α
        k=self.k    
        z=self.z
        return(s*z*self.k**α+(1-δ)*self.k)/(1+n)
    def update(self):
        self.k=self.h()

    def steady_state(self):
        n=self.n
        s=self.s    
        δ=self.δ
        α=self.α
        k=self.k    
        z=self.z
        return ((s*z)/(n+δ))**(1/(1-α)) 
    
    def generate_sequence(self,t):
        path=[]
        for i in range(t):
            path.append(self.k)
            self.update()
        return path
    

s1=Solow()
s2=Solow(k=8.0)
T=60
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot([s1.steady_state()]*T, 'k-', label='steady state')

for s in s1,s2:
    lb = f'capital series from initial state {s.k}'
    ax.plot(s.generate_sequence(T), 'o-', lw=2, alpha=0.6, label=lb)
    ax.legend(loc='upper right')

plt.show()




class Market:

    def __init__(self, ad, bd, az, bz):
     
        self.ad, self.bd, self.az, self.bz,  = ad, bd, az, bz
        if ad < az:
            raise ValueError('Insufficient demand.')

    def price(self):
        "Compute equilibrium price"
        return  (self.ad - self.az + self.bz * self.tax) / (self.bd + self.bz)

    def quantity(self):
        "Compute equilibrium quantity"
        return  self.ad - self.bd * self.price()
#################


class Demand:
    def __init__(self, ad, bd, ):
        self.ad = ad
        self.bd = bd

    def quantity(self, price):
        "Compute quantity demanded at given price"
        return self.ad - self.bd * price 

    def __str__(self):
        return f'Demand(ad={self.ad}, bd={self.bd})'

class Supply:
    def __init__(self, az, bz, ):
        self.az = az
        self.bz = bz

    def quantity(self, price):
        "Compute quantity supplied at given price"
        return self.az + self.bz * price 
    def __str__(self):
        return f'Supply(az={self.az}, bz={self.bz}, tax={self.tax})'

class Market:
    def __init__(self, demand, supply):
        self.demand = demand
        self.supply = supply

    def price(self):
        "Compute equilibrium price"
        return (self.demand.ad - self.supply.az + self.supply.bz * self.supply.tax) / (self.demand.bd + self.supply.bz)

    def quantity(self):
        "Compute equilibrium quantity"
        return self.demand.quantity(self.price())

    def __str__(self):
        return f'Market(demand={self.demand}, supply={self.supply})'
    
################## Create demand and supply curves
demand = Demand(ad=100, bd=0.5 )
supply = Supply(az=20, bz=0.3)

# Create market instance
market = Market(demand=demand, supply=supply)

# Compute equilibrium price and quantity
price = market.price()
quantity = market.quantity()

# Print results
print(f'Equilibrium price: {price:.2f}')
print(f'Equilibrium quantity: {quantity:.2f}')
print(market)