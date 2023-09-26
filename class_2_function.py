from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
###

######
#any()
#all()
def f(x):
    if x > 0:
        result = 1
    elif x == 0:
        result = 0
    else:
        result = -1
    return result

######
def f(x):
    value=2*x+1
    return value

result=f(1)  ### send back 3 to result
print(result)
##### return mean send back value

### no change under the hood
def changeFirstToZero_keepx(x):
    x_copy=x.copy()
    x_copy[0]=0
    return x_copy

#####list 是可以改變的

######change under the hood
def changeFirstToZero(x):
    x[0]=0

xx=[-3,1,4]
changeFirstToZero(xx) #x=xx
xx

yy=[-3,1,4]
changeFirstToZero_keepx(yy)
yy

import numpy as np
def f(x):
    if x<=0:
        return False
    return np.log(x)
f(-10)
#善用return可以使函數更有效率的提早結束


def f(x):
    def power3(z):
        return z**3 
    return power3(x)


x=4
def a():
    y=1
    def b(z): 
        print(y)
        print(x)
        print(z)
    b(3)

a()

########anonoymous function
f= lambda X:x**2

#######
def preference(x,y):
    return x**0.8*y**0.2

consumer = {
    'name': 'John Doe',
    'age': 30,
    'preference': preference
}

consumer = {
    'name': 'John Doe',
    'age': 30,
    'preference': lambda x,y: x**0.8*y**0.2
}

def cd(x,y,alpha):
    return x**alpha*y**(1-alpha)
alpha=0.3
cd(3,4)
########
cd(3,4,0.5)
########
def Cobb_Douglas(alpha):
    def cd(x, y):
        return x**alpha*y**(1-alpha)
    return cd
cd = Cobb_Douglas(0.3)
cd(1, 2), cd(3,7)
#######
def ces_para(sigma,alpha):
    def ces(x, y):
        return (alpha*x**(-sigma)+(1-alpha)*y**(-sigma))**(-1/sigma)
    return ces

ces=ces_para(0.5,0.7)
ces(1,2)

########CES general try
import numpy as np 
x = np.array([1.0, 2.0, 3.0])
sigma = 0.5
alpha = np.array([0.7, 0.2, 0.1])
x**(-sigma)*alpha
sum(x**(-sigma)*alpha)

def ces_ge_par(alpha,sigma):
    sigma = 0.5
    alpha = np.array([0.7, 0.2, 0.1])
    def ces_ge(X):
        X=np.array([])
    return sum(x**(-sigma)*alpha)^(-1/sigma)


def dev(f,x,h=0.000001):
    return (f(x+h)-f(x))/h

def Dev(f):
    def df(x,h=1e-5):
        return (f(x+h)-f(x))/h
    return df




########
class Consumer:
    def __init__(self, beta, u):
        self.beta = beta # self is mutable. 
        self.u = u
    def saving(self, w, R):
        return (self.beta/(1+self.beta))*w

import numpy as np
ce = Consumer(0.9, lambda c: np.log(c)) 
# a lambda function example
ce