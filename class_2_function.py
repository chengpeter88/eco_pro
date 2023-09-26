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
cd(3,4,0.5)