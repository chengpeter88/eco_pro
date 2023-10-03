import matplotlib.pyplot as plt
import numpy as np
 
n = 100 # number of points
# create a x vector
x = np.linspace(-5, 5, n) 
x
# compute y
y = x**2 + 3*x + 5
  
#
plt.plot(x, y)
plt.grid(True)
plt.xlabel('X')
plt.title('$f(x)=x^2+3x+5$')
plt.show()

######
x = np.arange(0, 7, 0.01) 
# formulate a function f
def f(x):
    return x**3 - 10*x**2 + 29*x - 20
fig = plt.figure(figsize=[6, 4])
plt.plot(x, f(x), linestyle = '--', color = 'r')
plt.grid(visible = True, color='g', linestyle='-', linewidth=0.5)
plt.xlabel('X'), plt.ylabel('Y')
plt.title('$f(x)=x^3-10x^2+29x-20$')
plt.show()

xmin, xmax = -5, 10
x = np.linspace(xmin, xmax, 100)
p = [1, -8, 16, -2, 8]
f = lambda x : np.polyval(p, x)
 
fig = plt.figure()
# use axis object
ax = plt.gca() # get current axis
ax.plot(x, f(x), linewidth = 3, color ='g', linestyle ='-.')
ax.grid(True)
ax.set_xlabel('X'), ax.set_ylabel('f(x)')
ax.set_title('The range of $x$ is not appropriate')
# ax.text(0, 2500, 'Re-define the range of $x$')
# save the current figure as an eps file
#plt.savefig('poly.eps', format='eps')
plt.show()



import numpy as np
  # 定義 n 的值
x = np.linspace(-5, 5,200)
#def f(x):
#    return 4 * np.pi  * (np.sin(x) / x)
def f(x):
    return 4 * np.pi  * (np.sin((2*x-1)*x) / (2*x-1)*x)
y = f(x)  # 計算 f(x) 的值
print(y) 
plt.plot(x, y)
plt.grid(True)
plt.show()




n = [1, 2, 3]
colors = ['r', 'g', 'b']
x = np.linspace(-3*np.pi, 3*np.pi, 200)
for i in n:         
    y=np.sin(i*x)   
    plt.plot(x, y, label = 'sin({}x)'.format(i), \
        color = colors[i - 1])
 
plt.legend()
plt.ylim(-2.5, 2.5), plt.grid(True), plt.xlabel('X')
plt.title('Demonstration of legend for each line')
plt.show()


####example
import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x) = sin(x) / x
def f(x):
    return np.sin(x) / x

def g(x):
    return np.sin(x**2)/x

def h(x):
    return np.sin(x*2)/x**2

# Generate x values from -10 to 10
x = np.linspace(-10, 10, 1000)

# Calculate y values for f(x)
y = f(x)

# Plot the function
plt.plot(x, y)

# Add a vertical line at x=0
plt.axvline(x=0, color='black', linestyle='--')

# Add a horizontal line at y=0
plt.axhline(y=0, color='black', linestyle='--')

# Set the x and y limits
plt.xlim(-10, 10)
plt.ylim(-0.5, 1)

# Add a title and axis labels
plt.title('f(x) = sin(x) / x')
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Define the functions f(x), g(x), and h(x)
def f(x):
    return np.sin(x) / x

def g(x):
    return np.sin(x**2) / x

def h(x):
    return np.sin(x*2) / x**2

import numpy as np
import matplotlib.pyplot as plt

# Define the functions f(x), g(x), and h(x)
def f(x):
    return np.sin(x) / x

def g(x):
    return np.sin(x**2) / x

def h(x):
    return np.sin(x*2) / x**2

# Generate x values from -10 to 10
x = np.linspace(-10, 10, 1000)

# Calculate y values for f(x), g(x), and h(x)
y1 = f(x)
y2 = g(x) 
y3 = h(x) 

# Plot the functions
plt.plot(x, y1, label='f(x)', linewidth=1, color='r')
plt.plot(x, y2, label='g(x)', linewidth=2, color='g',linestyle ='--')
plt.plot(x, y3, label='h(x)', linewidth=0.5, color='b',linestyle ='-.')

# Add a legend
plt.legend()

# Add a title and axis labels
plt.title('ex_1 sin(x)')
plt.xlabel('x')
plt.ylabel('y')

plt.ylim(-10, 10)
plt.grid(True)  

# Show the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)

def f(x, alpha):
    return np.exp(alpha*x) / (np.exp(alpha*x) + 1)

y = f(x, 1)

plt.plot(x, y)
plt.axhline(y=1, color='red', linestyle='--')

plt.grid(True)
plt.xlim(-10, 10)
plt.ylim(-2, 2)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
def f(x):
    return np.exp(-x/10) / (np.sin(x))

y = f(x)
plt.plot(x, y)
plt.grid(True)
plt.show()


import numpy as np  
import matplotlib.pyplot as plt 
def f(x):
    return 1/x-2
y = f(x)
np.setdiff1d(np.linspace(-5, 5, 1000),[0])
plt.plot(x, y)
plt.grid(True)
plt.ylim(-10, 10)
plt.legend(True)
plt.xlim(-5, 5)
plt.show()




import numpy as np      
import matplotlib.pyplot as plt 

x = np.linspace(-10, 10, 1000)
def f(x):   
    return 1/np.sqrt(2*np.pi)*np.exp((x-3)**2/2)
y = f(x)
plt.plot(x, y)
plt.grid(True)
plt.ylim(-10, 100)
plt.xlim(-2, 8)
plt.axvline(x=3, color='red', linestyle='--')
plt.show()

import numpy as np      
import matplotlib.pyplot as plt 
x = np.linspace(-10, 10, 1000)  
def f(x):
    return -x**4+3*x**3
y = f(x)
plt.plot(x, y, label='f(x)', linewidth=1, color='blue')    
plt.ylim(-100,30)
plt.xlim(-10, 10)
plt.axhline(y=3, color='red', linestyle='--')
plt.grid(True)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    if 1 <= x < 3:
        return 1
    elif 3 <= x < 5:
        return 2
    elif 5 <= x < 7:
        return 3
    else:
        return np.nan

# Generate x values from 0 to 8
x = np.linspace(0, 8, 1000)

# Calculate y values for f(x)
y = [f(xi) for xi in x]

# Plot the function
plt.plot(x, y)

# Add a title and axis labels
plt.title('Piecewise Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)  
# Show the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return x**3 + 2

# Define the inverse function of f(x)
def inv_f(x):
    return np.cbrt(x - 2)

# Generate x values from -10 to 10
x = np.linspace(-10, 10, 1000)

# Calculate y values for f(x) and f^{-1}(x)
y1 = f(x)
y2 = inv_f(x)

# Plot the functions
plt.plot(x, y1, label='f(x)')
plt.plot(x, y2, label='f$^{-1}$(x)')

# Set the x and y limits to be equal
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.gca().set_aspect('equal', adjustable='box')

# Add a legend
plt.legend()

# Add a title and axis labels
plt.title('Function and Its Inverse')
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()