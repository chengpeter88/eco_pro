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