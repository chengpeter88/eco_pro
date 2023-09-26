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