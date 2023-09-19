import pandas  as pd
import numpy as np 
import matplotlib.pyplot as plt
from numpy import array
####data type 
###str  int float list tuple dic 
### primtive type 
_string="hello world"
_integer=1
_float=1.0
_boolen=True
_none=None 
###collection data type 
_range= range(5)
_list=["apple","oragne",1,True]
####不限制資料裡面結構型態
_tuple=("apple",True)
_dictionary={"name":"peter","age":36}

###mutability 
###變數的存放 define the variable
a=1###記憶體某個位子回傳，a 是一個記憶體位子
b=a###b拿a的地址
b=2###把other value 放入b 但也會改變a記憶體位子東西
 
###immutability 
###有些特別例子

###list dic 都是mutability  注意會改記憶體中的東西

a=[1,2,3]
a[0]=10
###list 特性可以改裡面的東西

list_ex=[1,2,3,4,5]
dic_ex={"name":"peter","age":36,"birthday":"01/01/1995"}


import numpy as np
game={
    ("top","left"): np.array([2, 4]),
    ("top","right"): np.array([0, 2]),
    ("bottom","left"): np.array([4, 2]),
    ("bottom","right"): np.array([2, 0])
    }

game[("top","left")]