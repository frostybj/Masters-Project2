import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1
import matplotlib.pyplot as plt
import time

r_list  = np.linspace(0.00001, 9.0, 90000)   # spacing ~0.001
k1_list = k1(r_list) 

def besselfunc(k1_list,r_list,r):
    return np.interp(r,r_list,k1_list)


#check
j = np.linspace(0.00001,3,10000000)
actual = []
func = []
for i in j:
    actual.append(k1(i))
    start = time.time()
    func.append(besselfunc(k1_list,r_list,i))
    end = time.time()
    print("Time taken to Bessel ", end - start)

plt.plot(j,actual,label = "Actual")
plt.plot(j,func,label = "Func")
plt.xlabel("r")
plt.ylabel("k1")
plt.legend()
plt.show()





# def bessel_func(r):

   

    

