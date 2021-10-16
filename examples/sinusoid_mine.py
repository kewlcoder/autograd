from builtins import range, map

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

def sin_func(x):
    return np.sin(x)

d_dx = grad(sin_func)
d2_dx2 = grad(d_dx)

x = np.linspace(-10, 10, 200)


plt.xlim([-10, 10])
plt.ylim([-1.5, 1.5])
# plt.axis('off')

plt.plot(x, list(map(d_dx, x)), x, list(map(d_dx, x)), x, list(map(d2_dx2, x))  )
plt.savefig('sinusoid_mine.png')
plt.clf()


'''
Taylor approximation using the series - 
x - x^3/3! + x^5/5! - x^7/7! + ....
'''
def sin_taylor_func(x):
    tot = curr_val = x
    for i in range(1000):
        curr_val = -curr_val*(x**2)/( (2*i + 2)* (2*i + 3))
        tot += curr_val
        if np.abs(curr_val) < 0.2:
            break
    return tot

d_dx = grad(sin_taylor_func)
d2_dx2 = grad(d_dx)

plt.plot(x, list(map( sin_taylor_func, x)), x, list(map(d_dx,x)), x, list(map(d2_dx2, x)) ) 
plt.savefig('sinusoid_taylor_mine.png')
plt.clf()