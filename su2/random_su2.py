import numpy as np

from lalgebra import *

def generate_su2_pool(su2_pool_size,eps):
    su2_pool = np.zeros((su2_pool_size,2,2) , np.complex)   #su3 3x3 matrices for updating the links, including the inverses
   
    for i in range(su2_pool_size):
        
        r0  = np.random.uniform(-0.5,0.5)
        x0  = np.sign(r0)*np.sqrt(1-eps**2)
        
        r   = np.random.random((3)) - 0.5      
        x   = eps*r/np.linalg.norm(r)

        su2_pool[i] = x0*np.identity(2) + 1j*x[0]*sx + 1j*x[1]*sy + 1j*x[2]*sz     
    return su2_pool
    
def sort_from_pool(pool, pool_size):
    index = int( (pool_size-1)*np.random.random() )
    return pool[index]
