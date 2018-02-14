import numpy as np
import numpy.linalg as la

def inner(u,v):             #Cn inner product
    u = u.conj().T
    inner = np.inner(u,v)
    return inner

def cross(u,v):
    cross = np.cross(u,v)
    return cross
        
def three_product(u1,u2,u3):   #product of 3 complex matrices
    ans = np.dot( np.dot( u1 , u2 ), u3)
    return ans

def unitarize(matrix):
    u   = matrix[0,:]
    v   = matrix[1,:]
    uxv = matrix[2,:]
    
    u /= la.norm(u)
    v = v - u*(np.dot(v,u.conj()))           
    v /= la.norm(v)
    uxv = cross(u.conj(),v.conj())
    uxv /= la.norm(uxv)
    
    matrix[0,:] = u
    matrix[1,:] = v
    matrix[2,:] = uxv
    return matrix

