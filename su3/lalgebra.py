import numpy as np
import numpy.linalg as la

sx = np.array(((0, 1),( 1, 0)))
sy = np.array(((0, -1j),(1j, 0)))
sz = np.array(((1, 0),(0, -1)))

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

def generate_X2(su3_pool_size,eps):       
    su2_matrix = np.zeros((3*su3_pool_size,2,2) , np.complex128) 
    
    for i in range(3*su3_pool_size):
        
        r0  = np.random.uniform(-0.5,0.5)
        x0  = np.sign(r0)*np.sqrt(1-eps**2)
        
        r   = np.random.random((3)) -0.5
        x   = eps*r/la.norm(r)

        su2_matrix[i] = x0*np.identity(2) + 1j*x[0]*sx + 1j*x[1]*sy + 1j*x[2]*sz     
    return su2_matrix
