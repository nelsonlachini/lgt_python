import numpy as np
#import numpy.linalg as la
import matplotlib.pyplot as plt
#from sys import exit
#from scipy.optimize import curve_fit

from lalgebra import *
from random_su2 import *
from statistics import *
     
#######################################METHODS              
def init_lattice(parameter=0):
    for t in range(N):
        for x in range(N):
            for y in range(N):
                    for z in range(N):
                        for mi in range(4): 
                            if(np.random.uniform(0,1)>parameter):                             
                                U[t,x,y,z,mi] = np.identity(group_dim)    
                            else:
                                U[t,x,y,z,mi] = sort_from_pool(su2_pool , su2_pool_size)                            

def calculate_staple(t,x,y,z,mi):
    staple = np.zeros( (group_dim,group_dim) , np.complex )
    ami = [0,0,0,0]
    ami[mi] = 1
    
    for ni in range(4):
        if ni != mi :
            ani = [0,0,0,0]
            ani[ni] = 1  

            staple += three_product( 
            U[(t+ami[0])%N , (x+ami[1])%N , (y+ami[2])%N , (z+ami[3])%N , ni] ,
            U[(t+ani[0])%N , (x+ani[1])%N , (y+ani[2])%N , (z+ani[3])%N , mi].conj().T ,
            U[t , x , y , z , ni].conj().T ) 
            
            staple += three_product(
            U[(t+ami[0]-ani[0])%N , (x+ami[1]-ani[1])%N , (y+ami[2]-ani[2])%N , (z+ami[3]-ani[3])%N , ni].conj().T ,
            U[(t-ani[0])%N , (x-ani[1])%N , (y-ani[2])%N , (z-ani[3])%N , mi].conj().T ,
            U[(t-ani[0])%N , (x-ani[1])%N , (y-ani[2])%N , (z-ani[3])%N , ni]) 
        
    return staple        
    

def update_links_hb(t,x,y,z,mi):          
    A = calculate_staple(t,x,y,z,mi)
    a = np.sqrt(np.linalg.det(A)).real
    if(a == 0.):
        r0  = np.random.uniform(-0.5,0.5)
        x0  = np.sign(r0)*np.sqrt(1-eps**2)
        
        r   = np.random.random((3)) - 0.5      
        x   = eps*r/np.linalg.norm(r)

        U[t,x,y,z,mi] = x0*np.identity(2) + 1j*x[0]*sx + 1j*x[1]*sy + 1j*x[2]*sz
    else:

        ########generating x0
        r_g = np.random.random(3)        
        lamb2 = (np.log(r_g[0]) + (np.cos(2*np.pi*r_g[1])**2)*np.log(r_g[2]))/(-2*a*beta)    
        while(np.random.random()**2 > 1 - lamb2):
            r_g = np.random.random(3)
            lamb2 = (np.log(r_g[0]) + (np.cos(2*np.pi*r_g[1])**2)*np.log(r_g[2]))/(-2*a*beta)
        x0 = 1- 2*lamb2
        ########generating xvect
        xv = 2*np.random.random(3) - 1
        while( xv[0]**2 + xv[1]**2 + xv[2]**2 > 1):
            xv = 2*np.random.random(3) - 1        
        xv *= np.sqrt(1-x0**2)/np.sqrt(xv[0]**2 + xv[1]**2 + xv[2]**2)
        #######building X of SU(2)
        X = x0*np.identity(2) + 1j*xv[0]*sx + 1j*xv[1]*sy + 1j*xv[2]*sz        

        U[t,x,y,z,mi] = np.dot(X , A.conj().T/a)
    
def update_lattice():   
    for t in range(N):
        for x in range(N):
            for y in range(N):
                    for z in range(N):
                        for mi in range(4):                
                            update_links_hb(t,x,y,z,mi)
   
def MCloop(i,j):
    sum = 0.
    
    for t in range(N):
        for x in range(N):    
            for y in range(N):
                for z in range(N):                    
                    for ni in range(3):                        
                        ani = [0,0,0,0]   
                        ani[ni] = 1
                        for mi in range(ni+1,4): 
                            ami = [0,0,0,0]
                            ami[mi] = 1                            
                            I=0
                            J=0                            
                            
                            temp = np.identity(group_dim)
                                                                                 
                            for I in range(0,i):                                   
                                temp = np.dot(temp, U[(t+I*ami[0])%N , (x+I*ami[1])%N , (y+I*ami[2])%N , (z+I*ami[3])%N ,mi])

                            for J in range(0,j):                             
                                temp = np.dot(temp, U[(t+(I+1)*ami[0] + J*ani[0])%N , (x+(I+1)*ami[1]+ J*ani[1])%N , (y+(I+1)*ami[2]+ J*ani[2])%N , (z+(I+1)*ami[3]+ J*ani[3])%N , ni])

                            for I in range(I,-1,-1):
                                temp = np.dot(temp, U[(t+I*ami[0] + (J+1)*ani[0])%N , (x+I*ami[1]+ (J+1)*ani[1])%N , (y+I*ami[2]+ (J+1)*ani[2])%N , (z+I*ami[3]+ (J+1)*ani[3])%N , mi].conj().T)
                                
                            for J in range(J,-1,-1):
                                temp = np.dot(temp, U[(t+ J*ani[0])%N , (x+ (J)*ani[1])%N , (y+ (J)*ani[2])%N , (z+ (J)*ani[3])%N , ni].conj().T )
                            
                            sum += (np.trace(temp)).real/group_dim                      
    return sum/(6*N**4)  

def MCaverage():
    print("W11 measures:")
    print("")
    for i in range(N_cf):        
        update_lattice()
        measure11= MCloop(1,1)        
        loop11.append(measure11)        

        print(measure11 )        
   
#######################################PARAMETERS

beta = 2.2

group_dim = 2
    
N = 2                                                              # path length in lattice units
N_cf = 10                                                          # number of path configurations
N_hit = 10
eps = 0.24                                                         # random parameter: controls the accept ratio

su2_pool_size = 300                                                # include inverse matrices
U  = np.zeros( (N,N,N,N,4,group_dim,group_dim) , np.complex )      #4D lattice link variables
loop11 = []                                                        #contain all measurements taken along the run

#######################################EXECUTION
su2_pool = generate_su2_pool(su2_pool_size , eps)
init_lattice(0)                                                             #0 for totally cold start and 1 for totally hot start

MCaverage()

#######################################PLOT
x_cf = list(range(1,len(loop11)+1)) 
plt.plot(x_cf , loop11 , 'ro')

