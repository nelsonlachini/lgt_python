import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time

from lalgebra import *

def generate_su3_pool_lepage():   
    H = np.zeros( (int(su3_pool_size/2),3,3) , np.complex )             

    for i in range(int(su3_pool_size/2)):
        rand = 2*np.random.random((3,3))-1
        H[i] = (rand +rand.T)/2.   
        
    for i in range(int(su3_pool_size/2)):
        matrix = np.identity(3) + 1j*eps*H[i]
        matrix = unitarize(matrix)
        su3_pool[i] = matrix
        su3_pool[i+int(su3_pool_size/2)] = matrix.conj().T
       
def sort_from_pool():
    number = int( (su3_pool_size-1)*np.random.random() )
    return su3_pool[number]
     
#simulation              
def init_lattice(parameter=0):
    for t in range(N):
        for x in range(N):
            for y in range(N):
                    for z in range(N):
                        for mi in range(4): 
                            if(np.random.uniform(0,1)>parameter):                             
                                U[t,x,y,z,mi] = np.identity(3)    
                            else:
                                U[t,x,y,z,mi] = sort_from_pool()                            

def get_S(t,x,y,z,mi,gamma):
    ans = (np.trace( np.dot( U[t,x,y,z,mi] , gamma ) )).real
    return -beta*ans/3.
    
def get_staple(t,x,y,z,mi):
    staple = np.zeros( (3,3) , np.complex )
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
    

def update_links(t,x,y,z,mi):              
    staple = get_staple(t,x,y,z,mi)
    
    for i in range(N_hit): 
        old_S = get_S(t,x,y,z,mi,staple)
        old_link = U[t,x,y,z,mi].copy()
        
        U[t,x,y,z,mi] = np.dot(sort_from_pool(),old_link)
        
        new_S = get_S(t,x,y,z,mi,staple)       

        dS = new_S - old_S
        if ( (dS > 0.) and (np.exp(-dS) < np.random.uniform(0,1)) ):                
            U[t,x,y,z,mi] = old_link               
    
def update_lattice():   
    for t in range(N):
        for x in range(N):
            for y in range(N):
                    for z in range(N):
                        for mi in range(4):
                            update_links(t,x,y,z,mi)

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
                            
                            temp = np.identity(3)
                                                                                 
                            for I in range(0,i):                                   
                                temp = np.dot(temp, U[(t+I*ami[0])%N , (x+I*ami[1])%N , (y+I*ami[2])%N , (z+I*ami[3])%N ,mi])

                            for J in range(0,j):                             
                                temp = np.dot(temp, U[(t+(I+1)*ami[0] + J*ani[0])%N , (x+(I+1)*ami[1]+ J*ani[1])%N , (y+(I+1)*ami[2]+ J*ani[2])%N , (z+(I+1)*ami[3]+ J*ani[3])%N , ni])

                            for I in range(I,-1,-1):
                                temp = np.dot(temp, U[(t+I*ami[0] + (J+1)*ani[0])%N , (x+I*ami[1]+ (J+1)*ani[1])%N , (y+I*ami[2]+ (J+1)*ani[2])%N , (z+I*ami[3]+ (J+1)*ani[3])%N , mi].conj().T)
                                
                            for J in range(J,-1,-1):
                                temp = np.dot(temp, U[(t+ J*ani[0])%N , (x+ (J)*ani[1])%N , (y+ (J)*ani[2])%N , (z+ (J)*ani[3])%N , ni].conj().T )
                            
                            sum += (np.trace(temp)).real/3                      
    return sum/(6*N**4)  

def MCaverage():
    for i in range(N_cf): 
        print('\n')
        print('Updating lattice...')
        
        generate_su3_pool_lepage()
        
        for j in range(N_cor):
            update_lattice()
            
        measure1= MCloop(1,1)
        loop1.append(measure1)        
        
        print('Wilson Loop Measures:')
        print('a x a Wilson loop: ', measure1)
        
 
########################PARAMETERS##########################

beta = 5.5
 
N = 2                                                       # lattice size in lattice units
N_cf =  11                                                  # number of configurations
N_cor = 5                                                   # number of configurations to uncorrelate
N_hit = 10
eps = 0.24                                                  # random parameter: controls the acceptance ratio

su3_pool_size = 100                                         # include inverse matrices
su3_pool = np.zeros((su3_pool_size,3,3) , np.complex)       # SU3 3x3 matrices for updating links
U  = np.zeros( (N,N,N,N,4,3,3) , np.complex )               # 4D lattice link variables

init_lattice(0)

loop1 = []
MCaverage()

##################################################################################
n_cf = list(range(1,N_cf+1))
plt.plot(n_cf,loop1,'ro')
