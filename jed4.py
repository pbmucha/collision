from sys import argv
import numpy as np
import matplotlib.pyplot as pyp
from celluloid import Camera

fig = pyp.figure()
camera = Camera(fig)

M=10
N=2*M
T=5
dt=0.03
ST=1

xp=np.zeros((N,2))
xp1=np.zeros((N,2))
yp1=np.zeros((N,2))
yy=np.zeros((N,2))
xk=np.zeros((N,2))
xkk=np.zeros((N,2))
C1=np.zeros((N),int)
CC=np.zeros((N),int)
C=np.zeros((N),int)
kom=np.zeros((T,N),int)
f4=np.zeros((T,N,N))
xx=np.zeros((T,N,2))

def ID(N):
    for i in range(M):
        x = np.random.random()*200-300
        y = np.random.random()*100
        xp[i,0] = x
        xp[i,1] = y
    for i in range(M):
        x = np.random.random()*200+100
        y = np.random.random()*100
        xp[M+i,0] = x
        xp[M+i,1] = y    
    return xp
    
    
def L(yy):
    l=0
    for i in range(N):
        for k in range(N):
                l=l+ (yy[i,0]-yy[k,0])*(yy[i,0]-yy[k,0])+(yy[i,1]-yy[k,1])*(yy[i,1]-yy[k,1])
    return l/N
        
         
def krokC(xx):
    CC=np.zeros((N),int)
    xx4 = xx
    l=L(xx4)
    for k in range(N):
        yy=xx4
        CC[k]=k
        for i in range(N):
            ll=l
            yy[k]=(1-dt)*xx4[k]+dt*xx4[i]
            l1=L(yy)
            if ll < l1:
               l=l1
               CC[k]=i
    return CC  
            
    
def krokK(xk,CC):
    yk=np.zeros((N,2))
    for k in range(N):
        l3=CC[k]
        yk[k,0]=(1-dt)*xk[k,0]+dt*xk[l3,0] 
        yk[k,1]=(1-dt)*xk[k,1]+dt*xk[l3,1] 
    return yk             

xp=ID(N)
#print(xp,'initial data') 

C=krokC(xp)
#print(C1,'kom')  

xp1=xp
for t in range(T):
    yp1 = xp1
    pyp.scatter(yp1[:,0],yp1[:,1],s=10,c='blue')  
    camera.snap()
    #C=krokC(yp1)
    print(L(yp1),'L',t)
    xp1 = krokK(yp1,C)
    C=krokC(yp1)
    print(C,'C')
    
        
animation = camera.animate()


pyp.show()      
    

