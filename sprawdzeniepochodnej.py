from sys import argv
import numpy as np
import matplotlib.pyplot as pyp
from celluloid import Camera

fig = pyp.figure()
camera = Camera(fig)

M=10
N=2*M
T=20
dt=0.2
ST=1

xp=np.zeros((N,2))
C1=np.zeros((N),int)
kom=np.zeros((T,N),int)
f4=np.zeros((T,N,N))
xx=np.zeros((T,N,2))

def ID(N):
    for i in range(M):
        x = np.random.random()*200 
         #-300
        y = np.random.random()*200
        xp[i,0] = x
        xp[i,1] = y
    for i in range(M):
        x = np.random.random()*200  
        #+100
        y = np.random.random()*200
        xp[M+i,0] = x
        xp[M+i,1] = y    
    return xp
    
    
def sasrand(N,T):    
    g1=np.zeros((T,N),int)
    for t in range(T): 
        for k in range(N):
            q=np.random.randint(0,N)
            g1[t,k]=q
    #ll=np.linalg.norm(fi)
    return g1
    
# evolucja kaskady dla war pocz xp i komuniakcji fi
def ev(xp,gi,TT):
    yp=xp.copy()
    g1=gi.copy()
    xxx=np.zeros((T,N,2))
    for t in range(TT):
        if t==0:
           for k in range(N):
               xxx[0,k,0]=yp[k,0]
               xxx[0,k,1]=yp[k,1]
        if t>0:    
           for k in range(N):
              xxx[t,k,:]=(1-dt)*xxx[t-1,k,:] + dt*xxx[t-1,g1[t-1,k],:] 
    return xxx   
    
def L(yy):
    l=0
    for i in range(N):
        for k in range(N):
                l=l+ (yy[i,0]-yy[k,0])*(yy[i,0]-yy[k,0])+(yy[i,1]-yy[k,1])*(yy[i,1]-yy[k,1])
    return l/N
        
#to trzeba zrobic inaczej, to moze by poprosty zle... obliczyc g1[TT-1] i tyle.koszmarek slon za stary jest :-( -<--<
         
def krokC(xk):
    CC=np.zeros((N),int)
    l=L(xk)
    for k in range(N):
        yk=xk.copy()
        CC[k]=k
        for i in range(N):
            ll=l
            yk[k]=(1-dt)*xk[k]+dt*xk[i]
            l1=L(yk)
            #print(ll,l1,'stary-nowy',i)
            if ll < l1:
               l=l1
               CC[k]=i
               #print(ll,l1,'stary-nowy',i)
               #print(yk)
    return CC  
    
def krokC2(xk,kk):
    l=L(xk)    
    yk=xk.copy()
    cc=kk
    for i in range(N):
        ll=l
        yk[kk]=(1-dt)*xk[kk]+dt*xk[i]
        l1=L(yk)
        #print(ll,l1,'stary-nowy',i)
        if ll > l1:
               l=l1
               cc=i
               #print(ll,l1,'stary-nowy',i)
               #print(yk)
    return cc   
    
def krokK2(xk):
    yk=xk.copy()
    for k in range(N):
        yk1=yk
        yk1[k]=(1-dt)*yk[k]+dt*yk[krokC2(yk1,k)] 
        #print(L(yk1))
        yk=yk1 
    return yk            
    
def krokK(xk,CC):
    yk=np.zeros((N,2))
    for k in range(N):
        #l3=CC[k]
        yk[k]=(1-dt)*xk[k]+dt*xk[CC[k]]  
    return yk      


def xtidle(xk,C):
    xxk=xk.copy()
    xxk[1]= (1-dt)*xxk[1]+dt*xxk[C[1]]
    return xxk   

def pochodna(xk,C):
    l=0
    for k in range(N):
        l=l+(xk[1,0]-xk[k,0])*(xk[1,0]-xk[C[1],0])+  (xk[1,1]-xk[k,1])*(xk[1,1]-xk[C[1],1])
    return l            

xp=ID(N)
print(xp,'initial data') 

C1=krokC(xp)
print(C1,'kom')  

xp1=xp.copy()

for t in range(T):
    yp1=xp1 
    pyp.scatter(yp1[:,0],yp1[:,1],s=100,c='blue')  
    camera.snap()
    C=krokC(yp1)
    print('step',t,'L(x)',L(yp1),'L(tilde_x)-L(x)',L(xtidle(yp1,C))-L(yp1),'pochodna',pochodna(yp1,C)*(-4)*dt/N)
    #print(C,'C')
    xp1=krokK(yp1,C)
    
        
animation = camera.animate()

animation.save('antykolaps.gif', writer = 'imagemagick')   



pyp.show()      
    

