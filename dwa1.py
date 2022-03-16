from sys import argv
import numpy as np
import matplotlib.pyplot as pyp
from celluloid import Camera

fig = pyp.figure()
camera = Camera(fig)

M=5
N=2*M
T=100
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
        x = np.random.random()*20-300
        y = np.random.random()*10
        xp[i,0] = x
        xp[i,1] = y
    for i in range(M):
        x = np.random.random()*20+300
        y = np.random.random()*10
        xp[M+i,0] = x
        xp[M+i,1] = y    
    return xp
    
def srod(xx):
    l0=0
    l1=0
    ll0=0
    ll1=0
    for k in range(N):
        ll0=l0+xx[k,0]
        ll1=l1+xx[k,1]
        l0=ll0
        l1=ll0
    return [l0/N,l1/N]   
    
def sign(a,b):
    k=1
    if a > 0 and b <0:
             k=0
    if a < 0 and b >0:
             k=0
    return k       
    
def L(yy):
    l=0
    ss=srod(yy)
    for i in range(N):
        for k in range(N):
                l=l+ sign(yy[i,0]-ss[0],yy[k,0]-ss[0])*(yy[i,0]-yy[k,0])*(yy[i,0]-yy[k,0])
                +sign(yy[i,0]-ss[0],yy[k,0]-ss[0])*(yy[i,1]-yy[k,1])*(yy[i,1]-yy[k,1])
                #+ np.linalg.norm(yy[i])+np.linalg.norm(yy[k])
                #l=l+ sign(yy[i,0]-ss[0],yy[k,0]-ss[0])*abs(yy[i,0]-yy[k,0])+sign(yy[i,0]-ss[0],yy[k,0]-ss[0])*abs(yy[i,1]-yy[k,1])*(yy[i,1]-yy[k,1])
                #l=l+ (yy[i,0]-ss[0])*(yy[k,0]-ss[0])*(abs(yy[i,0]-yy[k,0])+abs(yy[i,1]-yy[k,1])*(yy[i,1]-yy[k,1]))
    return l/N
        
#to trzeba zrobic inaczej, to moze by poprosty zle... obliczyc g1[TT-1] i tyle.koszmarek slon za stary jest :-( -<--<
         
def krokC(xx):
    CC=np.zeros((N),int)
    #yyk=np.zeros((N,2))
    xx4 = xx
    l=L(xx4)
    for k in range(N):
        yy=xx4
        CC[k]=k
        for i in range(N):
            ll=l
            yy[k]=(1-dt)*xx4[k]+dt*xx4[i]
            l1=L(yy)
            #print(ll,l1,'stary-nowy',i)
            if ll > l1:
               l=l1
               CC[k]=i
               #print(ll,l1,'stary-nowy',i)
               #print(yk)
    return CC  
            
    
def krokK(xk,CC):
    yk=np.zeros((N,2))
    for k in range(N):
        l3=CC[k]
        yk[k,0]=(1-dt)*xk[k,0]+dt*xk[l3,0] 
        yk[k,1]=(1-dt)*xk[k,1]+dt*xk[l3,1] 
    return yk             

xp=ID(N)

print(srod(xp)[1],'srod')
#print(xp,'initial data') 

C=krokC(xp)
#print(C1,'kom')  

xp1=xp
for t in range(T):
    yp1 = xp1
    kk=srod(xp1)
    pyp.scatter(yp1[:,0],yp1[:,1],s=100,c='blue')  
    pyp.scatter(yp1[2,0],yp1[2,1],s=200,c='green')
    pyp.scatter(yp1[C[2],0],yp1[C[2],1],s=200,c='red')
    pyp.scatter(kk[0],kk[1],s=400,c='black')
    camera.snap()
    #print(yp1[:,0],'xp1-pezed')
    #C=krokC(yp1)
    print(L(yp1),'L',t)
    #print(C,'C')
    #print(yp1[:,0],'xp1-po')
    xp1 = krokK(yp1,C)
    C=krokC(yp1)
    print(C,'C')
    
        
animation = camera.animate()

#animation.save('kolaps04.gif', writer = 'imagemagick')   

#oo=y = np.random.random()*100
#ooo=(1-dt)*oo+dt*oo
#print(ooo,oo,'ooo')

pyp.show()      
    

