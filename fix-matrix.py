from sys import argv
import numpy as np
import matplotlib.pyplot as pyp
from celluloid import Camera

fig = pyp.figure()
camera = Camera(fig)

N=10
S=200
a=0.25
K=40
W=2800


mm=np.zeros((N,N))
mc=np.zeros((N,N))
m0=np.zeros((N,N))
xx=np.zeros((N))
ss=np.zeros((S,N))
vv=np.zeros((S))

xp=np.zeros((N,2))
C1=np.zeros((N),int)


#wybór macierzy

def RandM(N):
    mm=np.zeros((N,N))
    for i in range(N):
        for k in range(N):
            mm[i,k] = np.random.random()*20-10
    return mm   
    

def RandX(N):
    for i in range(N):
        xx[i]=np.random.random()*20-10
    return  xx  


def Product(xx,mm):
    l=0
    for i in range(N):
        for k in range(N):
            l=l+xx[k]*mm[k,i]*xx[i]
    return l 

def set(S):    
    sss=np.zeros((S,N))
    for s in range(S):
        sss[s,:]=RandX(N)
    return sss

def val(ss,mm):
    vv=np.zeros(S)
    for s in range(S):
        vv[s]=Product(ss[s,:],mm)
    return vv  

def LLL(ss,vv,mm):
    l=0
    for s in range(S):
        l=l+(Product(ss[s,:],mm) - vv[s])*(Product(ss[s,:],mm) - vv[s]) 
    return l/S

def matrkor(N):
    ee=np.zeros((N,N))
    for i in range(N):
        for k in range(N):
            ee[i,k]=np.random.random()-0.5
    #i=np.random.randint(0,N)
    #k=np.random.randint(0,N)
    #ee[i,k]=1
    return ee   

def kor(ss,vv,mm):
    mmm=mm.copy()
    mmmm=np.zeros((K,N,N))
    l=LLL(ss,vv,mmm)
    kor=0
    C=-1
    for k in range(K):
        mmmm[k,:,:]=mmm+a*matrkor(N)*(np.random.randint(0,2)-0.5)
        ll=LLL(ss,vv,mmmm[k,:,:])
        #print('k',k,'l',l,'ll',ll)
        if ll<l:
            l=ll
            C=k  
    #print('C',C)        
    if C>-1:
        mmm=mmmm[C,:,:]
    return mmm               

xx=RandX(N)
#print('x')
#print(xx)

mc=RandM(N)
#print('mm')
#print(mm)


ss=set(S)

#print('ss')
#print(ss)

vv=val(ss,mc)
#print('vv')
#print(vv)

ll=LLL(ss,vv,mc)
#print('LLL',ll)


m0=RandM(N)
#print('m0')
#print(m0)
kk=m0.copy()

for w in range(W):
    print('w',w)
    kkk=kor(ss,vv,kk)
    kk=kkk
    #print('kk')
    #print(kk)


print('mm')
print(mc+mc.T)

print('m0')
print(m0+m0.T)

print('m1')
print(kk+kk.T)

print('L(mm)',LLL(ss,vv,mc),'L(m0)',LLL(ss,vv,m0),'L(m1)',LLL(ss,vv,kk))





    
    

    
    


    
          
    
     


#for w in range(W):
#    xp=ID(N)
#    xp1=xp.copy()
#    for t in range(T):
#       yp1=xp1 
#       pyp.scatter(yp1[:,0],yp1[:,1],s=100,c='blue')  
#       camera.snap()
#       C=krokC(yp1)
#       print('step',t,'L(x+)-L(x)',L(krokK(yp1,C))-L(yp1),'L(tilde_x)-L(x)',L(xtidle(yp1,C))-L(yp1),'pochodna',pochodna(yp1,C)*(-4)*dt/N)
#       #print(C,'C')
#      xp1=krokK(yp1,C)
    
        
#animation = camera.animate()

#animation.save('antykolaps.gif', writer = 'imagemagick')   



#pyp.show()      
    

