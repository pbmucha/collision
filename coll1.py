from sys import argv
import numpy as np
import matplotlib.pyplot as pyp
from celluloid import Camera

fig = pyp.figure()
camera = Camera(fig)


#tu chcemy miec dwa dodatknie i ujemne

N=20
dt=0.3
T=15
GR=3000
step=3
alf=1.5
bar=1.5/N

xx = np.zeros((N,T,2))
xx0=np.zeros((N,T,2))
xx1=np.zeros((N,T,2))
zz1=np.zeros((N,T,2))
zz2=np.zeros((N,T,2))
xp = np.zeros((N,2))
xll = np.zeros((N,2))
Fi = np.zeros((N,N))
fi = np.zeros((N,N,T))
ff1 = np.zeros((N,N,T))
fii = np.zeros((N,N,T))
fi0=np.zeros((N,N))
xl = np.zeros((N,2))
hh=np.zeros((N,N))
ll=np.zeros((2))
yll=np.zeros((N))

def ID(N):
    for i in range(N):
        x = np.random.random()*200-100
        y = np.random.random()*100
        xp[i,0] = x
        xp[i,1] = y
    return xp

def firand(N):    
    for i in range(N): 
        for k in range(N):
            for t in range(T):
                fi[i,k,t]=np.random.random()
    #ll=np.linalg.norm(fi)
    return fi
    
def L1(yll):
    kk=0
    for i in range(N):
        ll=kk+yll[i]
        kk=ll
    return kk    
        
def normfi(fi):        
    for i in range(N): 
       for t in range(T):
           kk=L1(fi[i,:,t])
           for k in range(N):
               fii[i,k,t]=fi[i,k,t]/kk
    return fii           
               

def evol(xp,fi):
    fii=normfi(fi)
    for t in range(T):
        if t==0:
           for k in range(N):
               xx[k,0,0]=xp[k,0]
               xx[k,0,1]=xp[k,1]
        if t>0:    
           for k in range(N):
              ll=np.zeros((2))
              for l in range(N):
                  lll=  ll + fii[k,l,t-1]*xx[l,t-1,:]
                  ll = lll
              xx[k,t,:]=(1-dt)*xx[k,t-1,:] + dt*ll  
    return xx   
      


def LL(xl):
    l=0
    for i in range(N):
        for k in range(N):
            l=l+ (xl[i,0]-xl[k,0])*(xl[i,0]-xl[k,0])+(xl[i,1]-xl[k,1])*(xl[i,1]-xl[k,1])
    return l/(N*N)
    


def jjj(N,T):
    l1=np.random.randint(N)
    l2=np.random.randint(N)
    l3=np.random.randint(T)
    yy=np.zeros((N,N,T))
    yy[l1,l2,l3]=1
    return yy  


def bladzenie(fffi,xxp): 
    xpp=xxp
    xx=evol(xpp,fi)        
    xll=xx[:,T-1,:]  
    fff=fffi
    oo=LL(xll)
    for q in range(GR): 
        ff1=fffi+jjj(N,T)
        xx0=evol(xpp,ff1)
        ooo=LL(xx0[:,T-1,:])
        print(ooo,oo,'roznica')
        if ooo < oo:
                   fff=ff1
                   oo=ooo
    return fff
    
    
def opt(fi,xxp): 
    xpp=xxp
    xx=evol(xpp,fi)        
    xll=xx[:,T-1,:]  
    ff=fi
    oo=LL(xll)
    for q in range(step):
        print('step',q)
        ffff=bladzenie(ff,xpp)
        xx0=evol(xp,ffff)
        ooo=LL(xx0[:,T-1,:])
        if ooo<oo:
              ff=ffff
              oo=ooo
    ff          
            
    
xp=ID(N)    
print(xp,'xp')
ff=firand(N)
print(ff,'firand')
print(normfi(ff),'normed-fi')  


#xx=evol(xp,ff)        
#print(xx,'xx')    

#xl=xx[:,T-1,:]    
#print(LL(xl),'LL')

ffff=bladzenie(ff,xp)
#ffff=opt(ff,xp)


xx1=evol(xp,ffff) 
print(LL(xx1[:,T-1,:]),'nowe LL')         
     
  

