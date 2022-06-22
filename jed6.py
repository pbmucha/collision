from sys import argv
import numpy as np
import matplotlib.pyplot as pyp
from celluloid import Camera
import pickle

fig = pyp.figure()
camera = Camera(fig)


W=100   #wielość losować danych poczatkowcyh
M=10
N=2*M  #liczba czastek
T=20   #liczba iteracji w czasie
dt=0.2 #krok czasowy
ST=1

xp=np.zeros((N,2))
C1=np.zeros((N),int)
CCC=np.zeros((T*W,N),int)
kom=np.zeros((T,N),int)
f4=np.zeros((T,N,N))
xx=np.zeros((T,N,2))
XXX=np.zeros((T*W,N,2))


#losowanie danych
def ID(N):
    for i in range(M):
        x = np.random.random()*20
         #-300
        y = np.random.random()*20
        xp[i,0] = x
        xp[i,1] = y
    for i in range(M):
        x = np.random.random()*20
        #+100
        y = np.random.random()*20
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

#definiujemy komunikacje         
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
    
    
# def krokK2(xk):
#     yk=xk.copy()
#     for k in range(N):
#         yk1=yk
#         yk1[k]=(1-dt)*yk[k]+dt*yk[krokC2(yk1,k)] 
#         #print(L(yk1))
#         yk=yk1 
#     return yk            
    
def krokK(xk,CC):
    yk=np.zeros((N,2))
    for k in range(N):
        #l3=CC[k]
        yk[k]=(1-dt)*xk[k]+dt*xk[CC[k]]  
    return yk   


#losujemy dane poczatkowe
#xp=ID(N)

#liczymy komunikcjae dla dan. poczat.
#C1=krokC(xp)
#print(C1,'kom')  


#xp1=xp.copy()

for w in range(W):
    xp1=ID(N)
    print(w,'wielokrotnosc')
    for t in range(T):  
        yp1=xp1 
        C=krokC(yp1)
        print('step',t)
        #print(C,'C')
        xp1=krokK(yp1,C)
        XXX[t+w*T,:]=yp1
        CCC[t+w*T,:]=C
    #dodaj(yp1,C) 


    #pyp.scatter(yp1[:,0],yp1[:,1],s=100,c='blue')  
    
    # pyp.scatter(yp1[2,0],yp1[2,1],s=110,c='green')  
    # pyp.scatter(yp1[C[2],0],yp1[C[2],1],s=110,c='red')  
    # pyp.scatter(yp1[11,0],yp1[11,1],s=110,c='green')  
    # pyp.scatter(yp1[C[11],0],yp1[C[11],1],s=110,c='red')  
    # pyp.scatter(yp1[7,0],yp1[7,1],s=110,c='green')  
    # pyp.scatter(yp1[C[7],0],yp1[C[7],1],s=110,c='red') 
    # pyp.scatter(yp1[17,0],yp1[17,1],s=110,c='green')  
    # pyp.scatter(yp1[C[17],0],yp1[C[17],1],s=110,c='red') 
    # pyp.scatter(yp1[3,0],yp1[3,1],s=110,c='green')  
    # pyp.scatter(yp1[C[3],0],yp1[C[3],1],s=100,c='red') 
    #camera.snap()


np.savetxt('XXX.csv', np.reshape(XXX, (-1, N*2)), delimiter=',')
np.savetxt('CCC.csv', CCC, delimiter=',')


#animation = camera.animate()

#animation.save('antykolaps.gif', writer = 'imagemagick')   
#animation.save('anim5.mp4', writer="ffmpeg")


#pyp.show()      
    

#https://www.datacamp.com/community/tutorials/pickle-python-tutorial