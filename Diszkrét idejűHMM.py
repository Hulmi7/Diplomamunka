import numpy
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as asns
from glob import glob
import librosa
import librosa.display
#import Ipython.display as ipd
from itertools import cycle

num = 2 # állapotok száma
KezdElosz = [0.8, 0.2]
TM = np.array([[0.6, 0.4],
               [0.2, 0.8]])
ps = np.array([[0.7, 0.3],
               [0.2, 0.8]])

############################szimulálás
def MarkovChain(l,KezdElosz,TM,ps):
    markovchain = np.zeros(l,int)
    markovchain[0] = numpy.random.choice(numpy.arange(0, len(KezdElosz)), p=KezdElosz)
    for i in range(l-1):
        if markovchain[i] == 0:
            markovchain[i+1] = numpy.random.choice(numpy.arange(0, len(KezdElosz)), p=TM[0])
        if markovchain[i] == 1:
            markovchain[i+1] = numpy.random.choice(numpy.arange(0, len(KezdElosz)), p=TM[1])
    #ps-ek
    for i in range(l):
        if markovchain[i] == 0:
            markovchain[i] = numpy.random.choice(numpy.arange(0, len(KezdElosz)), p=ps[0])
        if markovchain[i] == 1:
            markovchain[i] = numpy.random.choice(numpy.arange(0, len(KezdElosz)), p=ps[1])
    return markovchain

jel = list(MarkovChain(100, KezdElosz, TM, ps))
print(jel)
#########################################Diszkrét eset#######################################################x


F = np.zeros((num, len(jel)))
def Forward(F, KezdElosz, ps, jel, TM):
    for i in range(len(F)):
        F[i][0] = KezdElosz[i] * ps[i][jel[0]]

    for i in range(len(F.transpose()) - 1):
        for j in range(len(F)):
            F[j][i + 1] = ps[j][jel[i + 1]] * np.dot(F[:, i], TM[:, j])
Forward(F, KezdElosz, ps, jel, TM)
print('Forward', F ,'\n')


B = np.zeros((num, len(jel)))
def Backward(B, KezdElosz, ps, jel, TM ):
    n = len(B.transpose())
    for i in range(len(B)):
        B[i][-1] = 1
        B[i][-2] = np.dot(TM[i], ps[:, jel[-1]])

    E = 0

    while n > 1:
        for i in range(len(B)):
            for j in range(len(B)):
                E = E + ps[j][jel[n - 1]] * B[j][n - 1] * TM[i][j]
            B[i][n - 2] = E
            E = 0
        n -= 1

Backward(B, KezdElosz, ps, jel, TM )
print('Backward','\n', B , '\n')


'''P = 0
for j in range(len(B)):
    P = P + ps[j][whatisee[0]] * B[j][0] * KezdElosz[j]

print('P(S^n=s_n)=', P, '\n')

for j in range(6):
    print(F[0][j]*B[0][j]+F[1][j]*B[1][j])
'''# F/B ellenőrzése a valószínűségekkel

''' # Viterbi
V = np.zeros((num, len(jel))) 

for i in range(len(F)):
    V[i][0] = KezdElosz[i] * ps[i][jel[0]]

m = np.zeros(len(TM))

for k in range(1, len(V.transpose()), 1):
    for j in range(len(V)):
            for i in range(len(TM)):
               m[i] = TM[j][i] * V[i][k - 1]
            V[j][k]=ps[j][jel[k]]*max(m)

print('Viterbi\n', V, '\n')


predictedstates = np.zeros(len(V.transpose()))
predictedstates[-1] = numpy.argmax(V[:,-1])
n = np.zeros(len(TM))

for j in range(1, len(jel)-1): 
    for i in range(len(TM)):
     n[i] = TM[i][int(predictedstates[-j])]*V[i][-j-1]

    predictedstates[-1-j] = numpy.argmax(n)


print('predicted states:\n', predictedstates, '\n')
''' #Viterbi


Kszi = numpy.zeros((len(jel)-1, num, num))

def kszi(Kszi, F, B, jel,TM, ps):
    for t in range(len(jel)-1):
        for i in range(len(Kszi.transpose())):
            for j in range(len(Kszi.transpose())):
                Kszi[t][i][j] = F[i][t]*TM[i][j]*ps[j][jel[t+1]]*B[j][t+1]
        Kszi[t] = Kszi[t]/sum(sum(Kszi[t]))

kszi(Kszi, F, B, jel,TM, ps)
#print('Kszi mátrixok\n', Kszi, '\n')


#GAMMA
Gamma = numpy.zeros((len(jel), num))
def gamma(Gamma, F, B):
    for t in range(len(Gamma)):
        for i in range(len(Gamma.transpose())):
            Gamma[t][i] = F[i][t]*B[i][t]
        Gamma[t] = Gamma[t] / sum(Gamma[t])
gamma(Gamma, F, B)
print('gammák\n', list(Gamma[0]), '\n')


#Paraméterbecslés
F = np.zeros((num, len(jel)))
B = np.zeros((num, len(jel)))
Kszi = numpy.zeros((len(jel)-1, num, num))
Gamma = numpy.zeros((len(jel), num))

###Kezdeti tipp az eloszlásra
KezdElosz = np.array([[0.2, 0.8], [0,0]])
TM = np.array([[0.7, 0.3],
               [0.2, 0.8]])
ps = np.array([[0.8, 0.2],
               [0.3, 0.7]])
def Reestimation(KezdElosz, TM, ps, n, jel):
    for q in range(n):
        Forward(F, KezdElosz[0], ps, jel, TM)
        Backward(B, KezdElosz[0], ps, jel, TM)
        kszi(Kszi, F, B, jel, TM, ps)
        gamma(Gamma, F, B)

        for i in range(num):
            KezdElosz[0][i] = Gamma[0][i]


        #TM
        for i in range(num):
            d = Gamma.sum(axis=0)[i]-Gamma[-1][i]
            for j in range(num):
                TM[i][j] =sum(Kszi[:, i, j])/d

        #ps-ek
        for i in range(num):
            d = Gamma.sum(axis=0)[i]
            for j in range(num):
                s = 0
                for k in range(len(jel)):
                    if j == jel[k]:
                        s = s+Gamma[k][i]
                        ps[i][j] = s/d
    return KezdElosz[0],TM,ps

#Szimulált jelen:
def Average(hossz,db,KezdElosz, TM, ps, n):
    a = [0, 0]
    b = np.array([[0, 0], [0, 0]])
    c = np.array([[0, 0], [0, 0]])
    for q in range(db):
        whatisee = list(MarkovChain(hossz,[0.8, 0.2],[[0.9, 0.1],[0, 1]],[[0.9, 0.1],[0.2, 0.8]]))
        Reestimation(KezdElosz, TM, ps, n)
        a = np.add(a, KezdElosz)
        b = np.add(b, TM)
        c = np.add(c, ps)
    return a/db, b/db, c/db




