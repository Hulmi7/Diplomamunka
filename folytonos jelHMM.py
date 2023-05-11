import numpy
import numpy as np
import time
import pandas as pd
import matplotlib.pylab as plt
import seaborn as asns
from glob import glob
import librosa
# import librosa.display
# import Ipython.display as ipd
from itertools import cycle
from scipy.stats import norm
import random
start = time.time()
random.seed(10)


num = 5  # állapotok száma
nyujtas = 100
##############################################hang##################################################################
jelek = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
                  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
                  [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], []])

jelek[0], f = librosa.core.load('jobbrateszt (1).mp3')
jelek[1], f = librosa.core.load('jobbrateszt (2).mp3')
jelek[2], f = librosa.core.load('jobbrateszt (3).mp3')
jelek[3], f = librosa.core.load('jobbrateszt (4).mp3')
jelek[4], f = librosa.core.load('jobbrateszt (5).mp3')
jelek[5], f = librosa.core.load('jobbrateszt (6).mp3')
jelek[6], f = librosa.core.load('jobbrateszt (7).mp3')
jelek[7], f = librosa.core.load('jobbrateszt (50).mp3')
jelek[8], f = librosa.core.load('jobbrateszt (8).mp3')
jelek[9], f = librosa.core.load('jobbrateszt (9).mp3')
jelek[10], f = librosa.core.load('jobbrateszt (10).mp3')
jelek[11], f = librosa.core.load('jobbrateszt (11).mp3')
jelek[12], f = librosa.core.load('jobbrateszt (12).mp3')
jelek[13], f = librosa.core.load('jobbrateszt (13).mp3')
jelek[14], f = librosa.core.load('jobbrateszt (14).mp3')
jelek[15], f = librosa.core.load('jobbrateszt (15).mp3')
jelek[16], f = librosa.core.load('jobbrateszt (16).mp3')
jelek[17], f = librosa.core.load('jobbrateszt (17).mp3')
jelek[18], f = librosa.core.load('jobbrateszt (18).mp3')
jelek[19], f = librosa.core.load('jobbrateszt (19).mp3')
jelek[20], f = librosa.core.load('jobbrateszt (20).mp3')
jelek[21], f = librosa.core.load('jobbrateszt (21).mp3')
jelek[22], f = librosa.core.load('jobbrateszt (22).mp3')
jelek[23], f = librosa.core.load('jobbrateszt (23).mp3')
jelek[24], f = librosa.core.load('jobbrateszt (24).mp3')
jelek[25], f = librosa.core.load('jobbrateszt (25).mp3')
jelek[26], f = librosa.core.load('jobbrateszt (26).mp3')
jelek[27], f = librosa.core.load('jobbrateszt (27).mp3')
jelek[28], f = librosa.core.load('jobbrateszt (28).mp3')
jelek[29], f = librosa.core.load('jobbrateszt (29).mp3')
jelek[30], f = librosa.core.load('jobbrateszt (30).mp3')
jelek[31], f = librosa.core.load('jobbrateszt (31).mp3')
jelek[32], f = librosa.core.load('jobbrateszt (32).mp3')
jelek[33], f = librosa.core.load('jobbrateszt (33).mp3')
jelek[34], f = librosa.core.load('jobbrateszt (34).mp3')
jelek[35], f = librosa.core.load('jobbrateszt (35).mp3')
jelek[36], f = librosa.core.load('jobbrateszt (36).mp3')
jelek[37], f = librosa.core.load('jobbrateszt (37).mp3')
jelek[38], f = librosa.core.load('jobbrateszt (38).mp3')
jelek[39], f = librosa.core.load('jobbrateszt (39).mp3')
jelek[40], f = librosa.core.load('jobbrateszt (40).mp3')
jelek[41], f = librosa.core.load('jobbrateszt (41).mp3')
jelek[42], f = librosa.core.load('jobbrateszt (42).mp3')
jelek[43], f = librosa.core.load('jobbrateszt (43).mp3')
jelek[44], f = librosa.core.load('jobbrateszt (44).mp3')
jelek[45], f = librosa.core.load('jobbrateszt (45).mp3')
jelek[46], f = librosa.core.load('jobbrateszt (46).mp3')
jelek[47], f = librosa.core.load('jobbrateszt (47).mp3')
jelek[48], f = librosa.core.load('jobbrateszt (48).mp3')
jelek[49], f = librosa.core.load('jobbrateszt (49).mp3') # tesztek, jelenleg a "jobbra"



##################Jellevágás########################
def cut(jel, hatar, l):
    for i in range(len(jel)):
        if abs(jel[i]) > hatar:
            maximum = max(abs(jel[i+l:i + l + 100]))
            maximum2 = max(abs(jel[i + l+500:i + l + 500 + 100]))
            if maximum > hatar and maximum2 > hatar:
                jel = jel[i:]
                break
    jel = jel[::-1]
    for i in range(len(jel)):
        if abs(jel[i]) > hatar:
            maximum = max(abs(jel[i+l:i + l + 100]))
            maximum2 = max(abs(jel[i + l + 500:i + l + 500 + 100]))
            if maximum > hatar and maximum2 > hatar:
                jel = jel[i:]
                break
    jel = jel[::-1]

    jel = jel *nyujtas
    return jel

for i in range(50):
    jelek[i] = cut(jelek[i], 0.0005, 500)
    #print(len(jelek[i]))
    #plt.plot(jelek[i])
    #plt.suptitle(i)
    #plt.show()



############################Normális###################################################
def norm(x, mu, sig):
    a = np.exp(-(1 / 2) * (np.power((x - mu) / sig, 2))) * 1 / (sig * np.sqrt(2 * np.pi))
    return a

def Norm(c, x, mu, sig):
    a = 0
    for i in range(5):
        a = a + c[i] * norm(x, mu[i], sig[i])
    return a

#########################Paraméterek#####################

KezdElosz = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
TM = np.array([[0.2, 0.2, 0.2,0.2, 0.2],
               [0.2, 0.2, 0.2,0.2,0.2],
               [0.2, 0.2, 0.2,0.2,0.2],
               [0.2, 0.2, 0.2,0.2,0.2],
               [0.2, 0.2, 0.2,0.2,0.2]], dtype='g')

mu = np.ones((5, num))
sig = np.ones((5, num))
c = np.ones((num, 5)) / 5

##########################Algoritmusok######################################
F = np.zeros((num, len(jelek[0])), dtype='g')
Fc = np.zeros(len(jelek[0]), dtype='g')
B = np.zeros((num, len(jelek[0])), float)
Kszi = numpy.zeros((len(jelek[0]) - 1, num, num), float)
Gamma = numpy.zeros((len(jelek[0]), 5, num), float)
mult = numpy.zeros((5, num), float)

def Forward(F, Fc, KezdElosz, whatisee, TM, c, mu, sig):
    for n in range(len(F)):
        F[n][0] = KezdElosz[n] * Norm(c[n], whatisee[0], mu[:, n], sig[:, n])
    Fc[0] = 1/sum(F[:, 0])
    for k in range(len(F)):
        F[k][0] = F[k][0]*(Fc[0])

    for i in range(len(F.transpose()) - 1):
        for j in range(len(F)):
            F[j][i + 1] = Norm(c[j], whatisee[i + 1], mu[:, j], sig[:, j]) * np.dot(F[:, i], TM[:, j])
        if F[0][i+1] == F[0][i+1] or F[1][i+1] < 0.01 or F[2][i+1] < 0.01: # lehet mindig kalapozni kell?
            Fc[i+1] = 1/sum(F[:, i+1])
            for k in range(len(F)):
                F[k][i+1] = F[k][i+1]*(Fc[i+1])
        else:
            Fc[i+1] = 1
#Forward(F, Fc, KezdElosz, jelek[0], TM, c, mu, sig)
#print('Forward:\n', F, '\n', Fc, '\n')
#print(-sum(np.log(Fc)))
#print(F.dtype)
def Backward(B, Fc,  KezdElosz, whatisee, TM, c, mu, sig):
    n = len(B.transpose())
    for i in range(len(B)):
        B[i][-1] = Fc[-1]
        for j in range(len(B)):
            B[i][-2] = TM[i][j] * Norm(c[j], whatisee[-1], mu[:, j], sig[:, j])*B[i][-1]* Fc[-2]

    e = 0

    while n > 1:
        for i in range(len(B)):
            for j in range(len(B)):
                e = e + Norm(c[j], whatisee[n - 1], mu[:, j], sig[:, j]) * B[j][n - 1] * TM[i][j]
            B[i][n - 2] = e * Fc[n-2]
            e = 0
        n -= 1
# Backward(B, Fc, KezdElosz, jelek[0], TM, c ,mu, sig)
# print('Backward\n',B)
def kszi(Kszi, F, B, whatisee, TM, c, mu, sig):
    for t in range(len(whatisee) - 1):
        for i in range(len(Kszi.transpose())):
            for j in range(len(Kszi.transpose())):
                Kszi[t][i][j] = F[i][t] * TM[i][j] * Norm(c[j], whatisee[t + 1], mu[:, j], sig[:, j]) * B[j][t + 1]
        Kszi[t] = Kszi[t] / sum(sum(Kszi[t]))
# kszi(Kszi, F, B, jelek[0], TM, c ,mu, sig)
# print('Kszi mátrixok\n', Kszi[-1], '\n')
def gamma(Gamma, F, B, whatisee, c, mu, sig, mult):
    for t in range(len(Gamma)):
        for j in range(len(Gamma.transpose())):
            for k in range(5):
                Gamma[t][k][j] = F[j][t] * B[j][t]
                mult[k][j] = (c[j][k] * norm(whatisee[t], mu[k][j], sig[k][j]) / Norm(c[j], whatisee[t], mu[:, j],
                                                                                      sig[:, j]))
        for h in range(5):
            Gamma[t][h] = Gamma[t][h] / sum(Gamma[t][h])
        Gamma[t] = Gamma[t] * mult
# gamma(Gamma, F, B, jelek[0], c ,mu, sig)
# print('gammák\n', Gamma[0], '\n\n')

#################################################################
def Reestimation(KezdElosz, TM, c, mu, sig, n, whatisee):
    F = np.zeros((num, len(whatisee)), float)
    Fc = np.zeros(len(whatisee), dtype='g')
    B = np.zeros((num, len(whatisee)), float)
    Kszi = numpy.zeros((len(whatisee) - 1, num, num), float)
    Gamma = numpy.zeros((len(whatisee), 5, num), float)
    mult = numpy.zeros((5, num), float)

    for q in range(n):
        Forward(F, Fc, KezdElosz, whatisee, TM, c, mu, sig)
        Backward(B, Fc, KezdElosz, whatisee, TM, c, mu, sig)
        kszi(Kszi, F, B, whatisee, TM, c, mu, sig)
        gamma(Gamma, F, B, whatisee, c, mu, sig, mult)

        KezdElosz = Gamma[0].sum(axis=0)

        # TM
        TM = Kszi.sum(axis=0)
        for i in range(num):
            TM[i, :] = TM[i, :] / ((sum(Gamma.sum(axis=0) - Gamma[-1]))[i])

        # sig
        sig = np.zeros((5, num))
        for j in range(len(Gamma.transpose())):
            for k in range(5):
                for t in range(len(whatisee)):
                    sig[k][j] = sig[k][j] + Gamma[t][k][j] * np.power(whatisee[t] - mu[k][j], 2)  # hova kell sqrt?
        sig = np.sqrt(sig / Gamma.sum(axis=0))

        # mu
        mu = np.zeros((5, num))
        for j in range(len(Gamma.transpose())):
            for k in range(5):
                for t in range(len(whatisee)):
                    mu[k][j] = mu[k][j] + Gamma[t][k][j] * whatisee[t]  # t+1 kell mert t=0 a kezdeti eloszlás van??
        mu = mu / Gamma.sum(axis=0)

        # c-k
        c = np.zeros((num, 5))
        for j in range(len(Gamma.transpose())):
            for k in range(5):
                c[j][k] = Gamma.sum(axis=0)[k][j] / ((Gamma.sum(axis=0)).sum(axis=0)[j])
    return KezdElosz, TM, c, mu, sig

# #############################################Folytonos jel szimulálása#######################
def MarkovChainContinuous(l, KezdElosz, TM):
    markovchain = np.zeros(l)
    markovchain[0] = numpy.random.choice(numpy.arange(0, len(KezdElosz)), p=KezdElosz)
    for i in range(l-1):
        if markovchain[i] == 0:
            markovchain[i+1] = numpy.random.choice(numpy.arange(0, len(KezdElosz)), p=TM[0])
        if markovchain[i] == 1:
            markovchain[i+1] = numpy.random.choice(numpy.arange(0, len(KezdElosz)), p=TM[1])
        if markovchain[i] == 2:
            markovchain[i + 1] = numpy.random.choice(numpy.arange(0, len(KezdElosz)), p=TM[2])
    for i in range(l):
        if markovchain[i] == 0:
            markovchain[i] = 0 + np.random.normal(loc=0.4, scale=0.5, size=None)
        if markovchain[i] == 1:
            markovchain[i] = 1 + np.random.normal(0.3, 0.4, size=None)
        if markovchain[i] == 2:
            markovchain[i] = 2 + np.random.normal(0.5, 0.6, size=None)
    return markovchain
##############Szimulálás########################
def Average(hossz, db, KezdElosz, TM, c, mu, sig, n):
    a = [0, 0, 0]
    b = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    cc = np.zeros((num, 5))
    d = np.zeros((5, num))
    e = np.zeros((5, num))
    for q in range(db):
        markovlanc = MarkovChainContinuous(hossz, [0.2, 0.2, 0.6],
                                           [[0.5, 0.2, 0.3], [0.4, 0.3, 0.3],
                                            [0.3, 0.4, 0.3]])
        modell = Reestimation(KezdElosz, TM, c, mu, sig, n, markovlanc)
        a = np.add(a, modell[0])
        b = np.add(b, modell[1])
        cc = np.add(cc, modell[2])
        d = np.add(d, modell[3])
        e = np.add(e, modell[4])
    return a / db, b / db, cc/db, d/db, e/db
#############################Felismerés################################x
KezdElosz = np.loadtxt("kezdetikirany6.txt", dtype=float).reshape((12, num))
TM = np.loadtxt("átmenetekirany6.txt", dtype=float).reshape((12, num, num))
c = np.loadtxt("cirany6.txt", dtype=float).reshape((12, num, 5))
mu = np.loadtxt("muirany6.txt", dtype=float).reshape((12, 5, num))
sig = np.loadtxt("sigirany6.txt", dtype=float).reshape((12, 5, num))

tesztelesihalmaz = {0,1,2,3,4,5,6,7,8,9,10,11}
def felismero(KezdElosz, jel, TM, c,mu, sig):
    val = list([13,13,13,13,13,13,13,13,13,13,13,13])
    indexek = list([13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])
    for i in tesztelesihalmaz:
        F = np.zeros((num, len(jel)), float)
        Fc = np.zeros(len(jel), dtype='g')
        Forward(F,Fc,KezdElosz[i],jel,TM[i],c[i],mu[i],sig[i])
        val[i] = -sum(np.log(Fc))
    print(val)
    for i in range(24):
        indexek[i] = val.index(max(val))
        val[indexek[i]] = -200000
        if indexek.count(0)== 1 and indexek.count(1)== 1 and indexek.count(2)== 1:
            print('nyert felfelé')
            break
        if indexek.count(3)== 1 and indexek.count(4)== 1 and indexek.count(5)== 1:
            print('nyert balra')
            break
        #if indexek.count(6)== 1 and indexek.count(7)== 1 and indexek.count(8)== 1:
          #  print('nyert le')
            #break
        if indexek.count(9)== 1 and indexek.count(10)== 1 and indexek.count(11)== 1:
            print('nyert jobbra')
            break

for i in range(50):
    print('hanyadik jelen:',i,'\n')
    felismero(KezdElosz, jelek[i], TM, c, mu, sig)
    print('\n','_____________________','\n')


end = time.time()
print(end-start)
