import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
import math

def c(a):
    """
    Hjelpefunksjon brukt i DCT
    """
    if a == 0:
        return 1/np.sqrt(2)
    else:
        return 1

def cos(i, j):
    """
    Regner ut cosinus-uttrykket i formelen for 2D DCT
    """
    return np.cos((((2*i) + 1)*j*np.pi)/(16))

def DCT(f, u, v, a, b):
    """
    Transformerer hver 8x8 blokk med den todimensjonale diskrete cosinus-transformen
    """
    sum = 0
    for x in range(8):
        for y in range(8):
            u = u%8
            v = v%8
            sum += f[x+a,y+b] * cos(x,u) * cos(y,v)
    sum *= (1/4) * c(u) * c(v)
    return sum

def IDCT(F, x, y, a, b):
    """
    Inverstransformerer hver 8x8 blokk med den inverse todimensjonale diskrete
    cosinus-transformen
    """
    sum = 0
    for u in range(8):
        for v in range(8):
            x = x%8
            y = y%8
            sum += F[u+a,v+b] * c(u) * c(v) * cos(x,u) * cos(y,v)
    sum *= 1/4
    return sum


def kvantifiseringsmatrise():
    Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    return Q

def kvantifiser(F, q, M, N):
    """
    Punktvis dividerer hver av de transformerte 8×8-blokkene med qQ,
    dvs. produktet av tallparameteren q og kvantifiseringsmatrisen over.
    Avrunder de resulterende verdiene til nærmeste heltall.
    """
    Q = kvantifiseringsmatrise()
    for a in range(0,M,8):
        for b in range(0,N,8):
            F[a:a+8,b:b+8] = np.rint(F[a:a+8,b:b+8]/(Q*q))
    return F


def rekvantifiser(F, q, M, N):
    """
    Punktvis multipliserer hver 8×8-blokk med qQ og avrunder resultatene.
    """
    Q = kvantifiseringsmatrise()
    for a in range(0,M,8):
        for b in range(0,N,8):
            F[a:a+8,b:b+8] = np.rint(F[a:a+8,b:b+8]*Q*q)
    return F


def entropi(F):
    """
    Beregner entropien av et datasett.
    """
    F = F.flatten()
    N_ = len(F)

    if N_ <= 1:
        return 0 # Hvis datasettet består av 0 eller ett element ->entropi lik 0

    s, n = np.unique(F, return_counts=True)
    p = n / N_ # Sannsynligheten til de unike elementene i datasettet

    n_nonzero = np.count_nonzero(p)
    if n_nonzero <= 1:
        return 0 # Hvis alle pikslene er like -> entropi lik 0

    H = 0.
    for i in p:
        H -= i * math.log(i, 2)

    return H

def test_cos_trans(f, F, M, N):
    """
    Funksjonen tar inn originalbildet og en cosinustransformert versjon,
    samt dimensjonene deres. Inverterer transformen og adderer 128 til alle
    pikselintensitetene, og sjekker om det rekonstruerte bildet er identisk
    med originalen.
    """
    f_re = np.zeros_like(f)

    for i in range(0,M,8):
        for j in range(0,N,8):
            for x in range(i,i+8):
                for y in range(j,j+8):
                    f_re[x,y] = round(IDCT(F, x, y, i, j))
    f_re += 128


    if f_re[i,j] != f[i,j]:
        print("Feil! Rekonstruert bilde samsvarer ikke med orginalen")
    else:
        print("Rekonstruert bilde samsvarer med orginalen")


def jpeg_kompresjon(filename, q):
    """
    Funksjonen beregner omtrent hvor stor lagringsplass
    bildefilen funnet i "filename" vil bruke etter JPEG-komprimering,
    finner hvilken kompresjonsrate dette tilsvarer og beregner entropien.
    Skriver rekonstruksjoner av bildet til nye filer.

    Argumenter:
        filename: filnavn som spesifiserer plasseringen til bildefilen som skal benyttes
        q: Ett tall som indirekte vil bestemme kompresjonsraten
    """
    # Steg 1 : Last inn bildet
    f = imread(filename, as_gray = True) # Last inn bilde


    M = f.shape[0] # Rader / vertikale piksler
    N = f.shape[1] # Kolonner / horisontale piksler


    # Steg 2 : Subtraher 128 fra alle pikselintensiteter
    f_new = np.zeros_like(f)
    for i in range(M):
        for j in range(N):
            f_new[i,j] = f[i,j] - 128

    F = np.zeros_like(f)
    m = int(M/8)
    n = int(N/8)

    # Steg 3 : Del opp bildet i 8×8-blokker.
    # Transformer hver blokk med den todimensjonale diskrete cosinus-transformen
    for i in range(0,M,8):
        for j in range(0,N,8):
            for u in range(i,i+8):
                for v in range(j,j+8):
                    F[u,v] = DCT(f_new, u, v, i, j)




    # Steg 4 : Rekonstruer det opprinnelige bildet ved å invertere transformen
    # fra forrige steg og verifiser at det rekonstruerte bildet er identisk med originalen
    test_cos_trans(f, F, M, N)

    # Steg 5 : Punktvis divider hver av de transformerte 8×8-blokkene fra steg 3
    # med produktet av tallparameteren q og kvantifiseringsmatrisen Q
    F = kvantifiser(F,q, M, N)

    # Steg 6 : Beregn entropien og bruk den til å estimere hvor stor
    # lagringsplass det spesifiserte bildet vil bruke etter JPEG-komprimeringen
    # og hvilken kompresjonsrate dette tilsvarer.
    H = entropi(F)
    print(f"Entropi når q = {q}: {H:.4f}")
    print(f"Estimert lagringsplass når q = {q}: {(H*M*N):.4f}")
    print(f"Kompresjonsrate når q = {q}: {(64/H):.4f}\n")

    # Steg 7 : Bruk de transformerte og kvantifiserte 8×8-blokkene fra Steg 5
    # til å rekonstruere en tilnærming av det opprinnelige bildet.
    F = rekvantifiser(F,q, M, N)

    f_re = np.zeros_like(f)
    for i in range(0,M,8):
        for j in range(0,N,8):
            for x in range(i,i+8):
                for y in range(j,j+8):
                    f_re[x,y] = round(IDCT(F, x, y, i, j))
    f_re += 128

    imwrite(f"q={q}.png", f_re) # Skriv resultatbildet til fil
    return 0



def main():
    filename = "uio.png"

    # Tester implementasjonen ved å anvende den på bildet uio.png med
    # følgende verdier av tallparameteren q:
    qs = [0.1, 0.5, 2, 8, 32]
    for q in qs:
        jpeg_kompresjon(filename, q)


    return 0

main()
