import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from scipy import signal
import time


def middelverdifilter(size):
    """
    Funksjonen returnerer et middelverdifilter av størrelse size x size
    """
    filter = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            filter[i,j] = 1 / (size*size)
    return filter


def main():
    filename = "cow.png"
    img = imread(filename, as_gray = True)
    M = img.shape[0] # Rader / vertikale piksler
    N = img.shape[1] # Kolonner / horisontale piksler
    
    # Oppgave 1.1:
    filter = middelverdifilter(15) # 15 x 15 middelverdifilter
    konvolusjon_romlig = signal.convolve2d(img, filter, "same") # romlig konvolusjon

    filter_frekvens = np.fft.fft2(filter, [M,N])
    F = np.fft.fft2(img)

    konvolusjon_frekvensdom = np.real(np.fft.ifft2(F*filter_frekvens)) # filtrering i frekvensdomenet

    plt.subplot(1,3,1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
    plt.title("Originalbilde")

    plt.subplot(1,3,2)
    plt.imshow(konvolusjon_romlig, cmap="gray", vmin=0, vmax=255, aspect="auto")
    plt.title("Romlig konvolusjon")

    plt.subplot(1,3,3)
    plt.imshow(konvolusjon_frekvensdom, cmap="gray", vmin=0, vmax=255, aspect="auto")
    plt.title("Konvolusjon i frekvensdomenet")
    plt.show()

    # Oppgave 1.3:
    # Middelverdifiltre med forskjellig størrelse:
    filter2 = middelverdifilter(2)
    filter3 = middelverdifilter(3)
    filter5 = middelverdifilter(5)
    filter7 = middelverdifilter(7)
    filter9 = middelverdifilter(9)
    filter15 = middelverdifilter(15)

    filtere = [filter2, filter3, filter5, filter7, filter9, filter15]
    tid_romlig = []
    tid_frekvens = []
    # Finner kjøretid ved direkte konvolusjon, og ved å gå til frekvensdomenet:
    for filter in filtere:
        start_romlig = time.time()
        konvolusjon_romlig = signal.convolve2d(img, filter, "same")
        total_romlig = time.time() - start_romlig
        tid_romlig.append(total_romlig)

        start_frekvens = time.time()
        filter_frekvens = np.fft.fft2(filter, [M,N])
        F = np.fft.fft2(img)
        konvolusjon_frekvensdom = np.real(np.fft.ifft2(F*filter_frekvens))
        total_frekvens = time.time() - start_frekvens
        tid_frekvens.append(total_frekvens)

    filter_størrelser = [2*2, 3*3, 5*5, 7*7, 9*9, 15*15]

    fontsize = 15
    ticksize = 13
    plt.plot(filter_størrelser, tid_romlig, label="Romlig konvolusjon")
    plt.plot(filter_størrelser, tid_frekvens, label="Konvolusjon i frekvensdomenet")
    plt.title("Kjøretid ved konvolusjon", fontsize=fontsize)
    plt.xlabel("filterstørrelse [# piksler]", fontsize=fontsize)
    plt.ylabel("tid [s]", fontsize=fontsize)
    plt.legend(prop={'size': fontsize})
    plt.xticks(size=ticksize)
    plt.yticks(size=ticksize)
    plt.show()


    return 0

main()
