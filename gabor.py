import numpy as np
import matplotlib.pyplot as plt

def g_sigma(t,sigma):
    return np.exp((-t**2)/(sigma**2))

u   = np.load('IMT2113_PIANO_SCALE_x8.NPY')
u=u[0:5555]
fs = 5512.5     
dt  = 1/fs
num_t = 2
ejef = np.linspace(-fs/2,fs/2,5555)
ejet = np.linspace(0,num_t,5555)

sigma = 10
XX,YY = np.meshgrid(ejef,ejet)
matrix = []
for t in ejet:
    gabor =  dt * np.fft.fftshift(np.fft.fft(g_sigma(ejet-t,sigma)*u))
    matrix.append(abs(gabor)**2)

plt.contourf(YY,XX/1000,matrix)
plt.xlabel('tiempo (s)')
plt.ylabel('frecuencia (kHz)')

plt.show()
