import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.transform import radon, iradon

imagen_path = 'im.png'
imagen = io.imread(imagen_path, as_gray=True)


angulos = np.linspace(0., 180., max(imagen.shape), endpoint=False)
sinograma = radon(imagen, theta=angulos)

imagen_reconstruida = iradon(sinograma, theta=angulos, filter_name='ramp')


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Imagen original')
plt.imshow(imagen, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Sinograma')
plt.imshow(sinograma, cmap='gray', aspect='auto', extent=(0, 180, 0, sinograma.shape[0]))

plt.subplot(1, 3, 3)
plt.title('Reconstrucción')
plt.imshow(imagen_reconstruida, cmap='gray')

plt.show()
