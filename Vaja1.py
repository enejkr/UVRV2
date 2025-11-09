from matplotlib import pyplot as plt

from naloga_2_resitve import *

slika = plt.imread("Primeri/primer_1.png")
koncnaSlika = binarna_segmentacija(slika)

plt.figure()
plt.subplot(1, 2, 1)
plt.title('Originalna slika')
plt.imshow(slika)
plt.subplot(1, 2, 2)
plt.title('Segmentirana slika')
plt.imshow(koncnaSlika, cmap='gray')
plt.show()
