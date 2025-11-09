from matplotlib import pyplot as plt

from naloga_2_resitve import *

slika = plt.imread("Primeri/primer_3.png")
koncnaSlika = detekcija_4_kotnikov(slika)
plt.figure()
plt.subplot(1, 2, 1)
plt.title('Originalna slika')
plt.imshow(slika)
plt.subplot(1, 2, 2)

plt.figure()
plt.title('Detekcija 4-kotnikov')

img = slika

rectangles = []
if isinstance(koncnaSlika, (list, tuple)):
    rectangles = [np.asarray(r) for r in koncnaSlika if r is not None]
elif isinstance(koncnaSlika, np.ndarray):
    if koncnaSlika.ndim == 3 and koncnaSlika.shape[-1] == 2:
        rectangles = [koncnaSlika]
    else:
        img = koncnaSlika

plt.imshow(img)
plt.axis('off')
for rect in rectangles:
    if rect.ndim == 2 and rect.shape[1] >= 2:
        plt.plot(rect[:, 1], rect[:, 0], 'ro', markersize=3)
plt.show()