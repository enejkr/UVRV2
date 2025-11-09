from matplotlib import pyplot as plt

from naloga_2_resitve import *
slika = plt.imread("Primeri/primer_2.png")
koncnaSlika = binarna_segmentacija(slika)
izrezane_regije = izrezi_regije(slika, koncnaSlika)

# Prikaz originalne in segmentirane slike
fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
axes1[0].set_title('Originalna slika')
axes1[0].imshow(slika)
axes1[0].axis('off')
axes1[1].set_title('Segmentirana slika')
axes1[1].imshow(koncnaSlika, cmap='gray')
axes1[1].axis('off')
fig1.tight_layout()

if len(izrezane_regije) > 0:
    n = len(izrezane_regije)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows * cols == 1:
        axes2 = [axes2]
    else:
        axes2 = axes2.flatten()
    for i, izrezek in enumerate(izrezane_regije):
        axes2[i].imshow(izrezek)
        axes2[i].set_title(f'Izrezek {i+1}')
        axes2[i].axis('off')
    for j in range(len(izrezane_regije), rows * cols):
        axes2[j].axis('off')
    fig2.tight_layout()

plt.show()