import numpy as np
from naloga_2_resitve import *


def test_binarna_segmentacija():
    '''Test binarne segmentacije.'''
    # Test 1: Sivinska slika
    slika = np.zeros((100, 100), dtype=np.float32)
    rezultat = binarna_segmentacija(slika)

    assert rezultat.shape == (100, 100), f'Napačna oblika: {rezultat.shape}'
    assert rezultat.dtype == bool, f'Napačen tip: {rezultat.dtype}'
    print('test_binarna_segmentacija: sivinska slika')

    # Test 2: Barvna slika
    slika = np.zeros((150, 180, 3), dtype=np.float32)
    rezultat = binarna_segmentacija(slika)

    assert rezultat.shape == (150, 180), f'Napačna oblika: {rezultat.shape}'
    print('test_binarna_segmentacija: različne dimenzije')


def test_izrezi_regije():
    '''Test izrezovanja regij.'''
    # Test 1: Slika z masko
    slika = np.random.rand(200, 200, 3)
    maska = np.zeros((200, 200), dtype=bool)
    maska[50:100, 50:100] = True
    maska[120:150, 120:180] = True

    rezultat = izrezi_regije(slika, maska)

    assert isinstance(rezultat, list), f'Rezultat mora biti seznam, ne {type(rezultat)}'
    assert len(rezultat) > 0, 'Pričakovan vsaj en vzorec'
    for i, vzorec in enumerate(rezultat):
        assert isinstance(vzorec, np.ndarray), f'Vzorec {i} ni numpy tabela'
        assert vzorec.ndim == 3, f'Vzorec {i} mora biti 3D tabela'
    print('test_izrezi_regije: osnovna maska')


def test_detekcija_4_kotnikov():
    '''Test detekcije 4-kotnikov.'''
    # Test 1: Slika s štirikotnikom
    slika = np.zeros((200, 200))
    # Narisemo preprost pravokotnik
    slika[50:150, 50:150] = 1

    rezultat = detekcija_4_kotnikov(slika)

    assert isinstance(rezultat, list), f'Rezultat mora biti seznam, ne {type(rezultat)}'
    for i, stirikonik in enumerate(rezultat):
        assert isinstance(stirikonik, np.ndarray), f'Štirikotnik {i} ni numpy tabela'
        assert stirikonik.shape == (4, 2), f'Štirikotnik {i} mora imeti obliko (4, 2), ne {stirikonik.shape}'
    print('test_detekcija_4_kotnikov: osnovna detekcija')


def test_detekcija_4_kotnikov_adaptivno():
    '''Test adaptivne detekcije 4-kotnikov.'''
    # Test 1: Slika s štirikotnikom
    slika = np.ones((200, 200)) * 128
    # Narisemo preprost pravokotnik z različno osvetlitvijo
    slika[50:150, 50:150] = 200

    rezultat = detekcija_4_kotnikov_adaptivno(slika)

    assert isinstance(rezultat, list), f'Rezultat mora biti seznam, ne {type(rezultat)}'
    for i, stirikonik in enumerate(rezultat):
        assert isinstance(stirikonik, np.ndarray), f'Štirikotnik {i} ni numpy tabela'
        assert stirikonik.shape == (4, 2), f'Štirikotnik {i} mora imeti obliko (4, 2), ne {stirikonik.shape}'
    print('test_detekcija_4_kotnikov_adaptivno: osnovna detekcija')


def test_detekcija_plus():
    '''Test detekcije '+' simbolov.'''
    # Test 1: Slika in filter
    slika = np.random.rand(100, 100)
    filter = np.random.rand(15, 15)  # Poljuben filter velikosti 15x15

    rezultat = detekcija_plus(slika, filter)

    assert isinstance(rezultat, np.ndarray), f'Rezultat mora biti numpy tabela, ne {type(rezultat)}'
    assert rezultat.ndim == 2, f'Rezultat mora biti 2D tabela, ne {rezultat.ndim}D'
    assert rezultat.shape[1] == 2, f'Rezultat mora imeti 2 stolpca (y, x), ne {rezultat.shape[1]}'
    print('test_detekcija_plus: osnovna detekcija')


if __name__ == '__main__':
    print('Začenjam teste...')

    print('=== Test 1: Binarna segmentacija ===')
    test_binarna_segmentacija()
    print()

    print('=== Test 2: Izrezovanje regij ===')
    test_izrezi_regije()
    print()

    print('=== Test 3: Detekcija 4-kotnikov ===')
    test_detekcija_4_kotnikov()
    print()

    print('=== Test 4: Detekcija 4-kotnikov adaptivno ===')
    test_detekcija_4_kotnikov_adaptivno()
    print()

    print('=== Test 5: Detekcija plus simbolov ===')
    test_detekcija_plus()
    print()

    print('Vsi testi so uspešno zaključeni!')