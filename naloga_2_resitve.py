import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import cv2
from matplotlib.pyplot import figure


def binarna_segmentacija(slika: np.ndarray) -> np.ndarray:
    #pridobim spekter
    siv = slika.mean(axis=2)
    T = 0.65
    #ohranim vrednosti nad 0.65 (0.65) da lahko ta nacin pridobivanje maske
    #deluje tudi pri drugi nalogi
    mask = siv > T
    return mask

def izrezi_regije(slika: np.ndarray, maska: np.ndarray) -> list[np.ndarray]:
    # invertam masko in poskrbim da je v bool
    maska = np.logical_not(maska.astype(bool))

    #build a square kernel ~5% of the smaller image dimension (min size = 1).
    kernel_size = max(1, int(min(maska.shape) * 0.05))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # prilagodim masko glede na kernel
    povecana_maska = cv2.dilate(maska.astype(np.uint8), kernel, iterations=1).astype(bool)

    # locim nepovezane objekte
    labeled, num_features = ndimage.label(povecana_maska)
    if num_features == 0:
        return []

    # for each labeled component, get bounding box slices.
    objekti = ndimage.find_objects(labeled)

    patches: list[np.ndarray] = []
    for slc in objekti:
        if slc is None:
            continue
        ys, xs = slc  #pridobim kordinate kvadrata okoli lika

        #pridobim in shranim kvadrat
        patch = slika[ys, xs, ...]
        patches.append(patch.copy())

    return patches


def detekcija_4_kotnikov(slika: np.ndarray) -> list[np.ndarray]:
    """
    Detect quadrilateral contours and return their 4 corner points in (y, x) order.

    Pipeline:
      1) Convert to grayscale if input is color.
      2) Normalize to 8-bit range [0, 255] for stable thresholding.
      3) Apply a fixed binary inverse threshold (value = 100) to separate foreground.
      4) Find external contours.
      5) Approximate each contour with a polygon (epsilon = 2% perimeter).
      6) Keep only 4-vertex polygons with sufficient area.
      7) Convert points from (x, y) to (y, x) and collect.

    Parameters:
      slika : np.ndarray
        Input image, grayscale or BGR color.

    Returns:
      list[np.ndarray]
        Each element is a (4, 2) array of corner coordinates in (y, x).
    """
    # spravim v greysale in
    gray = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY) if len(slika.shape) == 3 else slika

    #normaliziram tip v unit8
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #  binary inverse threshold:
    #   - pxels darker than 100 become 255 (foreground), others become 0 (background).
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # detect outermost contours on the thresholded image. aka dobim robe kvadratov
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kvadrilateri = []

    #za vsak lik preverim ce vstreza kvadratu
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        #obdrzim samo 4 robe kvadrata in odvrzem male like
        if len(approx) == 4 and cv2.contourArea(approx) > 100:
            # pretvorim array tock v format (4,2)
            pts = approx.reshape(-1, 2)

            # flip to (y, x)
            pts_yx = np.flip(pts, axis=1)

            kvadrilateri.append(pts_yx)

    return kvadrilateri


def detekcija_4_kotnikov_adaptivno(slika: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY) if slika.ndim == 3 else slika

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #glajenje ozadja z pomocjo gausa
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Choose an odd block size for local neighborhood:
    # - Use 51 on typical images; otherwise compute the largest odd number <= half of the smaller dimension.
    # - Larger values smooth over larger illumination gradients; smaller values preserve fine detail.
    block_size = 51
    C = 7  # Bias term: positive values make the threshold slightly stricter (favoring darker foreground).
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C
    )

    # Morphology: clean up the binary mask.
    # - Open: remove isolated white noise (small bright specks).
    # - Close: fill pinholes and connect thin gaps along edges.
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # pridobim robove likov
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # nesprejmem malih zaznanih likov
    h, w = thresh.shape
    min_area = 0.0005 * (h * w)
    quads: list[np.ndarray] = []

    def order_points_yx(pts_yx: np.ndarray) -> np.ndarray:
        #changes for a triangle
        # idx_y = np.argsort(pts_yx[:, 0])
        # top = pts_yx[idx_y[0]]
        # bottom = pts_yx[idx_y[1:]]
        # # sort bottom by x descending -> right first, then left
        # br, bl = bottom[np.argsort(bottom[:, 1])[::-1]]
        # return np.array([top, br, bl], dtype=int)

        # split into top/bottom by y, then sort each pair by x to get left/right.
        idx = np.argsort(pts_yx[:, 0])
        top_idx = idx[:2]
        bot_idx = idx[2:]
        top = pts_yx[top_idx]
        bot = pts_yx[bot_idx]
        tl, tr = top[np.argsort(top[:, 1])]
        bl, br = bot[np.argsort(bot[:, 1])]
        return np.array([tl, tr, br, bl], dtype=int)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # Ignore tiny detections/noise.

        # Approximate contour with a polygon; epsilon is 2% of perimeter.
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        #changes for a triangle
        # if approx.shape[0] != 3:
        #     continue

        # obdrzim samo kvadrate
        if approx.shape[0] != 4:
            continue

        # impose consistent corner ordering.
        pts = approx.reshape(4, 2)         # (x, y)
        pts_yx = np.flip(pts, axis=1)      # -> (y, x)
        ordered = order_points_yx(pts_yx)
        quads.append(ordered)

        ##changes for a triangle
        # pts = approx.reshape(3, 2)          # (x,y)

    return quads

#gray scale and float32 conversion
def _to_gray_f32(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32)

#priprava filterja za plus (input je x in plus)
def pripravi_filter_plus(plus_patches: list[np.ndarray], x_patches: list[np.ndarray]) -> np.ndarray:
    assert len(plus_patches) > 0 and len(x_patches) > 0, "Need positive and negative patches."

    P = np.stack([_to_gray_f32(p) for p in plus_patches], axis=0)
    X = np.stack([_to_gray_f32(p) for p in x_patches], axis=0)
    assert P.shape[1:] == X.shape[1:], "Patch sizes must match."

    # class mediane.
    mu_pos = P.mean(axis=0)
    mu_neg = X.mean(axis=0)

    #  podarim krizane pixle
    v = mu_pos - mu_neg

    # normaliziram uporaba formule
    norm = np.sqrt(np.sum(v * v))
    w = (v / norm).astype(np.float32)
    return w


def detekcija_plus(slika: np.ndarray, filter: np.ndarray) -> np.ndarray:
    # normaliziranje vhoda
    img = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY).astype(np.float32)

    tpl = filter.astype(np.float32)
    h, w = tpl.shape
    #korelacija lahko tudi uporabil coralate iz prejsnje vaje
    resp = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    figure()
    plt.imshow(resp)

    # flip if one side domenates TODO check why neccesary
    if abs(resp.min()) > abs(resp.max()):
        resp = -resp


    nms_radius = max(2, int(round(min(h, w) / 4)))
    #size = 2 * nms_radius + 1
    resp_max = ndimage.maximum_filter(resp, size=nms_radius, mode="nearest")
    is_max = resp == resp_max  # Boolean mask for local maxima.

    # threshold katere obdrzim
    thr = 0.4
    keep = is_max & (resp >= thr)

    # pridobim ostale vrednosti
    ys, xs = np.nonzero(keep)
    if ys.size == 0:
        return np.empty((0, 2), dtype=int)

    scores = resp[ys, xs]

    # pridobivanje centra
    ys = ys + h // 2
    xs = xs + w // 2

    # sort detections by descending score.
    order = np.argsort(-scores)
    coords = np.column_stack([ys[order].astype(int), xs[order].astype(int)])
    return coords