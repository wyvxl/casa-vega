"""Texturas fotográficas del modelo 3D (tex/).

Descarga fotos CC0 de Poly Haven (https://polyhaven.com, a 1k) y las convierte en las texturas de tex/, a 512 px,
cada una con su mapa de relieve (_n, normal map en convención OpenGL).

Para que la casa no cambie de colores, el color de cada foto se ajusta así:
- Repello, concreto, cerámica, bloque, asfalto, madera de marcos y corteza quedan en gris casi blanco ("detail"),
  porque el color lo pone el material en index.html (color × textura).
- Tierra, zacate, piedrilla, puerta y alero se recolorean al color promedio de la textura dibujada que reemplazan.
- La cerámica, el bloque, la baldosa de tapia, la fachaleta y la puerta se arman con el mismo tamaño y la misma
  distribución que las texturas de canvas de index.html (T_tile, T_block, ...), porque los UV de la página cuentan con eso.

Uso (desde cualquier carpeta):  python herramientas/texturas.py
Requiere Python 3 con numpy y Pillow. Las fotos descargadas quedan en herramientas/fuentes/ (no se suben: *.jpg).
Lo que sale es determinista: correrlo otra vez da las mismas imágenes.
"""
import os, random, json, urllib.request
import numpy as np
from PIL import Image, ImageFilter

AQUI = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(AQUI, 'fuentes')               # fotos originales de Poly Haven
O = os.path.join(AQUI, '..', 'tex')             # texturas que usa la página
N = 512
FOTOS = ['painted_plaster_wall', 'rough_concrete', 'asphalt_02', 'gravel_floor', 'brown_mud_dry', 'grass_ground',
         'oak_veneer_01', 'roof_planks', 'castle_wall_slates', 'pine_bark']


def descargar():
    os.makedirs(S, exist_ok=True)
    op = urllib.request.build_opener(); op.addheaders = [('User-Agent', 'casa-texturas/1.0')]
    for i in FOTOS:
        faltan = [k for k in ('diff', 'nor') if not os.path.exists(f'{S}/{i}_{k}.jpg')]
        if not faltan: continue
        f = json.load(op.open('https://api.polyhaven.com/files/' + i, timeout=15))
        for key, k in (('Diffuse', 'diff'), ('nor_gl', 'nor')):
            if k in faltan:
                with op.open(f[key]['1k']['jpg']['url'], timeout=15) as r, open(f'{S}/{i}_{k}.jpg', 'wb') as out: out.write(r.read())
        print('descargada', i)


def load(n, kind='diff', size=N):
    return np.asarray(Image.open(f'{S}/{n}_{kind}.jpg').convert('RGB').resize((size, size), Image.LANCZOS)).astype(np.float32) / 255
def save(a, name, q=84):
    Image.fromarray((np.clip(a, 0, 1) * 255 + .5).astype(np.uint8)).save(f'{O}/{name}.jpg', quality=q, optimize=True, progressive=True)
def lum(a): return a @ np.array([.299, .587, .114], np.float32)
def to_img(a): return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
def from_img(im): return np.asarray(im).astype(np.float32) / 255
def blur(a, r): return from_img(to_img(a).filter(ImageFilter.GaussianBlur(r)))

def detail(a, mean, std, warm=(1, 1, 1)):
    """Gris normalizado (promedio y contraste dados): el color lo pone el material."""
    l = lum(a); l = (l - l.mean()) / (l.std() + 1e-6) * std + mean
    return np.stack([l * warm[0], l * warm[1], l * warm[2]], -1)
def hp_detail(name, mean, std, r):
    """Como detail, pero sin las manchas grandes (pasa altos de radio r): en una pared de 2 m se veían como suciedad."""
    l = lum(load(name)); h = l - blur(l, r); h = h / (h.std() + 1e-6) * std + mean
    return np.stack([h, h, h], -1)
def recolor(a, target, std_scale=1.0):
    """Lleva el color promedio de la foto a target (RGB 0-255) y escala el contraste."""
    t = np.array(target, np.float32) / 255; m = a.reshape(-1, 3).mean(0)
    return (a - m) * std_scale + t
def height_normal(h, strength=2.0):
    """Mapa normal a partir de una altura, con bordes que se repiten."""
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * strength; gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * strength
    n = np.stack([-gx, gy, np.ones_like(h)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * .5 + .5
def blend_normal(a, b):
    """Combina dos mapas normales (whiteout)."""
    a = a * 2 - 1; b = b * 2 - 1
    n = np.stack([a[..., 0] + b[..., 0], a[..., 1] + b[..., 1], a[..., 2] * b[..., 2]], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * .5 + .5
def flatten_normal(n, k):
    n = n * 2 - 1; n[..., :2] *= k; n /= np.linalg.norm(n, axis=-1, keepdims=True); return n * .5 + .5
def resize(a, w, h): return from_img(to_img(a).resize((w, h)))


def generar():
    os.makedirs(O, exist_ok=True)
    # Repello pintado y concreto: solo el grano fino (T_plaster, T_concrete; en las cajas se repiten cada 2 m)
    save(hp_detail('painted_plaster_wall', .975, .02, 6), 'plaster'); save(flatten_normal(load('painted_plaster_wall', 'nor'), .8), 'plaster_n')
    save(hp_detail('rough_concrete', .94, .055, 40), 'concrete'); save(load('rough_concrete', 'nor'), 'concrete_n')

    # Piso cerámico (T_tile): 2 × 2 baldosas de esmalte casi liso con junta. Con mármol se veían juntas dentro de cada baldosa
    pp = load('painted_plaster_wall', 'diff', 1024)
    tile = np.zeros((N, N, 3), np.float32); h = np.zeros((N, N), np.float32); s = N // 2; rnd = random.Random(3)
    for i in range(2):
        for j in range(2):
            x, y = rnd.randrange(0, 1024 - s), rnd.randrange(0, 1024 - s)
            tile[j * s:(j + 1) * s, i * s:(i + 1) * s] = detail(pp[y:y + s, x:x + s], .95 + rnd.uniform(-.02, .02), .025)
    for k in range(0, N, s):
        for kk in range(-3, 3):
            idx = (k + kk) % N; tile[idx, :] = [.62, .58, .54]; tile[:, idx] = [.62, .58, .54]; h[idx, :] = -1; h[:, idx] = -1
    save(tile, 'tile'); save(height_normal(blur((h + 1) / 2, 1.2), 3.0), 'tile_n')

    # Bloque de 0.40 × 0.20 en petatillo (T_block: 2 bloques × 2 hiladas) y baldosa de tapia (T_slab: junta arriba)
    def masonry(name, W, H, joints, depth=4.0, dark=.58):
        a = detail(load('rough_concrete', 'diff', W)[:H], .9, .09)
        hh = np.ones((H, W), np.float32)
        for (x0, y0, x1, y1) in joints: a[y0:y1, x0:x1] *= dark; hh[y0:y1, x0:x1] = 0
        nb = load('rough_concrete', 'nor', max(W, H))[:H, :W]
        save(a, name); save(blend_normal(height_normal(blur(hh, 1.5), depth), nb), name + '_n')
    W, H, j = 512, 256, 3
    masonry('block', W, H, [(0, 0, W, j), (0, H // 2 - j // 2, W, H // 2 + j // 2 + 1), (0, 0, j, H // 2), (W // 2 - 1, 0, W // 2 + 2, H // 2),
                            (W // 4 - 1, H // 2, W // 4 + 2, H), (3 * W // 4 - 1, H // 2, 3 * W // 4 + 2, H)])
    masonry('slab', 512, 256, [(0, 0, 512, 4)], dark=.65)

    # Calle, piedrilla, tierra roja y zacate
    save(detail(load('asphalt_02'), .92, .11), 'asphalt'); save(load('asphalt_02', 'nor'), 'asphalt_n')
    save(recolor(load('gravel_floor'), (150, 146, 140), 1.8), 'gravel'); save(load('gravel_floor', 'nor'), 'gravel_n')
    save(recolor(load('brown_mud_dry'), (143, 80, 50), 1.1), 'earth'); save(load('brown_mud_dry', 'nor'), 'earth_n')
    save(recolor(load('grass_ground'), (108, 136, 76), 1.25), 'grass'); save(load('grass_ground', 'nor'), 'grass_n')

    # Puerta (T_door, 256 × 512): vetas de roble con los seis tableros en el mismo lugar que la textura dibujada
    d = recolor(resize(load('oak_veneer_01', 'diff', 1024), 256, 512), (201, 152, 103), 1.2)
    hh = np.ones((512, 256), np.float32)
    m_, gap = 26, 20; pw = (256 - 2 * m_ - gap) // 2; y = 24
    for ph in (120, 170, 130):
        for k in range(2):
            x = m_ + k * (pw + gap)
            d[y - 3:y + ph + 3, x - 3:x + pw + 3] *= .78; hh[y - 3:y + ph + 3, x - 3:x + pw + 3] = .4
            d[y - 1:y + ph + 3, x - 1:x + pw + 3] *= 1.08
            d[y + 6:y + ph - 6, x + 6:x + pw - 6] = d[y + 6:y + ph - 6, x + 6:x + pw - 6] / (1.08 * .78) * 1.03; hh[y + 6:y + ph - 6, x + 6:x + pw - 6] = .75
        y += ph + 18
    save(d, 'door'); save(height_normal(blur(hh, 1.6), 4.0), 'door_n')
    # Marcos de cedro (T_wood, cada 1 m) y corteza de los troncos (T_bark)
    save(detail(load('oak_veneer_01'), .92, .09, (1.03, 1, .96)), 'wood'); save(flatten_normal(load('oak_veneer_01', 'nor'), .6), 'wood_n')
    save(detail(load('pine_bark', 'diff', 256), .9, .16, (1.02, 1, .97)), 'bark'); save(load('pine_bark', 'nor', 256), 'bark_n')

    # Cielo del alero (T_soffit): tablilla, la textura cubre 0.40 m = 4 tablas
    save(recolor(resize(load('roof_planks', 'diff', 1024)[:, :512], 256, 256), (91, 59, 41), 1.0), 'soffit', 86)
    save(resize(load('roof_planks', 'nor', 1024)[:, :512], 256, 256), 'soffit_n', 86)

    # Fachaleta (T_stone, panel de 0.62 m): tiras de laja apiladas, cada una con un pedazo de piedra real
    stl = lum(load('castle_wall_slates', 'diff', 1024))
    cols = [(207, 174, 130), (220, 192, 153), (191, 154, 108), (227, 203, 166), (198, 162, 117), (212, 181, 138), (181, 141, 98)]
    A = np.zeros((N, N, 3), np.float32) + np.array([122, 97, 72]) / 255 * .85; Hh = np.zeros((N, N), np.float32); r = random.Random(8)
    y = 0
    while y < N:
        rh = min(N - y, 18 + r.randrange(24)); x = -r.random() * 100
        while x < N:
            ln = int(44 + r.random() * 180); ins = r.randrange(3)
            x0, x1 = int(max(0, x + 2)), int(min(N, x + ln - 2)); y0, y1 = y + 2 + ins, y + rh - 2
            if x1 > x0 and y1 > y0:
                sx, sy = r.randrange(0, 1024 - (x1 - x0)), r.randrange(0, 1024 - (y1 - y0))
                pat = stl[sy:sy + y1 - y0, sx:sx + x1 - x0]; pat = (pat - pat.mean()) / (pat.std() + 1e-6) * .09 + 1
                c = np.array(cols[r.randrange(len(cols))]) / 255
                A[y0:y1, x0:x1] = pat[..., None] * c
                A[y0:y0 + 3, x0:x1] *= 1.12; A[y1 - 3:y1, x0:x1] *= .86
                Hh[y0:y1, x0:x1] = .7 + pat * .25 + r.random() * .1
            x += ln
        y += rh
    A[:, 0:4] = np.array([.94, .92, .88])                      # junta entre paneles
    save(A, 'stone', 86); save(height_normal(blur(np.clip(Hh, 0, 1), 1.0), 5.0), 'stone_n', 86)


if __name__ == '__main__':
    descargar()
    generar()
    print('listo: texturas en', os.path.normpath(O))
