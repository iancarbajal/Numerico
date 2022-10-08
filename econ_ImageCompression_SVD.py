#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 11:10:01 2022
@author: Pablo Castañeda, ITAM

EJEMPLO CON SVD:
    La idea de este código es usar la Descomposición en
    Valores Singulares, o SVD por su sigla en inglés, para
    guardar menos información de una figura, por ejemplo.
    
    El código tiene todavía detalles a ser mejorados en
    el manejo de las imágenes. Los colores llamativos y
    disonantes en los resultados no es lo común, estos
    deberían ser matizados con los colores del entorno.

@author: Pablo Castañeda.
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image


""" Aquí hay varios ejemplos que pueden ser utilizados """
FNAME = "quetzalbalam.jpg"

img = Image.open(FNAME)
"""
 Atención especial hay que tener en cuenta pues las imágenes
 y los arreglos tienen las dimensiones en opuesto. Por eso
 tomamos n y m al contrario.
"""
m = img.size[0]
n = img.size[1]

# Queremos que las imágnes entren en la pantalla.
if n > 1500 or m > 1000:
    r = min(1500/n, 1000/m)
    n = int(r*n)
    m = int(r*m)
    img = img.resize((n, m))

# Es bueno tener una imagen original
img.show(title = 'Original')

# Separamos la imagen en sus colores...
fuente = img.split()

# ... y las representamos como arreglos.
imgR = np.array(fuente[0])
imgG = np.array(fuente[1])
imgB = np.array(fuente[2])

# Hay que darle las dimensiones correctas...
img_mat = np.empty([3*n, m])
# ... y concatenarlo en un único arreglo 
# con las tres bandas
img_mat[   :  n, :] = imgR
img_mat[  n:2*n, :] = imgG
img_mat[2*n:3*n, :] = imgB

# Aquí utilizamos la SVD implementada en python
U, s, V = np.linalg.svd(img_mat, full_matrices = False)
# Con el tamaño completo como falso, estamos calculando
# la descomposición, no la factorización; más económico.

# Es util tener Sigma como la matriz completa
tot = len(s)

plt.plot(np.log(s))
plt.show()
time.sleep(1)

# Utilizaremos distintos porcentajes de la 
# información proporcionada con SVD
for k in range(7):
    # A saber 1, 2, 4, 8, 16, 32, 64%
    res = int(tot*(2**k)/100.)
    
    # En cada una de estas resoluciones, tenemos:
#    Ak = np.uint8(np.round(U[:, :res]@S[:res, :res]@V[:res, :]))
    Ak = np.uint8( (U[:, :res]*s[:res])@V[:res, :] )
    # Debememos juntar cada una de bandas...
    Rk = Image.fromarray(Ak[   :  n, :], mode = None)
    Gk = Image.fromarray(Ak[  n:2*n, :], mode = None)
    Bk = Image.fromarray(Ak[2*n:3*n, :], mode = None)

    # ... en una úinca imagen que visualizamos.
    reconK = Image.merge('RGB', (Rk, Gk, Bk))
    reconK.show()
    time.sleep(0.1)