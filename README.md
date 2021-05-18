# Diplodatos


# Links útiles

- nbdime :
	https://anaconda.org/conda-forge/nbdime
	https://nbdime.readthedocs.io/en/latest/

Ejemplo para comparar copia local con branch

```
nbdiff origin/lior_main -- Entregable_Parte_1.ipynb
```
Ejemplo para comparar copia local con branch ignorando las celdas de salida

```
nbdiff-web --ignore-output origin/lior_main -- Entregable_Parte_1.ipynb
```
