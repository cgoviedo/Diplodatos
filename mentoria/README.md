# Como correr el notebook

- Correr el Notebook usando archivos pre-procesados: Si no se modifican variables, el notebook se puede correr directamente. Para ello el notebook carga los archivos **corpus.csv** y **agregated_corpus.csv**. Estos archivos tiene los corpus preprocesados. Se asume que jupyter notebook se lanza desde el directorio en donde se encuentra el notebook **Practico 1 - Analisis y Visualizacion** y los archivos csv.

- Correr el notebook usando los documentos de sentencias: En el caso de querer generar los corpus desde los archivos de sentencias, se debe setear la variable **load_preprocessed_files=False** y además, se debe indicar en la variable **root_path** el directorio raíz que contiene los subdirectorios PENAL, FAMILIA, LABORAL y MENORES.

# Carpetas

- mentoria: Contiene el Notebook de la mentoria y los csv con los corpus preprocesados.
- mentoria/lib: Contiene modulos python que desarrollamos con clases y funciones reutilizables.
- mentoria/misc: Notebooks varios que se utilizaron para hacer pruebas.


Convertir archivos

ffmpeg -i mentoria_v4.1.mov  -r 24 -b:a 50k  output.mp4 
