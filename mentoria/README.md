# Busqueda y Recomendacion para Textos Legales

La idea es poder recomendar a partir de un texto proveniente de un artículo, contrato, etc., sentencias y fallos judiciales, semejantes a dicho texto.

El principal atractivo de un sistema de Recomendación reside en que ofrece información relevante para el usuario en forma activa, acerca de la base de información del dominio en cuestión, sin necesidad de que el mismo tenga conocimientos sobre los artículos recomendados o la consulta a realizar. El valor de esta información proviene del análisis previo de los datos y su relación con los usuarios.

Además, podrás tener una primera aproximación al análisis de texto y el procesamiento del lenguaje natural integrando sobre un mismo dataset los temas vistos en las materias de la Diplomatura.

Trataremos de responder algunas de las siguientes preguntas
* ¿Se puede buscar y recomendar fallos y sentencias judiciales estableciendo automáticamente cuales son en un corpus de datos lo que tienen mayor similitud con el tema buscado? 
* ¿Se puede extender la búsqueda a leyes, decretos, resoluciones, etc.?
* ¿Cuáles son los procesos necesarios basados en Procesamiento de Lenguaje Natural (NLP) para buscar y brindar recomendaciones de fallos y sentencias judiciales? 
* ¿Se pueden hacer búsquedas y recomendaciones desde distintas fuentes de información, como artículos periodísticos, fragmentos de contratos, textos con sentido legal como tweets u opiniones de profesionales del ámbito legal?

Tenemos 2 carpetas:
* Datos: contiene fallos y sentencias divididos por fuero: Familia, Laboral, Menores y Penal. Son documentos en formato DOC, DOCX y PDF.
* Documentos: son los mismos documentos en formato TXT.

También se encuentra la notebook utilizada para la conversión del formato de los documentos.
