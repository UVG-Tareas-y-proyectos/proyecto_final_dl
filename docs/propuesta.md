# Entrega parcial: detección automática de enfermedades en hojas

**Curso:** Deep Learning, UVG. **Fecha:** 26 de octubre de 2026.
**Integrantes:** Ihan-Marroquin y dpatzan2.
**Repositorio:** https://github.com/UVG-Tareas-y-proyectos/proyecto_final_dl

## 1. Descripción del problema

La identificación visual de enfermedades en hojas requiere experiencia y puede
complicarse cuando diferentes afecciones presentan síntomas parecidos. Proponemos
un prototipo que clasifique imágenes de hojas para explorar cómo las redes
convolucionales pueden apoyar el análisis de sanidad vegetal.

## 2. Objetivo y pregunta experimental

Desarrollar una aplicación local que reciba una imagen y presente su clase predicha,
comparando una CNN desde cero con una red preentrenada.
¿MobileNetV2 obtiene mayor F1 macro que una CNN pequeña usando los mismos datos,
particiones y presupuesto de épocas? La hipótesis se evaluará, no se asume verdadera.

Objetivos específicos: construir particiones reproducibles; entrenar ambos modelos;
medir accuracy, precision, recall, F1 y confusiones; analizar errores y ofrecer una
interfaz de carga de imágenes con limitaciones visibles.

## 3. Datos y análisis

Se utilizará PlantVillage, descrito por sus autores con aproximadamente 54 mil imágenes,
14 cultivos y 38 clases de cultivo/estado sanitario. El inventario efectivo dependerá
de la versión descargada y de los filtros aplicados. Se empleará únicamente RGB color.
La ficha oficial declara CC BY-SA 3.0; se conservarán procedencia y atribución.

La unidad de separación será la hoja, usando leaf-map.json del proveedor. Se agrupan
también duplicados exactos mediante SHA-256. Las imágenes sin correspondencia de hoja
se excluyen, reportando su cantidad. No se garantiza detectar duplicados visualmente
similares ni errores del metadato original. Se inspeccionarán desbalance, archivos
dañados y cobertura de clases antes del entrenamiento.

## 4. Propuesta de solución

Preparación -> manifiesto CSV -> CNN / MobileNetV2 -> selección en validación ->
evaluación de test -> interfaz Streamlit.

CNN: tres bloques convolución/ReLU/pooling, agregación espacial y capa clasificadora.
MobileNetV2: extractor ImageNet congelado y nuevo clasificador; un ajuste fino posterior
será un experimento independiente. Las dos redes reciben imágenes de 224 x 224 con
normalización de ImageNet para mantener un preprocesamiento comparable.
Los aumentos aleatorios se aplican únicamente al entrenamiento.

## 5. Diseño experimental y métricas

Separación aproximada de grupos por clase: 70% entrenamiento, 15% validación y 15% test,
semilla 42. Los porcentajes de imágenes pueden variar por el tamaño de cada grupo.
Cada clase debe tener al menos tres grupos independientes. Se conserva el manifiesto
y su SHA-256, y se verifican hashes de las imágenes antes de entrenar/evaluar.

Configuración inicial propuesta: AdamW, learning rate 0.001, batch 32 y 15 épocas.
La comparación principal será F1 macro. Se reportarán además accuracy, precision y
recall macro/ponderados, métricas por clase y matriz de confusión. Se selecciona la
mejor época solo con validación; test queda reservado para la evaluación posterior.
Se prevé repetir con semillas 42, 43 y 44 y analizar media y dispersión.

## 6. Herramientas y alcance

Python, PyTorch/Torchvision, Pillow, scikit-learn y Streamlit; Git/GitHub para código
y trazabilidad. Todo el procesamiento se realiza localmente. La IA generativa se
utilizó como apoyo para estructurar el código y la documentación; su registro y
evaluación se incluyen en uso_ia.md y deben ampliarse durante el proyecto.

## 7. Avance y resultados

Se implementan preparación de datos por hoja, dos arquitecturas, entrenamiento,
evaluación y aplicación local. Los resultados técnicos verificados se registran en
validacion.md. No se presentan métricas inventadas ni resultados del conjunto completo.
El piloto, si se ejecuta, se identifica por su subconjunto y presupuesto reducido.

## 8. Próximos pasos y conclusión provisional

Adjuntar la captura real del correo de aprobación del tema a Canvas; completar la
descarga e inventario; ejecutar la comparación completa; revisar errores y fotos
externas; preparar documento final, video y presentación de hasta 10 diapositivas.
La implementación permite comprobar la viabilidad del flujo. La calidad predictiva
y la generalización agrícola siguen siendo preguntas experimentales pendientes.

## 9. Bibliografía

1. Mohanty, S. P., Hughes, D. P. y Salathé, M. (2016). Using Deep Learning for
   Image-Based Plant Disease Detection. https://doi.org/10.3389/fpls.2016.01419
2. PlantVillage Dataset, repositorio original. https://github.com/spMohanty/PlantVillage-Dataset
3. Ficha del dataset. https://huggingface.co/datasets/mohanty/PlantVillage
4. Torchvision, MobileNetV2. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html

