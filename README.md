# Detección de enfermedades en hojas con Deep Learning

Proyecto del curso de Deep Learning, Universidad del Valle de Guatemala.
Integrantes: **Ihan-Marroquin** y **dpatzan2**.

Se compara una CNN entrenada desde cero con MobileNetV2 preentrenada en ImageNet.
El objetivo es clasificar combinaciones de cultivo/enfermedad de PlantVillage
y ofrecer una interfaz local para cargar imágenes. Estado: entrega parcial.

## Instalación

Python 3.10 o superior (verificado con 3.12). Desde la raíz del repositorio:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

En Linux/macOS: `source .venv/bin/activate`.
PyTorch puede ejecutarse con CPU; GPU es opcional.

## Flujo reproducible

1. Obtener la variante **raw/color** del [repositorio original de PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
   y su archivo `leaf-map.json`. Mantener la estructura `data/raw/color/<clase>/<imagen>`.
   No mezclar variantes color, grayscale y segmented.
2. Preparar el manifiesto con grupos por hoja; los archivos sin metadatos se excluyen:

```powershell
python -m plant_disease.prepare --root data/raw/color --leaf-map data/raw/leaf-map.json --out data/processed
python -m plant_disease.train --root data/raw/color --manifest data/processed/manifest.csv --model cnn --out runs/cnn --epochs 15
python -m plant_disease.train --root data/raw/color --manifest data/processed/manifest.csv --model mobilenet --out runs/mobilenet --epochs 15
```

La primera ejecución de MobileNetV2 descarga los pesos de ImageNet (~14 MB).
Se selecciona el checkpoint por F1 macro de validación, sin consultar test.

```powershell
python -m plant_disease.evaluate --root data/raw/color --manifest data/processed/manifest.csv --checkpoint runs/cnn/best.pt --out runs/cnn/test
python -m plant_disease.evaluate --root data/raw/color --manifest data/processed/manifest.csv --checkpoint runs/mobilenet/best.pt --out runs/mobilenet/test
python -m streamlit run app.py -- --checkpoint runs/mobilenet/best.pt
python -m pytest -q
```

Se generan configuración, historial por época, pesos, métricas por clase,
accuracy, precision/recall/F1 macro y ponderado, matriz de confusión y predicciones.
Los experimentos deben reutilizar exactamente el mismo manifiesto y semilla.

## Demostración disponible

Se incluye `artifacts/pilot_cnn.pt`, una CNN entrenada durante cinco épocas con
81 imágenes de tres clases de papa; validación y test contienen 18 imágenes cada uno.
No representa el modelo final ni cubre los demás cultivos.

```powershell
python -m streamlit run app.py -- --checkpoint artifacts/pilot_cnn.pt
```

Para repetir el piloto usando la misma revisión de los datos:

```powershell
python scripts/download_pilot.py --ref 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a
python -m plant_disease.prepare --root data/raw/pilot/color --leaf-map data/raw/pilot/leaf-map.json --out data/processed/pilot
python -m plant_disease.train --root data/raw/pilot/color --manifest data/processed/pilot/manifest.csv --model cnn --out runs/pilot_cnn --epochs 5
python -m plant_disease.evaluate --root data/raw/pilot/color --manifest data/processed/pilot/manifest.csv --checkpoint runs/pilot_cnn/best.pt --out runs/pilot_cnn/test
```

Usar carpetas nuevas si ya existen resultados. La descarga de pesos de MobileNetV2
quedó bloqueada por el filtro de red del entorno; su implementación y dimensiones
se probaron con pesos aleatorios, pero su entrenamiento con transferencia queda pendiente.
Los datos y resultados auditables del piloto se encuentran en `experiments/pilot/`.

## Documentos de la entrega

- [Entrega parcial en PDF](docs/entrega_parcial.pdf).

La entrega parcial en Canvas requiere además la captura del correo que aprueba
el **tema**. Debe adjuntarla el equipo; no se dispone de esa evidencia en este repositorio.
El video y la presentación de máximo 10 diapositivas corresponden a la entrega final.

El reporte usa el formato de laboratorio del equipo: portada UVG, Times New Roman,
texto justificado y tablas sencillas. La portada corresponde a Deep Learning,
sección 10, catedrático Luis Alberto Suriano Saravia. Para regenerar el Word:
`python scripts/build_report.py --template ruta/al/formato.docx --out reporte.docx`;
después exportarlo a PDF desde Word. La plantilla se proporciona externamente.

## Límites

PlantVillage utiliza imágenes mayormente controladas. La precisión en ese conjunto
no demuestra generalización a fotografías de campo. La interfaz muestra predicciones
experimentales; softmax no es una probabilidad calibrada de diagnóstico.
No se han completado aún los experimentos con el conjunto completo.

## Fuentes

- Mohanty, Hughes y Salathé (2016). [Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419).
- [Datos y agrupación por hoja originales](https://github.com/spMohanty/PlantVillage-Dataset).
- [Ficha oficial del dataset y licencia CC BY-SA 3.0](https://huggingface.co/datasets/mohanty/PlantVillage).
- [Documentación de MobileNetV2 en Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html).

