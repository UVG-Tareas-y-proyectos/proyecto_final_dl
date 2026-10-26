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

## Documentos de la entrega

- [Propuesta y metodología](docs/propuesta.md).
- [Entrega parcial en PDF](docs/entrega_parcial.pdf).
- [Registro del uso de IA](docs/uso_ia.md).
- [Validación del avance](docs/validacion.md).

La entrega parcial en Canvas requiere además la captura del correo que aprueba
el **tema**. Debe adjuntarla el equipo; no se dispone de esa evidencia en este repositorio.
El video y la presentación de máximo 10 diapositivas corresponden a la entrega final.

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

