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
python -m pip install -e ".[dev,analysis]"
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

## Avance 1 de 3 · miércoles 28 de octubre

Se amplió el piloto a las **2,152 imágenes RGB de papa** de la misma revisión
de PlantVillage: 1,000 de tizón temprano, 1,000 de tizón tardío y 152 sanas.
Esto cubre tres clases; todavía no es el conjunto completo de 38 clases.
La descarga verifica cada imagen contra el objeto Git de la revisión fijada.

```powershell
python scripts/download_pilot.py --config configs/data_potato.json
python -m plant_disease.prepare --root data/raw/potato/color --leaf-map data/raw/potato/leaf-map.json --out data/processed/potato --seed 42
python -m plant_disease.audit --root data/raw/potato/color --manifest data/processed/potato/manifest.csv --summary data/processed/potato/summary.json --out runs/audit_potato --plots
```

Ejecutar cada comando después de terminar el anterior y usar salidas nuevas si
ya existen. El manifiesto definitivo se conserva en
`experiments/2026-10-28/manifest.csv`; úsalo también en los entrenamientos para
mantener las particiones. La auditoría rechaza imágenes nuevas sin contabilizar,
archivos modificados, hojas con etiquetas contradictorias y duplicados de imágenes.

| Partición | Imágenes | Hojas |
| --- | ---: | ---: |
| Entrenamiento | 1,524 | 372 |
| Validación | 312 | 78 |
| Prueba | 316 | 78 |

Las 528 hojas quedan separadas entre particiones. No se detectaron archivos dañados,
imágenes sin metadatos ni duplicados exactos. Todas las imágenes son RGB de 256 x 256.
El desbalance en entrenamiento es **6.07:1**. La clase sana solo dispone de cuatro
hojas en validación y cuatro en prueba: sus métricas tendrán alta variabilidad.
Estos controles no detectan todas las similitudes visuales ni errores del metadato.

Los resultados están en [la auditoría del miércoles](experiments/2026-10-28/audit/audit.json),
con [distribución por clase](experiments/2026-10-28/audit/class_distribution.png),
inventario, resumen de preparación y procedencia. Los pesos de clase sugeridos
se calcularon **solo con train**. La parte 2 documenta su aplicación en el
entrenamiento. El piloto de 117 imágenes permanece como experimento anterior.

| Parte | Fecha | Alcance |
| --- | --- | --- |
| 1 · Ihan | Miércoles 28 | Completado: descarga ampliada, manifiesto y auditoría de datos. |
| 2 · Diego | Viernes 30 | Completado: dos CNN con/sin pesos de clase y comparación de validación. MobileNetV2 pendiente por acceso a ImageNet. |
| 3 · Equipo | Viernes 30 | Pendiente: evaluar checkpoints seleccionados en test, revisar errores, integrar la interfaz y actualizar la entrega parcial. |

Comando de entrenamiento base sobre los datos ampliados:

```powershell
python -m plant_disease.train --root data/raw/potato/color --manifest experiments/2026-10-28/manifest.csv --model cnn --out runs/potato_cnn --epochs 15
```

La nueva partición contiene imágenes del piloto anterior. No comparar sus métricas
como si fueran el mismo test ni reutilizar `artifacts/pilot_cnn.pt` como modelo
final de este experimento. Las semillas y configuraciones se seleccionan en
validación; conservar test para la evaluación final.

## Avance 2 de 3 · Diego · viernes 30 de octubre

Se entrenaron dos CNN desde cero con el manifiesto fijado por Ihan: **1,524 imágenes
para entrenar y 312 para validar**, tres clases de papa. Ambas usan la misma
arquitectura de 23,779 parámetros, semilla 42, 15 épocas, batch 32, AdamW con
learning rate 0.001 y weight decay 0.01, y los mismos aumentos de entrenamiento.

Una variante utiliza entropía cruzada sin ponderación; la otra aplica pesos por
clase calculados exclusivamente con train: aproximadamente 0.7216 para cada
clase de enfermedad y 4.3793 para hojas sanas. La pérdida de validación siempre
se calcula sin ponderar para que las curvas sean comparables.

Resultados del mejor checkpoint de cada ejecución, seleccionado por F1 macro:

| CNN | Mejor época | Accuracy val. | F1 macro val. | Recall hojas sanas val. |
| --- | ---: | ---: | ---: | ---: |
| Sin ponderación | 14 | 0.9904 | 0.9932 | 1.0000 |
| Pesos por clase | 15 | 0.9840 | 0.9798 | 1.0000 |

El candidato seleccionado es **`cnn_unweighted`**. Su checkpoint está en
[artifacts/part2_cnn.pt](artifacts/part2_cnn.pt). Las métricas anteriores corresponden
a validación; **las 316 imágenes de test siguen reservadas para la parte 3**.
No son resultados de test ni se comparan directamente con el piloto de 117 imágenes.
La validación solo contiene cuatro hojas sanas independientes, por lo que el recall
de esa clase puede variar mucho. Se utilizó una sola semilla y aún falta repetirla.

Se conservan configuraciones, revisión del código, pesos por clase, historial por
época, predicciones de validación, hashes y constancias de finalización en
[experiments/2026-10-30](experiments/2026-10-30).
La [comparación](experiments/2026-10-30/comparison/comparison.json) y las
[curvas de aprendizaje](experiments/2026-10-30/comparison/learning_curves.png)
permiten revisar la selección. La herramienta rechaza ejecuciones incompletas,
artefactos modificados y comparaciones con particiones o presupuestos distintos.

Para repetir ambos entrenamientos y compararlos:

```powershell
python scripts/train_experiments.py --config configs/train_part2.json
python -m plant_disease.compare --runs runs/part2/cnn_unweighted runs/part2/cnn_balanced --out runs/part2/comparison --plots
```

Usar salidas nuevas si ya existen. Los checkpoints de ambas ejecuciones quedan
localmente en `runs/part2/`; el candidato seleccionado también se versiona en
`artifacts/part2_cnn.pt` para continuar sin volver a entrenar.

MobileNetV2 sigue pendiente porque la descarga oficial de ImageNet permanece
bloqueada por la red.

La parte 3 puede comenzar con la evaluación del candidato, sin retocar su selección
con el conjunto de test:

```powershell
python -m plant_disease.evaluate --root data/raw/potato/color --manifest experiments/2026-10-28/manifest.csv --checkpoint artifacts/part2_cnn.pt --out runs/part3_test
```

## Documentos de la entrega

- [Entrega parcial en PDF](docs/entrega_parcial.pdf).

El PDF incluye al final la captura proporcionada del mensaje del catedrático,
sin texto explicativo adicional.
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

