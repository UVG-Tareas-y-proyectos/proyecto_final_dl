# Uso crítico de IA generativa

Registro inicial del 26 de octubre de 2026. Debe ampliarse con los experimentos y
correcciones reales del equipo para el reporte final.

## Solicitud relevante del equipo

«Detección automática de enfermedades en hojas de plantas mediante redes neuronales
convolucionales (CNN) [...] utilizar el conjunto de datos PlantVillage [...] comparando
su desempeño con un modelo preentrenado [...] precisión, recall, F1-score y matriz de
confusión». Se solicitó preparar avances y la entrega parcial siguiendo el PDF del curso.

## Aportes y evaluación

Codex ayudó a estructurar propuesta, módulos, comandos reproducibles, pruebas e interfaz.
Se contrastó el diseño con el repositorio original de PlantVillage y la documentación
de Torchvision. La propuesta inicial del equipo no especificaba cómo evitar fuga entre
fotos de la misma hoja: se incorporó agrupación con metadatos y hashes.

Un resultado satisfactorio debe poder ejecutarse y producir archivos consistentes;
la plausibilidad del texto generado no basta para validar el programa. El registro de
verificación y sus límites se encuentra en validacion.md. Los datos sintéticos usados
en pruebas verifican software y no miden reconocimiento de enfermedades.

## Aspectos pendientes de evaluación humana

Revisar ejemplos y etiquetas, interpretar las confusiones de los modelos, explicar el
efecto del desbalance y contrastar fotos externas. El equipo debe entender cada módulo,
verificar las fuentes y registrar respuestas erróneas o experimentos fallidos conforme
se produzcan; no se atribuyen revisiones ni conclusiones experimentales no realizadas.

