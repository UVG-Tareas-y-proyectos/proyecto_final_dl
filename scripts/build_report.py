"""Crear el reporte editable a partir del formato de laboratorio conservado.

Uso: python scripts/build_report.py --template formato.docx --out reporte.docx
La exportación a PDF se realiza desde Word o un conversor DOCX compatible.
"""
import argparse
import copy
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS = {'w': W, 'r': R}

def tag(name):
    return '{' + W + '}' + name

def put(parent, name, **attributes):
    element = E.SubElement(parent, tag(name))
    for key, value in attributes.items():
        element.set(tag(key), str(value))
    return element

def replace_text(paragraph, text):
    first = paragraph.find('w:r/w:rPr', NS)
    properties = copy.deepcopy(first) if first is not None else None
    for child in list(paragraph):
        if child.tag != tag('pPr'):
            paragraph.remove(child)
    run = put(paragraph, 'r')
    if properties is not None:
        run.append(properties)
    for i, line in enumerate(text.split('\n')):
        if i:
            put(run, 'br')
        node = put(run, 't')
        node.text = line
    return paragraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with ZipFile(args.template) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    xml = E.fromstring(parts['word/document.xml'])
    body = xml.find('w:body', NS)
    original = list(body)
    paragraphs = body.findall('w:p', NS)
    cover = [copy.deepcopy(node) for node in original[:7]]
    body_pattern = copy.deepcopy(paragraphs[13])
    heading_pattern = copy.deepcopy(paragraphs[11])
    table_pattern = copy.deepcopy(body.find('w:tbl', NS))
    section = copy.deepcopy(body.find('w:sectPr', NS))
    for node in list(body):
        body.remove(node)
    replace_text(cover[1], 'DEEP LEARNING - 10')
    replace_text(cover[2], 'LUIS ALBERTO SURIANO SARAVIA')
    replace_text(cover[4], 'Entrega parcial del proyecto final')
    replace_text(cover[5], 'Ihan Marroquin - 23108\nDiego Patzan - 23525\n\n')
    replace_text(cover[6], 'GUATEMALA, 26 de octubre del 2026')
    for node in cover:
        body.append(node)

    def paragraph(text, bold=False, heading=False):
        node = replace_text(copy.deepcopy(heading_pattern if heading else body_pattern), text)
        prop = node.find('w:pPr', NS)
        if prop is None:
            prop = E.Element(tag('pPr'))
            node.insert(0, prop)
        if heading:
            put(prop, 'keepNext')
            spacing = prop.find('w:spacing', NS)
            if spacing is not None:
                spacing.set(tag('before'), '240')
                spacing.set(tag('after'), '120')
        for run in node.findall('w:r', NS):
            rp = run.find('w:rPr', NS)
            if rp is None:
                rp = E.Element(tag('rPr'))
                run.insert(0, rp)
            if bold or heading:
                put(rp, 'b')
            put(rp, 'color', val='000000')
            put(rp, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman')
            put(rp, 'sz', val='24')
        body.append(node)
        return node

    def page():
        node = E.SubElement(body, tag('p'))
        put(put(node, 'r'), 'br', type='page')

    def heading(text):
        paragraph(text, heading=True)

    def table(rows, widths):
        node = copy.deepcopy(table_pattern)
        template_row = copy.deepcopy(node.findall('w:tr', NS)[0])
        for old in node.findall('w:tr', NS):
            node.remove(old)
        grid = node.find('w:tblGrid', NS)
        for old in list(grid):
            grid.remove(old)
        for width in widths:
            put(grid, 'gridCol', w=width)
        for index, values in enumerate(rows):
            row = copy.deepcopy(template_row)
            trpr = row.find('w:trPr', NS)
            for old in list(trpr):
                trpr.remove(old)
            put(trpr, 'cantSplit')
            if index == 0:
                put(trpr, 'tblHeader')
            cells = row.findall('w:tc', NS)
            while len(cells) > len(values):
                row.remove(cells.pop())
            for j, (cell, value) in enumerate(zip(cells, values)):
                for old in list(cell):
                    if old.tag != tag('tcPr'):
                        cell.remove(old)
                cp = cell.find('w:tcPr', NS)
                width = cp.find('w:tcW', NS)
                if width is None:
                    width = put(cp, 'tcW', type='dxa')
                width.set(tag('w'), str(widths[j]))
                put(cp, 'vAlign', val='center')
                p = put(cell, 'p')
                pp = put(p, 'pPr')
                put(pp, 'jc', val='left' if j == 0 else 'center')
                put(pp, 'spacing', after='60', before='60', line='240', lineRule='auto')
                r = put(p, 'r')
                rp = put(r, 'rPr')
                put(rp, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman')
                put(rp, 'sz', val='24')
                if index == 0:
                    put(rp, 'b')
                put(r, 't').text = str(value)
            node.append(row)
        body.append(node)

    page()
    heading('Informe parcial')
    paragraph('Detección automática de enfermedades en hojas de plantas mediante redes neuronales convolucionales', bold=True)
    paragraph('https://github.com/UVG-Tareas-y-proyectos/proyecto_final_dl')
    heading('Descripción del problema')
    paragraph('La identificación de enfermedades en hojas requiere experiencia y puede complicarse cuando distintas afecciones presentan síntomas parecidos. Proponemos una aplicación local que reciba una fotografía y muestre una clasificación sugerida, como una exploración del uso de Deep Learning en sanidad vegetal.')
    heading('Objetivos')
    paragraph('Nuestro objetivo es comparar una CNN entrenada desde cero con MobileNetV2 preentrenada en ImageNet, utilizando las mismas particiones de PlantVillage. Prepararemos datos reproducibles, mediremos accuracy, precision, recall y F1, analizaremos la matriz de confusión e integraremos el modelo en una interfaz de carga de imágenes.')
    heading('Análisis de los datos')
    paragraph('PlantVillage contiene aproximadamente 54 mil imágenes, 14 cultivos y 38 clases de cultivo y estado sanitario. Trabajaremos con la variante RGB color de la fuente original y conservaremos su atribución. La ficha oficial declara licencia CC BY-SA 3.0. El inventario efectivo dependerá de la revisión descargada y de los filtros aplicados.')
    paragraph('En el primer piloto seleccionamos 20 hojas por clase, hasta dos fotos por hoja, de papa con tizón temprano, tizón tardío y hojas sanas. Obtuvimos 117 imágenes de 60 hojas. No encontramos archivos dañados, duplicados exactos ni metadatos de hoja faltantes en esta selección.')
    table([['Partición', 'Imágenes', 'Hojas', 'Uso'], ['Entrenamiento',81,42,'Ajustar pesos'],['Validación',18,9,'Seleccionar época'],['Prueba',18,9,'Evaluación final']], [2700,1500,1300,3395])

    page()
    heading('Propuesta de solución')
    paragraph('El flujo implementado prepara un manifiesto de imágenes, entrena una de las dos redes, selecciona un checkpoint mediante validación y evalúa posteriormente el conjunto de prueba. La interfaz Streamlit permite cargar un JPG o PNG y muestra la clase predicha y las puntuaciones softmax.')
    heading('Construcción de entrenamiento validación y prueba')
    paragraph('Utilizamos leaf-map.json del proveedor para mantener todas las fotografías de una hoja en una sola partición. Agrupamos también los duplicados exactos por SHA-256 y excluimos las imágenes sin metadatos de hoja. El reparto aproximado por grupos de cada clase es 70/15/15 con semilla 42; los porcentajes de imágenes pueden variar. Cada clase requiere al menos tres hojas independientes.')
    paragraph('Guardamos las rutas, etiquetas, grupos y hashes en un CSV. Antes de entrenar o evaluar, verificamos la integridad de las imágenes y que no existan hojas ni duplicados exactos compartidos entre particiones. Esto no garantiza detectar imágenes visualmente similares o errores del metadato original.')
    heading('Preprocesamiento de imágenes')
    paragraph('Convertimos las imágenes a RGB y las redimensionamos a 224 x 224. Aplicamos normalización de ImageNet para ambos modelos. El flip horizontal y las rotaciones de hasta 15 grados se aplican solamente durante entrenamiento; validación, prueba e inferencia usan transformaciones deterministas.')
    heading('Arquitecturas y entrenamiento')
    paragraph('La CNN utiliza tres bloques de convolución 3 x 3, ReLU y max-pooling, con 16, 32 y 64 canales; termina con agregación espacial, dropout de 0.2 y una capa clasificadora. MobileNetV2 conserva el extractor preentrenado congelado y sustituye su clasificador por una capa con el número de clases del manifiesto.')
    paragraph('La configuración inicial utiliza AdamW, learning rate de 0.001 y batch de 32. Proponemos 15 épocas para la comparación completa y repeticiones con semillas 42, 43 y 44. La mejor época se selecciona por F1 macro de validación. Prueba no se utiliza para ajustar hiperparámetros.')
    heading('Herramientas aplicadas')
    paragraph('Python, PyTorch y Torchvision para las redes; Pillow para las imágenes; scikit-learn para las métricas; Streamlit para la interfaz y GitHub para conservar el código. El procesamiento de las imágenes y la inferencia se realizan localmente.')

    page()
    heading('Resultados del primer piloto')
    metrics = json.loads((ROOT / 'experiments/pilot/cnn/metrics.json').read_text(encoding='utf-8'))
    paragraph('Entrenamos la CNN durante cinco épocas con 81 imágenes. La mejor época por F1 macro de validación fue la tercera, con 0.5556. Evaluamos ese checkpoint en las 18 imágenes de prueba y conservamos sus predicciones individuales para revisar errores.')
    table([['Métrica','Resultado CNN'],['Accuracy',f"{metrics['accuracy']:.4f}"],['Precision macro',f"{metrics['report']['macro avg']['precision']:.4f}"],['Recall macro',f"{metrics['report']['macro avg']['recall']:.4f}"],['F1 macro',f"{metrics['report']['macro avg']['f1-score']:.4f}"],['Imágenes de prueba',18]], [5200,3695])
    heading('Matriz de confusión')
    paragraph('Las filas representan la clase real y las columnas la clase predicha.')
    table([['Real y predicha','Temprano','Tardío','Sana'], *[[label,*row] for label,row in zip(['Tizón temprano','Tizón tardío','Sana'],metrics['confusion_matrix'])]], [3100,1900,1900,1995])
    paragraph('La CNN acertó nueve de las 18 imágenes y no reconoció correctamente ninguna hoja sana. Esto indica que el entrenamiento del piloto todavía es insuficiente para separar las tres clases. Con solo seis imágenes por clase en prueba, el resultado verifica el flujo y constituye una línea base, pero no permite estimar el desempeño general.')
    heading('Verificación del programa')
    paragraph('Pasaron nueve pruebas automatizadas sobre particiones por hoja, integridad de imágenes, dimensiones de ambas redes, congelamiento del extractor, persistencia del checkpoint, conversión RGB e interfaz con y sin modelo. El entrenamiento y la evaluación de la CNN se ejecutaron con imágenes reales.')
    heading('Estado de MobileNetV2')
    paragraph('La arquitectura se verificó con pesos aleatorios en las pruebas de software. La descarga de pesos ImageNet fue bloqueada por el filtro de red del entorno. No se completó un entrenamiento con transferencia ni se reportan métricas de ese modelo; será necesario ejecutarlo en una red que permita acceder al servidor oficial de PyTorch.')

    body.append(section)
    parts['word/document.xml'] = E.tostring(xml, xml_declaration=True, encoding='UTF-8', standalone=True)
    relationships = E.fromstring(parts['word/_rels/document.xml.rels'])
    for relation in list(relationships):
        if relation.get('Type','').endswith('/image') and relation.get('Id') != 'rId6':
            parts.pop('word/' + relation.get('Target'), None)
            relationships.remove(relation)
        elif relation.get('Type','').endswith('/hyperlink'):
            relationships.remove(relation)
    parts['word/_rels/document.xml.rels'] = E.tostring(relationships,xml_declaration=True,encoding='UTF-8',standalone=True)
    if 'docProps/core.xml' in parts:
        core = E.fromstring(parts['docProps/core.xml'])
        for element in core:
            if E.QName(element).localname in ('creator','lastModifiedBy'):
                element.text = 'Ihan Marroquin y Diego Patzan'
            if E.QName(element).localname == 'title':
                element.text = 'Entrega parcial del proyecto final de Deep Learning'
        parts['docProps/core.xml'] = E.tostring(core,xml_declaration=True,encoding='UTF-8',standalone=True)
    output = args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    print(output)

if __name__ == '__main__':
    main()
