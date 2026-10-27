"""Interfaz local; ejecutar con python -m streamlit run app.py -- --checkpoint ..."""
import argparse
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from plant_disease.predict import load_checkpoint, predict_image

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=Path("runs/mobilenet/best.pt"))
args, _ = parser.parse_known_args()

st.set_page_config(page_title="Hojas | Deep Learning", page_icon="🌿")
st.title("🌿 Clasificación de hojas")
st.write("Proyecto de Deep Learning · PlantVillage · UVG")
st.info("Prototipo académico. La clase sugerida depende de los datos del modelo y debe revisarse con criterio agronómico.")


@st.cache_resource
def cached_model(path, modified_ns):
    return load_checkpoint(path)


if not args.checkpoint.is_file():
    st.warning("Primero entrena un modelo y proporciona su checkpoint al iniciar la aplicación.")
    st.code("python -m streamlit run app.py -- --checkpoint runs/mobilenet/best.pt")
    st.stop()

try:
    model, checkpoint = cached_model(str(args.checkpoint.resolve()), args.checkpoint.stat().st_mtime_ns)
except (OSError, ValueError, KeyError, RuntimeError) as error:
    st.error(f"No se pudo cargar el checkpoint local: {error}")
    st.stop()

st.caption(f"Modelo: {checkpoint['model_name']} · {len(checkpoint['classes'])} clases · época seleccionada: {checkpoint.get('epoch', '?')}")
with st.expander("Clases que puede reconocer este modelo"):
    st.write(checkpoint["classes"])
uploaded = st.file_uploader("Carga una fotografía de una hoja", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    if uploaded.size > 10 * 1024 * 1024:
        st.error("Carga una imagen de hasta 10 MB.")
        st.stop()
    try:
        with Image.open(uploaded) as source:
            source.load()
            image = source.copy()
        st.image(image, caption="Imagen cargada", width=360)
        with st.spinner("Analizando imagen…"):
            predictions = predict_image(model, checkpoint["classes"], image)
        st.subheader(predictions[0]["class_name"].replace("___", " · ").replace("_", " "))
        st.metric("Puntuación softmax", f"{predictions[0]['score']:.1%}")
        st.caption("La puntuación no está calibrada como probabilidad de diagnóstico. El modelo siempre elige entre sus clases, incluso para imágenes ajenas a PlantVillage.")
        frame = pd.DataFrame(predictions[:5]).rename(columns={"class_name": "Clase", "score": "Puntuación"})
        st.dataframe(frame, hide_index=True, use_container_width=True)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        st.error("No se pudo procesar la imagen. Prueba con otro JPG o PNG válido.")

