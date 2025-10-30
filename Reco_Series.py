import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# === CARGA DE DATOS (Capa Batch) ===
# Usa la ruta completa a tu archivo CSV o asegúrate de que esté en el mismo directorio del script
data = pd.read_csv("ratings_series.csv")
# Título principal
st.title("🎬 Recomendador de Series (Arquitectura Lambda)")

# === PROCESAMIENTO BATCH ===
# Matriz usuario-serie
user_series_matrix = data.pivot_table(index='user', columns='series', values='rating').fillna(0)

# Similitud entre series
series_similarity = pd.DataFrame(
    cosine_similarity(user_series_matrix.T),
    index=user_series_matrix.columns,
    columns=user_series_matrix.columns
)

# === FUNCIÓN DE RECOMENDACIÓN (Capa de Servicio) ===
def recomendar_series(series_name, n_recomendaciones=5):
    if series_name not in series_similarity.columns:
        return ["Serie no encontrada en la base de datos"]
    similar_series = series_similarity[series_name].sort_values(ascending=False)[1:n_recomendaciones+1]
    return similar_series.index.tolist()

# === STREAMLIT INTERFAZ ===
st.subheader("Capa Batch + Capa de Velocidad + Capa de Servicio")

# Selección de serie
serie_input = st.selectbox("Selecciona una serie:", options=list(series_similarity.columns))

# Botón para recomendar
if st.button("Obtener Recomendaciones"):
    recomendaciones = recomendar_series(serie_input)
    st.success(f"✅ Recomendaciones basadas en '{serie_input}':")
    for r in recomendaciones:
        st.write(f"- {r}")

# === CAPA DE VELOCIDAD (Simulación de nuevas valoraciones) ===
st.subheader("🕒 Incorporar nueva valoración en tiempo real")
usuario_nuevo = st.text_input("ID de usuario nuevo:")
serie_nueva = st.selectbox("Serie a valorar:", options=list(series_similarity.columns))
nueva_valoracion = st.slider("Valoración (1 a 5)", 1, 5, 3)

if st.button("Agregar valoración"):
    nueva_fila = pd.DataFrame({
        'user': [usuario_nuevo],
        'series': [serie_nueva],
        'rating': [nueva_valoracion]
    })
    data = pd.concat([data, nueva_fila], ignore_index=True)
    st.success("⭐ Valoración añadida correctamente (simulada en tiempo real)")
    user_series_matrix = data.pivot_table(index='user', columns='series', values='rating').fillna(0)
    series_similarity = pd.DataFrame(
        cosine_similarity(user_series_matrix.T),
        index=user_series_matrix.columns,
        columns=user_series_matrix.columns
    )

st.info("💡 La nueva valoración se ha incorporado a los datos y actualizará las recomendaciones futuras.")
st.subheader("🔍 Recomendaciones Actualizadas")
if serie_input:
    recomendaciones_actualizadas = recomendar_series(serie_input)
    st.success(f"✅ Recomendaciones actualizadas basadas en '{serie_input}':")
    for r in recomendaciones_actualizadas:
        st.write(f"- {r}")

        


