
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar modelos y columnas
modelos = joblib.load("modelos_colorectal.pkl")
columnas = joblib.load("columnas_modelo_colorectal.pkl")

st.set_page_config(page_title="Predicción de Complicaciones y Estancia - Cirugía Colorectal", layout="wide")

st.title("🧠 Modelo de Predicción - Cirugía Colorectal")
st.markdown("Ingrese los datos del paciente para estimar el riesgo de complicaciones postoperatorias y estancia hospitalaria prolongada.")

# Crear formulario
with st.form("formulario_prediccion"):
    entradas = {}
    st.subheader("🩺 Variables Clínicas")

    columnas_pantalla = st.columns(2)
    for i, col in enumerate(columnas[:20]):  # Mostrar primeras 20 variables para interfaz sencilla
        with columnas_pantalla[i % 2]:
            valor = st.text_input(f"{col}", "")
            entradas[col] = valor

    submitted = st.form_submit_button("🔍 Calcular riesgo")

if submitted:
    st.subheader("📊 Resultados de predicción")

    # Crear dataframe de entrada
    entrada_df = pd.DataFrame([entradas])
    entrada_df = entrada_df.reindex(columns=columnas, fill_value=np.nan)

    # Preprocesamiento mínimo
    for c in entrada_df.columns:
        try:
            entrada_df[c] = pd.to_numeric(entrada_df[c])
        except:
            entrada_df[c] = entrada_df[c].astype(str)

    # Resultados
    for nombre, modelo in modelos.items():
        try:
            proba = modelo.predict_proba(entrada_df)[0, 1]
            st.write(f"**{nombre.replace('_',' ').title()}**: {proba*100:.2f}%")
        except Exception as e:
            st.warning(f"No se pudo calcular para {nombre}: {e}")
