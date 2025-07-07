
import streamlit as st
import pandas as pd
import joblib

# Título
st.title("🧠 Predicción de Complicaciones en Cirugía Colorrectal")

# Ingreso de datos del paciente
st.sidebar.header("📝 Datos del paciente")
edad = st.sidebar.number_input("Edad", 18, 100, 65)
sexo = st.sidebar.selectbox("Sexo", ["Masculino", "Femenino"])
asa = st.sidebar.selectbox("Clasificación ASA", [1, 2, 3, 4])
tipo_cirugia = st.sidebar.selectbox("Tipo de Cirugía", ["Electiva", "Urgencia"])
laparoscopia = st.sidebar.selectbox("Laparoscopia", ["Sí", "No"])
tiempo_qx = st.sidebar.number_input("Tiempo quirúrgico (min)", 30, 600, 180)
sangrado = st.sidebar.number_input("Sangrado estimado (ml)", 0, 5000, 300)
uci_postop = st.sidebar.selectbox("UCI postoperatoria", ["Sí", "No"])
estancia_preop = st.sidebar.number_input("Estancia preoperatoria (días)", 0, 30, 1)

# Codificación manual
sexo = 1 if sexo == "Masculino" else 0
tipo_cirugia = 1 if tipo_cirugia == "Electiva" else 0
laparoscopia = 1 if laparoscopia == "Sí" else 0
uci_postop = 1 if uci_postop == "Sí" else 0

# DataFrame con los datos
input_df = pd.DataFrame([{
    'Edad': edad,
    'Sexo': sexo,
    'ASA': asa,
    'Tipo_Cirugia': tipo_cirugia,
    'Laparoscopia': laparoscopia,
    'Tiempo_quirurgico': tiempo_qx,
    'Sangrado': sangrado,
    'UCI_postop': uci_postop,
    'Estancia_preop': estancia_preop
}])

# Cargar modelos
@st.cache_resource
def cargar_modelo(path):
    return joblib.load(path)

modelos = {
    "Fuga anastomótica": cargar_modelo("modelo_anastomotic.pkl"),
    "Sepsis": cargar_modelo("modelo_sepsis.pkl"),
    "Absceso intraabdominal": cargar_modelo("modelo_abscess.pkl"),
    "Infección urinaria": cargar_modelo("modelo_uti.pkl"),
    "Íleo postoperatorio": cargar_modelo("modelo_ileus.pkl")
}

# Botón para predecir
if st.button("Predecir complicaciones"):
    st.subheader("📊 Resultados de predicción:")
    for nombre, modelo in modelos.items():
        prob = modelo.predict_proba(input_df)[0][1]
        st.write(f"✅ Riesgo de {nombre}: {prob*100:.1f}%")
