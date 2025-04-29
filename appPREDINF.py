import streamlit as st
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt
import warnings

# =====================
# Quitar warnings molestos
# =====================
warnings.filterwarnings("ignore", category=UserWarning)

# =====================
# Función para poner imagen de fondo
# =====================
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background('fondo.jpg')

# =====================
# Cargar modelos y normalizador
# =====================
modelos = {
    "Regresión Logística": joblib.load("logreg_MODEL.pkl"),
    "KNN": joblib.load("knn_MODEL.pkl"),
    "Naive Bayes": joblib.load("GausianNB_MODEL.pkl"),
    "SVM": joblib.load("svm_linear_MODEL1.pkl"),
    "Red Neuronal (MLP)": joblib.load("mejor_modelo_random_mlp.pkl"),
    "Árbol de Decisión": joblib.load("ArbolDes_MODEL.pkl"),
}

scaler = joblib.load("normalizador.pkl")

# =====================
# Interfaz de usuario
# =====================
st.title("🔎 Predicción de Infarto del Miocardio")
st.write("Ingrese los datos clínicos del paciente para ver los resultados de todos los modelos:")

st.subheader("📝 Datos del paciente")


# =====================
# Crear el DataFrame de entrada
# =====================

age = st.number_input("Edad en años completos", min_value=0.0, step=1.0, format="%.0f")
ckmb = st.number_input("CK-MB Nota: Los valores normales suelen ser menos de 5 ng/mL ", min_value=0.0, step=0.0001, format="%.4f")
troponin = st.number_input("Troponina Nota: Los valores normales son por debajo de 0.04 ng/ml", min_value=0.0, step=0.0001, format="%.4f")
input_data = pd.DataFrame([[age, ckmb, troponin]], columns=["Age", "CK-MB", "Troponin"])
input_scaled = scaler.transform(input_data)

# =====================
# Predicción con todos los modelos
# =====================
if st.button("Predecir con todos los modelos"):
    st.subheader("🔍 Resultados de cada modelo:")
    probabilidades = {}

    for nombre_modelo, modelo in modelos.items():
        # Para evitar el warning, igualamos los nombres si es necesario
        try:
            input_df_named = pd.DataFrame(input_scaled, columns=modelo.feature_names_in_)
        except AttributeError:
            input_df_named = input_scaled  # si no tiene el atributo, usamos directamente

        pred = modelo.predict(input_df_named)[0]

        if hasattr(modelo, "predict_proba"):
            prob = modelo.predict_proba(input_df_named)[0][1]
            probabilidades[nombre_modelo] = prob
            st.write(f"**{nombre_modelo}** - Probabilidad de infarto: **{prob:.2%}**")
        else:
            st.write(f"**{nombre_modelo}** - No disponible probabilidad.")

        if pred == 1:
            st.error(f"🔴 {nombre_modelo}: ¡Alta probabilidad de infarto!")
        else:
            st.success(f"🟢 {nombre_modelo}: Baja probabilidad de infarto.")

    # =====================
    # Gráfico de barras horizontal con probabilidades
    # =====================
    if probabilidades:
        st.subheader("📊 Comparación de probabilidades de infarto")

        fig, ax = plt.subplots(figsize=(8, 4))
        modelos_nombres = list(probabilidades.keys())
        valores_probs = list(probabilidades.values())

        bars = ax.barh(
            modelos_nombres,
            valores_probs,
            color="#1f77b4",
            height=0.35,
            edgecolor='white'
        )

        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilidad de Infarto", fontsize=12, color='black')
        ax.set_ylabel("Modelo", fontsize=12, color='black')
        ax.set_title("Comparación entre modelos", fontsize=14, weight='bold')

        for i, v in enumerate(valores_probs):
            ax.text(v + 0.01, i, f"{v:.1%}", color='black', va='center', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.tick_params(colors='black')

        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        st.pyplot(fig)