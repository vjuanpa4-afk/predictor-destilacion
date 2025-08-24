
import joblib
import streamlit as st
import pandas as pd

# --- Configuración de la Página ---
# Esto debe ser lo primero que se ejecute en el script.
st.set_page_config(
    page_title="Predictor de Rendimiento de Destilación",
    page_icon="🧪",
    layout="wide"
)

# --- Carga del Modelo ---
# Usamos @st.cache_resource para que el modelo se cargue solo una vez y se mantenga en memoria,
# lo que hace que la aplicación sea mucho más rápida.
@st.cache_resource
def load_model(model_path):
    """Carga el modelo entrenado desde un archivo .joblib."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en {model_path}. Asegúrate de que el archivo del modelo esté en el directorio correcto.")
        return None

# Cargamos nuestro modelo campeón. Streamlit buscará en la ruta 'modelo_xgboost_final.joblib'.
model = load_model('model.joblib')

# --- Barra Lateral para las Entradas del Usuario ---
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    st.markdown("""
    Ajusta los deslizadores para que coincidan con los parámetros operativos de la columna de destilación.
    """)

    # Slider para el caudal de alimentación
    flowrate = st.slider(
        label='Caudal de Alimentación (m³/s)',
        min_value=100,
        max_value=500,
        value=300, # Valor inicial
        step=1
    )
    st.caption("Representa el volumen de la mezcla de alimentación que entra en la columna por segundo.")

    # Slider para la temperatura
    temperature = st.slider(
        label='Temperatura del Rehervidor (°C)',
        min_value=100,
        max_value=200,
        value=130,
        step=1
    )
    st.caption("La temperatura en el fondo de la columna. Es crucial para vaporizar los componentes.")

    # Slider para la diferencia de presión
    pressure = st.slider(
        label='Diferencia de Presión (psi)',
        min_value=-50,
        max_value=50,
        value=0,
        step=1
    )
    st.caption("La caída de presión a lo largo de la columna. Influye en los puntos de ebullición.")

# --- Contenido de la Página Principal ---
st.title("🧪 Predictor de Rendimiento de Columna de Destilación")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza un modelo de machine learning para predecir el rendimiento de un producto químico en una columna de destilación basándose en parámetros operativos clave.

**Esta herramienta puede ayudar a los ingenieros de procesos y operadores a:**
- **Optimizar** las condiciones de operación para obtener el máximo rendimiento.
- **Predecir** el impacto de los cambios en el proceso antes de implementarlos.
- **Solucionar** problemas potenciales simulando diferentes escenarios.
""")

# --- Lógica de Predicción ---
# Solo intentamos predecir si el modelo se ha cargado correctamente.
if model is not None:
    # El botón principal que el usuario presionará para obtener un resultado.
    if st.button('🚀 Predecir Rendimiento', type="primary"):
        # Creamos un DataFrame de pandas con las entradas del usuario.
        # ¡Es crucial que los nombres de las columnas coincidan exactamente con los que el modelo espera!
        df_input = pd.DataFrame({
            'PressureC1_diff': [pressure],
            'FlowC1': [flowrate],
            'Temp1': [temperature]
        })

        # Hacemos la predicción
        try:
            prediction_value = model.predict(df_input)
            st.subheader("📈 Resultado de la Predicción")
            # Mostramos el resultado en un cuadro de éxito, formateado a dos decimales.
            st.success(f"**Rendimiento Predicho:** `{prediction_value[0]:.2f}%`")
            st.info("Este valor representa el porcentaje estimado del producto deseado que se recuperará.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
else:
    st.warning("El modelo no pudo ser cargado. Por favor, verifica la ruta del archivo del modelo.")

st.divider()

# --- Sección de Explicación ---
with st.expander("ℹ️ Sobre la Aplicación"):
    st.markdown("""
    **¿Cómo funciona?**

    1.  **Datos de Entrada:** Proporcionas los parámetros operativos clave usando los deslizadores en la barra lateral.
    2.  **Predicción:** El modelo de machine learning pre-entrenado recibe estas entradas y las analiza basándose en los patrones que aprendió de datos históricos.
    3.  **Resultado:** La aplicación muestra el rendimiento final predicho como un porcentaje.

    **Detalles del Modelo:**

    * **Tipo de Modelo:** `Regression Model` (XGBoost Optimizado)
    * **Propósito:** Predecir el valor continuo del rendimiento de la destilación.
    * **Características Usadas:** Caudal de Alimentación, Temperatura del Rehervidor y Diferencia de Presión.
    """)
