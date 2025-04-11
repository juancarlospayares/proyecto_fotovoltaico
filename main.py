import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="DeepSolar", page_icon=":sunny:", layout="wide")
st.title("Deep Solar Plus")
st.markdown("---")


st.image("paneles.png", width=300)  # Reemplaza con la URL de tu logo
# Cargar el archivo CSV
@st.cache_data
def load_data():
    df = pd.read_csv("predicciones_modelo_LSTM.csv")
    df['Fecha y hora'] = pd.to_datetime(df['Fecha y hora'])
    return df

# Filtrar datos según el período seleccionado
def filter_data(df, option):
    today = datetime.now().date()
    
    if option == "Hoy y mañana":
        start_date = today
        end_date = today + timedelta(days=1)
    elif option == "Esta semana":
        start_date = today
        end_date = today + timedelta(days=6 - today.weekday())
    else:  # Próximos 15 días
        start_date = today
        end_date = today + timedelta(days=14)
    
    mask = (df['Fecha y hora'].dt.date >= start_date) & (df['Fecha y hora'].dt.date <= end_date)
    filtered_df = df[mask]
    num_days = len(filtered_df['Fecha y hora'].dt.date.unique())
    
    return filtered_df, num_days

# Sidebar
st.sidebar.header("Opciones de Predicción")
prediction_options = ["Hoy y mañana", "Esta semana", "Próximos 15 días"]
selected_option = st.sidebar.radio("SELECCIONA EL TIEMPO DE PREDICCION:", prediction_options)

# Cargar datos
data = load_data()

# Página principal
st.subheader(f"Predicciones de Producción Fotovoltaica: {selected_option}")
filtered_data, num_days = filter_data(data, selected_option)

if not filtered_data.empty:
    # Número de días
    st.markdown(f"**Número de días en la predicción:** {num_days}")
    
    # Tabla
    st.write("Datos de producción fotovoltaica:")
    st.dataframe(
        filtered_data.rename(columns={'Producción Fotovoltaica Wh': 'Producción Fotovoltaica (Wh)'}),
        use_container_width=True
    )
    
    # Gráfica
    fig = px.line(
        filtered_data,
        x='Fecha y hora',
        y='Producción Fotovoltaica Wh',
        title=f"Producción Fotovoltaica - {selected_option}",
        labels={'Producción Fotovoltaica Wh': 'Producción Fotovoltaica (Wh)', 'Fecha y hora': 'Fecha y Hora'}
    )
    fig.update_layout(
        xaxis_title="Fecha y Hora",
        yaxis_title="Producción Fotovoltaica (Wh)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No hay datos disponibles para el período seleccionado.")

# Pie de página
st.markdown("---")
st.markdown("DeepSolar - Predicciones basadas en modelo LSTM")