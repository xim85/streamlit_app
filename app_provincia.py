import pandas as pd
from sklearn.linear_model import LogisticRegression  # Modelo de clasificación
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st

# Cargar el DataFrame
df = pd.read_csv('df_pred.csv')

# Definir las características de entrada sin CCAA_Origen_Num y CCAA_Destino_Num
features = ['Año', 'Mes', 'Estancia_Media', 'TEMP_MIN_MEDIA', 'TEMP_MAX_MEDIA', 'TEMP_MEDIA', 
            'Provincia_Origen_Num', 'Gasto medio diario por persona']

X = df[features]  # Variables de entrada
y = df['Provincia_Destino_Num']  # Cambiado: Variable de salida es ahora Provincia_Destino_Num

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de clasificación
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model, 'model_provincia_destino.pkl')

# Cargar el modelo y encoders
model = joblib.load('model_provincia_destino.pkl')
label_encoder_provincia_origen = joblib.load('label_encoder_provincia_origen.pkl')
label_encoder_provincia_destino = joblib.load('label_encoder_provincia_destino.pkl')

# Función para predecir la provincia destino
def predecir_provincia_destino(año, mes, estancia_media, temp_media, gasto_medio_diario, provincia_origen):
    # Convertir las categorías a números usando los encoders
    provincia_origen_num = label_encoder_provincia_origen.transform([provincia_origen])[0]
    
    # Crear un DataFrame con los inputs del usuario
    input_data = pd.DataFrame({
        'Año': [año],
        'Mes': [mes],
        'Estancia_Media': [estancia_media],
        'TEMP_MIN_MEDIA': [10],  # Usa un valor default o estimado si no está disponible
        'TEMP_MAX_MEDIA': [30],  # Usa un valor default o estimado si no está disponible
        'TEMP_MEDIA': [temp_media],
        'Provincia_Origen_Num': [provincia_origen_num],
        'Gasto medio diario por persona': [gasto_medio_diario]
    })

    # Hacer la predicción con el modelo entrenado
    prediccion_num = model.predict(input_data)[0]
    
    # Convertir el resultado a la categoría original
    prediccion = label_encoder_provincia_destino.inverse_transform([prediccion_num])[0]
    
    return prediccion

# Interfaz de usuario con Streamlit
st.title("Predicción de la Provincia de Destino")

# Obtener los inputs del usuario
año = st.number_input('Año', min_value=2019, max_value=2026, value=2024)
mes = st.number_input('Mes', min_value=1, max_value=12, value=7)
estancia_media = st.number_input('Estancia Media', min_value=1, max_value=15, format="%d")
temp_media = st.number_input('Temperatura Media')
gasto_medio_diario = st.number_input('Gasto medio diario por persona', min_value=0.0, step=0.1)

# Obtener las categorías de los encoders para los selectboxes
provincia_origen_options = label_encoder_provincia_origen.classes_

# Crear selectbox para la categoría de Provincia Origen
provincia_origen = st.selectbox('Provincia Origen', options=provincia_origen_options)

# Hacer la predicción cuando el usuario haga clic en el botón
if st.button('Predecir'):
    resultado = predecir_provincia_destino(año, mes, estancia_media, temp_media, gasto_medio_diario, provincia_origen)
    st.write(f'La provincia de destino para viajar es: {resultado}')