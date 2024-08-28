import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st

# Cargar el DataFrame
df = pd.read_csv('df_pred.csv')

# Definir todas las columnas que quieres usar para entrenar el modelo
features = ['Año', 'Mes', 'Estancia_Media', 'TEMP_MIN_MEDIA', 'TEMP_MAX_MEDIA', 'TEMP_MEDIA', 
            'CCAA_Origen_Num', 'CCAA_Destino_Num', 'Provincia_Origen_Num', 'Provincia_Destino_Num']

X = df[features]  # Variables de entrada
y = df['Gasto medio diario por persona']  # Variable de salida

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model, 'model.pkl')

# Cargar el modelo y encoders
model = joblib.load('model.pkl')
label_encoder_provincia_origen = joblib.load('label_encoder_provincia_origen.pkl')
label_encoder_provincia_destino = joblib.load('label_encoder_provincia_destino.pkl')

# Función para predecir el gasto medio diario
def predecir_gasto(año, mes, estancia_media, temp_media, provincia_origen, provincia_destino):
    # Convertir las categorías a números usando los encoders
    provincia_origen_num = label_encoder_provincia_origen.transform([provincia_origen])[0]
    provincia_destino_num = label_encoder_provincia_destino.transform([provincia_destino])[0]

    # Asignar valores por defecto o estimados para CCAA_Origen_Num y CCAA_Destino_Num
    ccaa_origen_num = 0  # Puedes cambiar este valor por uno estimado o dejarlo como 0
    ccaa_destino_num = 0  # Puedes cambiar este valor por uno estimado o dejarlo como 0

    # Crear un DataFrame con los inputs del usuario
    input_data = pd.DataFrame({
        'Año': [año],
        'Mes': [mes],
        'Estancia_Media': [estancia_media],
        'TEMP_MIN_MEDIA': [0],  # Usa un valor default o estimado si no está disponible
        'TEMP_MAX_MEDIA': [0],  # Usa un valor default o estimado si no está disponible
        'TEMP_MEDIA': [temp_media],
        'CCAA_Origen_Num': [ccaa_origen_num],
        'CCAA_Destino_Num': [ccaa_destino_num],
        'Provincia_Origen_Num': [provincia_origen_num],
        'Provincia_Destino_Num': [provincia_destino_num]
    })

    # Hacer la predicción con el modelo entrenado
    prediccion = model.predict(input_data)
    return prediccion[0]

# Interfaz de usuario con Streamlit
st.title("Predicción del Gasto Medio Diario por Persona")

# Obtener los inputs del usuario
año = st.number_input('Año', min_value=2019, max_value=2028, value=2024)
mes = st.number_input('Mes', min_value=1, max_value=12, value=7)
estancia_media = st.number_input('Estancia Media', min_value=1, max_value=15, format="%d")  # Asegura que se ingrese un número entero
temp_media = st.number_input('Temperatura Media')

# Obtener las categorías de los encoders para los selectboxes
provincia_origen_options = label_encoder_provincia_origen.classes_
provincia_destino_options = label_encoder_provincia_destino.classes_

# Crear selectboxes para las categorías de provincia
provincia_origen = st.selectbox('Provincia Origen', options=provincia_origen_options)
provincia_destino = st.selectbox('Provincia Destino', options=provincia_destino_options)

# Hacer la predicción cuando el usuario haga clic en el botón
if st.button('Predecir'):
    resultado = predecir_gasto(año, mes, estancia_media, temp_media,
                               provincia_origen, provincia_destino)
    st.write(f'El gasto medio diario predicho es: {resultado:.2f}')