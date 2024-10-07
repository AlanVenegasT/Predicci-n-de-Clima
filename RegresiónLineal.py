# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Generar más datos para hacer el modelo más robusto
# Añadimos más variables como temperatura máxima, mínima, humedad y velocidad del viento
data = {
    'day_of_year': np.arange(1, 366),  # Días del año
    'temperature_max': np.random.uniform(22, 35, 365),  # Temperatura máxima en grados Celsius
    'temperature_min': np.random.uniform(15, 22, 365),  # Temperatura mínima
    'humidity': np.random.uniform(30, 80, 365),  # Humedad en porcentaje
    'wind_speed': np.random.uniform(1, 10, 365),  # Velocidad del viento en m/s
    'temperature': np.random.uniform(20, 30, 365)  # Temperatura promedio (nuestra variable objetivo)
}

# Crear DataFrame
df = pd.DataFrame(data)

# Visualizar los primeros datos
print(df.head())

# Dividir los datos en características (X) y variable objetivo (y)
X = df[['day_of_year', 'temperature_max', 'temperature_min', 'humidity', 'wind_speed']]  # Nuevas características
y = df['temperature']  # Temperatura como objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Validación cruzada para verificar el desempeño del modelo
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Error medio en validación cruzada: {-np.mean(scores)}")

# Hacer predicciones con los datos de prueba
y_pred = model.predict(X_test)

# Calcular el error
error = mean_absolute_error(y_test, y_pred)
print(f"Error absoluto medio: {error}")

# Predecir la temperatura del día actual (366)
today_prediction = model.predict([[366, 28, 20, 50, 5]])  # Ejemplo con valores del día actual
print(f"Predicción de temperatura para el día de hoy: {today_prediction[0]}")

# Graficar los resultados
plt.scatter(X_test['day_of_year'], y_test, color='blue', label='Valores reales')
plt.scatter(X_test['day_of_year'], y_pred, color='red', label='Predicciones')
plt.xlabel('Día del año')
plt.ylabel('Temperatura')
plt.title('Predicciones vs Valores reales')
plt.legend()
plt.show()
