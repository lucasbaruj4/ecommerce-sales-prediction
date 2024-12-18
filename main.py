# Importa las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar el dataset
data = pd.read_csv("Dataset.csv")

# Exploración inicial
print("Primeras filas del dataset:")
print(data.head())
print("\nInformación del dataset:")
print(data.info())

# Limpieza de datos
# Verificar valores nulos
print("\nValores nulos en cada columna:")
print(data.isnull().sum())

# Imputar o eliminar valores nulos (suponiendo que son pocos para simplificar)
data = data.dropna()

# Crear nueva columna de ventas totales
data['total_sales'] = data['Units_Sold'] * data['Price']

# Convertir la fecha a datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['day_of_week'] = data['Date'].dt.dayofweek

# Codificar variables categóricas
categorical_cols = ['Product_Category', 'Customer_Segment']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Dividir en variables independientes y dependientes
X = data.drop(columns=['Date', 'total_sales'])
y = data['total_sales']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Entrenar y evaluar modelos
results = {}
for name, model in models.items():
    print(f"\nEntrenando {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R^2': r2}

    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R^2: {r2:.2f}")

# Comparar resultados
print("\nResultados finales:")
for name, metrics in results.items():
    print(f"{name}: {metrics}")

# Visualización de predicciones para el mejor modelo (supongamos XGBoost)
best_model = models['XGBoost']
predictions = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Ventas Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Ventas Reales - XGBoost')
plt.show()

# Importancia de las características
importances = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.title("Importancia de las Características - XGBoost")
plt.show()

# Comparación de métricas entre modelos
metrics_df = pd.DataFrame(results).T
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title("Comparación de Modelos")
plt.ylabel("Métricas")
plt.legend(loc='upper right')
plt.show()

# Análisis de elasticidad de precio
price_sales_corr = data['Price'].corr(data['total_sales'])
print(f"\nCorrelación entre Precio y Ventas Totales: {price_sales_corr:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Price', y='total_sales', data=data, alpha=0.5)
plt.title('Relación entre Precio y Ventas Totales')
plt.xlabel('Precio del Producto')
plt.ylabel('Ventas Totales')
plt.show()

# Predicción por segmentos usando columnas dummies
print("\nAnálisis por segmento usando columnas dummies:")
segment_columns = [col for col in data.columns if 'Customer_Segment' in col]
print(f"Columnas de segmentos encontradas: {segment_columns}")

for col in segment_columns:
    print(f"\nAnalizando el segmento representado por: {col}")
    segment_data = data[data[col] == 1]  # Filtra datos donde la columna dummy sea 1
    X_segment = segment_data.drop(columns=['Date', 'total_sales'])
    y_segment = segment_data['total_sales']

    X_train, X_test, y_train, y_test = train_test_split(X_segment, y_segment, test_size=0.2, random_state=42)
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    print(f"R² para el segmento ({col}): {r2:.2f}")

# Simulación de escenarios de marketing
marketing_increase = X_test.copy()
marketing_increase['Marketing_Spend'] *= 1.2
predictions_new = best_model.predict(marketing_increase)
improvement = predictions_new.mean() - y_test.mean()
print(f"\nIncremento promedio de ventas al aumentar Marketing un 20%: {improvement:.2f}")