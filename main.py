import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


# funcion para cargarv el modelo
def load_model(model_name, **kwargs):
    if model_name == 'Random Forest':
        return RandomForestClassifier(**kwargs)         
    elif model_name == 'Gradient Boosting':
        return GradientBoostingClassifier(**kwargs)
    elif model_name == 'Logistic Regression':
        return LogisticRegression(**kwargs)
    else:
        raise ValueError(f'Model not recognized: {model_name}')

def main():
    st.header("""
    Aplicación de predicción de flores de iris

    This app predicts the **Iris flower** type!
    """)

    st.sidebar.header('Parámetros de entrada del usuario')

    def user_input_features():
        sepal_length = st.sidebar.slider('Longitud del sépalo', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Ancho del sépalo', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Longitud del pétalo', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Ancho del pétalo', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()

    st.subheader('Parámetros de entrada del usuario')
    st.write(df)

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # Widget para elegirnuetro modelo 
    model_name = st.sidebar.selectbox('Seleccione el modelo', ['Random Forest', 'Gradient Boosting', 'Logistic Regression'])
    
    # Añadimos los  controles deslizantes para ajustar hiperparámetros
    if model_name == 'Random Forest':
        n_estimators = st.sidebar.slider('Número de estimadores (n_estimators)', 1, 100, 10)
        clf = load_model(model_name, n_estimators=n_estimators)
    elif model_name == 'Gradient Boosting':
        learning_rate = st.sidebar.slider('Tasa de aprendizaje (learning_rate)', 0.01, 1.0, 0.1)
        clf = load_model(model_name, learning_rate=learning_rate)
    elif model_name == 'Logistic Regression':
        C = st.sidebar.slider('Parámetro de regularización (C)', 0.01, 10.0, 1.0)
        clf = load_model(model_name, C=C)
    else:
        raise ValueError(f'Model not recognized: {model_name}')

    # Realizamos  validación cruzada
    cv_scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')

    st.subheader('Rendimiento del Modelo (Validación Cruzada)')
    st.write(f'Precisión media: {cv_scores.mean()}')
    st.write(f'Desviación estándar de la precisión: {cv_scores.std()}')

    st.subheader('Resultados de la Validación Cruzada')
    st.line_chart(pd.DataFrame(cv_scores, columns=['Precisión']))

    clf.fit(X, Y)

    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)

    st.subheader('Etiquetas de clase y sus números de índice correspondientes')
    st.write(iris.target_names)

    st.subheader('Predicción')    
    st.write(iris.target_names[prediction])

    st.subheader('Probabilidad de predicción')
    st.write(prediction_proba)

    # Cargar el conjunto de datos Iris
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target

    # Gráfico de dispersión (scatter plot)
    st.subheader('Scatter Plot del Conjunto de Datos de Iris')
    scatter_fig = sns.pairplot(iris_df, hue='target')
    scatter_fig.fig.savefig('scatter_plot.png')  # Guardar la figura
    st.pyplot(scatter_fig.fig)  # Mostrar la figura en Streamlit

    # Histograma de las características
    st.subheader('Distribución de las Características del Conjunto de Datos de Iris')
    for feature in iris.feature_names:
        plt.figure(figsize=(8, 5))
        hist_fig = sns.histplot(iris_df[feature], kde=True)
        plt.title(f'Distribución de {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frecuencia')
        plt.savefig(f'{feature}_hist.png')  # Guardar la figura
        st.image(f'{feature}_hist.png')  # Mostrar la figura en Streamlit
        plt.close()  # Cerrar la figura para liberar recursos
        
    Export_results = st.checkbox('Exportar resultados a CSV')
    if Export_results:
        results_df = pd.DataFrame({
            'Predicción':[iris.target_names[prediction]],
            'Probabilidad de predicción':[prediction_proba[0]]
        })
        download_csv = results_df.to_csv(index=False)
        st.markdown(download_csv, unsafe_allow_html=True)
        st.download_button(
            label="Descargar los resultados",
            data=download_csv,
            file_name="Resultados de la Predicción.csv",
            mime="text/csv")
        
        
if __name__ == '__main__':
    main()