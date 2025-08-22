import streamlit as st
import numpy as np
import tensorflow as tf
from keras.initializers import RandomUniform
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import backend
#from keras.utils.vis_utils import plot_model
import warnings
from keras.models import Sequential
import matplotlib.pyplot as plt
import graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from tensorflow.keras.metrics import Recall
from sklearn.metrics import recall_score, confusion_matrix
import os


st.set_page_config(layout="wide")

#st.markdown(
#    """
#    <style>
#    header {visibility: hidden;}
#    </style>
#    """,
#    unsafe_allow_html=True
#)

def detalles_capa(capa):
    #st.write("\n\n"+"="*20 + str(capa.name)+ "="*20)
    #st.write(capa.name, capa.kernel.shape)
    #st.write("\nInicializacion de los pesos")
    #st.write(capa.kernel_initializer.minval,capa.kernel_initializer.maxval)
    #st.write("\nDetalles de capa")
    #st.write(capa.kernel)
    #st.write("\nValores de pesos y sesgos")
    weights, biases = capa.get_weights()
    #st.write(weights)
    #st.write(biases)
    return weights, biases


def pasando_por_capa(entrada, weights, biases, final=False, use_bias=True):
    """
    Calcula la salida de una capa, con la opción de incluir o no el sesgo.
    """
    combi_sin_sesgo = np.matmul(entrada, weights)
    
    if use_bias:
        combinacion = combi_sin_sesgo + biases
    else:
        combinacion = combi_sin_sesgo

    if final:
        sal_capa = 1.0 / (1.0 + np.exp(-1 * combinacion))
    else:
        sal_capa = np.maximum(0, combinacion)
        
    return sal_capa, combinacion

warnings.filterwarnings('ignore')

#np.set_st.writeoptions(suppress=True)

backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

if 'modelo_redcita' not in st.session_state:
    st.session_state.modelo_redcita = Sequential([
        Dense(4, 
              kernel_initializer=RandomUniform(seed=42), 
              bias_initializer=RandomUniform(seed=42), 
              activation='relu', 
              name="Oculta_1", 
              input_dim=8),
        Dense(2, 
              kernel_initializer=RandomUniform(seed=42), 
              bias_initializer=RandomUniform(seed=42), 
              activation='relu', 
              name="Oculta_2"),
        Dense(1, 
              kernel_initializer=RandomUniform(seed=42), 
              bias_initializer=RandomUniform(seed=42), 
              activation='sigmoid', 
              name="Salida")
    ], name="Redcita")
    st.session_state.modelo_redcita.build()  # Construye las capas con input_shape

model = st.session_state.modelo_redcita

import pandas as pd

# Dataset Pima Indians Diabetes tiene 9 columnas:
# 8 features + 1 target (Outcome)
columnas = [
    "Embarazos",
    "Glucosa",
    "PresionSanguinea",
    "EspesorPiel",
    "Insulina",
    "IMC",
    "DiabetesPedigree",
    "Edad",
    "Diabetes"   # la etiqueta
]

dataset = pd.read_csv("pima-indians-diabetes.csv",delimiter=",",
    names=columnas,
    header=None)

st.write("Dataset")

col1, col2 = st.columns([2,1.5])

with col1:
    st.dataframe(dataset.head(13), hide_index=True, height=500)

# Separar features (X) y target (Y) en numpy float32
X = dataset.drop("Diabetes", axis=1).to_numpy(dtype="float32")
Y = dataset["Diabetes"].to_numpy(dtype="float32")

# Elegir índice de registro
with col2:
    idx = st.number_input(
        "Elige el índice del registro:", 
        min_value=0, max_value=len(dataset)-1, value=0, step=1
    )

    # Mostrar registro completo con nombres de columna
    registro = dataset.iloc[int(idx)]
    st.write("Registro elegido:")
    st.write(registro)

# Obtener solo X_sample e y_sample como float32
X_sample = X[int(idx)]       # shape (8,)
y_sample = Y[int(idx)]       # scalar

#st.dataframe(dataset)

##########################################################################################################################################################################################################################################################################################


# Estado inicial
if "Contenido1" not in st.session_state:
    st.session_state.Contenido1 = False

# Función para alternar estado
def toggle_radiografia():
    st.session_state.Contenido1 = not st.session_state.Contenido1

# Layout del encabezado
col1, col2 = st.columns([1, 30])
with col1:
    st.button(
        "▼" if not st.session_state.Contenido1 else "▲",
        key="btn_rad",
        on_click=toggle_radiografia
    )
with col2:
    st.markdown("### **1. Radiografía del script deep.py**")

# Mostrar contenido si está expandido
if st.session_state.Contenido1:

    cols_btn = st.columns(4)
    if 'seccion_actual' not in st.session_state:
        st.session_state.seccion_actual = 'A'

    with cols_btn[0]:
        if st.button("A. Función de Capas", use_container_width=True):
            st.session_state.seccion_actual = 'A'
    with cols_btn[1]:
        if st.button("B. Distribución Pesos", use_container_width=True):
            st.session_state.seccion_actual = 'B'
    with cols_btn[2]:
        if st.button("C. Activación Final", use_container_width=True):
            st.session_state.seccion_actual = 'C'
    with cols_btn[3]:
        if st.button("D. Impacto Sesgo", use_container_width=True):
            st.session_state.seccion_actual = 'D'


    # --- Contenido según la sección ---
    if st.session_state.seccion_actual == 'A':
            st.markdown('''#### A) Describa qué estructura de datos devuelve cada función. Señale qué elementos de la salida están destinados exclusivamente a la inspección manual y no son utilizados en el flujo interno de la red.''',unsafe_allow_html=True)
            st.code('''           
            def detalles_capa(capa):
                print("="*20 + str(capa.name)+ "="*20)
                print(capa.name, capa.kernel.shape)
                print(Inicializacion de los pesos")
                print(capa.kernel_initializer.minval,capa.kernel_initializer.maxval)
                print(Detalles de capa")
                print(capa.kernel)
                print(Valores de pesos y sesgos")
                weights, biases = capa.get_weights()
                print(weights)
                print(biases)
                return weights, biases
            
            pesos_capa1, sesgos_capa1 = detalles_capa(model.layers[0])

            print(f" Tipo de 'pesos_capa1': {type(pesos_capa1)}")
            >>> Tipo de 'pesos_capa1': <class 'numpy.ndarray'>
            
            print(f" Tipo de 'sesgos_capa1': {type(sesgos_capa1)}")
            >>> Tipo de 'sesgos_capa1': <class 'numpy.ndarray'>
            
            print(f" Forma de los pesos retornados: {pesos_capa1.shape}")
            >>> Forma de los pesos retornados: (8, 4)
                    
            def pasando_por_capa(entrada, weights,biases, final = False):
                combi_sin_sesgo = np.matmul(entrada,weights)
                print("Entrada: ",entrada)
                print("Suma ponderada sin sesgo: ",combi_sin_sesgo)
                combinacion = combi_sin_sesgo+biases
                print("Suma ponderada con sesgo: ",combinacion)
                if final:
                sal_capa =1.0 / (1.0 + np.exp(-1*combinacion))
                else:
                sal_capa = np.maximum(0,combinacion)
                print("Aplicando Activacion: ",sal_capa)
                return sal_capa
                    
            salida_capa1 = pasando_por_capa(X_sample, pesos_capa1, sesgos_capa1)

            print(f" Tipo de 'salida_capa1': {type(salida_capa1)}")
            >>> Tipo de 'salida_capa1': <class 'numpy.ndarray'>
            
            print(f" Contenido de la salida retornada: {salida_capa1}")
            >>> Contenido de la salida retornada: [0.       0.       6.813616 0.      ]        
            
            
            ''', language='python')

    elif st.session_state.seccion_actual == 'B':
            st.markdown('''#### B) Analice cóomo se distribuyen los valores iniciales de los pesos y sesgos. ¿Qué patrones pueden observarse en términos de magnitud y dirección (positivos o negativos)? ¿Coinciden con el tipo de inicializador utilizado?''',unsafe_allow_html=True)

            # Extraemos los pesos y sesgos usando la función original
            pesos_capa1, sesgos_capa1 = detalles_capa(model.layers[0])

            import seaborn as sns
            import matplotlib.pyplot as plt

            sns.kdeplot(pesos_capa1)
            st.pyplot(plt.gcf())

            # Ahora, analizamos las variables retornadas
            st.write(f"\nAnálisis programático de los PESOS de la capa '{model.layers[0].name}':")
            st.write(f"  - Valor Mínimo:  {np.min(pesos_capa1):.6f}")
            st.write(f"  - Valor Máximo:  {np.max(pesos_capa1):.6f}")
            st.write(f"  - Valor Promedio:{np.mean(pesos_capa1):.6f}")

            st.write(f"\nAnálisis programático de los SESGOS de la capa '{model.layers[0].name}':")
            st.write(f"  - Valor Mínimo:  {np.min(sesgos_capa1):.6f}")
            st.write(f"  - Valor Máximo:  {np.max(sesgos_capa1):.6f}")
            st.write(f"  - Valor Promedio:{np.mean(sesgos_capa1):.6f}")

            st.write("\nConclusión: Todos los valores están dentro del rango esperado [-0.05, 0.05] de RandomUniform.")

    elif st.session_state.seccion_actual == 'C':
            st.markdown('#### C) Explique la diferencia funcional al usar `final=True` en `pasando_por_capa`.')
            st.markdown("""
            El parámetro `final` actúa como un interruptor para la función de activación:

            - **`final=False` (Predeterminado):** Se aplica la función de activación **ReLU** (`np.maximum(0, combinacion)`). Esta función es ideal para las capas ocultas porque ayuda a mitigar el problema del desvanecimiento del gradiente y es computacionalmente eficiente. Devuelve `0` para entradas negativas y la propia entrada para valores positivos.

            - **`final=True`:** Se aplica la función de activación **Sigmoide** (`1 / (1 + np.exp(-x))`). Esta función comprime cualquier valor de entrada a un rango entre 0 y 1. Es perfecta para la capa de salida en problemas de clasificación binaria, ya que el resultado puede interpretarse directamente como una probabilidad.
            """)

            # --- Visualización Interactiva de la Capa de Salida ---
            st.subheader("Visualización Interactiva de la Capa de Salida")

            # Realizar el forward pass para obtener la entrada a la última capa
            pesos1, sesgos1 = model.layers[0].get_weights()
            salida1, _ = pasando_por_capa(X_sample, pesos1, sesgos1)
            
            pesos2, sesgos2 = model.layers[1].get_weights()
            entrada_final, _ = pasando_por_capa(salida1, pesos2, sesgos2)

            pesos_salida, sesgos_salida = model.layers[2].get_weights()

            # Función para crear el grafo
            def crear_grafo(entrada, pesos, sesgo, final=False):
                dot = graphviz.Digraph(comment='Capa de Salida')
                dot.attr(rankdir='LR', splines='line')

                # Nodos de entrada
                with dot.subgraph(name='cluster_0') as c:
                    c.attr(style='filled', color='lightgrey')
                    c.node_attr.update(style='filled', color='white')
                    for i, val in enumerate(entrada):
                        c.node(f'in_{i}', f'{val:.3f}')
                    c.attr(label='Entrada (desde Oculta_2)')

                # Nodo de suma
                with dot.subgraph(name='cluster_1') as c:
                    c.attr(color='blue')
                    c.node_attr.update(style='filled')
                    _, combinacion = pasando_por_capa(entrada, pesos, sesgo, final)
                    c.node('sum', f'Suma Ponderada\n+ Sesgo\n= {combinacion[0]:.3f}', shape='box')
                    c.attr(label='Combinación Lineal')

                # Nodo de activación y salida
                salida, _ = pasando_por_capa(entrada, pesos, sesgo, final)
                if final:
                    activation_name = 'Sigmoide'
                    act_color = 'darkorange'
                else:
                    activation_name = 'ReLU'
                    act_color = 'olivedrab'
                    
                dot.node('act', f'Activación\n{activation_name}', shape='box', style='filled', color=act_color)
                dot.node('out', f'Salida Final\n{salida[0]:.3f}', shape='doublecircle', style='filled', color='lightblue')
                
                # Conexiones
                for i in range(len(entrada)):
                    dot.edge(f'in_{i}', 'sum', label=f'x {pesos[i][0]:.5f}')
                dot.edge('sum', 'act')
                dot.edge('act', 'out')
                
                return dot

            # Estado para los botones
            if 'vista_final' not in st.session_state:
                st.session_state.vista_final = 'relu'

            # Botones
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button('Ver como Capa Oculta (final=False)', use_container_width=True):
                    st.session_state.vista_final = 'relu'
            with col_btn2:
                if st.button('Ver como Capa de Salida (final=True)', use_container_width=True):
                    st.session_state.vista_final = 'sigmoid'

            # Mostrar el grafo correspondiente
            
            _, col, _ = st.columns([1, 3, 1])

            with col:
                if st.session_state.vista_final == 'relu':
                    st.write("#### Simulación con Activación ReLU (`final=False`)")
                    st.graphviz_chart(crear_grafo(entrada_final, pesos_salida, sesgos_salida, final=False))
                else:
                    st.write("#### Simulación con Activación Sigmoide (`final=True`)")
                    st.graphviz_chart(crear_grafo(entrada_final, pesos_salida, sesgos_salida, final=True))

    elif st.session_state.seccion_actual == 'D':

        # --- Sección D: Análisis del Sesgo ---
        st.markdown("### **D. Análisis del Impacto del Sesgo (Bias)**")
        st.markdown("""
        Comente temporalmente la línea que incorpora los sesgos dentro de `pasando_por_capa`. Observe y documente los efectos sobre la salida de cada neurona. Determine si algunas neuronas permanecen inactivas o con valores constantes, y analice las posibles causas.

        **Análisis:** El **sesgo** (bias) añade un grado de libertad a cada neurona, permitiéndole ajustar su salida independientemente de sus entradas ponderadas. Es como el punto de corte en una regresión lineal (`y = mx + b`, donde `b` es el sesgo). Sin él, la función de activación de la neurona siempre pasaría por el origen, limitando severamente lo que la red puede aprender.

        Usa los botones a continuación para visualizar el flujo de datos entre las capas. Observa cómo cambian los valores de salida y el estado de activación (color de la neurona) cuando se elimina el sesgo.
        """)

        # --- Cálculos Forward Pass (ambos escenarios) ---
        p1, b1 = model.layers[0].get_weights()
        p2, b2 = model.layers[1].get_weights()
        p3, b3 = model.layers[2].get_weights()

        # CON Sesgo
        s1_con, c1_con = pasando_por_capa(X_sample, p1, b1, use_bias=True)
        s2_con, c2_con = pasando_por_capa(s1_con, p2, b2, use_bias=True)
        s3_con, c3_con = pasando_por_capa(s2_con, p3, b3, final=True, use_bias=True)

        # SIN Sesgo
        s1_sin, c1_sin = pasando_por_capa(X_sample, p1, b1, use_bias=False)
        s2_sin, c2_sin = pasando_por_capa(s1_sin, p2, b2, use_bias=False)
        s3_sin, c3_sin = pasando_por_capa(s2_sin, p3, b3, final=True, use_bias=False)

        # --- Lógica de la Interfaz Gráfica ---
        if 'vista_capa' not in st.session_state:
            st.session_state.vista_capa = 'capa_1'

        cols_btn = st.columns(3)
        with cols_btn[0]:
            if st.button("Capa 1 (Entrada → Oculta 1)", use_container_width=True):
                st.session_state.vista_capa = 'capa_1'
        with cols_btn[1]:
            if st.button("Capa 2 (Oculta 1 → Oculta 2)", use_container_width=True):
                st.session_state.vista_capa = 'capa_2'
        with cols_btn[2]:
            if st.button("Capa 3 (Oculta 2 → Salida)", use_container_width=True):
                st.session_state.vista_capa = 'capa_3'

        def crear_grafo_capa(dot, titulo, input_labels, input_values, output_values, pre_activation_values, weights, biases=None, es_final=False):
            """Función genérica para dibujar una transición entre capas sin bordes de cluster ni labels de pesos."""
            # Subgrafo para el grafo completo
            with dot.subgraph(name=f'cluster_{titulo.replace(" ", "_")}') as c:
                c.attr(label=titulo, fontsize='20', fontcolor='black', color='transparent')  # <-- sin borde
                
                # Nodos de entrada
                with c.subgraph(name=f'cluster_input_{titulo.replace(" ", "_")}') as in_c:
                    in_c.attr(label='Capa Anterior', style='filled', color='lightgrey', penwidth='0', fontcolor='black')
                    for i, (label, val) in enumerate(zip(input_labels, input_values)):
                        in_c.node(f'in_{i}_{titulo}', f'{label}\nValor: {val:.3f}', style='filled', color='white', penwidth='0')

                # Nodos de salida
                with c.subgraph(name=f'cluster_output_{titulo.replace(" ", "_")}') as out_c:
                    out_c.attr(label='Capa Actual', style='filled', color='lightgrey', penwidth='0', fontcolor='black')
                    for i, (out_val, pre_val) in enumerate(zip(output_values, pre_activation_values)):
                        # Determinar color por activación
                        color = 'darkorange' if es_final else ('lightblue' if out_val > 0 else 'gray88')
                        
                        out_c.node(f'out_{i}_{titulo}', f'Neurona {i}\nPre-Act: {pre_val:.3f}\nSalida: {out_val:.3f}', 
                                style='filled', color=color, shape='ellipse', penwidth='0')

                # Edges (conexiones con pesos) sin label
                for i in range(len(output_values)):
                    for j in range(len(input_values)):
                        dot.edge(f'in_{j}_{titulo}', f'out_{i}_{titulo}')

                # Nodo y Edges de Sesgo (si aplica)
                if biases is not None:
                    c.node(f'bias_{titulo}', 'Bias\n(1.0)', shape='box', style='filled', color='khaki', penwidth='0')
                    for i in range(len(output_values)):
                        dot.edge(f'bias_{titulo}', f'out_{i}_{titulo}', style='dashed')

        # --- Renderizado de Grafos ---
        _, col_con, _, col_sin, _ = st.columns([1.5,3,1.5,3,1.5])

        with col_con:
            dot_con = graphviz.Digraph()
            dot_con.attr(rankdir='LR')

        with col_sin:
            dot_sin = graphviz.Digraph()
            dot_sin.attr(rankdir='LR')

        # Lógica para seleccionar qué capa mostrar
        if st.session_state.vista_capa == 'capa_1':
            input_labels = dataset.columns[:-1]
            crear_grafo_capa(dot_con, "CON Sesgo", input_labels, X_sample, s1_con, c1_con, p1, b1)
            crear_grafo_capa(dot_sin, "SIN Sesgo", input_labels, X_sample, s1_sin, c1_sin, p1)

        elif st.session_state.vista_capa == 'capa_2':
            input_labels = [f"N_Oculta1_{i}" for i in range(4)]
            crear_grafo_capa(dot_con, "CON Sesgo", input_labels, s1_con, s2_con, c2_con, p2, b2)
            crear_grafo_capa(dot_sin, "SIN Sesgo", input_labels, s1_sin, s2_sin, c2_sin, p2)

        elif st.session_state.vista_capa == 'capa_3':
            input_labels = [f"N_Oculta2_{i}" for i in range(2)]
            crear_grafo_capa(dot_con, "CON Sesgo", input_labels, s2_con, s3_con, c3_con, p3, b3, es_final=True)
            crear_grafo_capa(dot_sin, "SIN Sesgo", input_labels, s2_sin, s3_sin, c3_sin, p3, es_final=True)

        with col_con:
            st.graphviz_chart(dot_con)

        with col_sin:
            st.graphviz_chart(dot_sin)

        # --- Comparación Final ---
        st.subheader("Comparación de la Salida Final")
        final_col1, final_col2 = st.columns(2)
        with final_col1:
            st.metric(label="Predicción Final CON Sesgo", value=f"{s3_con[0]:.6f}")
        with final_col2:
            st.metric(label="Predicción Final SIN Sesgo", value=f"{s3_sin[0]:.6f}")

        st.write("**Conclusión:** El sesgo actúa como un 'ajuste fino' en cada capa. Sin él, la red pierde flexibilidad y su capacidad de aprendizaje se ve severamente limitada, lo que resulta en una predicción final diferente y, generalmente, peor.")

########################################################################################################################################################################################################################################################################################

# Estado inicial
if "Contenido2" not in st.session_state:
    st.session_state.Contenido2 = False

# Función para alternar estado
def toggle_radiografia():
    st.session_state.Contenido2 = not st.session_state.Contenido2

# Layout del encabezado
col1, col2 = st.columns([1, 30])
with col1:
    st.button(
        "▼" if not st.session_state.Contenido2 else "▲",
        key="btn_rad2",
        on_click=toggle_radiografia
    )
with col2:
    st.markdown("### **2. ¿Qué neuronas se activan más?**")

_, col, _ = st.columns([1.5, 3, 1.5])

# Mostrar contenido si está expandido
with col:
    if st.session_state.Contenido2:
        from graphviz import Digraph

        def crear_grafo_red_ordenado(model, X_sample):
            """
            Grafo compacto con nodos de entrada renombrados y neuronas en orden ascendente.
            """
            dot = Digraph(format="png")
            dot.attr(rankdir="LR", nodesep="0.1", ranksep="0.15")

            # Entradas
            input_values = X_sample.flatten()
            prev_nodes = [f"in_{i}" for i in range(len(input_values))]

            for i, (label, val) in enumerate(zip(columnas[:-1], input_values)):
                dot.node(prev_nodes[i], f"{label}\n{val:.3f}", shape="box", style="filled", color="lightgrey")

            # Capas
            for idx_capa, layer in enumerate(model.layers):
                pesos, sesgos = layer.get_weights()
                n_inputs, n_outputs = pesos.shape

                # Pre-activación y activación
                preactivacion = np.dot(input_values, pesos) + sesgos
                salida = layer.activation(preactivacion).numpy()

                # Crear nodos de salida en orden ascendente
                curr_nodes = [f"h{idx_capa+1}_{i}" for i in range(n_outputs)]
                for i in range(n_outputs):
                    color = "darkorange" if idx_capa == len(model.layers)-1 else ("lightblue" if salida[i] > 0 else "gray88")
                    dot.node(curr_nodes[i],
                            f"Neuron {i}\nPre: {preactivacion[i]:.3f}\nAct: {salida[i]:.3f}",
                            style="filled",
                            color=color,
                            shape="ellipse")

                # Conexiones simples
                for i in range(n_outputs):
                    for j in range(n_inputs):
                        dot.edge(prev_nodes[j], curr_nodes[i])

                # Bias
                bias_node = f"bias_{idx_capa+1}"
                dot.node(bias_node, "Bias\n(1.0)", shape="box", style="filled", color="khaki")
                for i in range(n_outputs):
                    dot.edge(bias_node, curr_nodes[i], style="dashed")

                # Preparar siguiente capa
                input_values = salida
                prev_nodes = curr_nodes

            return dot

        # --- Uso ---
        X_sample = X[int(idx)]
        dot = crear_grafo_red_ordenado(model, X_sample)
        st.graphviz_chart(dot, use_container_width=True)


########################################################################################################################################################################################################################################################################################

import matplotlib.pyplot as plt
import seaborn as sns

# Estado inicial
if "Contenido3" not in st.session_state:
    st.session_state.Contenido3 = False

# Función para alternar estado
def toggle_contenido3():
    st.session_state.Contenido3 = not st.session_state.Contenido3

# Layout del encabezado
col1_c3, col2_c3 = st.columns([1, 30])
with col1_c3:
    st.button("▼" if not st.session_state.Contenido3 else "▲", on_click=toggle_contenido3, key="btn_c3")
with col2_c3:
    st.markdown("### **3. Análisis Detallado de Pesos y Activaciones (Primera Capa)**")

if st.session_state.Contenido3:
    st.markdown("Obtenga la matriz de pesos correspondiente a la primera capa del modelo. A partir de esta información, realice los siguientes análisis:")

    # --- Preparación de datos ---
    pesos_capa1, sesgos_capa1 = model.layers[0].get_weights()
    nombres_variables = dataset.columns[:-1].tolist()
    
    # --- Navegación ---
    if 'vista_seccion3' not in st.session_state:
        st.session_state.vista_seccion3 = 'a'

    cols_btn_c3 = st.columns(4)
    with cols_btn_c3[0]:
        if st.button("a) Mapa de Pesos", use_container_width=True): st.session_state.vista_seccion3 = 'a'
    with cols_btn_c3[1]:
        if st.button("b) Influencia Alta", use_container_width=True): st.session_state.vista_seccion3 = 'b'
    with cols_btn_c3[2]:
        if st.button("c) Influencia Baja", use_container_width=True): st.session_state.vista_seccion3 = 'c'
    with cols_btn_c3[3]:
        if st.button("d) Comparar Registros", use_container_width=True): st.session_state.vista_seccion3 = 'd'

    # --- FUNCIÓN DE GRAFO REFACTORIZADA ---
    def crear_grafo_capa1(input_data, pesos, sesgos, titulo, highlight_neurona=None, mostrar_etiquetas_pesos=True, estilizar_conexiones=True, curved_lines=False, grosor_variable=True):
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR', label=titulo, fontsize='35', nodesep='0.02', ranksep='2', size='8,6!')
        dot.attr(splines='curved' if curved_lines else 'line')

        pesos_abs = np.abs(pesos)
        max_peso_abs = np.max(pesos_abs) if np.max(pesos_abs) > 0 else 1

        with dot.subgraph(name='cluster_input') as c:
            c.attr(label='Variables de Entrada', style='filled', color='lightgrey')
            for i, nombre in enumerate(nombres_variables):
                c.node(f'in_{i}', f'{nombre}\n({input_data[i]:.2f})')

        with dot.subgraph(name='cluster_output') as c:
            c.attr(label='Capa Oculta 1', style='filled', color='lightgrey')
            salidas, _ = pasando_por_capa(input_data, pesos, sesgos)
            for i in range(4):
                color_neurona = 'lightblue' if salidas[i] > 0 else 'gray88'
                if highlight_neurona is not None and i == highlight_neurona:
                    color_neurona = 'yellow'
                c.node(f'out_{i}', f'Neurona {i}\nSalida: {salidas[i]:.3f}', style='filled', color=color_neurona)
        
        for i in range(4):
            if highlight_neurona is not None and i != highlight_neurona:
                continue
            for j in range(8):
                peso = pesos[j, i]
                
                if estilizar_conexiones:
                    penwidth = str(0.8 + 4 * (abs(peso) / max_peso_abs)) if grosor_variable else '1.5'
                    color = 'firebrick' if peso < 0 else 'forestgreen'
                else:
                    penwidth = '1.0'
                    color = 'gray50'
                
                etiqueta = f'{peso:.2f}' if mostrar_etiquetas_pesos else ''
                dot.edge(f'in_{j}', f'out_{i}', label=etiqueta, penwidth=penwidth, color=color, fontcolor=color, decorate='true', labelangle='-25', labeldistance='2.0')
        return dot

    # --- Lógica de visualización ---
    if st.session_state.vista_seccion3 == 'a':
        st.markdown("#### (a) Mapa de Pesos de la Primera Capa (Heatmap)")
        st.write("Un mapa de calor para visualizar la matriz de pesos. Colores más intensos significan mayor influencia.")
        
        # --- AJUSTE DE TAMAÑO ---
        # Se reduce figsize y el tamaño de las fuentes
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.heatmap(
            pesos_capa1,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=.2,
            ax=ax,
            yticklabels=nombres_variables,
            xticklabels=[f"N {i}" for i in range(4)],
            cbar_kws={'label': 'Valor del Peso'},
            annot_kws={"size": 5} # Letra más pequeña para los números
        )
        ax.set_title("Pesos: Entrada -> Oculta 1", fontsize=8)
        ax.set_xlabel("Neuronas Capa Oculta 1", fontsize=6)
        ax.set_ylabel("Variables de Entrada", fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=5) # Letra más pequeña para los ejes
        st.pyplot(fig)


    elif st.session_state.vista_seccion3 in ['b', 'c']:
        if st.session_state.vista_seccion3 == 'b':
            st.markdown("#### (b) Analice el efecto de la variable con mayor peso absoluto.")
        else:
            st.markdown("#### (c) Compare el efecto con una variable de peso absoluto bajo.")

        neurona_idx = st.radio("Seleccione una neurona para analizar:", [0, 1, 2, 3], horizontal=True, key="neurona_select")
        
        pesos_neurona = pesos_capa1[:, neurona_idx]
        pesos_abs_neurona = np.abs(pesos_neurona)
        
        idx_max_peso = np.argmax(pesos_abs_neurona)
        idx_min_peso = np.argmin(pesos_abs_neurona)

        var_idx = idx_max_peso if st.session_state.vista_seccion3 == 'b' else idx_min_peso
        tipo_analisis = "mayor" if st.session_state.vista_seccion3 == 'b' else "menor"

        st.write(f"Análisis para la **Neurona {neurona_idx}**. La variable con **{tipo_analisis}** peso absoluto es **{nombres_variables[var_idx]}** (Peso: {pesos_neurona[var_idx]:.3f}).")
        
        col_grafo, col_control = st.columns([2,1])
        
        with col_control:
            st.write("#### Modifique el valor:")
            valor_original = X_sample[var_idx]
            
            nuevo_valor = st.number_input(
                f"Nuevo valor para '{nombres_variables[var_idx]}'", 
                value=float(valor_original),
                step=1.0,
                format="%.2f",
                key=f"input_{var_idx}_{neurona_idx}"
            )

            X_modificado = X_sample.copy()
            X_modificado[var_idx] = nuevo_valor
            salidas_mod, _ = pasando_por_capa(X_modificado, pesos_capa1, sesgos_capa1)
            salida_neurona_mod = salidas_mod[neurona_idx]
            salidas_orig, _ = pasando_por_capa(X_sample, pesos_capa1, sesgos_capa1)
            salida_neurona_orig = salidas_orig[neurona_idx]

            st.metric(label=f"Activación Neurona {neurona_idx}", value=f"{salida_neurona_mod:.4f}", delta=f"{salida_neurona_mod - salida_neurona_orig:.4f}")
            
            if salida_neurona_mod > 0:
                st.info(f"**Interpretación:** La neurona está **activa**. Su sensibilidad a '{nombres_variables[var_idx]}' es {'alta' if tipo_analisis == 'mayor' else 'baja'}.")

        with col_grafo:
            st.graphviz_chart(crear_grafo_capa1(X_modificado, pesos_capa1, sesgos_capa1, f"Análisis de Neurona {neurona_idx}", highlight_neurona=neurona_idx), use_container_width=True)

    elif st.session_state.vista_seccion3 == 'd':
        st.markdown("#### (d) Compare las activaciones de dos registros distintos.")
        st.write("Observe cómo diferentes entradas generan distintos patrones de activación. El color de las flechas indica el signo del peso (influencia positiva o negativa).")
        
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            idx1 = st.number_input("Índice del Registro 1:", min_value=0, max_value=len(X)-1, value=0, step=1, key="idx1_d")
            st.dataframe(pd.DataFrame(X[idx1], index=nombres_variables, columns=['Valor']).T, height=60)
        with col_sel2:
            idx2 = st.number_input("Índice del Registro 2:", min_value=0, max_value=len(X)-1, value=7, step=1, key="idx2_d")
            st.dataframe(pd.DataFrame(X[idx2], index=nombres_variables, columns=['Valor']).T, height=60)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            # LLAMADA: Sin etiquetas, con estilo (solo color), sin grosor variable y con curvas
            st.graphviz_chart(crear_grafo_capa1(X[idx1], pesos_capa1, sesgos_capa1, f"Activaciones para Registro {idx1}", mostrar_etiquetas_pesos=False, estilizar_conexiones=True, curved_lines=True, grosor_variable=False), use_container_width=True)
        with col_g2:
            st.graphviz_chart(crear_grafo_capa1(X[idx2], pesos_capa1, sesgos_capa1, f"Activaciones para Registro {idx2}", mostrar_etiquetas_pesos=False, estilizar_conexiones=True, curved_lines=True, grosor_variable=False), use_container_width=True)
#####################################################################################################################################################################################################################################################################

# --- Estado y Lógica de Entrenamiento ---
if "pesos_antes_entrenamiento" not in st.session_state:
    st.session_state.pesos_antes_entrenamiento = model.get_weights()
    st.session_state.pesos_despues_entrenamiento = None
    st.session_state.entrenado = False

# Layout del encabezado
col1_c4, col2_c4 = st.columns([1, 30])
with col1_c4:
    if st.button("▼" if "Contenido4" not in st.session_state or not st.session_state.Contenido4 else "▲", key="btn_c4"):
        st.session_state.Contenido4 = not st.session_state.get("Contenido4", False)
with col2_c4:
    st.markdown("### **4. Análisis de Pesos: Antes vs. Después del Entrenamiento**")

# Contenido de la sección
if st.session_state.get("Contenido4", False):

    # Botón para entrenar el modelo
    if not st.session_state.entrenado:
        st.warning("El modelo aún no ha sido entrenado. Los pesos 'antes' y 'después' serán idénticos.")
        if st.button("🚀 Entrenar Modelo (1 epoch)"):
            with st.spinner("Entrenando por una época..."):
                # Compilar el modelo
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                # Entrenar
                model.fit(X, Y, epochs=1, batch_size=32, verbose=0)
                # Guardar los nuevos pesos
                st.session_state.pesos_despues_entrenamiento = model.get_weights()
                st.session_state.entrenado = True
                st.success("¡Entrenamiento completado! Los pesos han sido actualizados.")
                st.rerun() # Forzar un refresco de la app para mostrar los resultados
    else:
        st.info("El modelo ya ha sido entrenado. Ahora puedes comparar los pesos.")

    # --- Interfaz de visualización ---
    if 'vista_pesos' not in st.session_state:
        st.session_state.vista_pesos = 'antes'

    cols_btn_c4 = st.columns(2)
    with cols_btn_c4[0]:
        if st.button("Ver Pesos ANTES de Entrenar", use_container_width=True):
            st.session_state.vista_pesos = 'antes'
    with cols_btn_c4[1]:
        if st.button("Ver Pesos DESPUÉS de Entrenar", use_container_width=True, disabled=not st.session_state.entrenado):
            st.session_state.vista_pesos = 'despues'

    # Selector de capa
    nombres_capas = [layer.name for layer in model.layers]
    capa_seleccionada = st.selectbox("Selecciona una capa para analizar:", nombres_capas, key="select_capa_c4")
    idx_capa_sel = nombres_capas.index(capa_seleccionada)

    # --- Preparación de datos para la capa seleccionada ---
    pesos_antes, sesgos_antes = st.session_state.pesos_antes_entrenamiento[idx_capa_sel*2], st.session_state.pesos_antes_entrenamiento[idx_capa_sel*2+1]
    
    if st.session_state.entrenado:
        pesos_despues, sesgos_despues = st.session_state.pesos_despues_entrenamiento[idx_capa_sel*2], st.session_state.pesos_despues_entrenamiento[idx_capa_sel*2+1]
    else:
        # Si no se ha entrenado, los pesos 'despues' son los mismos que 'antes'
        pesos_despues, sesgos_despues = pesos_antes, sesgos_antes

    # Seleccionar qué datos mostrar basado en el botón presionado
    pesos_a_mostrar = pesos_despues if st.session_state.vista_pesos == 'despues' else pesos_antes
    
    # --- Visualizaciones ---
    st.markdown(f"#### Visualización para la capa: **{capa_seleccionada}** ({st.session_state.vista_pesos.upper()})")
    
    col_graf, col_tabla = st.columns([1.5, 1])

    with col_graf:
        st.markdown("**a) Distribución de Magnitud de Pesos**")
        
        # Aplanar los pesos para el gráfico de barras
        pesos_flat = pesos_a_mostrar.flatten()
        
        # Crear etiquetas para cada peso
        labels = []
        if pesos_a_mostrar.ndim > 1: # Capas densas
            for i in range(pesos_a_mostrar.shape[1]): # Neurona de salida
                for j in range(pesos_a_mostrar.shape[0]): # Neurona de entrada
                    labels.append(f'N{i}_In{j}')
        else: # Capa de salida con una neurona
             for j in range(pesos_a_mostrar.shape[0]):
                    labels.append(f'N0_In{j}')

        df_pesos = pd.DataFrame({'valor': pesos_flat, 'label': labels})
        df_pesos['color'] = ['#3498db' if x > 0 else '#e74c3c' for x in df_pesos['valor']]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(df_pesos['label'], df_pesos['valor'], color=df_pesos['color'])
        ax.set_xlabel("Valor del Peso")
        ax.set_ylabel("Conexión (Neurona_Entrada)")
        ax.set_title(f"Pesos de la Capa '{capa_seleccionada}'")
        ax.axvline(0, color='grey', linewidth=0.8) # Línea en cero
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        st.pyplot(fig)

    with col_tabla:
        st.markdown("**b) Comparación Numérica Detallada**")

        # Crear un DataFrame para la comparación detallada
        diff_pesos = np.abs(pesos_despues - pesos_antes)
        
        # Aplanar todas las matrices
        data = {
            'Peso Antes': pesos_antes.flatten(),
            'Peso Después': pesos_despues.flatten(),
            'Diferencia Abs.': diff_pesos.flatten()
        }
        df_detalle = pd.DataFrame(data)
        df_detalle.index.name = "ID del Peso"
        
        st.dataframe(df_detalle, height=450)

    st.divider()

    # --- Análisis y Conclusiones ---
    st.markdown("### **Conclusiones del Entrenamiento**")
    
    if st.session_state.entrenado:
        # (b) Determinar la capa con cambios más significativos
        cambios_por_capa = []
        for i, layer in enumerate(model.layers):
            p_antes, b_antes = st.session_state.pesos_antes_entrenamiento[i*2], st.session_state.pesos_antes_entrenamiento[i*2+1]
            p_despues, b_despues = st.session_state.pesos_despues_entrenamiento[i*2], st.session_state.pesos_despues_entrenamiento[i*2+1]
            
            diff_pesos_norm = np.linalg.norm(p_despues - p_antes)
            diff_sesgos_norm = np.linalg.norm(b_despues - b_antes)
            cambios_por_capa.append({'Capa': layer.name, 'Cambio en Pesos (Norma L2)': diff_pesos_norm, 'Cambio en Sesgos (Norma L2)': diff_sesgos_norm})

        df_cambios = pd.DataFrame(cambios_por_capa)
        capa_mas_cambio = df_cambios.loc[df_cambios['Cambio en Pesos (Norma L2)'].idxmax()]

        st.markdown(f"**b) Capa con Mayor Cambio:** La capa **'{capa_mas_cambio['Capa']}'** fue la que presentó la modificación más significativa en sus pesos, con una diferencia (norma L2) de **{capa_mas_cambio['Cambio en Pesos (Norma L2)']:.6f}**.")
        st.table(df_cambios.set_index('Capa'))

        # (c) Identificar neuronas con pocos cambios
        min_diff = float('inf')
        info_min_diff = {}
        
        for i in range(len(model.layers)):
            p_antes = st.session_state.pesos_antes_entrenamiento[i*2]
            p_despues = st.session_state.pesos_despues_entrenamiento[i*2]
            diff = np.abs(p_despues - p_antes)
            
            if np.min(diff) < min_diff:
                min_diff = np.min(diff)
                coords = np.unravel_index(np.argmin(diff), diff.shape)
                info_min_diff = {
                    'capa': model.layers[i].name,
                    'neurona_in': coords[0],
                    'neurona_out': coords[1] if len(coords) > 1 else 0,
                    'cambio': min_diff
                }
        
        st.markdown(f"**c) La Neurona 'Perezosa':** El peso que menos cambió en toda la red fue el que conecta la neurona de entrada **#{info_min_diff['neurona_in']}** con la neurona de salida **#{info_min_diff['neurona_out']}** en la capa **'{info_min_diff['capa']}'**. Su valor cambió solo en **{info_min_diff['cambio']:.8f}**.")
        st.markdown("Un cambio tan pequeño puede ocurrir si esa conexión específica contribuyó muy poco al error general de la red durante la época de entrenamiento, o si la activación de la neurona de entrada correspondiente fue consistentemente baja o cero para los datos de entrenamiento, haciendo que su gradiente fuera casi nulo.")
    else:
        st.info("Entrena el modelo para generar las conclusiones.")
#########################################################################################################################################################################################
# --- Estado y Lógica de la Sección 5 ---
if "Contenido5" not in st.session_state:
    st.session_state.Contenido5 = False

def toggle_contenido5():
    st.session_state.Contenido5 = not st.session_state.Contenido5

# Layout del encabezado
col1_c5, col2_c5 = st.columns([1, 30])
with col1_c5:
    st.button("▼" if not st.session_state.Contenido5 else "▲", on_click=toggle_contenido5, key="btn_c5")
with col2_c5:
    st.markdown("### **5. Diseño y Comparación de un Modelo Mejorado**")

# --- Contenido Principal de la Sección 5 ---
if st.session_state.Contenido5:

    # --- Funciones de Ayuda (Definidas aquí para mantener el código organizado) ---
    def crear_modelo_mejorado():
        """Define la arquitectura del nuevo modelo mejorado."""
        modelo = Sequential([
            Dense(6, kernel_initializer='he_uniform', input_dim=8, activation='tanh', name="Oculta_1"),
            Dense(3, kernel_initializer='he_uniform', activation='relu', name="Oculta_2"),
            Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid', name="Salida")
            ], name="RedMejorada")
        modelo.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.Recall(name="recall")]  # type: ignore
        )
        return modelo

    @st.cache_data
    def entrenar_y_evaluar_modelos():
        """
        Esta función encapsula todo el proceso de entrenamiento.
        """
        # 1. Cargar y dividir los datos
        X_raw = dataset.drop("Diabetes", axis=1).to_numpy(dtype="float32")
        y_raw = dataset["Diabetes"].to_numpy(dtype="float32")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
        )

        # 2. Estandarizar los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # --- Escenario 1: Entrenamiento SIN estandarización ---
        st.write("Entrenando modelos con datos SIN estandarizar...")
        modelo_base_raw = st.session_state.modelo_redcita
        modelo_base_raw.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.Recall(name="recall")]  # type: ignore
        )
        hist_base_raw = modelo_base_raw.fit(
            X_train_raw, y_train,
            epochs=100, batch_size=32, verbose=0, validation_split=0.1
        )
        
        modelo_mejorado_raw = crear_modelo_mejorado()
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        hist_mejorado_raw = modelo_mejorado_raw.fit(
            X_train_raw, y_train,
            epochs=150, batch_size=32, verbose=0, validation_split=0.1,
            callbacks=[early_stopping]
        )

        # --- Escenario 2: Entrenamiento CON estandarización ---
        st.write("Entrenando modelos con datos CON estandarización...")
        modelo_base_scaled = tf.keras.models.clone_model(st.session_state.modelo_redcita)
        modelo_base_scaled.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.Recall(name="recall")]  # type: ignore
        )
        hist_base_scaled = modelo_base_scaled.fit(
            X_train_scaled, y_train,
            epochs=100, batch_size=32, verbose=0, validation_split=0.1
        )
        
        modelo_mejorado_scaled = crear_modelo_mejorado()
        hist_mejorado_scaled = modelo_mejorado_scaled.fit(
            X_train_scaled, y_train,
            epochs=150, batch_size=32, verbose=0, validation_split=0.1,
            callbacks=[early_stopping]
        )
        
        return {
            "sin_estandarizar": {
                "base": {"modelo": modelo_base_raw, "historial": hist_base_raw.history},
                "mejorado": {"modelo": modelo_mejorado_raw, "historial": hist_mejorado_raw.history},
                "X_test": X_test_raw, "y_test": y_test
            },
            "con_estandarizacion": {
                "base": {"modelo": modelo_base_scaled, "historial": hist_base_scaled.history},
                "mejorado": {"modelo": modelo_mejorado_scaled, "historial": hist_mejorado_scaled.history},
                "X_test": X_test_scaled, "y_test": y_test
            }
        }

    def visualizar_modelo(modelo, nombre_archivo):
        """Genera y muestra una imagen de la arquitectura del modelo."""
        try:
            if not os.path.exists("model_images"):
                os.makedirs("model_images")
            ruta_completa = os.path.join("model_images", nombre_archivo)
            plot_model(modelo, to_file=ruta_completa, show_shapes=True, show_layer_names=True, show_layer_activations=True)
            st.image(ruta_completa)
        except Exception as e:
            st.error(f"Error al generar la imagen del modelo: {e}")
            st.warning("Asegúrate de tener instaladas las librerías `pydot` y `graphviz`.")

    def plot_matriz_confusion(y_true, y_pred, titulo):
        """Grafica una matriz de confusión usando Seaborn."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        ax.set_title(titulo)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor Real')
        st.pyplot(fig)
    
    def plot_curvas_entrenamiento(hist, titulo):
        """Grafica las curvas de Loss y Recall del entrenamiento."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(titulo, fontsize=16)
        
        ax1.plot(hist['loss'], label='Pérdida (Entrenamiento)')
        ax1.plot(hist['val_loss'], label='Pérdida (Validación)')
        ax1.set_title('Evolución de la Pérdida')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(hist['recall'], label='Recall (Entrenamiento)')
        ax2.plot(hist['val_recall'], label='Recall (Validación)')
        ax2.set_title('Evolución del Recall')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Recall')
        ax2.legend()
        ax2.grid(True)
        
        st.pyplot(fig)

    # --- Navegación de la Sección 5 ---
    if 'vista_seccion5' not in st.session_state:
        st.session_state.vista_seccion5 = 'arquitectura'

    cols_btn_c5 = st.columns(3)
    with cols_btn_c5[0]:
        if st.button("1. Comparación de Arquitectura", use_container_width=True):
            st.session_state.vista_seccion5 = 'arquitectura'
    with cols_btn_c5[1]:
        if st.button("2. Comparación de Desempeño", use_container_width=True):
            st.session_state.vista_seccion5 = 'desempeno'
    with cols_btn_c5[2]:
        if st.button("3. Comparación de Activaciones", use_container_width=True):
            st.session_state.vista_seccion5 = 'activaciones'

    st.divider()

    # --- Lógica de renderizado según la vista seleccionada ---
    if st.session_state.vista_seccion5 == 'arquitectura':
        st.header("🔬 Comparación de Arquitecturas")
        col_arq1, col_arq2 = st.columns(2)
        with col_arq1:
            st.subheader("Modelo Base")
            visualizar_modelo(st.session_state.modelo_redcita, "modelo_base.png")
        with col_arq2:
            st.subheader("Modelo Mejorado")
            modelo_para_visualizar = crear_modelo_mejorado()
            visualizar_modelo(modelo_para_visualizar, "modelo_mejorado.png")

    elif st.session_state.vista_seccion5 == 'desempeno':
        st.header("📊 Comparación de Desempeño")
        
        if st.button("🚀 Iniciar Entrenamiento y Evaluación de Modelos"):
            with st.spinner("Realizando todo el proceso de entrenamiento... Esto puede tardar un momento."):
                st.session_state.resultados_entrenamiento = entrenar_y_evaluar_modelos()
            st.success("¡Entrenamiento y evaluación completados!")

        if 'resultados_entrenamiento' in st.session_state:
            escenario = st.radio(
                "Selecciona el escenario de datos a visualizar:",
                ('sin_estandarizar', 'con_estandarizacion'),
                format_func=lambda x: "Sin Estandarizar" if x == 'sin_estandarizar' else "Con Estandarización",
                horizontal=True
            )
            
            resultados = st.session_state.resultados_entrenamiento[escenario]
            modelo_base = resultados["base"]["modelo"]
            modelo_mejorado = resultados["mejorado"]["modelo"]
            hist_base = resultados["base"]["historial"]
            hist_mejorado = resultados["mejorado"]["historial"]
            X_test_escenario = resultados["X_test"]
            y_test_escenario = resultados["y_test"]

            pred_base = (modelo_base.predict(X_test_escenario) > 0.5).astype("int32")
            pred_mejorado = (modelo_mejorado.predict(X_test_escenario) > 0.5).astype("int32")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown("#### Modelo Base")
                st.metric("Recall en Test", f"{recall_score(y_test_escenario, pred_base):.2%}")
                plot_matriz_confusion(y_test_escenario, pred_base, "Matriz de Confusión (Base)")
                plot_curvas_entrenamiento(hist_base, "Curvas de Entrenamiento (Base)")

            with col_res2:
                st.markdown("#### Modelo Mejorado")
                st.metric("Recall en Test", f"{recall_score(y_test_escenario, pred_mejorado):.2%}")
                plot_matriz_confusion(y_test_escenario, pred_mejorado, "Matriz de Confusión (Mejorado)")
                plot_curvas_entrenamiento(hist_mejorado, "Curvas de Entrenamiento (Mejorado)")
        else:
            st.info("Haz clic en el botón de arriba para entrenar los modelos y ver la comparación de desempeño.")

    elif st.session_state.vista_seccion5 == 'activaciones':
        st.header("🧠 Comparación de Activaciones de Neuronas")
        if 'resultados_entrenamiento' in st.session_state:
            resultados_scaled = st.session_state.resultados_entrenamiento["con_estandarizacion"]
            modelo_base_final = resultados_scaled["base"]["modelo"]
            modelo_mejorado_final = resultados_scaled["mejorado"]["modelo"]

            X_sample_raw = X[int(idx)].reshape(1, -1)
            scaler_completo = StandardScaler().fit(dataset.drop("Diabetes", axis=1).to_numpy(dtype="float32"))
            X_sample_scaled = scaler_completo.transform(X_sample_raw)
            
            # Modelo Base
            pesos_b1, sesgos_b1 = modelo_base_final.layers[0].get_weights()
            salida_b1_pre = np.dot(X_sample_scaled, pesos_b1) + sesgos_b1
            salida_b1 = np.maximum(0, salida_b1_pre)  # ReLU
            pesos_b2, sesgos_b2 = modelo_base_final.layers[1].get_weights()
            salida_b2_pre = np.dot(salida_b1, pesos_b2) + sesgos_b2
            salida_b2 = np.maximum(0, salida_b2_pre)

            # Modelo Mejorado
            pesos_m1, sesgos_m1 = modelo_mejorado_final.layers[0].get_weights()
            salida_m1_pre = np.dot(X_sample_scaled, pesos_m1) + sesgos_m1
            salida_m1 = np.tanh(salida_m1_pre)
            pesos_m2, sesgos_m2 = modelo_mejorado_final.layers[1].get_weights()
            salida_m2_pre = np.dot(salida_m1, pesos_m2) + sesgos_m2
            salida_m2 = np.maximum(0, salida_m2_pre)  # ReLU

            col_act1, col_act2 = st.columns(2)
            def plot_activations(data, title, ax):
                neuronas = [f'N{i}' for i in range(data.shape[1])]
                colores = ['#2ecc71' if val > 0 else '#e74c3c' for val in data[0]]
                ax.bar(neuronas, data[0], color=colores)
                ax.set_title(title)
                ax.set_ylabel('Valor de Activación')
                ax.grid(axis='y', linestyle='--', alpha=0.7)

            with col_act1:
                st.markdown("#### Activaciones del Modelo Base")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
                plot_activations(salida_b1, "Capa Oculta 1", ax1)
                plot_activations(salida_b2, "Capa Oculta 2", ax2)
                fig.tight_layout()
                st.pyplot(fig)

            with col_act2:
                st.markdown("#### Activaciones del Modelo Mejorado")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
                plot_activations(salida_m1, "Capa Oculta 1 (6 neuronas)", ax1)
                plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
                plot_activations(salida_m2, "Capa Oculta 2 (3 neuronas)", ax2)
                plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Debes entrenar los modelos en la pestaña '2. Comparación de Desempeño' primero para poder analizar las activaciones.")
########################################################################################################################################################################################################################################################################################

# Estado inicial para la sección y los pesos/sesgos modificables
if "Contenido6" not in st.session_state:
    st.session_state.Contenido6 = False
    # Guardamos una copia editable de los pesos y sesgos de la primera capa
    pesos_orig, sesgos_orig = model.layers[0].get_weights()
    st.session_state.pesos_modificados_c6 = pesos_orig.copy()
    st.session_state.sesgos_modificados_c6 = sesgos_orig.copy()

# Función para alternar la visibilidad de la sección
def toggle_contenido6():
    st.session_state.Contenido6 = not st.session_state.Contenido6

# Función para restaurar los pesos a su estado original
def restaurar_pesos_originales():
    pesos_orig, sesgos_orig = model.layers[0].get_weights()
    st.session_state.pesos_modificados_c6 = pesos_orig.copy()
    st.session_state.sesgos_modificados_c6 = sesgos_orig.copy()


# Layout del encabezado de la sección
col1_c6, col2_c6 = st.columns([1, 30])
with col1_c6:
    st.button("▼" if not st.session_state.Contenido6 else "▲", on_click=toggle_contenido6, key="btn_c6")
with col2_c6:
    st.markdown("### **6. Activar o Apagar Neuronas Manualmente**")

# --- Preparación de datos ---
    pesos_capa1, sesgos_capa1 = model.layers[0].get_weights()
    nombres_variables = dataset.columns[:-1].tolist()
# Contenido principal de la sección
if st.session_state.Contenido6:
    st.markdown("""
    Modifique manualmente los pesos y/o sesgos de las neuronas de la primera capa oculta para forzar que una permanezca **inactiva** (salida cero) y que otra esté **constantemente activa** (salida positiva), independientemente del registro de entrada.
    """)

    # Dividimos la interfaz en dos columnas: grafo a la izquierda, controles a la derecha
    col_grafo, col_controles = st.columns([2, 1])

    with col_controles:
        st.markdown("#### Controles de la Neurona")
        
        # Selector para elegir la neurona a modificar
        neurona_idx = st.radio(
            "Selecciona la neurona a modificar:",
            options=[0, 1, 2, 3],
            horizontal=True,
            key="neurona_select_c6"
        )
        
        st.write("---")

        # Slider para el sesgo de la neurona seleccionada
        st.session_state.sesgos_modificados_c6[neurona_idx] = st.slider(
            f"**Sesgo (Bias) de la Neurona {neurona_idx}**",
            min_value=-2.0,
            max_value=2.0,
            value=float(st.session_state.sesgos_modificados_c6[neurona_idx]),
            step=0.01,
            key=f"bias_slider_{neurona_idx}"
        )
        
        st.write("---")
        st.markdown("**Pesos de Entrada:**")

        # Creamos sliders para cada uno de los 8 pesos de entrada de la neurona seleccionada
        for i, nombre_var in enumerate(nombres_variables):
            st.session_state.pesos_modificados_c6[i, neurona_idx] = st.slider(
                nombre_var,
                min_value=-1.0,
                max_value=1.0,
                value=float(st.session_state.pesos_modificados_c6[i, neurona_idx]),
                step=0.01,
                key=f"peso_slider_{i}_{neurona_idx}"
            )
        
        # Botón para restaurar los valores por defecto
        if st.button("Restaurar Pesos Originales", use_container_width=True):
            restaurar_pesos_originales()
            st.rerun()

    # Usamos la función `crear_grafo_capa1` existente para la visualización
    def crear_grafo_capa1(input_data, pesos, sesgos, titulo, highlight_neurona=None, mostrar_etiquetas_pesos=True, estilizar_conexiones=True, curved_lines=False, grosor_variable=True):
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR', label=titulo, fontsize='35', nodesep='0.02', ranksep='2', size='8,6!')
        dot.attr(splines='curved' if curved_lines else 'line')

        pesos_abs = np.abs(pesos)
        max_peso_abs = np.max(pesos_abs) if np.max(pesos_abs) > 0 else 1

        with dot.subgraph(name='cluster_input') as c:
            c.attr(label='Variables de Entrada', style='filled', color='lightgrey')
            for i, nombre in enumerate(nombres_variables):
                c.node(f'in_{i}', f'{nombre}\n({input_data[i]:.2f})')

        with dot.subgraph(name='cluster_output') as c:
            c.attr(label='Capa Oculta 1', style='filled', color='lightgrey')
            salidas, _ = pasando_por_capa(input_data, pesos, sesgos)
            for i in range(4):
                color_neurona = 'lightblue' if salidas[i] > 0 else 'gray88'
                if highlight_neurona is not None and i == highlight_neurona:
                    color_neurona = 'yellow'
                c.node(f'out_{i}', f'Neurona {i}\nSalida: {salidas[i]:.3f}', style='filled', color=color_neurona)
        
        for i in range(4):
            if highlight_neurona is not None and i != highlight_neurona:
                continue
            for j in range(8):
                peso = pesos[j, i]
                
                if estilizar_conexiones:
                    penwidth = str(0.8 + 4 * (abs(peso) / max_peso_abs)) if grosor_variable else '1.5'
                    color = 'firebrick' if peso < 0 else 'forestgreen'
                else:
                    penwidth = '1.0'
                    color = 'gray50'
                
                etiqueta = f'{peso:.2f}' if mostrar_etiquetas_pesos else ''
                dot.edge(f'in_{j}', f'out_{i}', label=etiqueta, penwidth=penwidth, color=color, fontcolor=color, decorate='true', labelangle='-25', labeldistance='2.0')
        return dot
    # Le pasamos los pesos y sesgos modificados desde el session_state
    with col_grafo:
        st.markdown(f"#### Visualización Interactiva para el Registro `{idx}`")
        grafo = crear_grafo_capa1(
            X[idx],
            st.session_state.pesos_modificados_c6,
            st.session_state.sesgos_modificados_c6,
            titulo=f"Estado de Activación (Neurona {neurona_idx} seleccionada)",
            highlight_neurona=neurona_idx,
            mostrar_etiquetas_pesos=True,
            grosor_variable=True
        )
        st.graphviz_chart(grafo, use_container_width=True)

    st.divider()

   
    # Conclusiones finales para los puntos (c) y (d)
    st.markdown("### Conclusiones")
    st.markdown("""
    **(c) ¿Qué efecto tiene sobre la salida del modelo el hecho de que una neurona permanezca constantemente apagada?**

    Una neurona que siempre está apagada se conoce como una **"neurona muerta" (Dead Neuron)**. Su efecto es equivalente a eliminarla de la red. No contribuye con ninguna información a las capas posteriores, ya que su salida es siempre cero. Esto reduce la **capacidad del modelo**, es decir, su habilidad para aprender patrones complejos. Si demasiadas neuronas mueren durante el entrenamiento, el rendimiento de la red se degrada significativamente.

    **(d) ¿Qué implicaciones tiene que una neurona esté constantemente activa?**

    Una neurona que siempre está activa (salida > 0) no es necesariamente un problema, pero puede ser un síntoma de uno. Si está constantemente activa sin importar la entrada, significa que ha dejado de discriminar entre diferentes patrones de datos. En lugar de agregar no linealidad, simplemente pasa una versión transformada linealmente de sus entradas a la siguiente capa. Esto **reduce la no linealidad** de la red en ese punto, limitando su poder de representación. En casos extremos, si el valor de activación es muy grande y constante, puede llevar a problemas como la **explosión de gradientes** durante el entrenamiento.
    """)