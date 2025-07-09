import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from skimage.morphology import skeletonize
import base64
from io import BytesIO

# ======== Funciones personalizadas ========
def Weighted_Cross_Entropy(beta):
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=beta)
        return tf.reduce_mean(loss)

    return loss

class RepeatChannels(Layer):
    def __init__(self, rep, **kwargs):
        super().__init__(**kwargs)
        self.rep = rep

    def call(self, inputs):
        return tf.tile(inputs, [1, 1, 1, self.rep])

# ======== Cargar modelos ========
@st.cache_resource
def cargar_segmentador():
    return load_model('/content/modelo3_b2.h5', custom_objects={'RepeatChannels': RepeatChannels, 'loss': Weighted_Cross_Entropy(10.0)})

@st.cache_resource
def cargar_clasificador():
    return load_model('/content/clasificador_superficie_SA.h5')

model_segmentador = cargar_segmentador()
model_clasificador = cargar_clasificador()

# ======== Clasificador de superficie ========
def predecir_superficie_streamlit(img_rgb, modelo):
    ALTO = 128
    ANCHO = 128
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (ALTO, ANCHO))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = np.expand_dims(img_norm, axis=-1)
    img_norm = np.expand_dims(img_norm, axis=0)
    pred = modelo.predict(img_norm)[0][0]
    clase = "Macizo (1)" if pred >= 0.5 else "Pandereta (0)"
    return clase, pred

# ======== Interfaz ========
st.title("Detector de Grietas")

st.markdown("""
### Instrucciones para subir imágenes:
1. Suba **imágenes cuadradas** (idealmente 1:1).
2. El sistema **solo detecta grietas** visibles en la imagen.
3. **No extrapolable** a otros materiales (madera, metal, etc.).
4. **No garantiza buenos resultados** en todos los tipos de albañilería.
""")

# Parámetros
st.markdown("### Parámetros")
umbral = st.slider("Umbral para binarización de la máscara predicha", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
ancho_mm = st.number_input("Ancho real de la escala cuadrada (mm)", min_value=1.0, max_value=1000.0, value=50.0, step=1.0)

# Subida de imagen
uploaded_file = st.file_uploader("Sube una imagen (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((512, 512))
    img_input = np.expand_dims(resized_image, axis=0)

    # Segmentación
    prediction = model_segmentador.predict(img_input)[0]
    if prediction.shape[-1] == 1:
        prediction = prediction[:, :, 0]
    mask = (prediction > umbral).astype(np.uint8)

    # Esqueletización
    skeleton = skeletonize(mask).astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    crack_width_map = dist_transform * skeleton * 2
    mean_width = crack_width_map[crack_width_map > 0].mean() if np.any(crack_width_map > 0) else 0
    max_idx = np.unravel_index(np.argmax(crack_width_map), crack_width_map.shape)
    max_width = crack_width_map[max_idx]

    # Detección de escala
    cv_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask1 = cv2.inRange(hsv, lower_green, upper_green)
    contornos, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    escala_detectada = False
    pixels_per_mm = None
    if contornos:
        contorno_mayor = max(contornos, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contorno_mayor, True)
        approx = cv2.approxPolyDP(contorno_mayor, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            escala_detectada = True
            pixels_per_mm = np.mean([w, h]) / ancho_mm
    img_escala_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Clasificación de superficie
    clase_superficie, confianza = predecir_superficie_streamlit(resized_image, model_clasificador)

    # Mostrar resultados: imagen original y máscara
    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_image, caption="Imagen original", use_container_width=True)
    with col2:
        st.image(mask * 255, caption=f"Máscara predicha (umbral: {umbral})", use_container_width=True)

    # Mapa de ancho de grietas
    fig_width, ax_width = plt.subplots(figsize=(5, 4))
    im = ax_width.imshow(crack_width_map, cmap='jet')
    ax_width.scatter(max_idx[1], max_idx[0], color='white', s=80, edgecolors='black', label='Ancho máximo')
    ax_width.set_title("Mapa de ancho de grietas")
    ax_width.axis('off')
    plt.colorbar(im, ax=ax_width, fraction=0.046, pad=0.04, label='Ancho (píxeles)')
    ax_width.legend()

    buf_width = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf_width, format="png")
    plt.close(fig_width)

    # Preparar imagen de escala como base64
    buffer = BytesIO()
    Image.fromarray(img_escala_rgb).save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    caption_escala = "Escala detectada" if escala_detectada else "Escala no detectada"

    # Segunda sección: columna izquierda (mapa + escala) y columna derecha (resumen)
    col_izq, col_der = st.columns(2)
    with col_izq:
        st.image(buf_width.getvalue(), caption="Mapa de ancho de grietas", use_container_width=True)
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_base64}" width="250"/>
                <p><strong>{caption_escala}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_der:
        st.markdown("### Estimación del ancho de grieta")
        st.markdown(f"**Valor promedio:** {mean_width:.2f} píxeles")
        st.markdown(f"**Ancho máximo:** {max_width:.2f} píxeles")

        if escala_detectada and pixels_per_mm:
            mean_mm = mean_width / pixels_per_mm
            max_mm = max_width / pixels_per_mm
            st.markdown(f"**Promedio:** {mean_mm:.2f} mm")
            st.markdown(f"**Máximo:** {max_mm:.2f} mm")
            st.markdown(f"**Escala estimada:** {pixels_per_mm:.2f} píxeles/mm")
        else:
            st.markdown("*No se pudo calcular en mm (escala no detectada)*")

        st.markdown("### Predicción de superficie")
        st.markdown(f"**Predicción:** Albañilería con ladrillo tipo {clase_superficie}")
        st.markdown(f"**Confianza:** {confianza:.2f}")

