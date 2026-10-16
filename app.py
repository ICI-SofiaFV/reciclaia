# ♻️ ReciclaIA · App web para clasificar residuos y recomendar cómo reciclarlos
# Flujo: 📷 Imagen → 🧠 YOLO (clase + confianza) → 📊 Resultado visual → 💬 Gemini → ✅ Recomendación

# ==================== BLOQUE 1 · Librerías y configuración ====================
import io, json, hashlib, os                  # io: leer bytes como imagen · json: métricas · hashlib: huella de la foto · os: carpetas
import requests                                # requests: llamar a la API de Gemini
import pandas as pd                            # pandas: tablas en la pestaña del modelo
import streamlit as st                         # streamlit: la interfaz web
from PIL import Image, ImageOps                # PIL: abrir y enderezar fotos
from pillow_heif import register_heif_opener   # permite abrir fotos HEIC del iPhone
from ultralytics import YOLO                   # YOLO: nuestro modelo entrenado

register_heif_opener()
st.set_page_config(page_title="ReciclaIA", page_icon="♻️", layout="wide")

CARPETA_MODELO = "modelo"
CARPETA_EJEMPLOS = "ejemplos"
MODELO_GEMINI = "gemini-flash-lite-latest"     # modelo gratuito y rápido de Google AI Studio

# ==================== BLOQUE 2 · Información de cada clase ====================
# nombre en español, ícono, dónde va y cómo prepararlo (los colores de contenedor varían por comuna)
INFO_CLASES = {
    "cardboard": {"nombre": "Cartón", "icono": "📦", "reciclable": True,
                  "destino": "Contenedor de papel y cartón (habitualmente azul)",
                  "consejo": "Desármalo, aplánalo y verifica que esté limpio y seco."},
    "glass":     {"nombre": "Vidrio", "icono": "🍾", "reciclable": True,
                  "destino": "Contenedor de vidrio (habitualmente verde)",
                  "consejo": "Enjuágalo y retira tapas. No incluyas vidrios rotos de ventanas ni cerámica."},
    "metal":     {"nombre": "Metal", "icono": "🥫", "reciclable": True,
                  "destino": "Contenedor de latas y metales",
                  "consejo": "Enjuaga la lata y aplástala si puedes, para ahorrar espacio."},
    "paper":     {"nombre": "Papel", "icono": "📄", "reciclable": True,
                  "destino": "Contenedor de papel y cartón (habitualmente azul)",
                  "consejo": "Debe estar limpio y seco. Papel con grasa o comida no se recicla."},
    "plastic":   {"nombre": "Plástico", "icono": "🧴", "reciclable": True,
                  "destino": "Contenedor de plásticos (habitualmente amarillo)",
                  "consejo": "Enjuágalo, aplástalo y revisa el número de plástico en la base del envase."},
    "trash":     {"nombre": "Basura (no reciclable)", "icono": "🗑️", "reciclable": False,
                  "destino": "Basura común (residuos no reciclables)",
                  "consejo": "No la mezcles con reciclables: un solo residuo así puede contaminar un lote completo."},
}

# ==================== BLOQUE 3 · Cargar modelo, clases y métricas ====================
@st.cache_resource                             # se carga una sola vez, no en cada clic
def cargar_modelo():
    return YOLO(f"{CARPETA_MODELO}/modelo_yolo.pt")

@st.cache_data
def cargar_json(nombre):
    with open(f"{CARPETA_MODELO}/{nombre}", encoding="utf-8") as f:
        return json.load(f)

modelo = cargar_modelo()
CLASES = cargar_json("clases.json")            # orden de las clases tal como las entrega el modelo
METRICAS = cargar_json("metricas.json")        # desempeño medido en el conjunto de prueba
UMBRAL = METRICAS["umbral_confianza"]          # 0,60: bajo este valor se pide revisión humana

def clasificar(imagen):
    """Devuelve las 6 clases ordenadas de mayor a menor probabilidad."""
    r = modelo(imagen, imgsz=224, verbose=False)[0]
    probs = r.probs.data.cpu().numpy()         # probabilidades (softmax) de las 6 clases
    return sorted(zip(CLASES, probs), key=lambda x: x[1], reverse=True)

# ==================== BLOQUE 4 · Recomendación con Gemini ====================
def obtener_clave():
    try:
        return st.secrets["GEMINI_API_KEY"]    # la clave vive en Secrets, nunca en el código
    except Exception:
        return None

def pedir_recomendacion(ranking):
    clase, conf = ranking[0]
    clase2, conf2 = ranking[1]
    prompt = f"""Eres un asistente educativo de reciclaje para hogares en Chile.
Un modelo de clasificación de imágenes analizó la foto de un residuo:
- Clase más probable: {INFO_CLASES[clase]['nombre']} (confianza {conf:.0%})
- Segunda opción: {INFO_CLASES[clase2]['nombre']} (confianza {conf2:.0%})
- Umbral de confianza para revisión humana: {UMBRAL:.0%}

Instrucciones:
1. Responde en español, en un máximo de 120 palabras, con 3 viñetas: cómo preparar el residuo, dónde depositarlo y un error común a evitar.
2. Si la confianza es menor a {UMBRAL:.0%}, empieza advirtiendo que el resultado es incierto, menciona ambas opciones y pide revisar el material a simple vista antes de depositarlo. Si la confianza es igual o mayor a {UMBRAL:.0%}, NO digas que es incierto ni menciones la segunda opción: da la recomendación directamente para la clase más probable.
3. Habla solo de reciclaje y manejo de residuos. No inventes cifras ni normativas.
4. Recuerda que los colores de los contenedores pueden variar según la comuna o el punto limpio."""

    respuesta = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO_GEMINI}:generateContent",
        headers={"x-goog-api-key": obtener_clave(), "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.3, "maxOutputTokens": 400}},
        timeout=30,
    )
    if respuesta.status_code == 429:
        return "⏳ Se alcanzó el límite gratuito de Gemini. Intenta de nuevo en un minuto."
    respuesta.raise_for_status()
    return respuesta.json()["candidates"][0]["content"]["parts"][0]["text"]

# ==================== BLOQUE 5 · Interfaz ====================
st.title("♻️ ReciclaIA")
st.markdown("**Sube la foto de un residuo y te diremos de qué material es y cómo reciclarlo.**")

with st.sidebar:
    st.header("📸 Para un mejor resultado")
    st.markdown("- **Un solo objeto** por foto\n- **Fondo liso y claro** (mesa o papel blanco)\n"
                "- **Buena luz**, sin sombras fuertes\n- Toma la foto a **30–40 cm**: que se vea el **objeto completo** "
                "con fondo alrededor, no solo la etiqueta")
    st.header("⚠️ Aviso")
    st.info("Prototipo **educativo**. Ante la duda, revisa el residuo a simple vista o sigue la "
            "señalética de tu punto limpio.")
    st.caption("ReciclaIA solo reconoce **6 materiales**: si subes otro objeto (ropa, comida, mascotas), "
               "igual elegirá uno de ellos.")
    st.caption("Certamen I · IA Generativa para la Gestión Industrial · UDD\n\n"
               "Felipe Alonso · Sofía Fariña · Vicente Lagunas")

tab_clasificar, tab_modelo = st.tabs(["♻️ Clasificar", "📊 Sobre el modelo"])

# ---------- Pestaña 1: clasificar ----------
with tab_clasificar:
    col_foto, col_resultado = st.columns(2, gap="large")

    with col_foto:
        st.subheader("1. Carga una imagen")
        origen = st.radio("Origen", ["📁 Subir foto", "📷 Cámara", "🖼️ Ejemplo"],
                          horizontal=True, label_visibility="collapsed")
        datos = None
        if origen == "📁 Subir foto":
            archivo = st.file_uploader("Foto del residuo", type=["jpg", "jpeg", "png", "webp", "heic"])
            datos = archivo.getvalue() if archivo else None
        elif origen == "📷 Cámara":
            foto = st.camera_input("Toma una foto del residuo")
            datos = foto.getvalue() if foto else None
        else:
            ejemplos = sorted(os.listdir(CARPETA_EJEMPLOS))
            elegido = st.selectbox("Imagen de ejemplo", ejemplos)
            with open(f"{CARPETA_EJEMPLOS}/{elegido}", "rb") as f:
                datos = f.read()

        if datos:
            imagen = ImageOps.exif_transpose(Image.open(io.BytesIO(datos))).convert("RGB")
            st.image(imagen, width=400)                       # tamaño fijo: la foto no empuja el resultado hacia abajo

    with col_resultado:
        st.subheader("2. Resultado")
        if not datos:
            st.info("👈 Carga una imagen para ver el resultado.")
        else:
            ranking = clasificar(imagen)
            clase, conf = ranking[0]
            info = INFO_CLASES[clase]

            st.markdown(f"## {info['icono']} {info['nombre']}")
            st.metric("Confianza en esta imagen", f"{conf:.0%}")

            if conf < UMBRAL:
                st.warning(f"⚠️ **Confianza baja (menor a {UMBRAL:.0%}).** El modelo no está seguro: "
                           f"revisa el residuo a simple vista antes de depositarlo. Segunda opción: "
                           f"**{INFO_CLASES[ranking[1][0]]['nombre']}** ({ranking[1][1]:.0%}).")
            elif info["reciclable"]:
                st.success(f"✅ **Reciclable** · {info['destino']}")
            else:
                st.error(f"🚫 **No reciclable** · {info['destino']}")
            st.caption(f"💡 {info['consejo']}")

            st.markdown("**Probabilidad de cada clase**")
            for c, p in ranking:
                st.progress(float(p), text=f"{INFO_CLASES[c]['icono']} {INFO_CLASES[c]['nombre']}: {p:.1%}")

            # ---------- Recomendación con IA ----------
            st.subheader("3. Recomendación con IA")
            huella = hashlib.md5(datos).hexdigest()           # identifica la foto actual
            if st.session_state.get("huella") != huella:      # si cambió la foto, borra la recomendación anterior
                st.session_state.huella = huella
                st.session_state.recomendacion = None

            if not obtener_clave():
                st.error("Falta configurar la clave GEMINI_API_KEY en Secrets.")
            elif st.button("💬 Pedir recomendación a Gemini", type="primary"):
                with st.spinner("Gemini está preparando la recomendación..."):
                    try:
                        st.session_state.recomendacion = pedir_recomendacion(ranking)
                    except Exception:
                        st.session_state.recomendacion = "❌ No se pudo conectar con Gemini. Intenta de nuevo."
            if st.session_state.recomendacion:
                st.info(st.session_state.recomendacion)
                st.caption("Texto generado por IA (Google Gemini). Puede contener errores.")

# ---------- Pestaña 2: sobre el modelo ----------
with tab_modelo:
    st.subheader("🧠 Modelo: YOLO-cls nano v2 (Ultralytics)")
    st.markdown("Entrenado con *transfer learning* sobre **TrashNet** (2.524 imágenes, 6 clases), "
                "con aumento de datos para que se fije en el objeto y no en el fondo.")

    st.markdown(f"#### Desempeño en el conjunto de prueba ({METRICAS['n_test']} imágenes nunca vistas)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{METRICAS['accuracy']:.1%}")
    c2.metric("Accuracy balanceada", f"{METRICAS['accuracy_balanceada']:.1%}")
    c3.metric("Responde sin pedir revisión", f"{METRICAS['cobertura_test']:.1%}")
    st.info("📌 **No confundir:** la *accuracy* mide qué tan bien funciona el modelo en general (sobre "
            "cientos de imágenes de prueba). La *confianza* de la pestaña Clasificar es la seguridad "
            "del modelo en **una sola foto**.")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("**Recall por clase** (de cada clase real, cuántas acierta)")
        st.dataframe(pd.DataFrame({
            "Clase": [INFO_CLASES[c]["nombre"] for c in METRICAS["recall_por_clase"]],
            "Recall": [f"{v:.0%}" for v in METRICAS["recall_por_clase"].values()]}),
            hide_index=True, width="stretch")
    with col_b:
        st.markdown("**Comparación de enfoques** (misma prueba y métricas)")
        st.dataframe(pd.DataFrame([
            {"Enfoque": nombre, "Accuracy": f"{r['accuracy']:.1%}",
             "Acc. balanceada": f"{r['accuracy_balanceada']:.1%}"}
            for nombre, r in METRICAS["comparacion"].items()]),
            hide_index=True, width="stretch")

    propias = METRICAS["fotos_propias"]
    st.warning(f"⚠️ **Limitación:** con {propias['n']} fotos propias tomadas en fondos y luces reales, la "
               f"accuracy balanceada baja a **{propias['accuracy_balanceada']:.1%}**. El modelo aprendió "
               "con fotos sobre cartulina blanca, por eso pedimos fotografiar el residuo sobre un fondo liso y claro.")
    st.caption("Datos: TrashNet (Thung y Yang, Stanford 2016, licencia MIT) · huggingface.co/datasets/garythung/trashnet")