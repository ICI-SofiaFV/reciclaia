# ♻️ ReciclaIA · Clasificador de residuos con IA

**App publicada:** [https://reciclaia-udd.streamlit.app](https://reciclaia-udd.streamlit.app)

App web que, a partir de la foto de un residuo, identifica su material (cartón, vidrio, metal, papel, plástico o basura no reciclable), muestra la confianza del modelo y entrega una recomendación de reciclaje generada con **Google Gemini**.

📷 Imagen → 🧠 YOLO (clase + confianza) → 📊 Resultado visual → 💬 Gemini → ✅ Recomendación

**Certamen I · IA Generativa para la Gestión Industrial · Universidad del Desarrollo**
**Integrantes:** Felipe Alonso · Sofía Fariña · Vicente Lagunas

---

## 🎯 Problema
Separar mal los residuos contamina lotes completos de material reciclable. En el contexto de la **Ley REP (Ley 20.920)**, la app orienta a personas y hogares sobre **en qué contenedor depositar cada residuo**.

**Error más costoso:** clasificar basura no reciclable como reciclable, porque contamina el lote de reciclaje.

## 📂 Datos
| Aspecto | Detalle |
|---|---|
| Dataset | **TrashNet** · Gary Thung y Mindy Yang (Stanford, 2016) |
| Cómo se obtuvo | Fotos con iPhone de un objeto a la vez sobre cartulina blanca, redimensionadas a 512 × 384 px |
| Licencia | MIT |
| Enlace | https://huggingface.co/datasets/garythung/trashnet · https://github.com/garythung/trashnet |
| Imágenes | 2.527 (2.524 tras eliminar 3 duplicados exactos) |
| Clases | cartón 403 · vidrio 501 · metal 410 · papel 594 · plástico 482 · **basura 137 (5,4 %)** |
| División | Estratificada 70 / 15 / 15 → 1.766 entrenamiento · 379 validación · 379 prueba |

**Fotos propias (prueba externa):** 60 fotos de residuos chilenos tomadas con iPhone, usadas solo para evaluar, no para entrenar. [Ver carpeta en Google Drive](https://drive.google.com/drive/folders/1zD7NmAYxCInv2HMbGeIRnUt8rQcECzDp?usp=drive_link)

**Limitaciones:** desbalance (`trash` tiene solo 20 imágenes de prueba), condiciones controladas (fondo blanco), una sola fuente (2016) y posibles fotos casi duplicadas.
Además, el modelo siempre elige uno de los 6 materiales, aunque la foto no sea un residuo.

> El dataset **no se sube** a este repositorio; se descarga desde Hugging Face en el notebook. Las imágenes de `ejemplos/` son del conjunto de prueba de TrashNet (licencia MIT).

## 🧠 Modelo y resultados
Comparamos tres enfoques con las **mismas 379 imágenes de prueba** y las mismas métricas:

| Enfoque | Accuracy | Acc. balanceada | Basura → reciclaje |
|---|---|---|---|
| CLIP sin entrenar (línea base) | 60,2 % | 64,4 % | 3 de 20 |
| EfficientNet-B0 (timm) | 85,0 % | 84,7 % | 4 de 20 |
| YOLO-cls nano v1 (Ultralytics) | 88,1 % | 86,5 % | 5 de 20 |
| **YOLO-cls nano v2 · modelo final** | **91,6 %** | **91,1 %** | **2 de 20** |

- **Elegimos YOLO-cls nano:** mejor desempeño, menos reciclables enviados a la basura y solo 3,2 MB.
- **YOLO v2** se reentrenó con aumento de datos más fuerte (recortes y acercamientos), después de que una prueba con fotos propias revelara un **cambio de dominio**: el modelo había aprendido las condiciones de TrashNet más que el material.
- **Umbral de confianza: 60 %.** Bajo ese valor, la app pide revisión humana. En prueba responde sola en el 95,8 % de los casos, con 93,4 % de acierto.

**Limitación principal:** con fotos reales (fondos y luces distintas), la accuracy balanceada baja a 33,1 %. Por eso la app guía al usuario a fotografiar el residuo completo sobre un fondo liso y claro.
Además, el modelo siempre elige uno de los 6 materiales, aunque la foto no sea un residuo.

## 📱 La app
- Carga de imagen: subir foto, cámara o ejemplos.
- Clase predicha, confianza y probabilidades de las 6 clases.
- Desempeño del modelo en prueba (pestaña "Sobre el modelo"), diferenciado de la confianza de cada foto.
- Recomendación con **Gemini** (`gemini-flash-lite-latest`), con un *prompt* responsable: limita el tema, advierte cuando la confianza es baja y no inventa cifras ni normativas.
- Advertencias: aviso de uso educativo y revisión humana bajo el 60 % de confianza.

## 🗂️ Estructura
```
├── app.py                      ← aplicación Streamlit
├── requirements.txt            ← librerías de Python
├── packages.txt                ← librerías del sistema para Streamlit Cloud (OpenCV)
├── notebooks/entrenamiento.ipynb ← datos, comparación de modelos y métricas
├── modelo/                     ← modelo_yolo.pt · clases.json · metricas.json
├── ejemplos/                   ← imágenes de prueba (2-3 por clase, para mostrar variedad dentro de cada clase)
└── .streamlit/config.toml      ← tema visual
```

## ▶️ Cómo ejecutarla localmente
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Para la recomendación con IA, crea `.streamlit/secrets.toml` con `GEMINI_API_KEY = "tu-clave"`. Este archivo está en `.gitignore` y **nunca se sube** al repositorio; en Streamlit Cloud la clave se guarda en *Settings → Secrets*.

## 🤖 Declaración de uso de IA
- **Claude (Anthropic):** apoyo para planificar el proyecto según la rúbrica, explicar conceptos y generar código base del notebook y de `app.py`, que el grupo revisó, ejecutó, adaptó y comentó.
- **Google Gemini:** genera las recomendaciones dentro de la app.

Todas las decisiones (dataset, comparación de modelos, mejora v2, umbral y limitaciones) se justifican con los resultados del notebook.