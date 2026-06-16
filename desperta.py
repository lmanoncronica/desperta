import os
import re
import io
import wave
import json
import time
import threading
from datetime import datetime

import requests
import streamlit as st

# Reconocimiento de voz en el navegador (gratis, no necesita API).
# Si no está instalado, la app sigue funcionando sin la escucha.
try:
    from streamlit_mic_recorder import speech_to_text
    STT_DISPONIBLE = True
except Exception:
    STT_DISPONIBLE = False

# Captura de audio continua (escucha siempre activa). Opcional.
try:
    import av  # noqa: F401
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_DISPONIBLE = True
except Exception:
    WEBRTC_DISPONIBLE = False

# ============================================================
#  BCN DESPERTA PRO  ·  Fact-checking editorial en directo
#  Versión actualizada (junio 2026)
# ============================================================

# --- 1. CONFIGURACIÓN Y LLAVES (¡YA NO VAN EN EL CÓDIGO!) ---
# Las claves se leen de .streamlit/secrets.toml o de variables de entorno.
# Crea un archivo  .streamlit/secrets.toml  junto a este script con:
#
#   GEMINI_API_KEY      = "tu_clave"
#   CLAUDE_API_KEY      = "tu_clave"
#   ELEVENLABS_API_KEY  = "tu_clave"
#   PERPLEXITY_API_KEY  = "tu_clave"   # opcional
#   ELEVENLABS_VOICE_ID = "z0BZOmPMDRocOWMwLB5J"

def get_secret(name, default=""):
    """Lee primero de st.secrets, luego de variables de entorno."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

API_KEY_GEMINI     = get_secret("GEMINI_API_KEY")
API_KEY_CLAUDE     = get_secret("CLAUDE_API_KEY") or get_secret("ANTHROPIC_API_KEY")
ELEVEN_API_KEY     = get_secret("ELEVENLABS_API_KEY")
API_KEY_PERPLEXITY = get_secret("PERPLEXITY_API_KEY")
VOICE_ID           = get_secret("ELEVENLABS_VOICE_ID", "z0BZOmPMDRocOWMwLB5J")
SUPABASE_DB_URL    = get_secret("SUPABASE_DB_URL")   # opcional: persistencia entre reinicios

# Modelo de voz para directo. Flash v2.5 ≈ <75ms (ideal en vivo);
# Multilingual v2 = más calidad pero más lento.
VOZ_RAPIDA   = "eleven_flash_v2_5"
VOZ_CALIDAD  = "eleven_multilingual_v2"

# Modelo de Claude. Sonnet 4.6 = rápido y fiable para verificar en directo.
CLAUDE_MODEL = "claude-sonnet-4-6"

# Modelo de Perplexity (opcional). sonar-pro = mejor veracidad.
PERPLEXITY_MODEL = "sonar-pro"

# --- APARIENCIA POR DEFECTO ---
# Puedes cambiar estos valores aquí (quedan fijos) o en vivo desde el panel
# "🎨 Apariencia" de la barra lateral (se aplican al instante en la pantalla).
BRANDING_DEFAULT = {
    "titulo": "BCN DESPERTA",          # nombre del programa (cabecera del control)
    "fuente": "Arial",                 # tipografía de la pantalla
    "color_fondo": "#000000",          # fondo de la pantalla
    "color_titular": "#FFFFFF",        # color del titular
    "color_verificacion": "#FF0000",   # color de la verificación
    "color_acento": "#FF0000",         # color del borde lateral
    "grosor_borde": 25,                # grosor del borde lateral (0 = sin borde)
    "tam_titular": 80,                 # tamaño del titular (px)
    "tam_verificacion": 38,            # tamaño de la verificación (px)
    "mayusculas": True,                # titular en MAYÚSCULAS
    "alineacion": "left",              # "left" o "center"
    "logo_url": "",                    # URL de imagen para el logo
    "tam_logo": 130,                   # alto del logo (px)
}

FUENTES = ["Arial", "Oswald", "Montserrat", "Georgia", "Verdana", "Tahoma",
           "Trebuchet MS", "Impact", "Times New Roman", "Courier New"]


# --- 2. MEMORIA CENTRAL ---
@st.cache_resource
class SharedState:
    def __init__(self):
        self.en_pantalla = {"t": "", "v": ""}   # arranca en blanco
        self.cola_pendientes = []
        self.cola_resultados = []
        self.archivo_historico = []
        self.sugerencias = []          # titulares propuestos por la IA al escuchar
        self.transcript_log = []       # historial de lo transcrito
        self.branding = dict(BRANDING_DEFAULT)   # apariencia editable
        self.cargado = False                     # ¿ya se cargó el estado guardado?
        self.lock = threading.Lock()   # para escribir desde el hilo de voz sin corromper datos

gs = SharedState()


# --- FECHA DINÁMICA EN ESPAÑOL (sin depender del locale del sistema) ---
_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
_MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
          "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

def fecha_hoy_es():
    n = datetime.now()
    return f"{_DIAS[n.weekday()]} {n.day} de {_MESES[n.month - 1]} de {n.year}"


# --- 3. MOTOR IA: AUTODETECCIÓN DE MODELO VIGENTE ---
@st.cache_resource
def discover_model(api_key):
    """Descubre un modelo válido en tu cuenta, priorizando los más rápidos/actuales.
    Gemini 1.5 ya está apagado (404), por eso el respaldo es 2.5-flash."""
    preferidos = ["gemini-3-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    if not api_key:
        return "models/gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        r = requests.get(url, timeout=10).json()
        disponibles = [
            m["name"] for m in r.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        for pref in preferidos:
            for name in disponibles:
                if pref in name:
                    return name
        if disponibles:
            return disponibles[0]
    except Exception:
        pass
    return "models/gemini-2.5-flash"

WORKING_MODEL = discover_model(API_KEY_GEMINI)


# --- JERARQUÍA DE FUENTES Y FORMATEADOR (compartido por Gemini y Claude) ---
INSTRUCCION_FUENTES = (
    "Verifica el dato buscando en la web. Prioriza fuentes en este orden estricto: "
    "1) Crónica Global (cronicaglobal.elespanol.com), "
    "2) Metrópoli Abierta (metropoliabierta.elespanol.com), "
    "3) El Español (elespanol.com). "
    "Solo si esos tres medios no tienen nada sobre el tema, busca en el resto de la "
    "prensa fiable e indícalo. Sé preciso: NO afirmes nada que no aparezca en una "
    "fuente concreta; si algo no se puede confirmar, dilo claramente. "
    "FORMATO: Línea 1 = titular claro y conciso. Líneas siguientes = verificación breve "
    "(qué es cierto, qué es matizable o falso). No escribas etiquetas como 'TITULAR:'."
)


def formatear_fuentes(pares):
    """Construye un bloque 'Fuentes:' a partir de [(titulo, url)], sin duplicados."""
    vistos, lineas = set(), []
    for titulo, url in pares:
        if not url or url in vistos:
            continue
        vistos.add(url)
        lineas.append(f"- {titulo}: {url}")
    if not lineas:
        return "\n\n⚠️ FUENTES: no se obtuvo ninguna fuente web. Verificar a mano."
    return "\n\nFuentes:\n" + "\n".join(lineas)


def limpiar_para_voz(texto):
    """Quita el bloque de fuentes y cualquier URL para que la voz no las lea."""
    for marca in ["\nFuentes:", "\n⚠️ FUENTES", "\nFUENTES:", "Fuentes:", "⚠️ FUENTES"]:
        idx = texto.find(marca)
        if idx != -1:
            texto = texto[:idx]
    texto = re.sub(r"https?://\S+", "", texto)   # por si quedara alguna URL suelta
    return texto.strip()


def consultar_ia(texto, feedback=None, historial=None):
    if not API_KEY_GEMINI:
        return "⚠️ Falta GEMINI_API_KEY en secrets.toml."

    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"{WORKING_MODEL}:generateContent?key={API_KEY_GEMINI}")

    prompt = f"Hoy es {fecha_hoy_es()}. Eres verificador de BCN DESPERTA. {INSTRUCCION_FUENTES}\n"

    if feedback:
        contenido = (f"ANÁLISIS PREVIO: {historial}. "
                     f"CORRECCIÓN DEL EDITOR: {feedback}. "
                     "Genera la nueva versión final, volviendo a buscar si hace falta.")
    else:
        contenido = f"Verifica esto: {texto}"

    payload = {
        "contents": [{"parts": [{"text": prompt + contenido}]}],
        # Búsqueda real con Google (grounding): trae fuentes web auténticas.
        "tools": [{"google_search": {}}],
    }
    try:
        res = requests.post(url, json=payload, timeout=30)
        if res.status_code != 200:
            return f"Error IA {res.status_code}: {res.text[:300]}"
        cand = res.json()["candidates"][0]
        partes = cand.get("content", {}).get("parts", [])
        raw = "".join(p.get("text", "") for p in partes).strip()
        raw = raw.replace("TITULAR:", "").replace("VERIFICACION:", "").strip()
        # Fuentes reales devueltas por el grounding
        fuentes = []
        for ch in cand.get("groundingMetadata", {}).get("groundingChunks", []):
            web = ch.get("web", {})
            titulo = web.get("title") or web.get("domain") or "Fuente"
            fuentes.append((titulo, web.get("uri", "")))
        return raw + formatear_fuentes(fuentes)
    except Exception as e:
        return f"Error conexión: {str(e)}"


# --- 4. MOTOR PERPLEXITY ---
def consultar_perplexity(texto):
    if not API_KEY_PERPLEXITY or "TU_API" in API_KEY_PERPLEXITY:
        return "⚠️ Configura PERPLEXITY_API_KEY para usar la búsqueda en vivo."
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY_PERPLEXITY}",
               "Content-Type": "application/json"}
    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {"role": "system",
             "content": (f"Hoy es {fecha_hoy_es()}. "
                         "Prioriza Crónica Global, Metrópoli Abierta y El Español. "
                         "Responde con un titular, verificación breve y las fuentes.")},
            {"role": "user", "content": f"Verifica: {texto}"},
        ],
    }
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=30)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        return f"Error Perplexity {res.status_code}: {res.text[:300]}"
    except Exception as e:
        return f"Error Perplexity: {str(e)}"


# --- 4b. ESCUCHA: EXTRAER TITULARES DE LA TRANSCRIPCIÓN ---
def extraer_sugerencias(transcript):
    """Pide a Gemini que saque titulares y datos verificables de lo escuchado."""
    if not API_KEY_GEMINI or not transcript.strip():
        return []
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"{WORKING_MODEL}:generateContent?key={API_KEY_GEMINI}")
    prompt = (
        f"Hoy es {fecha_hoy_es()}. Eres asistente de redacción en un evento en directo. "
        "A partir de esta transcripción de lo que se acaba de decir, detecta como máximo 4 "
        "afirmaciones noticiables o verificables. "
        "Devuelve SOLO un array JSON, sin markdown ni texto extra, con este formato exacto: "
        '[{"titular": "...", "dato": "..."}]. '
        "Si no hay nada relevante, devuelve []."
        f"\n\nTRANSCRIPCIÓN:\n{transcript}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(url, json=payload, timeout=20)
        raw = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)][:4]
    except Exception:
        pass
    return []


# --- ESCUCHA CONTINUA: audio PCM -> WAV -> transcripción con ElevenLabs Scribe ---
SCRIBE_MODEL = "scribe_v2"   # modelo de voz-a-texto de ElevenLabs

def _pcm_a_wav(pcm, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)      # 16 bits
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribir_elevenlabs(wav_bytes):
    """Transcribe un bloque de audio con ElevenLabs Scribe (misma clave que la voz)."""
    if not ELEVEN_API_KEY:
        return ""
    try:
        r = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVEN_API_KEY},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model_id": SCRIBE_MODEL},
            timeout=25)
        if r.status_code == 200:
            return (r.json().get("text") or "").strip()
    except Exception:
        pass
    return ""


# --- 4c. MOTOR CLAUDE (con búsqueda web real) ---
def consultar_claude(texto, feedback=None, historial=None):
    if not API_KEY_CLAUDE:
        return "⚠️ Falta CLAUDE_API_KEY en secrets.toml."
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": API_KEY_CLAUDE,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    sistema = (
        f"Hoy es {fecha_hoy_es()}. Eres verificador editorial de BCN DESPERTA. "
        + INSTRUCCION_FUENTES
    )
    if feedback:
        contenido = (f"Análisis previo: {historial}. "
                     f"Corrección del editor: {feedback}. Da la versión final.")
    else:
        contenido = f"Verifica esto: {texto}"
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "system": sistema,
        "messages": [{"role": "user", "content": contenido}],
        # Búsqueda web nativa: Claude consulta fuentes reales antes de responder.
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=45)
        if res.status_code != 200:
            return f"Error Claude {res.status_code}: {res.text[:300]}"
        bloques = res.json().get("content", [])
        texto_final = "".join(
            b.get("text", "") for b in bloques if b.get("type") == "text"
        ).strip()
        # Fuentes reales: resultados de búsqueda + citas dentro del texto
        fuentes = []
        for b in bloques:
            if b.get("type") == "web_search_tool_result":
                for r in (b.get("content") or []):
                    if isinstance(r, dict) and r.get("type") == "web_search_result":
                        fuentes.append((r.get("title") or r.get("url"), r.get("url")))
            if b.get("type") == "text":
                for c in (b.get("citations") or []):
                    if c.get("url"):
                        fuentes.append((c.get("title") or c.get("url"), c.get("url")))
        return (texto_final or "Sin respuesta de Claude.") + formatear_fuentes(fuentes)
    except Exception as e:
        return f"Error Claude: {str(e)}"


# --- ANÁLISIS EN SEGUNDO PLANO (no bloquea la interfaz) ---
def _thread_analisis(item_id, texto, motor, feedback, historial):
    """Ejecuta la verificación en un hilo y rellena la tarjeta al terminar."""
    if motor == "claude":
        res = consultar_claude(texto, feedback=feedback, historial=historial)
    elif motor == "perplexity":
        res = consultar_perplexity(texto)
    else:
        res = consultar_ia(texto, feedback=feedback, historial=historial)
    with gs.lock:
        for r in gs.cola_resultados:
            if r["id"] == item_id:
                r["res"] = res
                r["estado"] = "LISTO"
                if feedback is not None:
                    r["v"] += 1
                break


def lanzar_analisis(orig, motor, h=None, feedback=None, historial=None, item_id=None):
    """Crea (o reutiliza) una tarjeta y lanza el análisis en segundo plano."""
    if item_id is None:
        item_id = time.time()
        gs.cola_resultados.insert(0, {
            "res": "⏳ Analizando…", "h": h or time.strftime("%H:%M:%S"),
            "id": item_id, "audio": None, "status_voz": "IDLE",
            "orig": orig, "v": 0, "motor": motor, "estado": "ANALIZANDO",
        })
    else:
        with gs.lock:
            for r in gs.cola_resultados:
                if r["id"] == item_id:
                    r["estado"] = "ANALIZANDO"
                    break
    threading.Thread(target=_thread_analisis,
                     args=(item_id, orig, motor, feedback, historial),
                     daemon=True).start()
    return item_id


# --- COPIA DE SEGURIDAD: exportar / restaurar / persistir ---
def exportar_estado():
    """Serializa todo el estado a JSON (sin los bytes de audio)."""
    with gs.lock:
        data = {
            "exportado": time.strftime("%Y-%m-%d %H:%M:%S"),
            "en_pantalla": gs.en_pantalla,
            "pendientes": gs.cola_pendientes,
            "resultados": [{k: v for k, v in r.items() if k != "audio"}
                           for r in gs.cola_resultados],
            "historico": gs.archivo_historico,
            "sugerencias": gs.sugerencias,
            "transcripciones": gs.transcript_log,
            "branding": gs.branding,
        }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _aplicar_estado(data):
    """Repuebla el estado a partir de un JSON (de la BD o de un archivo subido)."""
    with gs.lock:
        gs.en_pantalla = data.get("en_pantalla", gs.en_pantalla)
        gs.cola_pendientes = data.get("pendientes", [])
        resultados = data.get("resultados", [])
        for r in resultados:
            r.setdefault("v", 0)
            r.setdefault("motor", "gemini")
            if r.get("estado") == "ANALIZANDO":   # quedó a medias por un reinicio
                r["res"] = "(Análisis interrumpido por un reinicio. Pulsa Reprocesar.)"
            r["estado"] = "LISTO"
            r["audio"] = None
            r["status_voz"] = "IDLE"
        gs.cola_resultados = resultados
        gs.archivo_historico = data.get("historico", [])
        gs.sugerencias = data.get("sugerencias", [])
        gs.transcript_log = data.get("transcripciones", [])
        gs.branding = {**BRANDING_DEFAULT, **data.get("branding", {})}


@st.cache_resource
def get_engine():
    """Conexión a la base de datos (Supabase/Postgres). None si no está configurada."""
    if not SUPABASE_DB_URL:
        return None
    try:
        from sqlalchemy import create_engine, text
        eng = create_engine(SUPABASE_DB_URL, pool_pre_ping=True)
        with eng.begin() as conn:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS estado_app (id TEXT PRIMARY KEY, data TEXT)"))
        return eng
    except Exception:
        return None


def guardar_estado():
    """Guarda el estado en la base de datos (si está configurada)."""
    eng = get_engine()
    if not eng:
        return
    try:
        from sqlalchemy import text
        payload = exportar_estado()
        with eng.begin() as conn:
            conn.execute(text(
                "INSERT INTO estado_app (id, data) VALUES ('principal', :d) "
                "ON CONFLICT (id) DO UPDATE SET data = :d"), {"d": payload})
    except Exception:
        pass


def cargar_estado():
    """Carga el estado desde la base de datos al arrancar (si la hay)."""
    eng = get_engine()
    if not eng:
        return
    try:
        from sqlalchemy import text
        with eng.connect() as conn:
            row = conn.execute(
                text("SELECT data FROM estado_app WHERE id='principal'")).fetchone()
        if row and row[0]:
            _aplicar_estado(json.loads(row[0]))
    except Exception:
        pass


# --- 5. MOTOR AUDIO (HILO INDEPENDIENTE) ---
def thread_voz(texto, item_id, model_id):
    texto = limpiar_para_voz(texto)   # la voz no lee las fuentes ni las URLs
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    data = {
        "text": texto,
        "model_id": model_id,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    def _marcar(estado, audio=None):
        with gs.lock:
            for res in gs.cola_resultados:
                if res["id"] == item_id:
                    res["status_voz"] = estado
                    if audio is not None:
                        res["audio"] = audio
                    break

    try:
        response = requests.post(url, json=data, headers=headers, timeout=60)
        if response.status_code == 200:
            _marcar("LISTO", response.content)
        else:
            _marcar("ERROR")
    except Exception:
        _marcar("ERROR")


# --- 6. INTERFAZ ---
# Carga el estado guardado (de la BD) una sola vez al arrancar el contenedor
if not gs.cargado:
    cargar_estado()
    gs.cargado = True

st.set_page_config(page_title=gs.branding["titulo"], layout="wide")

modo = st.sidebar.radio("MODO:", ["🛠️ CONTROL", "👂 ESCUCHA", "📺 ESCENARIO"])

# Aviso si faltan claves
if not API_KEY_GEMINI:
    st.sidebar.error("Falta GEMINI_API_KEY")
if not API_KEY_CLAUDE:
    st.sidebar.warning("Falta CLAUDE_API_KEY (motor Claude desactivado)")
if not ELEVEN_API_KEY:
    st.sidebar.warning("Falta ELEVENLABS_API_KEY (voz desactivada)")

# Selector de voz para directo
voz_modo = st.sidebar.radio("🎙️ Voz:", ["Rápida (directo)", "Calidad"])
MODELO_VOZ = VOZ_RAPIDA if voz_modo.startswith("Rápida") else VOZ_CALIDAD

# Perplexity queda en el código pero apagado por defecto. Actívalo aquí si lo necesitas.
mostrar_pplx = st.sidebar.checkbox("Mostrar Perplexity (opcional)", value=False)

# --- 🎨 APARIENCIA EDITABLE (solo en CONTROL; se aplica antes de pintar el CSS) ---
if modo == "🛠️ CONTROL":
    with st.sidebar.expander("🎨 Apariencia (pantalla)"):
        b = gs.branding
        b["titulo"] = st.text_input("Nombre del programa", b["titulo"])
        _idx = FUENTES.index(b["fuente"]) if b["fuente"] in FUENTES else 0
        b["fuente"] = st.selectbox("Tipografía", FUENTES, index=_idx)
        c1, c2 = st.columns(2)
        b["color_fondo"]        = c1.color_picker("Fondo", b["color_fondo"])
        b["color_titular"]      = c2.color_picker("Titular", b["color_titular"])
        b["color_verificacion"] = c1.color_picker("Verificación", b["color_verificacion"])
        b["color_acento"]       = c2.color_picker("Borde", b["color_acento"])
        b["grosor_borde"]     = st.slider("Grosor del borde", 0, 60, b["grosor_borde"])
        b["tam_titular"]      = st.slider("Tamaño titular", 30, 140, b["tam_titular"])
        b["tam_verificacion"] = st.slider("Tamaño verificación", 18, 80, b["tam_verificacion"])
        b["mayusculas"] = st.checkbox("Titular en MAYÚSCULAS", b["mayusculas"])
        b["alineacion"] = st.radio(
            "Alineación", ["left", "center"],
            index=0 if b["alineacion"] == "left" else 1,
            format_func=lambda x: "Izquierda" if x == "left" else "Centro",
            horizontal=True)
        b["logo_url"]  = st.text_input("Logo (URL de imagen)", b["logo_url"])
        b["tam_logo"]  = st.slider("Tamaño del logo", 40, 320, b["tam_logo"])
        if st.button("↩️ Restaurar diseño original"):
            gs.branding = dict(BRANDING_DEFAULT)
            st.rerun()

    # --- 🛟 COPIA DE SEGURIDAD Y RESTAURACIÓN ---
    with st.sidebar.expander("🛟 Copia de seguridad"):
        _eng = get_engine()
        if _eng:
            st.success("Base de datos conectada: se guarda solo.")
        elif SUPABASE_DB_URL:
            st.error("No se pudo conectar a la base de datos. Revisa SUPABASE_DB_URL.")
        else:
            st.info("Sin base de datos: usa la descarga/restauración manual.")
        st.download_button(
            "💾 Descargar copia (JSON)",
            data=exportar_estado(),
            file_name=f"bcndesperta_{time.strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )
        _subido = st.file_uploader("Restaurar desde copia (.json)", type=["json"],
                                   key="restore_file")
        if _subido is not None and st.button("♻️ Restaurar esta copia",
                                             use_container_width=True):
            try:
                _aplicar_estado(json.loads(_subido.read().decode("utf-8")))
                guardar_estado()
                st.success("Copia restaurada correctamente.")
                st.rerun()
            except Exception as e:
                st.error(f"No se pudo restaurar: {e}")

st.sidebar.caption(f"Modelo IA: {WORKING_MODEL.split('/')[-1]}")

# CSS: el panel de control queda oscuro y legible; la PANTALLA usa toda la apariencia
b = gs.branding
_align_items = "center" if b["alineacion"] == "center" else "flex-start"
_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Montserrat:wght@600;800&display=swap');
    .main { background-color: #000 !important; color: #FFF !important; }
    .stApp { background-color: #000; }
    .box-escena { background: __FONDO__; padding: 60px; border-left: __BORDE__px solid __ACENTO__;
                  height: 100vh; display: flex; flex-direction: column; justify-content: center;
                  align-items: __ALIGNITEMS__; text-align: __ALIGN__; }
    .tit-live { font-family: __FUENTE__; font-size: __TAMTIT__px; font-weight: 700;
                text-transform: __TRANSFORM__; line-height: 1.1; margin-bottom: 20px;
                color: __TITULAR__; }
    .ver-live { font-family: __FUENTE__; font-size: __TAMVER__px; color: __VERIF__;
                font-style: italic; border-top: 2px solid #333; padding-top: 20px;
                white-space: pre-wrap; }
    .card-mem { background: #111; padding: 12px; border: 1px solid #333;
                margin-bottom: 10px; border-radius: 5px; }
    .logo-escena { max-height: __TAMLOGO__px; margin-bottom: 35px; }
    </style>
"""
_css = (_css.replace("__FONDO__", b["color_fondo"])
            .replace("__ACENTO__", b["color_acento"])
            .replace("__BORDE__", str(b["grosor_borde"]))
            .replace("__ALIGNITEMS__", _align_items)
            .replace("__ALIGN__", b["alineacion"])
            .replace("__FUENTE__", f"'{b['fuente']}', Arial, sans-serif")
            .replace("__TAMTIT__", str(b["tam_titular"]))
            .replace("__TRANSFORM__", "uppercase" if b["mayusculas"] else "none")
            .replace("__TITULAR__", b["color_titular"])
            .replace("__TAMVER__", str(b["tam_verificacion"]))
            .replace("__VERIF__", b["color_verificacion"])
            .replace("__TAMLOGO__", str(b["tam_logo"])))
st.markdown(_css, unsafe_allow_html=True)

if modo == "🛠️ CONTROL":
    # Auto-refresco mientras haya algo en curso (análisis o voz), para ver cuándo acaban
    ocupado = any(r.get("estado") == "ANALIZANDO" or r.get("status_voz") == "GENERANDO"
                  for r in gs.cola_resultados)
    if ocupado:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=1500, key="busy_refresh")

    st.title(f"{gs.branding['titulo']} — Control Editorial")

    col_in, col_edit = st.columns([1, 1.3])

    with col_in:
        # --- 🎤 ESCUCHA ACTIVA ---
        st.subheader("🎤 Escucha activa")
        st.caption("Pulsa Escuchar, deja que hable el ponente y pulsa Pausar para que proponga "
                   "titulares. Son sugerencias que se SUMAN a lo que metas a mano.")
        if STT_DISPONIBLE:
            stt_text = speech_to_text(
                language="es",
                start_prompt="▶️ Escuchar",
                stop_prompt="⏸️ Pausar y proponer",
                just_once=True,
                use_container_width=True,
                key="stt_live",
            )
            if stt_text and stt_text != st.session_state.get("ultimo_stt"):
                st.session_state["ultimo_stt"] = stt_text
                gs.transcript_log.insert(0, {"h": time.strftime("%H:%M:%S"), "txt": stt_text})
                with st.spinner("Buscando titulares..."):
                    nuevas = extraer_sugerencias(stt_text)
                for k, s in enumerate(nuevas):
                    s["id"] = time.time() + k
                    gs.sugerencias.insert(0, s)
                st.rerun()

            for s in list(gs.sugerencias):
                st.markdown(
                    f"<div class='card-mem'>💡 <b>{s.get('titular','')}</b><br>"
                    f"<small>{s.get('dato','')}</small></div>",
                    unsafe_allow_html=True,
                )
                cs1, cs2 = st.columns(2)
                if cs1.button("🧠 Verificar", key=f"sug_v_{s['id']}"):
                    lanzar_analisis(f"{s.get('titular','')}. {s.get('dato','')}", "gemini")
                    gs.sugerencias = [x for x in gs.sugerencias if x["id"] != s["id"]]
                    st.rerun()
                if cs2.button("✖️ Descartar", key=f"sug_x_{s['id']}"):
                    gs.sugerencias = [x for x in gs.sugerencias if x["id"] != s["id"]]
                    st.rerun()
        else:
            st.caption("Instala streamlit-mic-recorder y usa Chrome para activar la escucha.")

        st.write("---")
        st.subheader("📥 Memoria")
        if "input_buffer" not in st.session_state:
            st.session_state.input_buffer = ""
        txt_input = st.text_area("Pegar intervención:", height=150,
                                 key="widget_input", value=st.session_state.input_buffer)
        if st.button("💾 ENVIAR A MEMORIA", use_container_width=True):
            if txt_input:
                gs.cola_pendientes.insert(0, {"texto": txt_input,
                                              "h": time.strftime("%H:%M:%S"),
                                              "id": time.time()})
                st.session_state.input_buffer = ""
                st.rerun()

        st.write("---")
        for item in list(gs.cola_pendientes):
            st.markdown(f"<div class='card-mem'><b>[{item['h']}]</b> "
                        f"{item['texto'][:100]}...</div>", unsafe_allow_html=True)
            c_g, c_c = st.columns(2)
            if c_g.button("🧠 GEMINI", key=f"gem_{item['id']}"):
                lanzar_analisis(item["texto"], "gemini", item["h"])
                gs.cola_pendientes = [x for x in gs.cola_pendientes if x["id"] != item["id"]]
                st.rerun()
            if c_c.button("🟣 CLAUDE", key=f"cla_{item['id']}"):
                lanzar_analisis(item["texto"], "claude", item["h"])
                gs.cola_pendientes = [x for x in gs.cola_pendientes if x["id"] != item["id"]]
                st.rerun()
            if mostrar_pplx:
                if st.button("🔍 PERPLEXITY", key=f"pplx_{item['id']}",
                             use_container_width=True):
                    lanzar_analisis(item["texto"], "perplexity", item["h"])
                    gs.cola_pendientes = [x for x in gs.cola_pendientes if x["id"] != item["id"]]
                    st.rerun()

    with col_edit:
        st.subheader("📝 Validación y Reanálisis")
        if st.button("🚨 LIMPIAR ESCENARIO", use_container_width=True):
            gs.en_pantalla = {"t": "", "v": ""}

        for res_item in list(gs.cola_resultados):
            _motor = res_item.get("motor", "?").upper()
            _analizando = res_item.get("estado") == "ANALIZANDO"
            _cab = "⏳ Analizando…" if _analizando else "Resultado"
            with st.expander(f"[{_motor}] {_cab} - {res_item['h']}", expanded=True):
                if _analizando:
                    st.info("Analizando en segundo plano. Puedes seguir trabajando en otros titulares.")
                    if st.button("🗑️ Quitar", key=f"del_{res_item['id']}"):
                        gs.cola_resultados = [x for x in gs.cola_resultados
                                              if x["id"] != res_item["id"]]
                        st.rerun()
                    continue

                txt_edit = st.text_area("Caja Editorial:", value=res_item["res"],
                                        key=f"txt_{res_item['id']}_v{res_item['v']}", height=180)

                feedback = st.text_input("Corrección para reanalizar:",
                                         key=f"feed_{res_item['id']}")
                if st.button("🔄 REPROCESAR E ITERAR", key=f"re_{res_item['id']}"):
                    lanzar_analisis(res_item["orig"], res_item.get("motor", "gemini"),
                                    feedback=feedback, historial=res_item["res"],
                                    item_id=res_item["id"])
                    st.rerun()

                st.write("---")
                c1, c2, c3 = st.columns(3)
                if c1.button("📺 PANTALLA", key=f"pub_{res_item['id']}"):
                    lineas = txt_edit.split("\n")
                    gs.en_pantalla["t"] = lineas[0].strip()
                    gs.en_pantalla["v"] = "\n".join(lineas[1:]).strip()
                    gs.archivo_historico.insert(0, {"t": gs.en_pantalla["t"],
                                                    "h": res_item["h"], "tipo": "PANTALLA"})

                if res_item["status_voz"] == "IDLE":
                    if c2.button("🎙️ VOZ", key=f"tts_{res_item['id']}"):
                        if not ELEVEN_API_KEY:
                            st.error("Falta ELEVENLABS_API_KEY")
                        else:
                            res_item["status_voz"] = "GENERANDO"
                            threading.Thread(target=thread_voz,
                                             args=(txt_edit, res_item["id"], MODELO_VOZ),
                                             daemon=True).start()
                            st.rerun()
                elif res_item["status_voz"] == "GENERANDO":
                    c2.info("🎙️ Generando…")
                elif res_item["status_voz"] == "LISTO":
                    st.audio(res_item["audio"], format="audio/mp3")
                elif res_item["status_voz"] == "ERROR":
                    c2.error("Error de voz")

                if c3.button("🗑️ ARCHIVAR", key=f"arc_{res_item['id']}"):
                    gs.archivo_historico.insert(0, {"t": "ARC: " + txt_edit[:40],
                                                    "h": res_item["h"], "tipo": "ARCH"})
                    gs.cola_resultados = [x for x in gs.cola_resultados
                                          if x["id"] != res_item["id"]]
                    st.rerun()

    st.write("---")
    st.subheader("📂 Registro")
    for h in gs.archivo_historico:
        st.markdown(f"<small>[{h['h']}] {h['tipo']} | {h['t']}</small>",
                    unsafe_allow_html=True)

    # Guarda automáticamente el estado (si hay base de datos configurada)
    guardar_estado()

elif modo == "👂 ESCUCHA":
    st.title("👂 Escucha")
    st.caption("Pulsa START para activar el micrófono. Las sugerencias aparecen en el modo "
               "CONTROL. Mantén esta pestaña abierta en un equipo cerca del audio del acto.")

    if not WEBRTC_DISPONIBLE:
        st.error("Falta instalar streamlit-webrtc y av. Sube el requirements.txt nuevo "
                 "y deja que Streamlit reinstale.")
    elif not ELEVEN_API_KEY:
        st.warning("Falta ELEVENLABS_API_KEY en los Secrets (es la misma clave de la voz).")
    else:
        modo_escucha = st.radio(
            "Modo:",
            ["🎙️ Por ráfagas (ahorra saldo)", "🔴 Continua"],
            help="Por ráfagas: solo transcribe (y gasta) cuando pulsas Capturar. "
                 "Continua: transcribe todo el rato mientras esté abierto.")

        webrtc_ctx = webrtc_streamer(
            key="escucha-webrtc",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        estado = st.empty()
        trans_box = st.empty()

        def _procesar_pcm(pcm):
            """Transcribe un bloque de audio y vuelca sugerencias a CONTROL."""
            if not pcm:
                return ""
            txt = transcribir_elevenlabs(_pcm_a_wav(pcm, 16000))
            if txt:
                gs.transcript_log.insert(0, {"h": time.strftime("%H:%M:%S"), "txt": txt})
                for k, s in enumerate(extraer_sugerencias(txt)):
                    s["id"] = time.time() + k
                    gs.sugerencias.insert(0, s)
                guardar_estado()
            return txt

        if not webrtc_ctx.state.playing:
            estado.info("Pulsa START para empezar. Acepta el micrófono en el navegador (Chrome).")

        # ---------- MODO POR RÁFAGAS ----------
        elif modo_escucha.startswith("🎙️"):
            estado.success("🟢 Micrófono activo. Captura cuando quieras (solo gastas en cada ráfaga).")
            dur = st.slider("Segundos por ráfaga", 5, 30, 15)
            if st.button("🎙️ Capturar ráfaga", use_container_width=True, type="primary"):
                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                pcm = b""
                with st.spinner(f"Grabando {dur}s y transcribiendo…"):
                    # descarta el audio viejo para capturar "desde ahora"
                    try:
                        webrtc_ctx.audio_receiver.get_frames(timeout=0.2)
                    except Exception:
                        pass
                    t0 = time.time()
                    while time.time() - t0 < dur:
                        try:
                            frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                        except Exception:
                            break
                        for f in frames:
                            for rf in resampler.resample(f):
                                pcm += bytes(rf.planes[0])
                    txt = _procesar_pcm(pcm)
                if txt:
                    st.success("Ráfaga procesada. Sugerencias enviadas a CONTROL.")
                    trans_box.markdown(f"**Última transcripción:** {txt}")
                else:
                    st.warning("No se captó audio claro. Acerca el micro y reintenta.")

        # ---------- MODO CONTINUA ----------
        else:
            estado.success("🔴 Escuchando en continuo… (consume saldo mientras esté abierto)")
            if "buf_audio" not in st.session_state:
                st.session_state.buf_audio = b""
                st.session_state.t_ultimo = time.time()
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            while True:
                try:
                    frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                except Exception:
                    break
                for f in frames:
                    for rf in resampler.resample(f):
                        st.session_state.buf_audio += bytes(rf.planes[0])
                if time.time() - st.session_state.t_ultimo > 5 and st.session_state.buf_audio:
                    pcm = st.session_state.buf_audio
                    st.session_state.buf_audio = b""
                    st.session_state.t_ultimo = time.time()
                    txt = _procesar_pcm(pcm)
                    if txt:
                        trans_box.markdown(f"**Transcripción reciente:** {txt}")

else:  # --- MODO ESCENARIO ---
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, key="stage_refresh")
    _logo = gs.branding.get("logo_url", "")
    _logo_html = f'<img src="{_logo}" class="logo-escena">' if _logo else ""
    _t = gs.en_pantalla.get("t") or "&nbsp;"
    _v = gs.en_pantalla.get("v") or "&nbsp;"
    _html = (
        '<div class="box-escena">'
        + _logo_html
        + f'<div class="tit-live">{_t}</div>'
        + f'<div class="ver-live">{_v}</div>'
        + '</div>'
    )
    st.markdown(_html, unsafe_allow_html=True)
