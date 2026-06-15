import os
import re
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

# Modelo de voz para directo. Flash v2.5 ≈ <75ms (ideal en vivo);
# Multilingual v2 = más calidad pero más lento.
VOZ_RAPIDA   = "eleven_flash_v2_5"
VOZ_CALIDAD  = "eleven_multilingual_v2"

# Modelo de Claude. Sonnet 4.6 = rápido y fiable para verificar en directo.
CLAUDE_MODEL = "claude-sonnet-4-6"

# Modelo de Perplexity (opcional). sonar-pro = mejor veracidad.
PERPLEXITY_MODEL = "sonar-pro"


# --- 2. MEMORIA CENTRAL ---
@st.cache_resource
class SharedState:
    def __init__(self):
        self.en_pantalla = {"t": "BCN DESPERTA", "v": "SISTEMA LISTO"}
        self.cola_pendientes = []
        self.cola_resultados = []
        self.archivo_historico = []
        self.sugerencias = []          # titulares propuestos por la IA al escuchar
        self.transcript_log = []       # historial de lo transcrito
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
        "A partir de esta transcripción de lo que se acaba de decir, detecta como máximo 3 "
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
            return [d for d in data if isinstance(d, dict)][:3]
    except Exception:
        pass
    return []


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


# --- HELPER: guardar un resultado en la cola de validación ---
def push_resultado(texto_res, orig, motor, h=None):
    gs.cola_resultados.insert(0, {
        "res": texto_res, "h": h or time.strftime("%H:%M:%S"), "id": time.time(),
        "audio": None, "status_voz": "IDLE", "orig": orig, "v": 0, "motor": motor,
    })


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
st.set_page_config(page_title="BCN DESPERTA PRO", layout="wide")

st.markdown("""
    <style>
    .main { background-color: black !important; color: white !important; }
    .stApp { background-color: black; }
    .box-escena { background: black; padding: 60px; border-left: 25px solid #FF0000;
                  height: 100vh; display: flex; flex-direction: column; justify-content: center; }
    .tit-live { font-size: 80px; font-weight: bold; text-transform: uppercase;
                line-height: 1.1; margin-bottom: 20px; }
    .ver-live { font-size: 38px; color: #FF0000; font-style: italic;
                border-top: 2px solid #333; padding-top: 20px; white-space: pre-wrap; }
    .card-mem { background: #111; padding: 12px; border: 1px solid #333;
                margin-bottom: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

modo = st.sidebar.radio("MODO:", ["🛠️ CONTROL", "📺 ESCENARIO"])

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

st.sidebar.caption(f"Modelo IA: {WORKING_MODEL.split('/')[-1]}")

if modo == "🛠️ CONTROL":
    # Auto-refresco SOLO mientras hay audio generándose (para ver el cambio a LISTO)
    if any(r.get("status_voz") == "GENERANDO" for r in gs.cola_resultados):
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=1500, key="audio_refresh")

    st.title("BCN DESPERTA - Control Editorial")

    col_in, col_edit = st.columns([1, 1.3])

    with col_in:
        # --- 🎤 ESCUCHA EN DIRECTO ---
        st.subheader("🎤 Escucha en directo")
        if STT_DISPONIBLE:
            stt_text = speech_to_text(
                language="es",
                start_prompt="🔴 Escuchar",
                stop_prompt="⏹️ Parar y analizar",
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
                    with st.spinner("Gemini analizando..."):
                        res = consultar_ia(
                            f"{s.get('titular','')}. {s.get('dato','')}"
                        )
                    push_resultado(res, s.get("titular", ""), "gemini")
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
                with st.spinner("Gemini analizando..."):
                    res = consultar_ia(item["texto"])
                push_resultado(res, item["texto"], "gemini", item["h"])
                gs.cola_pendientes = [x for x in gs.cola_pendientes if x["id"] != item["id"]]
                st.rerun()
            if c_c.button("🟣 CLAUDE", key=f"cla_{item['id']}"):
                with st.spinner("Claude verificando en la web..."):
                    res = consultar_claude(item["texto"])
                push_resultado(res, item["texto"], "claude", item["h"])
                gs.cola_pendientes = [x for x in gs.cola_pendientes if x["id"] != item["id"]]
                st.rerun()
            if mostrar_pplx:
                if st.button("🔍 PERPLEXITY", key=f"pplx_{item['id']}",
                             use_container_width=True):
                    with st.spinner("Perplexity buscando..."):
                        res = consultar_perplexity(item["texto"])
                    push_resultado(res, item["texto"], "perplexity", item["h"])
                    gs.cola_pendientes = [x for x in gs.cola_pendientes if x["id"] != item["id"]]
                    st.rerun()

    with col_edit:
        st.subheader("📝 Validación y Reanálisis")
        if st.button("🚨 LIMPIAR ESCENARIO", use_container_width=True):
            gs.en_pantalla = {"t": "", "v": ""}

        for res_item in list(gs.cola_resultados):
            _motor = res_item.get("motor", "?").upper()
            with st.expander(f"[{_motor}] Resultado - {res_item['h']}", expanded=True):
                txt_edit = st.text_area("Caja Editorial:", value=res_item["res"],
                                        key=f"txt_{res_item['id']}_v{res_item['v']}", height=180)

                feedback = st.text_input("Corrección para reanalizar:",
                                         key=f"feed_{res_item['id']}")
                if st.button("🔄 REPROCESAR E ITERAR", key=f"re_{res_item['id']}"):
                    motor = res_item.get("motor", "gemini")
                    with st.spinner("Re-evaluando..."):
                        if motor == "claude":
                            res_item["res"] = consultar_claude(res_item["orig"],
                                                               feedback=feedback,
                                                               historial=res_item["res"])
                        elif motor == "perplexity":
                            res_item["res"] = consultar_perplexity(res_item["orig"])
                        else:
                            res_item["res"] = consultar_ia(res_item["orig"],
                                                           feedback=feedback,
                                                           historial=res_item["res"])
                        res_item["v"] += 1
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

else:  # --- MODO ESCENARIO ---
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, key="stage_refresh")
    st.markdown(f"""
        <div class="box-escena">
            <div class="tit-live">{gs.en_pantalla['t']}</div>
            <div class="ver-live">{gs.en_pantalla['v']}</div>
        </div>
    """, unsafe_allow_html=True)
