# -*- coding: utf-8 -*-
"""
ChatBot Caso de Estudio (Seguros) - Streamlit
Incluye:
- Parser robusto de fechas (parse_effective_to_date)
- Fix "truth value ambiguous" (coalesce_pandas)
- Fix "too many values to unpack" (iterrows en coberturas)
- FAQ dinámico con cifras
- Filtros en sidebar que SOLO afectan 3 gráficas interactivas
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import io, requests


# ------------------------------
# Configuración / Carga de archivo (vía RAW GitHub o uploader)
# ------------------------------

# URL RAW del archivo en GitHub (no la página HTML)
EXCEL_URL = "https://github.com/wgherrerar19/Caso-de-Estudio-Marketing-Analityc/blob/main/casodeestudio.xlsx"

@st.cache_data(show_spinner=False)
def load_excel_from_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content))  # requiere openpyxl en requirements

df_raw = None

# 1) Intentar por URL RAW
try:
    df_raw = load_excel_from_url(EXCEL_URL)
    origen = "URL RAW de GitHub"
except Exception as e:
    st.warning(f"No se pudo leer desde la URL RAW: {e}")

# 2) Fallback: uploader manual si falla la URL
if df_raw is None:
    up = st.file_uploader("📥 Sube 'casodeestudio.xlsx'", type=["xlsx", "xls"])
    if up is not None:
        try:
            df_raw = pd.read_excel(up)
            origen = "archivo subido por el usuario"
        except Exception as e:
            st.error(f"No pude leer el archivo subido: {e}")

# 3) Validación final
if df_raw is not None:
    st.success(f"✅ Base cargada correctamente desde {origen}")
    st.dataframe(df_raw.astype(str), use_container_width=True)
else:
    st.error("⚠️ No se pudo cargar 'casodeestudio.xlsx'. Verifica la URL RAW o sube el archivo.")
    st.stop()

# ------------------------------
# Helpers de formato y parsing
# ------------------------------
def money(x, currency="$", decimals=0):
    try:
        return f"{currency}{x:,.{decimals}f}"
    except Exception:
        return str(x)

def pct(x, decimals=1):
    try:
        return f"{x*100:.{decimals}f}%"
    except Exception:
        return str(x)

def coalesce_pandas(x, fallback):
    """Devuelve fallback si x es None o está vacío (Series/DataFrame)."""
    if x is None:
        return fallback
    if isinstance(x, (pd.DataFrame, pd.Series)):
        try:
            if x.empty:
                return fallback
        except Exception:
            return fallback
    return x

def parse_effective_to_date(series: pd.Series) -> pd.Series:
    """
    Parser robusto para fechas con formatos mixtos:
    1) Números de Excel (origen 1899-12-30)
    2) Formatos explícitos comunes
    3) Fallback genérico
    """
    s = series.copy()

    # 1) Números Excel -> fecha
    num = pd.to_numeric(s, errors="coerce")
    dt = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")

    # 2) Intentos vectorizados con formatos explícitos
    s_str = s.astype(str).str.strip()
    fmt_list = ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%Y/%m/%d"]
    for fmt in fmt_list:
        mask = dt.isna()
        if not mask.any():
            break
        dt.loc[mask] = pd.to_datetime(s_str.loc[mask], format=fmt, errors="coerce")

    # 3) Fallback genérico
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(s_str.loc[mask], errors="coerce")

    return dt

# ------------------------------
# Limpieza principal
# ------------------------------
def prepare_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [c.strip() for c in df.columns]

    # Fechas
    if "Effective To Date" in df.columns:
        df["Effective To Date"] = parse_effective_to_date(df["Effective To Date"])
        df["Effective_Month"] = df["Effective To Date"].dt.to_period("M").astype(str)

    # Numéricos clave
    for c in [
        "Customer Lifetime Value", "Income", "Monthly Premium Auto",
        "Months Since Last Claim", "Months Since Policy Inception",
        "Number of Open Complaints", "Number of Policies", "Total Claim Amount"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Categóricos frecuentes
    for c in ["Coverage", "Policy Type", "Policy", "Sales Channel",
              "Vehicle Class", "Vehicle Size", "Response", "Renew Offer Type", "State"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df

df = prepare_df(df_raw)

# ==========================================================
#  Sidebar (SOLO afecta las 3 gráficas)
# ==========================================================
st.sidebar.header("🔍 Filtros (solo gráficas)")
state_sel = st.sidebar.multiselect(
    "Estado",
    options=sorted(df["State"].dropna().unique()) if "State" in df.columns else []
)
channel_sel = st.sidebar.multiselect(
    "Canal de Venta",
    options=sorted(df["Sales Channel"].dropna().unique()) if "Sales Channel" in df.columns else []
)
month_sel = st.sidebar.multiselect(
    "Mes de Vigencia",
    options=sorted(df["Effective_Month"].dropna().unique()) if "Effective_Month" in df.columns else []
)
vehicle_sel = st.sidebar.multiselect(
    "Clase de Vehículo",
    options=sorted(df["Vehicle Class"].dropna().unique()) if "Vehicle Class" in df.columns else []
)

def df_for_charts(base: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros SOLO para las gráficas."""
    dfx = base.copy()
    if "State" in dfx.columns and state_sel:
        dfx = dfx[dfx["State"].isin(state_sel)]
    if "Sales Channel" in dfx.columns and channel_sel:
        dfx = dfx[dfx["Sales Channel"].isin(channel_sel)]
    if "Effective_Month" in dfx.columns and month_sel:
        dfx = dfx[dfx["Effective_Month"].isin(month_sel)]
    if "Vehicle Class" in dfx.columns and vehicle_sel:
        dfx = dfx[dfx["Vehicle Class"].isin(vehicle_sel)]
    return dfx

# ------------------------------
# Guardar conversaciones (apendea en el mismo Excel)
# ------------------------------
def save_interaction(user_msg, bot_response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame({
        "timestamp": [timestamp],
        "usuario": [user_msg],
        "bot": [bot_response]
    })
    if os.path.exists(EXCEL_FILE):
        try:
            df_existing = pd.read_excel(EXCEL_FILE)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_final = df_new
    else:
        df_final = df_new
    df_final.to_excel(EXCEL_FILE, index=False)

# ------------------------------
# Métricas auxiliares
# ------------------------------
def _top_margin_by(df_in, group_col, top_k=3):
    """Proxy de margen = sum(primas) - sum(reclamos)."""
    if group_col not in df_in.columns:
        return None
    tmp = df_in.groupby(group_col, dropna=False).agg(
        primas=("Monthly Premium Auto", "sum"),
        reclamos=("Total Claim Amount", "sum"),
        n=("Customer", "count")
    )
    tmp["margen"] = tmp["primas"] - tmp["reclamos"]
    tmp = tmp.sort_values("margen", ascending=False)
    return tmp.head(top_k)

def _acceptance_by_offer(df_in):
    """Tasa de aceptación por tipo de oferta de renovación."""
    if "Renew Offer Type" not in df_in.columns or "Response" not in df_in.columns:
        return None
    tmp = df_in.assign(accepted=(df_in["Response"].astype(str).str.strip().str.lower() == "yes").astype(int))
    acc = tmp.groupby("Renew Offer Type", observed=True)["accepted"].mean().sort_values(ascending=False)
    return acc

# ------------------------------
# Helper para formatear líneas (coberturas)
# ------------------------------
def _format_top_margin_lines(top: pd.DataFrame) -> str:
    if not isinstance(top, pd.DataFrame) or top.empty:
        return "Sin datos suficientes."
    lines = []
    for idx, row in top.iterrows():  # iterrows devuelve (index, Series)
        n_val = int(row["n"]) if pd.notna(row["n"]) else 0
        lines.append(
            f"- {idx}: margen {money(row['margen'],'$',0)} "
            f"(primas {money(row['primas'],'$',0)}, reclamos {money(row['reclamos'],'$',0)}, n={n_val})"
        )
    return "\n".join(lines)

# ------------------------------
# FAQ dinámico (generadores por intent) - con cifras
# ------------------------------
faq_generators = {
    "coberturas": lambda df_in: (
        (lambda top:
            "🛡️ **Coberturas con mejor margen estimado (prima - reclamos)**:\n"
            + _format_top_margin_lines(top)
            + "\n\n**Oportunidad:** Si *Basic* domina con margen bajo, revisar deducibles y precios; "
              "si *Premium* muestra margen negativo, evaluar condiciones y segmentación."
        )(coalesce_pandas(_top_margin_by(df_in, "Coverage"), pd.DataFrame()))
    ),

    "vigencia": lambda df_in: (
        (lambda vc:
            "📅 **Altas por mes (últimos 6):** "
            + (", ".join([f"{k}: {int(v)}" for k, v in vc.tail(6).items()]) if not vc.empty else "Sin datos.")
            + "\n**Oportunidad:** Programar campañas en meses de baja alta y retención proactiva 30 días antes de renovación."
        )(df_in["Effective_Month"].value_counts().sort_index() if "Effective_Month" in df_in.columns else pd.Series(dtype=int))
    ),

    "pago_mensual": lambda df_in: (
        (lambda g:
            "💳 **Prima mensual promedio por cobertura**:\n"
            + ("\n".join([f"- {idx}: {money(val,'$',0)}" for idx, val in g.items()]) if not g.empty else "Sin datos.")
            + "\n**Oportunidad:** Clientes con prima alta y quejas/reclamos elevados: ajustar precio/deducible; "
              "con prima baja y buen CLV: ofrecer add-ons."
        )(df_in.groupby("Coverage")["Monthly Premium Auto"].mean().sort_values(ascending=False).round(0)
          if "Monthly Premium Auto" in df_in.columns and "Coverage" in df_in.columns else pd.Series(dtype=float))
    ),

    "clv": lambda df_in: (
        (lambda g:
            "📈 **CLV promedio por tipo de póliza**:\n"
            + ("\n".join([f"- {idx}: {money(val,'$',0)}" for idx, val in g.items()]) if not g.empty else "Sin datos.")
            + "\n**Oportunidad:** Priorizar fidelización y beneficios en segmentos con mayor CLV; "
              "en bajo CLV con alta siniestralidad, optimizar condiciones."
        )(df_in.groupby("Policy Type")["Customer Lifetime Value"].mean().sort_values(ascending=False).round(0)
          if "Customer Lifetime Value" in df_in.columns and "Policy Type" in df_in.columns else pd.Series(dtype=float))
    ),

    "num_polizas": lambda df_in: (
        (lambda counts:
            "📦 **Clientes por número de pólizas (top valores):** "
            + (", ".join([f"{int(k)} pól.: {int(v)}" for k, v in counts.items()]) if not counts.empty else "Sin datos.")
            + "\n**Oportunidad:** Detectar clientes con 1 póliza y buen CLV para campañas de cross-sell/bundling."
        )(df_in["Number of Policies"].value_counts().sort_index().head(10)
          if "Number of Policies" in df_in.columns else pd.Series(dtype=int))
    ),

    "reclamos": lambda df_in: (
        (lambda t:
            "🧾 **Top estados por reclamos (suma)**:\n"
            + ("\n".join([f"- {idx}: {money(val,'$',0)}" for idx, val in t.head(5).items()]) if not t.empty else "Sin datos.")
            + "\n**Oportunidad:** Ajustar deducibles/precio y programas de mitigación de riesgo en estados con alta severidad."
        )(df_in.groupby("State")["Total Claim Amount"].sum().sort_values(ascending=False).round(0)
          if "Total Claim Amount" in df_in.columns and "State" in df_in.columns else pd.Series(dtype=float))
    ),

    "quejas": lambda df_in: (
        (lambda t:
            "❗ **Quejas abiertas promedio por canal**:\n"
            + ("\n".join([f"- {idx}: {val:.3f}" for idx, val in t.items()]) if not t.empty else "Sin datos.")
            + "\n**Oportunidad:** Reducir TAT, mejorar scripts/capacitación en canales con mayor queja promedio."
        )(df_in.groupby("Sales Channel")["Number of Open Complaints"].mean().sort_values(ascending=False).round(3)
          if "Sales Channel" in df_in.columns and "Number of Open Complaints" in df_in.columns else pd.Series(dtype=float))
    ),

    "canales": lambda df_in: (
        (lambda t:
            "🏪 **Distribución por canal (n de clientes)**:\n"
            + ("\n".join([f"- {idx}: {int(val)}" for idx, val in t.items()]) if not t.empty else "Sin datos.")
            + "\n**Oportunidad:** Reforzar inversión/capacitación en canales con mayor conversión y menor queja/siniestralidad."
        )(df_in["Sales Channel"].value_counts() if "Sales Channel" in df_in.columns else pd.Series(dtype=int))
    ),

    "vehiculo": lambda df_in: (
        (lambda t:
            "🚗 **Prima promedio por clase de vehículo (top 5)**:\n"
            + ("\n".join([f"- {idx}: {money(val,'$',0)}" for idx, val in t.items()]) if not t.empty else "Sin datos.")
            + "\n**Oportunidad:** Ajustar tarifas/deducibles en clases con prima alta y reclamos elevados; "
              "diseñar coberturas específicas por segmento."
        )(df_in.groupby("Vehicle Class")["Monthly Premium Auto"].mean().sort_values(ascending=False).head(5).round(0)
          if "Monthly Premium Auto" in df_in.columns and "Vehicle Class" in df_in.columns else pd.Series(dtype=float))
    ),

    "renovacion": lambda df_in: (
        (lambda acc:
            "🔁 **Tasa de aceptación por tipo de oferta**:\n"
            + ("\n".join([f"- {idx}: {pct(val,1)}" for idx, val in acc.items()]) if not acc.empty else "Sin datos.")
            + "\n**Oportunidad:** Personalizar oferta por CLV/quejas y testear beneficios (asistencia, deducible) donde la aceptación es baja."
        )(coalesce_pandas(_acceptance_by_offer(df_in), pd.Series(dtype=float)))
    ),
}

# ------------------------------
# Training phrases (orientadas a oportunidades)
# ------------------------------
training_phrases = {
    "coberturas": [
        "oportunidades por tipo de cobertura",
        "cómo mejorar el mix de coberturas",
        "migrar clientes a premium o extended",
        "qué cobertura genera mejor margen"
    ],
    "vigencia": [
        "picos de altas por mes",
        "qué meses conviene hacer campañas",
        "oportunidades de retención por vigencia",
        "cohortes por fecha de inicio"
    ],
    "pago_mensual": [
        "clientes con prima alta para ajuste",
        "dónde ofrecer add-ons por prima",
        "asequibilidad de la prima",
        "mejorar pricing mensual"
    ],
    "clv": [
        "segmentación por CLV",
        "priorizar clientes de alto valor",
        "oportunidades de up-sell por CLV",
        "cómo mejorar el CLV"
    ],
    "num_polizas": [
        "oportunidades de cross sell",
        "clientes con una sola póliza",
        "bundling de productos",
        "aumentar share of wallet"
    ],
    "reclamos": [
        "hotspots de siniestros",
        "dónde bajar severidad de reclamos",
        "frecuencia y severidad por estado",
        "ajustes de deducible y precio"
    ],
    "quejas": [
        "reducir quejas abiertas",
        "mejorar experiencia por canal",
        "oportunidades para bajar TAT",
        "priorizar acciones de servicio"
    ],
    "canales": [
        "qué canal vende mejor con buen margen",
        "canales con mejor conversión",
        "dónde invertir en ventas",
        "desempeño por canal"
    ],
    "vehiculo": [
        "pricing por clase de vehículo",
        "segmentos de mayor riesgo",
        "oportunidades por tamaño del vehículo",
        "ajustar tarifas por vehículo"
    ],
    "renovacion": [
        "mejor oferta de renovación",
        "tasa de aceptación por oferta",
        "A/B test de renovación",
        "cómo subir la renovación"
    ]
}

# ------------------------------
# Entrenamiento NLP (TF-IDF + KNN)
# ------------------------------
X, y = [], []
for intent, phrases in training_phrases.items():
    for phrase in phrases:
        X.append(phrase)
        y.append(intent)

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X_vec)

def predict_intent(user_input: str):
    user_vec = vectorizer.transform([user_input])
    dist, idx = model.kneighbors(user_vec)
    intent = y[idx[0][0]]
    confidence = 1 - dist[0][0]
    return intent if confidence >= 0.5 else None

# ==========================================================
# DASHBOARD: 3 GRÁFICAS (interactivas con filtros)
# ==========================================================
def draw_dashboard(df_filtered: pd.DataFrame):
    st.header("📊 Panel (3 gráficas) — Segmentado por filtros")

    if df_filtered.empty:
        st.info("No hay datos con los filtros seleccionados.")
        return

    # 1) Prima mensual promedio por cobertura
    st.subheader("1) Prima mensual promedio por cobertura")
    if "Monthly Premium Auto" in df_filtered.columns and "Coverage" in df_filtered.columns:
        g1 = (
            df_filtered.dropna(subset=["Monthly Premium Auto"])
            .groupby("Coverage", observed=True)["Monthly Premium Auto"]
            .mean().sort_values(ascending=False).round(0)
            .rename("Prima promedio").to_frame()
        )
        st.bar_chart(g1)
    else:
        st.info("No se encontraron columnas 'Monthly Premium Auto' y/o 'Coverage'.")

    # 2) Total de reclamos por estado (Top 10)
    st.subheader("2) Total de reclamos por estado (Top 5)")
    if "Total Claim Amount" in df_filtered.columns and "State" in df_filtered.columns:
        g2 = (
            df_filtered.dropna(subset=["Total Claim Amount"])
            .groupby("State", observed=True)["Total Claim Amount"]
            .sum().sort_values(ascending=False).head(10).round(0)
            .rename("Reclamos").to_frame()
        )
        st.bar_chart(g2)
    else:
        st.info("No se encontraron columnas 'Total Claim Amount' y/o 'State'.")

    # 3) Tasa de aceptación por tipo de oferta de renovación
    st.subheader("3) Tasa de aceptación por tipo de oferta de renovación")
    if "Response" in df_filtered.columns and "Renew Offer Type" in df_filtered.columns:
        temp = df_filtered.copy()
        temp["accepted"] = np.where(temp["Response"].astype(str).str.strip().str.lower() == "yes", 1, 0)
        g3 = (
            temp.groupby("Renew Offer Type", observed=True)["accepted"]
            .mean().sort_values(ascending=False)
            .rename("Tasa aceptación").to_frame()
        )
        st.bar_chart(g3)
    else:
        st.info("No se encontraron columnas 'Response' y/o 'Renew Offer Type'.")

# ------------------------------
# Interfaz Streamlit
# ------------------------------
st.title("🚗 ChatBot Caso de Estudio (Seguros)")
st.write("Los **filtros de la izquierda** modifican **solo las 3 gráficas**. Las respuestas de texto usan toda la base.")

# Historial
if "history" not in st.session_state:
    st.session_state["history"] = []

# Entrada
user_input = st.text_input("👤 ¿Qué deseas preguntar?")

if user_input:
    intent = predict_intent(user_input)
    if intent:
        # Respuesta de texto con la base completa (sin filtros)
        response = faq_generators[intent](df)
    else:
        response = "❓ No entendí tu consulta, por favor intenta con otra formulación."

    # Guardar y mostrar
    save_interaction(user_input, response)
    st.session_state["history"].append((user_input, response))

# Mostrar historial
for user_msg, bot_msg in st.session_state["history"]:
    st.markdown(f"👤 **Tú:** {user_msg}")
    st.markdown(f"🤖 **Bot:** {bot_msg}")
    st.divider()

# Mostrar las 3 gráficas con filtros aplicados
df_charts = df_for_charts(df)
draw_dashboard(df_charts)


# ----------------------- 
# ==============================
# Ejecutar en la Terminal

# cd  "C:\Users\Sala_\Downloads"
# py -m streamlit run chatbot_streamlit_casodeestudio.py
