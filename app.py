"""
Aplicación Streamlit del proyecto integrador CRISP-DM.
Clasificación de la categoría de discapacidad en personas certificadas de Armenia (Quindío).

Archivos necesarios en la misma carpeta (generados en el cuaderno de Colab):
    - modelo_discapacidad.joblib  (pipeline completo serializado)
    - metadatos_modelo.json       (opciones válidas, métricas y versiones)
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RUTA = Path(__file__).parent
ARCHIVO_MODELO = RUTA / "modelo_discapacidad.joblib"
ARCHIVO_METADATOS = RUTA / "metadatos_modelo.json"

# Desempeño por clase en el conjunto de prueba (cuaderno, sección 5.3).
# Si se reentrena el modelo, estos valores deben actualizarse.
DESEMPENO_POR_CLASE = {
    "Física":      {"precision": 0.294, "recall": 0.315, "f1": 0.304, "casos": 200},
    "Intelectual": {"precision": 0.462, "recall": 0.620, "f1": 0.530, "casos": 208},
    "Múltiple":    {"precision": 0.437, "recall": 0.163, "f1": 0.238, "casos": 337},
    "Psicosocial": {"precision": 0.275, "recall": 0.450, "f1": 0.341, "casos": 111},
    "Sensorial":   {"precision": 0.183, "recall": 0.286, "f1": 0.223, "casos": 98},
}
F1_REFERENCIA_AZAR = 0.207  # F1 macro de la referencia aleatoria estratificada en prueba

ETIQUETAS = {
    "edad": "Edad (años)",
    "vigencia": "Vigencia de la certificación",
    "genero": "Género",
    "comuna": "Comuna de residencia",
    "eps": "EPS",
    "regimen": "Régimen de afiliación",
    "condicion_victima": "Víctima del conflicto armado",
}

COLOR_PRINCIPAL = "#1F5F8B"
COLOR_SECUNDARIO = "#B8C7D3"

st.set_page_config(
    page_title="Categoría de discapacidad en Armenia",
    page_icon="🧭",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Carga del modelo y de los metadatos
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load(ARCHIVO_MODELO)


@st.cache_data
def cargar_metadatos():
    with open(ARCHIVO_METADATOS, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Funciones de apoyo (independientes de la interfaz)
# ---------------------------------------------------------------------------
def normalizar_texto(serie):
    """Aplica la misma limpieza del cuaderno: espacios y etiqueta única para el valor no reportado."""
    serie = serie.astype(str).str.strip()
    return serie.where(serie.str.upper() != "NO REPORTA", "No reporta")


def pct(x):
    """Porcentaje con espacio antes del signo (norma del español)."""
    return f"{x * 100:.0f} %"


def nivel_confianza(p_max, n_clases):
    """Clasifica la probabilidad máxima frente al reparto uniforme entre clases."""
    uniforme = 1 / n_clases
    if p_max < uniforme + 0.10:
        return "baja", (f"La probabilidad más alta ({pct(p_max)}) está cerca del {pct(uniforme)} que tendría "
                        "un reparto uniforme entre las categorías. El perfil no permite distinguir una categoría.")
    if p_max < 0.50:
        return "moderada", (f"La categoría predicha concentra el {pct(p_max)} de la probabilidad. Otras categorías "
                            "conservan probabilidades relevantes y conviene revisarlas.")
    return "relativamente alta", (f"La categoría predicha concentra el {pct(p_max)} de la probabilidad, un valor alto "
                                  "para este modelo, cuyo desempeño general es moderado.")


def predecir(modelo, datos):
    """Devuelve la tabla de entrada con la predicción, la confianza y la probabilidad de cada clase."""
    resultado = datos.copy()
    resultado["prediccion"] = modelo.predict(datos)
    if hasattr(modelo, "predict_proba"):
        prob = pd.DataFrame(modelo.predict_proba(datos), columns=modelo.classes_, index=datos.index)
        resultado["confianza"] = prob.max(axis=1).round(3)
        resultado = pd.concat([resultado, prob.round(3).add_prefix("p_")], axis=1)
    return resultado


def preparar_lote(df, meta):
    """Valida un archivo de registros. Devuelve (datos listos, errores, advertencias)."""
    errores, advertencias = [], []
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    faltantes = [c for c in meta["predictoras"] if c not in df.columns]
    if faltantes:
        errores.append("Faltan columnas obligatorias: " + ", ".join(faltantes) + ".")
        return None, errores, advertencias
    if df.empty:
        errores.append("El archivo no contiene registros.")
        return None, errores, advertencias

    datos = df[meta["predictoras"]].copy()

    # Variables numéricas: texto vacío o 999 se tratan como no reportado (el pipeline imputa la mediana)
    for col in meta["variables_numericas"]:
        valores = pd.to_numeric(datos[col], errors="coerce")
        invalidos = valores.isna() & datos[col].notna() & (datos[col].astype(str).str.strip() != "")
        if invalidos.any():
            advertencias.append(f"{ETIQUETAS.get(col, col)}: {int(invalidos.sum())} valor(es) no numérico(s) "
                                "se trataron como no reportados.")
        if col == "edad":
            valores = valores.where(valores != 999)
        datos[col] = valores
    if datos["vigencia"].isna().any():
        errores.append("La vigencia es obligatoria y debe ser un año (por ejemplo, 2026).")

    # Variables categóricas: valores no vistos en el entrenamiento se agrupan como infrecuentes
    for col in meta["variables_categoricas"]:
        datos[col] = normalizar_texto(datos[col])
        nuevos = sorted(set(datos[col]) - set(meta["opciones"][col]))
        if nuevos:
            advertencias.append(f"{ETIQUETAS.get(col, col)}: valores no vistos en el entrenamiento "
                                f"({', '.join(nuevos[:5])}{'...' if len(nuevos) > 5 else ''}). "
                                "El modelo los trata como categoría infrecuente.")
    return datos, errores, advertencias


def grafico_probabilidades(probabilidades, predicha):
    tabla = pd.DataFrame({"categoria": list(probabilidades.keys()),
                          "probabilidad": list(probabilidades.values())})
    tabla["predicha"] = np.where(tabla["categoria"] == predicha, "Predicha", "Otras")
    uniforme = 1 / len(tabla)

    barras = alt.Chart(tabla).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("probabilidad:Q", title="Probabilidad", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, max(0.7, tabla["probabilidad"].max() + 0.05)])),
        y=alt.Y("categoria:N", title=None, sort="-x"),
        color=alt.Color("predicha:N", legend=None,
                        scale=alt.Scale(domain=["Predicha", "Otras"], range=[COLOR_PRINCIPAL, COLOR_SECUNDARIO])),
        tooltip=[alt.Tooltip("categoria:N", title="Categoría"),
                 alt.Tooltip("probabilidad:Q", title="Probabilidad", format=".1%")],
    )
    etiquetas = barras.mark_text(align="left", dx=4).encode(text=alt.Text("probabilidad:Q", format=".0%"),
                                                           color=alt.value("#2B3A46"))
    regla = alt.Chart(pd.DataFrame({"x": [uniforme]})).mark_rule(strokeDash=[4, 4], color="#7A8894").encode(x="x:Q")
    texto_regla = alt.Chart(pd.DataFrame({"x": [uniforme], "t": ["Reparto uniforme"]})).mark_text(
        align="left", dx=4, dy=-8, color="#7A8894", fontSize=11).encode(x="x:Q", y=alt.value(0), text="t:N")
    return (barras + etiquetas + regla + texto_regla).properties(height=230)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
if not ARCHIVO_MODELO.exists() or not ARCHIVO_METADATOS.exists():
    st.error("No se encontraron los archivos del modelo. La carpeta de la aplicación debe contener "
             "modelo_discapacidad.joblib y metadatos_modelo.json, descargados desde el cuaderno de Colab.")
    st.stop()

modelo = cargar_modelo()
meta = cargar_metadatos()
clases = meta["clases"]
vigencias = meta["vigencias_entrenamiento"]

st.title("Clasificación de la categoría de discapacidad")
st.markdown(
    "Personas con discapacidad certificadas en Armenia (Quindío), 2021 a 2025. El modelo estima la categoría "
    "de discapacidad más probable a partir del perfil sociodemográfico y de aseguramiento en salud."
)
st.info(
    "Herramienta de apoyo a la planeación poblacional. No sustituye la valoración del equipo "
    "multidisciplinario de certificación ni debe usarse para decisiones sobre personas concretas.",
    icon="ℹ️",
)

pestana_individual, pestana_lote, pestana_modelo = st.tabs(
    ["Predicción individual", "Predicción por lote", "Sobre el modelo"])

# --- Predicción individual -------------------------------------------------
with pestana_individual:
    col_form, col_resultado = st.columns([1, 1.3], gap="large")

    with col_form:
        with st.form("formulario"):
            st.subheader("Perfil de la persona")
            edad_no_reportada = st.checkbox("Edad no reportada")
            edad = st.number_input(ETIQUETAS["edad"], min_value=0, max_value=int(meta["rango_edad"][1]),
                                   value=35, step=1, disabled=edad_no_reportada)
            opciones_vigencia = list(range(min(vigencias), max(vigencias) + 2))
            vigencia = st.selectbox(ETIQUETAS["vigencia"], opciones_vigencia,
                                    index=len(opciones_vigencia) - 1)
            entradas = {}
            for col in meta["variables_categoricas"]:
                opciones = meta["opciones"][col]
                entradas[col] = st.selectbox(ETIQUETAS.get(col, col), opciones)
            enviado = st.form_submit_button("Estimar categoría", type="primary", width="stretch")

    with col_resultado:
        if not enviado:
            st.subheader("Resultado")
            st.write("El resultado aparece al completar el perfil y presionar **Estimar categoría**: la categoría "
                     "más probable y la probabilidad de cada una.")
        else:
            registro = pd.DataFrame([{
                "edad": np.nan if edad_no_reportada else edad,
                "vigencia": vigencia,
                **entradas,
            }])[meta["predictoras"]]
            fila = predecir(modelo, registro).iloc[0]
            predicha = fila["prediccion"]

            st.subheader("Resultado")
            st.metric("Categoría más probable", predicha)

            if "confianza" in fila:
                probabilidades = {c: float(fila[f"p_{c}"]) for c in clases}
                nivel, explicacion = nivel_confianza(fila["confianza"], len(clases))
                st.altair_chart(grafico_probabilidades(probabilidades, predicha), width="stretch")
                st.markdown(f"**Confianza {nivel}.** {explicacion}")

            if predicha in DESEMPENO_POR_CLASE:
                d = DESEMPENO_POR_CLASE[predicha]
                st.caption(f"En el conjunto de prueba, cuando el modelo predijo \"{predicha}\" acertó en el "
                           f"{pct(d['precision'])} de los casos, y reconoció el {pct(d['recall'])} de las personas "
                           f"que realmente pertenecían a esa categoría.")
            if edad_no_reportada:
                st.caption("Sin edad reportada, el modelo usa la mediana de edad del entrenamiento.")
            if vigencia > max(vigencias):
                st.caption(f"El modelo se entrenó con vigencias {min(vigencias)} a {max(vigencias)}. Para "
                           f"{vigencia} supone que la tendencia temporal observada se mantiene.")

# --- Predicción por lote ---------------------------------------------------
with pestana_lote:
    st.subheader("Predicción para varios registros")
    st.write(
        "El archivo CSV debe tener una fila por persona y las columnas "
        + ", ".join(f"`{c}`" for c in meta["predictoras"])
        + ". La edad puede quedar vacía o con 999 cuando no se reporta. La plantilla incluye un ejemplo."
    )
    plantilla = pd.DataFrame([{c: (35 if c == "edad" else max(vigencias) + 1 if c == "vigencia"
                                   else meta["opciones"][c][0]) for c in meta["predictoras"]}])
    st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8-sig"),
                       file_name="plantilla_lote.csv", mime="text/csv")

    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if archivo is not None:
        try:
            df_lote = pd.read_csv(archivo, sep=None, engine="python", encoding="utf-8-sig")
        except Exception as e:  # archivo ilegible
            st.error(f"No fue posible leer el archivo como CSV ({e}). Conviene verificar el separador y la codificación.")
            df_lote = None

        if df_lote is not None:
            datos, errores, advertencias = preparar_lote(df_lote, meta)
            for e in errores:
                st.error(e)
            for a in advertencias:
                st.warning(a)
            if not errores:
                resultado = predecir(modelo, datos)
                st.success(f"Predicción completada para {len(resultado)} registros.")
                resumen = resultado["prediccion"].value_counts().rename_axis("categoria").reset_index(name="registros")
                c1, c2 = st.columns([1, 2])
                c1.dataframe(resumen, hide_index=True, width="stretch")
                c2.altair_chart(alt.Chart(resumen).mark_bar(color=COLOR_PRINCIPAL).encode(
                    x=alt.X("registros:Q", title="Registros"),
                    y=alt.Y("categoria:N", title=None, sort="-x")).properties(height=200),
                    width="stretch")
                st.dataframe(resultado, width="stretch")
                st.download_button("Descargar resultados", resultado.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="predicciones_discapacidad.csv", mime="text/csv")

# --- Sobre el modelo -------------------------------------------------------
with pestana_modelo:
    m = meta["metricas_prueba"]
    st.subheader(f"Modelo: {meta['modelo']}")
    st.write("Métricas en el conjunto de prueba (30 % de los registros, no usado durante el entrenamiento).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 macro", f"{m['f1_macro']:.3f}", help="Métrica principal: promedia el F1 de las cinco "
              "categorías con igual peso.")
    c2.metric("Accuracy", f"{m['accuracy']:.3f}")
    c3.metric("Precision macro", f"{m['precision_macro']:.3f}")
    c4.metric("Recall macro", f"{m['recall_macro']:.3f}")
    st.write(f"La referencia aleatoria obtiene un F1 macro de {F1_REFERENCIA_AZAR:.3f}; el modelo la supera de "
             "forma estadísticamente significativa.")

    st.markdown("**Desempeño por categoría**")
    st.dataframe(pd.DataFrame(DESEMPENO_POR_CLASE).T.rename_axis("Categoría").reset_index().rename(columns={
        "precision": "Precision", "recall": "Recall", "f1": "F1", "casos": "Casos en prueba"}),
        hide_index=True, width="stretch")

    st.markdown("**Limitaciones**")
    st.markdown(
        "- Las variables describen el perfil sociodemográfico y de aseguramiento; no incluyen información "
        "clínica, que es la que determina la categoría de discapacidad.\n"
        "- El conjunto incluye solo personas certificadas; la población con discapacidad sin certificar queda fuera.\n"
        "- La discapacidad múltiple tiene un recall bajo: el modelo reconoce pocos de esos casos.\n"
        "- El tipo de documento se excluyó por redundancia con la edad; con ello, el modelo no distingue a la "
        "población migrante.\n"
        "- Las predicciones para vigencias posteriores a 2025 suponen que la tendencia temporal se mantiene."
    )

    with st.expander("Detalles técnicos"):
        st.write("Predictoras:", ", ".join(meta["predictoras"]))
        st.write("Hiperparámetros:", meta["hiperparametros"])
        st.write("Versiones de entrenamiento:", meta["versiones"])
    st.caption("Fuente de datos: Personas con discapacidad certificadas en Armenia, portal Datos Abiertos "
               "Colombia (datos.gov.co). Proyecto integrador con metodología CRISP-DM.")
