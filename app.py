"""App de Streamlit: pronóstico de la demanda eléctrica horaria de Bangladesh con 24 h de anticipación.

Solo carga los artefactos de model/ y data/ (generados por train_model.py y los scripts prepare_*.py) y predice.
Uso: streamlit run app.py
"""
import os
from functools import partial

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from components import (LIGHT, bar_chart, callout, caption, cards, fold_chart, heading, inject_styles,
                        pdf_button, presentation_controls, presentation_styles, references, series_chart, table, theme_tokens,
                        toc)
from references import REFERENCES
from train_model import FEATURES, add_time_features

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "demand_mw"
MENDELEY_URL = "https://doi.org/10.17632/vpk8spw2mm.1"

st.set_page_config(page_title="Demanda eléctrica de Bangladesh — pronóstico a 24 h", layout="centered")

tokens = theme_tokens()
inject_styles(tokens)

SECTIONS = [
    ("introduccion", "Introducción"),
    ("inspeccion", "Inspección inicial"),
    ("objetivo", "Variable objetivo y fuga de información"),
    ("particion", "Limpieza, rezagos y partición"),
    ("eda", "Análisis exploratorio (Train)"),
    ("pipeline", "Preprocesamiento"),
    ("cv", "Validación cruzada y persistencia"),
    ("lineales", "Modelos lineales y polinomial"),
    ("kernels", "Comparación de kernels"),
    ("optimizacion", "Optimización de C y γ"),
    ("test", "Evaluación final en Test"),
    ("ensambles", "Ensambles: metodología"),
    ("ensambles_cv", "Ensambles: validación cruzada"),
    ("ensambles_test", "Ensambles: Test"),
    ("nuevos", "Validación con datos nuevos"),
    ("discusion", "Discusión y estado del arte"),
    ("conclusiones", "Conclusiones"),
    ("referencias", "Referencias"),
]
LABELS = {
    TARGET: "Real", "prediction": "SVR RBF", "prediction_persistence": "Persistencia 24 h",
    "prediction_stacking": "Stacking", "prediction_linear": "Regresión lineal", "prediction_ridge": "Ridge",
    "prediction_poly": "Polinomial", "prediction_svr_linear": "SVR lineal", "prediction_gb": "Gradient Boosting",
    "prediction_ada": "AdaBoost", "prediction_voting_simple": "Voting simple", "prediction_voting": "Voting ponderado",
    "prediction_voting_hard": "Voting hard",
}
heading = partial(heading, sections=SECTIONS)
series_chart = partial(series_chart, labels=LABELS)


def metrics(y_true, y_pred):
    """RMSE, MAE y MAPE en MW y %, y R²; el MAPE es interpretable porque la demanda nunca es cero."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }


def residuals_by_hour(frame, column):
    """Residuo medio (real − pronóstico) y RMSE por hora del día."""
    error = frame[TARGET] - frame[column]
    hours = frame["datetime"].dt.hour
    return pd.DataFrame({"Residuo medio": error.groupby(hours).mean(),
                         "RMSE": error.groupby(hours).apply(lambda e: np.sqrt((e ** 2).mean()))})


@st.cache_resource
def load_artifact(name):
    return joblib.load(os.path.join(HERE, "model", name))


def predict_all(df, train_median):
    """Agrega una columna por modelo. Los modelos base del Voting dan también las predicciones de GB y AdaBoost."""
    features = df[FEATURES]
    linear, voting = load_artifact("linear.joblib"), load_artifact("voting.joblib")
    base = voting["simple"].transform(features)  # columnas: SVR RBF, Gradient Boosting, AdaBoost
    return df.assign(
        prediction=load_artifact("svr_rbf.joblib")["model"].predict(features),
        prediction_persistence=df["demand_lag_24h"].fillna(train_median),
        prediction_stacking=load_artifact("stacking.joblib")["model"].predict(features),
        prediction_linear=linear["linear"].predict(features),
        prediction_ridge=linear["ridge"].predict(features),
        prediction_poly=linear["poly"].predict(features),
        prediction_svr_linear=load_artifact("svr_linear.joblib")["model"].predict(features),
        prediction_gb=base[:, 1],
        prediction_ada=base[:, 2],
        prediction_voting_simple=base.mean(axis=1),
        prediction_voting=voting["weighted"].predict(features),
        # Hard voting en regresión: mediana de las predicciones de los tres modelos base
        prediction_voting_hard=np.median(base, axis=1),
    )


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(HERE, "data", "study_period.csv"), parse_dates=["datetime"])
    train_median = float(df.loc[df["split"] == "train", TARGET].median())
    return predict_all(df, train_median), train_median


@st.cache_data
def load_new_data(train_median):
    # Horas posteriores al periodo de estudio (jun 2025 – mar 2026); se generan con prepare_new_data.py
    df = pd.read_csv(os.path.join(HERE, "data", "new_period.csv"), parse_dates=["datetime"])
    return predict_all(df, train_median)


MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
          "noviembre", "diciembre"]
DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def month_year(ts):
    return f"{MONTHS[ts.month - 1]} de {ts.year}"


svr_artifact = load_artifact("svr_rbf.joblib")
model, params = svr_artifact["model"], svr_artifact["params"]
stacking = load_artifact("stacking.joblib")["model"]
linear_artifact = load_artifact("linear.joblib")
voting_artifact = load_artifact("voting.joblib")
df, train_median = load_data()
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
y_true = test[TARGET]
svr_test = metrics(y_true, test["prediction"])
persistence_test = metrics(y_true, test["prediction_persistence"])

with st.sidebar:
    slide = presentation_controls(SECTIONS)
    if slide is None:
        toc(SECTIONS, "Reporte técnico")
    else:
        presentation_styles(SECTIONS, slide)
        st.caption("Se aplica a la pestaña Reporte técnico.")
    st.divider()
    pdf_button("Descargar PDF", tokens)
    st.caption("Abre la pestaña Reporte técnico y elige «Guardar como PDF» en el diálogo de impresión: portada y una "
               "sección por página, en horizontal.")

tab_product, tab_report = st.tabs(["Predicción", "Reporte técnico"])

with tab_report:
    st.html("""
    <header>
      <h1 class="nb-title">Pronosticando la demanda eléctrica de Bangladesh con un día de anticipación</h1>
      <p class="nb-description">Modelos lineales, Support Vector Regression y métodos de ensamble sobre la demanda de días
      anteriores, el calendario y el clima para pronosticar los MW que requerirá la red nacional en cada hora.</p>
      <p class="nb-meta"><span>Andrea Rodríguez · 395699</span><span>Universidad Autónoma de Chihuahua</span><span>Machine Learning</span></p>
      <ul class="nb-tags"><li>machine learning</li><li>series de tiempo</li><li>SVR</li><li>ensambles</li><li>scikit-learn</li></ul>
    </header>
    """)

    heading("introduccion")
    st.markdown(
        "Cuando la demanda supera a la generación disponible, la Power Grid Company of Bangladesh (PGCB) desconecta "
        "carga de forma programada: el **load shedding**. Anticipar la demanda del día siguiente es el insumo básico para "
        "planificar qué centrales despachar, cuánto combustible e importaciones asegurar y cómo distribuir los cortes. "
        "El objetivo es pronosticar la demanda de cada hora (`demand_mw`, MW) **con 24 h de anticipación**, usando solo "
        "información disponible el día anterior: un problema de **regresión** supervisada.\n\n"
        "Flujo: Datos → Inspección → Variable objetivo y fuga → Rezagos y partición → EDA solo en Train → Pipeline → "
        "Validación cruzada con persistencia → Modelos lineales → SVR → Ensambles → Test → Datos nuevos."
    )
    callout(
        "result", "Resultado",
        f"El SVR RBF optimizado obtiene en Test un MAPE de <mark>{svr_test['mape']:.2f}%</mark> (RMSE de "
        f"{svr_test['rmse']:,.0f} MW), frente a {persistence_test['mape']:.2f}% de la persistencia de 24 h: "
        f"{1 - svr_test['rmse'] / persistence_test['rmse']:.0%} menos error. También fue el mejor en validación "
        "cruzada (RMSE de 940.61 MW) y mantiene su desempeño en datos posteriores al periodo de estudio.",
    )

    heading("inspeccion")
    RAW_MISSING = [("solar", 22133, 23.9), ("wind", 73974, 79.8), ("india_adani", 85312, 92.1),
                   ("nepal", 87299, 94.2), ("remarks", 86257, 93.1)]
    st.markdown(
        "- **Dataset:** PGCB Hourly Generation Dataset (UCI): 92,650 registros horarios y 15 columnas, de 2015-04 a "
        "2025-06. Se complementa con clima horario de Open-Meteo (promedio de las 8 capitales de división).\n"
        "- **Objetivo:** `demand_mw`, sin faltantes.\n"
        "- **Duplicados:** 160 registros completos y 432 marcas de tiempo repetidas.\n"
        "- **Valores imposibles:** `demand_mw` = 156,050 MW o `generation_mw` = 64,526,500 MW, muy por encima de la "
        "capacidad de la red (< 17,000 MW)."
    )
    table(["Variable", "Faltantes", "%"], RAW_MISSING,
          note="Valores faltantes en el histórico completo: fuentes que empezaron a registrarse tarde. Ninguna es "
               "predictor del modelo de demanda.")

    heading("objetivo")
    with_shedding = df[df["load_shedding"] >= 50]
    ratio = (with_shedding[TARGET] - with_shedding["generation_mw"]) / with_shedding["load_shedding"]
    ratio_by_year = ratio.groupby(with_shedding["datetime"].dt.year).median()
    st.markdown(
        "La demanda de PGCB **no es una medición independiente**. En el 84.8% de los registros del histórico completo "
        "se cumple `demand_mw = generation_mw + load_shedding` (±1 MW). En las horas con cortes, desde 2023 la "
        "diferencia entre demanda y generación equivale a un factor casi constante sobre el load shedding:"
    )
    table(["Año", "(demanda − generación) / load shedding", "Horas"],
          [(str(year), f"{value:.3f}", int((with_shedding["datetime"].dt.year == year).sum()))
           for year, value in ratio_by_year.items()],
          note="Mediana en las horas del periodo de estudio con al menos 50 MW de load shedding.")
    callout(
        "warning", "Fuga de información",
        "Con la generación o el load shedding de la misma hora, un modelo solo tendría que sumarlos. Por eso se "
        "excluyen todas las variables medidas en la misma hora: generación, load shedding, generación por fuente, "
        "importaciones y <code>remarks</code>. Solo se usa información disponible el día anterior.",
    )
    callout(
        "tip", "¿Por qué la demanda y no el load shedding?",
        "Primero se intentó predecir el load shedding. Con variables de la misma hora, el modelo solo reconstruía la "
        "identidad contable (<i>nowcasting</i>); con 24 h de anticipación, el mejor modelo apenas mejoró a la persistencia "
        "en validación (279 frente a 294 MW) y no la superó en prueba. La demanda, en cambio, sigue patrones de "
        "calendario y clima que sí pueden anticiparse, y es la variable que reporta la literatura.",
    )

    heading("particion")
    st.markdown(
        "Reglas fijas antes de la partición (no se calculan con los datos, así que no generan fuga de información):\n"
        "- Eliminación de duplicados y marcas de tiempo repetidas, y orden cronológico.\n"
        "- Eliminación de 168 registros con valores físicamente imposibles (límites de capacidad por columna).\n"
        "- **Rezagos:** `demand_lag_24h` (misma hora del día anterior) y `demand_lag_168h` (misma hora de la semana "
        "anterior), calculados sobre la serie horaria completa y solo con valores del pasado.\n"
        "- **Periodo de estudio:** últimos 3 años, que reflejan el nivel de demanda y la operación más recientes.\n\n"
        f"Partición **cronológica** 80/20 (`shuffle=False`): Train con {len(train):,} registros "
        f"({train['datetime'].min():%Y-%m-%d} → {train['datetime'].max():%Y-%m-%d}) y Test con {len(test):,} "
        f"({test['datetime'].min():%Y-%m-%d} → {test['datetime'].max():%Y-%m-%d})."
    )
    context = (
        df.set_index("datetime")[[TARGET, "generation_mw"]].resample("D").mean()
        .rename(columns={TARGET: "Demanda", "generation_mw": "Generación"})
        .reset_index().melt(id_vars="datetime", var_name="Serie", value_name="MW")
    )
    lines = alt.Chart(context).mark_line(strokeWidth=1).encode(
        x=alt.X("datetime:T", title=None), y=alt.Y("MW:Q", title="MW (promedio diario)", scale=alt.Scale(zero=False)),
        color=alt.Color("Serie:N", legend=alt.Legend(orient="top", title=None)),
    )
    split_rule = alt.Chart(pd.DataFrame({"datetime": [test["datetime"].min()]})).mark_rule(strokeDash=[4, 4]).encode(x="datetime:T")
    st.altair_chart((lines + split_rule).properties(height=300), width="stretch")
    caption("Demanda y generación en el periodo de estudio (promedios diarios). La línea punteada marca el inicio de "
            "Test. La brecha entre ambas es el load shedding.")
    callout(
        "warning", "Tendencia y estacionalidad",
        "La demanda tiene una estacionalidad anual marcada (máximo en verano, mínimo en invierno) y una tendencia de "
        "crecimiento. Los modelos que no extrapolan, como los árboles, quedan en desventaja frente a esa tendencia.",
    )

    heading("eda")
    missing_lags = train[["demand_lag_24h", "demand_lag_168h"]].isna().sum()
    counts, edges = np.histogram(train[TARGET], bins=40)
    hist = pd.DataFrame({"MW": edges[:-1].round(0).astype(int), "Frecuencia": counts})
    st.altair_chart(
        alt.Chart(hist).mark_bar().encode(x=alt.X("MW:Q", title="demand_mw (MW)"), y="Frecuencia:Q").properties(height=220),
        width="stretch",
    )
    caption(
        f"Distribución de la demanda en Train: media de {train[TARGET].mean():,.1f} MW, desviación estándar de "
        f"{train[TARGET].std():,.1f} MW y mediana de {train[TARGET].median():,.1f} MW. Es aproximadamente simétrica y no "
        f"tiene ceros, por lo que el MAPE es interpretable. Hay {(train[TARGET] < 5000).sum()} horas con demanda menor "
        "de 5,000 MW (registros anómalos), ninguna en Test."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        by_hour = train.groupby("Hour")[TARGET].mean().reset_index()
        bar_chart(by_hour, "Hour", TARGET, "Hora del día", "MW promedio", zero=False)
    with col_b:
        by_month = train.groupby("Month")[TARGET].mean().reset_index()
        bar_chart(by_month, "Month", TARGET, "Mes", "MW promedio", zero=False)
    caption(f"Demanda promedio por hora y por mes (Train): mínimo a las {by_hour.loc[by_hour[TARGET].idxmin(), 'Hour']:02d}:00 "
            f"y máximo a las {by_hour.loc[by_hour[TARGET].idxmax(), 'Hour']:02d}:00; mínimo en "
            f"{MONTHS[by_month.loc[by_month[TARGET].idxmin(), 'Month'] - 1]} y máximo en "
            f"{MONTHS[by_month.loc[by_month[TARGET].idxmax(), 'Month'] - 1]}. Las relaciones no son monótonas, por lo que "
            "la hora, el mes y el día de la semana se codifican con one-hot.")

    sample = train.dropna(subset=["demand_lag_24h"]).sample(4000, random_state=42)
    st.altair_chart(
        alt.Chart(sample).mark_circle(size=8, opacity=0.3).encode(
            x=alt.X("demand_lag_24h:Q", title="Demanda 24 h antes (MW)", scale=alt.Scale(zero=False)),
            y=alt.Y(f"{TARGET}:Q", title="Demanda (MW)", scale=alt.Scale(zero=False)),
        ).properties(height=260),
        width="stretch",
    )
    caption(f"Demanda frente a la demanda de la misma hora del día anterior (muestra de 4,000 horas de Train). Relación "
            "casi lineal: los puntos alejados de la diagonal son cambios bruscos entre días (feriados, lluvias) y "
            f"registros anómalos. Solo {int(missing_lags.max())} horas de Train no tienen rezago y se imputan con la mediana.")

    numeric = [c for c in FEATURES if c not in ("datetime", "Hour", "Month", "DayOfWeek")]
    corr = train[numeric].corrwith(train[TARGET]).sort_values(key=abs, ascending=False)
    corr_frame = corr.rename("r").reset_index().rename(columns={"index": "Variable"})
    st.altair_chart(
        alt.Chart(corr_frame).mark_bar().encode(
            x=alt.X("r:Q", title="Correlación de Pearson con demand_mw"),
            y=alt.Y("Variable:N", sort=None, title=None),
            tooltip=["Variable", alt.Tooltip("r:Q", format=".3f")],
        ).properties(height=320),
        width="stretch",
    )
    caption("Correlación con la demanda en Train. El rezago de 24 h es el predictor más informativo, seguido de la "
            "sensación térmica: el uso de ventiladores y aire acondicionado depende del calor percibido. No se aplica un "
            "umbral de correlación, porque la hora y el día de la semana tienen relaciones no monótonas.")

    WEATHER_LABELS = {"apparent_temperature": "Sensación térmica", "temperature_2m": "Temperatura",
                      "relative_humidity_2m": "Humedad relativa", "cloud_cover": "Nubosidad", "precipitation": "Precipitación"}

    @st.cache_data
    def load_city_correlations(train_hours, train_target):
        """Correlación de Pearson entre el clima de cada capital y la demanda en Train (prepare_city_weather.py)."""
        weather = pd.read_csv(os.path.join(HERE, "data", "weather_by_city.csv"), parse_dates=["time"])
        target = pd.Series(train_target.to_numpy(), index=train_hours.dt.floor("h").to_numpy(), name=TARGET)
        rows = []
        for (city, lat, lon), group in weather.groupby(["city", "latitude", "longitude"]):
            merged = group.set_index("time").join(target, how="inner")
            rows.append({"Ciudad": city, "lat": lat, "lon": lon,
                         **{var: merged[var].corr(merged[TARGET]) for var in WEATHER_LABELS}})
        return pd.DataFrame(rows)

    st.markdown(
        "**Variables exógenas: clima de las 8 capitales de división.** Las variables de clima del modelo son el promedio "
        "simple, hora por hora, de las 8 capitales (Open-Meteo). El mapa muestra la correlación de cada ciudad con la "
        "demanda nacional en Train."
    )
    city_corr = load_city_correlations(train["datetime"], train[TARGET])
    map_var = st.segmented_control("Variable", list(WEATHER_LABELS), default="apparent_temperature",
                                   format_func=WEATHER_LABELS.get, key="map_var") or "apparent_temperature"
    national_r = float(train[map_var].corr(train[TARGET]))
    r_min, r_max = city_corr[map_var].min(), city_corr[map_var].max()
    accent = tuple(int(tokens["accent"][i:i + 2], 16) for i in (1, 3, 5))
    soft = tuple(int(tokens["mark"][i:i + 2], 16) for i in (1, 3, 5))
    span = (r_max - r_min) or 1
    city_map = city_corr.assign(
        r=city_corr[map_var].round(3),
        color=[[int(s + (a - s) * (r - r_min) / span) for s, a in zip(soft, accent)] + [230] for r in city_corr[map_var]],
        radius=12000 + 30000 * (city_corr[map_var].abs() / max(city_corr[map_var].abs().max(), 1e-9)),
    )[["Ciudad", "lat", "lon", "r", "color", "radius"]]
    st.pydeck_chart(
        pdk.Deck(
            layers=[
                pdk.Layer("ScatterplotLayer", city_map, get_position=["lon", "lat"], get_fill_color="color",
                          get_radius="radius", get_line_color=[40, 30, 60, 200], line_width_min_pixels=1, stroked=True,
                          pickable=True),
                pdk.Layer("TextLayer", city_map, get_position=["lon", "lat"], get_text="Ciudad", get_size=13,
                          get_color=[110, 100, 130, 255] if tokens is LIGHT else [220, 215, 235, 255],
                          get_pixel_offset=[0, -22]),
            ],
            initial_view_state=pdk.ViewState(latitude=23.8, longitude=90.3, zoom=5.6),
            tooltip={"text": "{Ciudad}\nr = {r}"},
            map_style=None,
        ),
        height=420,
    )
    caption(f"Correlación de {WEATHER_LABELS[map_var].lower()} con la demanda (Train) en cada capital: de "
            f"{r_min:.3f} a {r_max:.3f}. El promedio nacional que usa el modelo tiene r = {national_r:.3f}. "
            "El color y el tamaño del círculo indican la magnitud de la correlación.")

    heat = city_corr.melt(id_vars=["Ciudad", "lat", "lon"], var_name="var", value_name="r")
    heat = pd.concat([heat, pd.DataFrame({"Ciudad": "Promedio nacional", "var": list(WEATHER_LABELS),
                                          "r": [train[v].corr(train[TARGET]) for v in WEATHER_LABELS]})])
    heat["Variable"] = heat["var"].map(WEATHER_LABELS)
    city_order = ["Promedio nacional", *city_corr.sort_values(map_var, ascending=False)["Ciudad"]]
    heat_base = alt.Chart(heat).encode(
        x=alt.X("Variable:N", sort=list(WEATHER_LABELS.values()), title=None,
                axis=alt.Axis(labelAngle=0, orient="top", labelOverlap=False, labelExpr="split(datum.label, ' ')")),
        y=alt.Y("Ciudad:N", sort=city_order, title=None, axis=alt.Axis(labelLimit=200)),
    )
    st.altair_chart(
        (heat_base.mark_rect().encode(
            color=alt.Color("r:Q", scale=alt.Scale(scheme="purpleorange", domain=[-0.8, 0.8], reverse=True),
                            legend=alt.Legend(title="r")),
            tooltip=["Ciudad", "Variable", alt.Tooltip("r:Q", format=".3f")],
        ) + heat_base.mark_text(fontSize=11).encode(
            text=alt.Text("r:Q", format=".2f"),
            color=alt.condition(alt.datum.r > 0.5, alt.value("white"), alt.value("black")),
        )).properties(height=300),
        width="stretch",
    )
    best = city_corr.loc[city_corr["apparent_temperature"].idxmax()]
    caption("Correlación de cada variable de clima con la demanda en Train, por ciudad y para el promedio nacional. "
            f"La sensación térmica es la más correlacionada (máximo en {best['Ciudad']}, r = "
            f"{best['apparent_temperature']:.3f}). El promedio nacional queda dentro del rango de las ciudades: promediar "
            "no pierde señal y evita depender del clima de una sola ciudad.")

    heading("pipeline")
    numeric_features = svr_artifact["numeric_features"]
    st.markdown(
        "Decisiones tomadas a partir del EDA:\n"
        "- **Predictores:** rezagos de 24 y 168 h, calendario y clima de la hora pronosticada. El clima es el observado: "
        "en operación vendría de un pronóstico meteorológico, así que se supone un pronóstico perfecto.\n"
        "- **Imputación:** mediana para las numéricas (solo faltan rezagos en ~0.4% de las horas).\n"
        "- **Escalamiento:** `StandardScaler`, obligatorio para SVR: los kernels dependen de distancias.\n"
        "- **Codificación:** one-hot para `Hour`, `Month` y `DayOfWeek`.\n"
        "- **Objetivo:** en SVR, `y` se estandariza con `TransformedTargetRegressor` para que `epsilon` y `C` estén en una "
        "escala razonable. Las métricas se reportan en MW.\n\n"
        "Todas las transformaciones se ajustan **solo** con la parte de entrenamiento de cada fold."
    )
    

    heading("cv")
    st.markdown(
        "`TimeSeriesSplit(n_splits=5, gap=24)`: ventana creciente; cada fold se entrena solo con horas anteriores y "
        "valida con el bloque siguiente. El `gap` de 24 registros evita que la autocorrelación entre horas contiguas "
        "infle el desempeño. La referencia principal es la **persistencia de 24 h**: predecir que la demanda será igual "
        "a la de la misma hora del día anterior. Un modelo solo es útil si la supera."
    )
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=5, gap=24).split(train), start=1):
        fold_dates = train["datetime"].iloc
        fold_rows.append((f"Fold {fold}", f"{fold_dates[train_idx[0]]:%Y-%m-%d} → {fold_dates[train_idx[-1]]:%Y-%m-%d}",
                          len(train_idx), f"{fold_dates[val_idx[0]]:%Y-%m-%d} → {fold_dates[val_idx[-1]]:%Y-%m-%d}",
                          len(val_idx)))
    table(["Fold", "Entrenamiento", "Registros", "Validación", "Registros"], fold_rows)
    REFERENCE_CV = [
        ("Dummy (media)", 2694.50, 286.88, 2306.09, -1.247),
        ("Persistencia 24 h", 1153.40, 341.92, 710.68, 0.629),
    ]
    table(["Referencia", "RMSE", "RMSE std", "MAE", "R²"], REFERENCE_CV, best=1,
          note="Promedio de los 5 folds de validación. Métricas en MW. La persistencia es una referencia exigente: "
               "explica 63% de la varianza sin entrenar nada.")

    heading("lineales")
    # Resultados de LoadSheddingDemandaAllModels.ipynb (mismo preprocesamiento y mismos folds)
    LINEAR_CV = [
        ("Persistencia 24 h", 1153.40, 341.92, 710.68, 0.629, "—"),
        ("LinearRegression", 1388.95, 810.90, 1037.63, -0.029, 831.75),
        ("Ridge (α=1)", 1372.84, 782.13, 1024.20, 0.014, 831.78),
        ("Ridge (α=1000)", 1005.35, 288.07, 730.05, 0.702, 873.59),
        ("Polinomial grado 2 (α=1)", 1315.42, 816.81, 957.26, 0.030, 713.69),
        ("Polinomial grado 2 (α=10)", 1234.22, 671.96, 923.39, 0.231, 716.58),
    ]
    LINEAR_FOLD_RMSE = {
        "Persistencia 24 h": [787.2, 1605.0, 827.6, 1509.6, 1037.6],
        "LinearRegression": [2923.4, 1380.6, 636.8, 1186.3, 817.6],
        "Ridge (α=1000)": [865.1, 1444.5, 635.8, 1218.6, 862.7],
        "Polinomial (α=10)": [2463.8, 1369.4, 548.5, 1010.1, 779.3],
    }
    RIDGE_GRID = [(0.001, 1388.51), (0.01, 1387.90), (0.1, 1386.41), (1, 1372.84), (10, 1294.23), (100, 1162.71),
                  (1000, 1005.35), (10000, 1360.03)]
    POLY_GRID_TOP = [(2, 10, 1234.22, 716.58), (2, 1, 1315.42, 713.69), (2, 1000, 1319.77, 768.24),
                     (2, 100, 1403.43, 724.29), (3, 1000, 2765.55, 696.98)]
    st.markdown(
        "Modelos lineales con el mismo `preprocessor` y los mismos folds, sin escalar `y`:\n"
        "- **LinearRegression:** minimiza solo la suma de errores al cuadrado.\n"
        "- **Ridge:** añade la penalización L2 α·Σβ², que estabiliza los coeficientes de predictores correlacionados "
        "(temperatura y sensación térmica, r = 0.97; los dos rezagos). α se optimizó con `GridSearchCV`.\n"
        "- **Regresión polinomial:** sigue siendo lineal en los parámetros. `PolynomialFeatures` amplía las numéricas "
        "escaladas con cuadrados e interacciones (p. ej., temperatura × demanda del día anterior) y se ajusta un Ridge."
    )
    table(["Modelo", "RMSE", "RMSE std", "MAE", "R²", "RMSE Train"], LINEAR_CV, best=3,
          note="Promedio de los 5 folds de validación. Métricas en MW.")
    col_a, col_b = st.columns(2)
    with col_a:
        fold_chart(LINEAR_FOLD_RMSE, height=260)
    with col_b:
        ridge_grid = pd.DataFrame(RIDGE_GRID, columns=["alpha", "RMSE"])
        st.altair_chart(
            alt.Chart(ridge_grid).mark_line(point=True).encode(
                x=alt.X("alpha:Q", scale=alt.Scale(type="log"), title="α (escala log)"),
                y=alt.Y("RMSE:Q", scale=alt.Scale(zero=False), title="RMSE CV (MW)"),
                tooltip=["alpha", alt.Tooltip("RMSE:Q", format=",.2f")],
            ).properties(height=260),
            width="stretch",
        )
    caption("Izquierda: RMSE por fold. Las versiones sin regularización fallan en el fold 1, entrenado con solo cinco "
            "meses. Derecha: RMSE CV de Ridge según α; el mínimo está en α = 1000.")
    table(["Grado", "α", "RMSE val", "RMSE Train"], POLY_GRID_TOP, best=0,
          note="Regresión polinomial: mejores combinaciones de grado × α en <code>GridSearchCV</code>. El grado 3 "
               "sobreajusta de forma extrema (RMSE de validación de hasta 13,158 MW).")
    ridge_test, poly_test = metrics(y_true, test["prediction_ridge"]), metrics(y_true, test["prediction_poly"])
    l1, l2, l3 = st.columns(3)
    l1.metric("MAPE Test Ridge", f"{ridge_test['mape']:.2f}%")
    l2.metric("MAPE Test Polinomial", f"{poly_test['mape']:.2f}%")
    l3.metric("MAPE Test Persistencia", f"{persistence_test['mape']:.2f}%")
    series_chart(test.set_index("datetime")[[TARGET, "prediction_persistence", "prediction_ridge", "prediction_poly"]]
                 .resample("D").mean(), "MW (promedio diario)")
    caption(f"Demanda real frente a la persistencia, Ridge (α = {linear_artifact['ridge_alpha']}) y la regresión "
            f"polinomial (grado {linear_artifact['poly_params']['degree']}, α = {linear_artifact['poly_params']['alpha']}) "
            "en Test (promedios diarios).")
    st.markdown(
        "- **La persistencia gana a todos los lineales sin regularizar:** la regresión lineal tiene R² ≈ 0 en validación "
        "porque en el fold 1 extrapola mal al invierno siguiente (RMSE de 2,923 MW frente a 787 MW de la persistencia).\n"
        "- **Ridge con α = 1000 es la mayor mejora por optimización (−27%)** y ya supera a la persistencia en CV.\n"
        "- **La regresión polinomial ayuda poco:** el grado 2 mejora a la regresión lineal, pero no a Ridge, y el grado 3 "
        "sobreajusta."
    )
    callout(
        "tip", "Motivación para SVR",
        "La demanda responde de forma no lineal al calor y a la hora del día. Un kernel RBF modela esas relaciones sin "
        "elegir a mano qué términos crear, con una respuesta acotada lejos de los datos de entrenamiento.",
    )

    heading("kernels")
    KERNEL_CV = [
        ("Persistencia 24 h", 1153.40, 341.92, 710.68, 0.629, "—"),
        ("SVR Lineal", 1363.54, 793.42, 978.74, 0.020, 864.69),
        ("SVR Polinomial", 1726.57, 1549.29, 1269.73, -1.377, 576.25),
        ("SVR RBF", 1044.34, 381.10, 755.38, 0.608, 567.54),
    ]
    KERNEL_FOLD_RMSE = {
        "Persistencia 24 h": [787.2, 1605.0, 827.6, 1509.6, 1037.6],
        "SVR Lineal": [2839.2, 1403.7, 593.8, 1211.4, 769.6],
        "SVR Polinomial": [4749.9, 1550.2, 548.9, 1049.1, 734.7],
        "SVR RBF": [1394.5, 1532.7, 535.4, 1040.5, 718.6],
    }
    st.markdown("Los tres kernels con parámetros por defecto (`C = 1`, `epsilon = 0.1`, `gamma = \"scale\"`, "
                "`degree = 3`) frente a la persistencia de 24 h.")
    table(["Modelo", "RMSE", "RMSE std", "MAE", "R²", "RMSE Train"], KERNEL_CV, best=3,
          note="Promedio de los 5 folds de validación. Métricas en MW.")
    fold_chart(KERNEL_FOLD_RMSE)
    caption("RMSE por fold de cada kernel y de la persistencia.")
    st.markdown(
        "- **Lineal:** como la regresión lineal, falla en el fold 1 (2,839 MW) y queda peor que la persistencia.\n"
        "- **Polinomial:** el peor modelo; en el fold 1 su error es seis veces el de la persistencia.\n"
        "- **RBF:** el único que supera a la persistencia con sus parámetros por defecto; es el kernel que se optimiza."
    )

    heading("optimizacion")
    GRID_GAMMAS = ["scale", "0.001", "0.01", "0.1", "1", "10"]
    GRID_RMSE = {
        "0.01": [1344.4, 1906.6, 1180.3, 1434.1, 2692.1, 2725.1],
        "0.1": [1098.4, 1033.4, 940.6, 1159.5, 2445.2, 2716.9],
        "1": [1044.3, 1008.0, 969.7, 1099.6, 2012.3, 2669.6],
        "10": [1113.2, 1146.7, 1007.4, 1160.7, 1999.9, 2634.1],
        "100": [1227.9, 1254.4, 1410.0, 1277.4, 2000.3, 2634.1],
    }
    st.markdown(
        "`GridSearchCV` con los mismos folds y RMSE como métrica. `C` controla el compromiso entre regularización y "
        "errores fuera del tubo ε; `gamma` define el alcance del kernel exp(−γ‖x − x'‖²)."
    )
    grid = pd.DataFrame(GRID_RMSE, index=GRID_GAMMAS).reset_index(names="gamma").melt(id_vars="gamma", var_name="C", value_name="RMSE")
    base = alt.Chart(grid).encode(
        x=alt.X("gamma:O", sort=GRID_GAMMAS, title="gamma", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("C:O", sort=list(GRID_RMSE), title="C"),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color("RMSE:Q", scale=alt.Scale(scheme="purples", reverse=True), legend=alt.Legend(title="RMSE")),
        tooltip=["C", "gamma", alt.Tooltip("RMSE:Q", format=",.1f")],
    )
    labels = base.mark_text(fontSize=11).encode(
        text=alt.Text("RMSE:Q", format=",.0f"),
        color=alt.condition(alt.datum.RMSE < 1300, alt.value("white"), alt.value("black")),
    )
    st.altair_chart((heatmap + labels).properties(height=260), width="stretch")
    caption("RMSE de validación (MW) para cada combinación de C y gamma.")
    n_support = model.regressor_.named_steps["model"].support_.size
    callout(
        "result", "Mejor configuración",
        f"<mark>C = {params['C']}, gamma = {params['gamma']}</mark>: RMSE CV de 940.61 ± 293.45 MW, MAE de 644.60 MW y "
        f"R² de 0.734, 10% menos que el RBF por defecto y 18% menos que la persistencia. El modelo reentrenado en todo "
        f"Train usa {n_support:,} vectores de soporte ({n_support / len(train):.1%} de Train).",
    )
    st.markdown(
        "- **γ:** con γ ≥ 1 el kernel se vuelve demasiado local y el error supera 2,000 MW (sobreajuste).\n"
        "- **C:** el mínimo está en C = 0.1–1; con C = 100 el error vuelve a subir. Una regularización fuerte generaliza "
        "mejor entre periodos, porque el rezago de 24 h ya explica la mayor parte de la señal.\n"
        "- **Interacción:** un γ pequeño necesita un C moderado, por eso se buscan juntos."
    )

    heading("test")
    st.markdown("Primer uso de Test. El Pipeline aplica la imputación, el escalamiento y la codificación aprendidos "
                "con Train.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE Test", f"{svr_test['rmse']:,.0f} MW", f"{svr_test['rmse'] - persistence_test['rmse']:+,.0f} vs persistencia",
              delta_color="inverse")
    c2.metric("MAE Test", f"{svr_test['mae']:,.0f} MW")
    c3.metric("MAPE Test", f"{svr_test['mape']:.2f}%", f"{svr_test['mape'] - persistence_test['mape']:+.2f} pp",
              delta_color="inverse")
    c4.metric("R² Test", f"{svr_test['r2']:.3f}")

    split_choice = st.segmented_control("Periodo", ["Test", "Train", "Todo"], default="Test", label_visibility="collapsed")
    subset = {"Test": test, "Train": train, "Todo": df}[split_choice or "Test"]
    series_chart(subset.set_index("datetime")[[TARGET, "prediction"]].resample("D").mean(), "MW (promedio diario)")
    caption(f"Demanda real frente al pronóstico del SVR ({split_choice or 'Test'}, promedios diarios).")

    by_hour = residuals_by_hour(test, "prediction")
    bars = by_hour.reset_index(names="Hora").melt(id_vars="Hora", var_name="Métrica", value_name="MW")
    st.altair_chart(
        alt.Chart(bars).mark_bar().encode(
            x=alt.X("Hora:O", axis=alt.Axis(labelAngle=0)), xOffset="Métrica:N", y=alt.Y("MW:Q"),
            color=alt.Color("Métrica:N", legend=alt.Legend(orient="top", title=None)),
            tooltip=["Hora", "Métrica", alt.Tooltip("MW:Q", format=",.1f")],
        ).properties(height=240),
        width="stretch",
    )
    residual = float((y_true - test["prediction"]).mean())
    caption(f"RMSE y residuo medio (real − pronóstico) por hora en Test. El residuo es positivo en todas las horas "
            f"({residual:,.0f} MW en promedio, {residual / y_true.mean():.1%} de la demanda media): el modelo subestima "
            "ligeramente, sobre todo por la mañana, coherente con la tendencia de crecimiento de la demanda.")

    st.markdown("**Detalle por día**")
    dates = sorted(df["datetime"].dt.date.unique())
    day = st.date_input("Día", value=test["datetime"].dt.date.iloc[-1], min_value=dates[0], max_value=dates[-1])
    hourly = df[df["datetime"].dt.date == day]
    if hourly.empty:
        st.info("No hay registros para ese día.")
    else:
        series_chart(hourly.set_index("datetime")[[TARGET, "prediction", "prediction_persistence"]], "MW")
        caption(f"Demanda hora por hora el {day:%d/%m/%Y}, con el pronóstico del SVR y la persistencia de 24 h.")
        with st.expander("Ver datos del día"):
            st.dataframe(
                hourly[["datetime", "demand_lag_24h", "temperature_2m", "apparent_temperature", TARGET, "prediction"]]
                .rename(columns={"prediction": "SVR"}).round(1),
                hide_index=True, width="stretch",
            )

    heading("ensambles")
    voting_weights = voting_artifact["weights"]
    ADA_GRID_TOP = [
        (6, 1.0, "linear", 50, 1127.52, 691.08),
        (6, 0.1, "linear", 50, 1128.67, 693.58),
        (6, 0.1, "square", 50, 1149.64, 710.08),
        (6, 0.1, "linear", 200, 1154.93, 678.01),
        (3, 0.1, "linear", 200, 1350.57, 1050.45),
    ]
    GB_GRID_TOP = [
        (0.05, 5, 100, 1.0, 1075.12, 543.38),
        (0.05, 5, 100, 0.8, 1082.51, 538.60),
        (0.1, 5, 100, 1.0, 1116.61, 454.44),
        (0.05, 5, 300, 1.0, 1125.41, 393.65),
        (0.05, 3, 100, 1.0, 1135.30, 696.73),
    ]
    st.markdown(
        "Se reutilizan la partición, el `preprocessor` y los 5 folds, para comparar directamente con el SVR. Los árboles "
        "no necesitan escalamiento, pero `StandardScaler` no cambia el orden de los valores y no les afecta.\n"
        "- **AdaBoost** (AdaBoost.R2): entrena árboles de forma secuencial y aumenta el peso de las muestras con mayor "
        "error. La predicción es la mediana ponderada de los árboles.\n"
        "- **Gradient Boosting:** F<sub>m</sub>(x) = F<sub>m−1</sub>(x) + ν·h<sub>m</sub>(x); cada árbol se ajusta a los "
        "residuos del ensamble actual.\n"
        "- **Stacking:** un meta-modelo `LinearRegression(positive=True)` combina el SVR RBF, el Gradient Boosting y el "
        "AdaBoost optimizados. Sus predicciones internas se generan con `KFold(5, shuffle=False)`, porque "
        "`StackingRegressor` no admite `TimeSeriesSplit`; la evaluación externa sigue siendo temporal.\n"
        "- **Voting:** combina los mismos tres modelos base, sin meta-modelo. **Soft:** media de las predicciones, con "
        "pesos iguales o **ponderada** por 1/RMSE² de CV (" + " · ".join(
            f"{name}: {voting_weights[key]:.2f}" for key, name in [("svr", "SVR"), ("gb", "GB"), ("ada", "AdaBoost")])
        + "). **Hard:** mediana de las tres predicciones.",
        unsafe_allow_html=True,
    )
    st.markdown("**Optimización con `GridSearchCV`** (mismos folds, RMSE como métrica)")
    col_a, col_b = st.columns(2)
    with col_a:
        table(["depth", "lr", "loss", "n", "RMSE val", "RMSE Train"], ADA_GRID_TOP, best=0,
              note="AdaBoost: mejores de 16 combinaciones. La profundidad del árbol base es lo más importante.")
    with col_b:
        table(["lr", "depth", "n", "subsample", "RMSE val", "RMSE Train"], GB_GRID_TOP, best=0,
              note="Gradient Boosting: mejores de 16 combinaciones. Ganan las configuraciones con menos árboles y una "
                   "tasa de aprendizaje baja.")

    heading("ensambles_cv")
    ENSEMBLE_CV = [
        ("SVR RBF optimizado", 940.61, 293.45, 644.60, 0.734, 814.93),
        ("Voting hard (mediana)", 943.41, 281.60, 663.03, 0.726, 598.84),
        ("Voting ponderado (1/MSE)", 963.59, 282.21, 685.25, 0.713, 630.59),
        ("Voting soft (pesos iguales)", 973.71, 280.48, 695.34, 0.706, 616.38),
        ("Ridge (α=1000)", 1005.35, 288.07, 730.05, 0.702, 873.59),
        ("Stacking (SVR + GB + AdaBoost)", 1048.57, 334.67, 772.91, 0.612, 683.93),
        ("Gradient Boosting optimizado", 1075.12, 356.77, 760.95, 0.595, 543.38),
        ("XGBoost (max_depth=5)", 1110.46, 399.41, 801.56, 0.560, 414.47),
        ("AdaBoost optimizado", 1127.52, 247.24, 824.20, 0.616, 691.08),
        ("Persistencia 24 h", 1153.40, 341.92, 710.68, 0.629, "—"),
        ("Gradient Boosting (por defecto)", 1228.74, 528.24, 875.24, 0.352, 634.40),
        ("AdaBoost (por defecto)", 1504.42, 406.67, 1167.94, 0.372, 1210.49),
    ]
    ENSEMBLE_FOLD_RMSE = {
        "Persistencia 24 h": [787.2, 1605.0, 827.6, 1509.6, 1037.6],
        "SVR RBF optimizado": [862.4, 1389.0, 561.8, 1144.7, 745.2],
        "Gradient Boosting optimizado": [1422.3, 1477.6, 571.1, 1138.9, 765.6],
        "Stacking": [1434.8, 1373.1, 592.3, 1101.6, 741.0],
        "Voting hard": [935.2, 1359.0, 541.1, 1116.6, 765.1],
    }
    table(["Modelo", "RMSE", "RMSE std", "MAE", "R²", "RMSE Train"], ENSEMBLE_CV, best=0,
          note="Promedio de los 5 folds de validación. Métricas en MW.")
    fold_chart(ENSEMBLE_FOLD_RMSE)
    caption("RMSE por fold de la persistencia y de los mejores modelos. Los folds 2 y 4 validan en primavera y verano, "
            "cuando la demanda crece rápido con el calor. En el fold 1, entrenado con solo cinco meses, la persistencia "
            "sigue siendo la mejor opción.")
    meta_weights = dict(zip(["SVR", "Gradient Boosting", "AdaBoost"], stacking.final_estimator_.coef_))
    callout(
        "warning", "Pesos del meta-modelo",
        " · ".join(f"{name}: <mark>{weight:.2f}</mark>" for name, weight in meta_weights.items())
        + f" · intercepto: {stacking.final_estimator_.intercept_:,.0f} MW. El meta-modelo favorece a Gradient Boosting "
        "porque sus predicciones internas vienen de <code>KFold</code>: los modelos base también se entrenan con datos "
        "posteriores, y Gradient Boosting reproduce bien lo que vio pero extrapola peor. Por eso el Stacking queda por "
        "detrás de su mejor componente.",
    )
    gb_fitted = stacking.named_estimators_["gb"]
    importances = (
        pd.Series(gb_fitted.named_steps["model"].feature_importances_,
                  index=gb_fitted.named_steps["preprocessing"].get_feature_names_out())
        .sort_values(ascending=False).head(10)
        .rename(lambda name: name.split("__")[-1])
        .rename_axis("Variable").rename("Importancia").reset_index()
    )
    st.altair_chart(
        alt.Chart(importances).mark_bar().encode(
            x=alt.X("Importancia:Q"), y=alt.Y("Variable:N", sort=None, title=None),
            tooltip=["Variable", alt.Tooltip("Importancia:Q", format=".3f")],
        ).properties(height=260),
        width="stretch",
    )
    caption("Gradient Boosting optimizado: 10 variables más importantes. El rezago de 24 h concentra casi toda la "
            "importancia; el resto de la señal viene de la sensación térmica y del rezago semanal.")
    st.markdown(
        "- **El SVR RBF es el mejor modelo en CV**, prácticamente empatado con el Voting hard (2.8 MW de diferencia, "
        "muy por debajo de la desviación entre folds).\n"
        "- **Los Voting superan a sus componentes de boosting**, pero no al SVR: promediar ayuda cuando los errores son "
        "distintos, y la mediana descarta la predicción extrema de cada hora.\n"
        "- **El Stacking queda detrás del SVR** (1,048.57 MW) por la validación interna no temporal.\n"
        "- **AdaBoost y Gradient Boosting** solo superan a la persistencia después de optimizarse."
    )

    heading("ensambles_test")
    TEST_MODELS = {
        "SVR RBF optimizado": "prediction", "Stacking": "prediction_stacking", "Voting hard": "prediction_voting_hard",
        "SVR lineal": "prediction_svr_linear", "Polinomial (opt.)": "prediction_poly",
        "Gradient Boosting (opt.)": "prediction_gb", "Ridge (α=1000)": "prediction_ridge",
        "Regresión lineal": "prediction_linear", "Voting ponderado": "prediction_voting",
        "Voting simple": "prediction_voting_simple", "Persistencia 24 h": "prediction_persistence",
        "AdaBoost (opt.)": "prediction_ada",
    }
    rows = [(name, *(round(v, 2) for v in metrics(y_true, test[col]).values())) for name, col in TEST_MODELS.items()]
    rows.append(("XGBoost (max_depth=5)", 840.63, 572.11, 4.90, 0.86))
    rows = sorted(rows, key=lambda r: r[1])
    table(["Modelo", "RMSE", "MAE", "MAPE (%)", "R²"], rows, best=0,
          note="Métricas en Test (MW), calculadas con los modelos de la app; XGBoost se toma del notebook. Todos los "
               "modelos excepto AdaBoost superan a la persistencia.")
    series_chart(test.set_index("datetime")[[TARGET, "prediction", "prediction_stacking", "prediction_voting_hard"]]
                 .resample("D").mean(), "MW (promedio diario)")
    caption("Demanda real frente al SVR, el Stacking y el Voting hard en Test (promedios diarios).")
    callout(
        "warning", "AdaBoost en Test",
        f"AdaBoost optimizado obtiene un MAPE de {metrics(y_true, test['prediction_ada'])['mape']:.2f}%, el único "
        "modelo peor que la persistencia. El Stacking le asigna peso 0, y el Voting hard lo descarta cuando su "
        "predicción es la extrema; el Voting soft, con pesos fijos, conserva su error.",
    )

    heading("nuevos")
    new_period = load_new_data(train_median)
    new_true = new_period[TARGET]
    st.markdown(
        f"Los modelos se evaluaron **sin reentrenar** en {len(new_period):,} horas posteriores al periodo de estudio "
        f"({new_period['datetime'].min():%Y-%m-%d} → {new_period['datetime'].max():%Y-%m-%d}), de Shekh y Rafi (2026). "
        "Los rezagos se calculan con la serie completa, siempre con valores del pasado, y el clima se descarga de "
        "Open-Meteo con el mismo procedimiento."
    )
    new_rows = sorted([(name, *(round(v, 2) for v in metrics(new_true, new_period[col]).values()))
                       for name, col in TEST_MODELS.items()], key=lambda r: r[1])
    table(["Modelo", "RMSE", "MAE", "MAPE (%)", "R²"], new_rows, best=0,
          note="Métricas en MW sobre las horas nuevas. Ningún modelo vio estos datos.")
    series_chart(new_period.set_index("datetime")[[TARGET, "prediction", "prediction_svr_linear", "prediction_persistence"]]
                 .resample("D").mean(), "MW (promedio diario)")
    caption("Demanda real frente al SVR RBF, el SVR lineal y la persistencia en las horas nuevas (promedios diarios).")
    new_svr = metrics(new_true, new_period["prediction"])
    callout(
        "result", "El desempeño se mantiene",
        f"El SVR RBF obtiene un MAPE de <mark>{new_svr['mape']:.2f}%</mark> en las horas nuevas, incluso menor que en "
        "Test. El SVR lineal, mediocre en CV, es aquí el mejor: con la tendencia de crecimiento, extrapolar de forma "
        "lineal favorece a los modelos que no están acotados al rango de entrenamiento.",
    )

    heading("discusion")
    SOTA = [
        ("Chen et al. (2004)", "Demanda máxima diaria (31 días)", "SVR", "EUNITE 2001", "MAPE 1.95%"),
        ("Islam et al. (2025)", "Demanda horaria", "FB-Prophet, LSTM", "PGCB 2015–2023, clima y economía", "MAPE 2.3%"),
        ("Miah et al. (2024)", "Demanda horaria", "Bi-LSTM", "PGCB", "MAE 376.72"),
        ("Akhter et al. (2024)", "Demanda diaria de Dhaka", "CNN + BiLSTM", "Consumo diario 2016–2021", "MAPE 1.64%"),
        ("Karmakar et al. (2026)", "Demanda pico diaria, día siguiente", "Híbrido RF–Naive", "Bangladesh 2019–2022",
         "MAPE 5.14%"),
        ("Este trabajo", "Demanda horaria, 24 h antes", "SVR RBF", "PGCB 2022–2025 y clima",
         f"MAPE {svr_test['mape']:.2f}%"),
    ]
    table(["Autor / Año", "Objetivo", "Modelo", "Dataset", "Resultado"], SOTA, best=len(SOTA) - 1,
          note="Comparación con el estado del arte. Los estudios difieren en resolución, horizonte, periodo y protocolo.")
    st.markdown(
        "- **¿Qué modelo fue el mejor y por qué?** El SVR RBF optimizado, en CV y en Test. Combina una función suave con "
        "una regularización fuerte: el rezago de 24 h explica la mayor parte de la señal y el SVR la corrige de forma no "
        "lineal con el calendario y el clima, sin sobreajustar a periodos concretos.\n"
        "- **Base frente a optimizados:** la optimización mejoró a todos los modelos, y en todos favoreció "
        "configuraciones más regularizadas (α grande en Ridge, C pequeño en SVR, menos árboles en Gradient Boosting).\n"
        "- **Características de los datos:** la tendencia de crecimiento favorece a los modelos que extrapolan de forma "
        "lineal y explica la subestimación del SVR en Test. El calendario no incluye feriados.\n"
        "- **Comparación con la literatura:** el MAPE está en el rango de los estudios revisados y es menor que el de "
        "Karmakar et al., el único con un protocolo que también evita la fuga de información. Los estudios con menor MAPE "
        "pronostican datos diarios o periodos cortos y estables.\n"
        "- **Limitaciones:** el clima observado se usa como pronóstico perfecto; la demanda publicada es una estimación "
        "construida con la generación y el load shedding; la variación entre folds es grande frente a las diferencias "
        "entre los mejores modelos; y el Stacking no respeta el orden temporal en su validación interna."
    )

    heading("conclusiones")
    st.markdown(
        "- **Se cumplió el objetivo:** pronosticar la demanda horaria con 24 h de anticipación usando solo información "
        "disponible el día anterior.\n"
        f"- **Mejor modelo:** SVR RBF (C = {params['C']}, γ = {params['gamma']}), con MAPE de {svr_test['mape']:.2f}% en "
        f"Test, {1 - svr_test['rmse'] / persistence_test['rmse']:.0%} menos error que la persistencia, y "
        f"{new_svr['mape']:.2f}% en datos posteriores.\n"
        "- **La persistencia es una referencia exigente:** los modelos lineales y los ensambles por defecto no la "
        "superan.\n"
        "- **Revisar cómo se construye el objetivo:** la demanda de PGCB equivale casi a generación + load shedding; "
        "usar variables de la misma hora convierte el pronóstico en una suma.\n"
        "- **Los ensambles no garantizan una mejora** sobre su mejor componente cuando la validación interna no respeta "
        "el orden temporal.\n"
        "- **Trabajo futuro:** pronósticos meteorológicos archivados, feriados y Ramadán, "
        "Stacking con validación temporal, combinar la demanda con un pronóstico de la "
        "generación para anticipar el load shedding."
    )

    heading("referencias")
    references(REFERENCES.values())

with tab_product:
    st.html("""
    <section class="pd-hero">
      <p class="pd-kicker">Red eléctrica de Bangladesh</p>
      <h1 class="pd-title">¿Cuánta electricidad necesitará el país mañana?</h1>
      <p class="pd-lead">La predicción de demanda es un proceso crucial para administrar la capacidad generativa de la red eléctrica del país, todas las decisiones de aprovisionamiento dependen de un pronóstico preciso de la demanda del sistema: cuánto
      combustible e importaciones asegurar y dónde y cuándo ejecutar cortes estratégicos del servicio. Esta herramienta
      pronostica, <b>con un día de anticipación</b>, los megawatts que demandará el país en cada hora, utilizando condiciones climatológicas y temporales para aumentar la precisión de las estimaciones.</p>
    </section>
    """)
    shedding_hours = (train["load_shedding"] > 0).mean()
    IMPACT_STATS = [
        ("71.4%", "de las empresas formales sufrieron apagones en 2022, con un promedio de <b>26.2 cortes al mes</b>."
                  "<sup>1</sup>"),
        ("2,245 GWh", "de demanda quedaron sin cubrir en el año fiscal 2023-24; el 29 de abril de 2024 faltó el "
                      "<b>12.6%</b> de la energía demandada.<sup>2</sup>"),
        ("Tk 382,890 M", "(US$3,220 M) de subsidio al organismo eléctrico estatal (BPDB) en 2023-24, que además paga "
                         "capacidad a plantas ociosas.<sup>2</sup>"),
        ("3,612 MW", "de corte máximo registrado un sábado de finales de agosto de 2026, a las 3 de la tarde.<sup>3</sup>"),
    ]
    SOLUTION_STEPS = [
        ("1 · La demanda reciente", "Cuánta energía se usó a la misma hora ayer y hace una semana."),
        ("2 · El clima previsto", "Temperatura, sensación térmica, humedad y nubosidad en las 8 capitales de división."),
        ("3 · El pronóstico", "Un modelo de ML, entrenado con tres años de registros, estima la demanda de "
                              "cada hora futura."),
    ]
    st.html(f"""
    <h2 class="pd-h2"><small>El reto</small>Cuando la demanda supera a la generación, hay apagones conocidos como desprendimiento de carga</h2>
    <p class="pd-text">Entre 2022 y 2024, la red de Bangladesh atendió en promedio <b>{train[TARGET].mean():,.0f} MW en
    cada hora</b>, con picos de hasta <b>{train[TARGET].max():,.0f} MW</b> en verano. Cuando la generación disponible no
    alcanza, la Power Grid Company of Bangladesh corta el servicio en zonas completas: según sus propios registros, hubo
    cortes en <b>{shedding_hours:.0%} de las horas</b> del periodo de entrenamiento.</p>
    <p class="pd-text">Saber con anticipación cuánta energía se necesitará permite planificar la generación, comprar el
    combustible a tiempo y, cuando el déficit es inevitable, programar los cortes e informar a la población.</p>

    <h2 class="pd-h2"><small>Lo que está en juego</small>Implicaciones económicas y sociales</h2>
    {cards("stat", IMPACT_STATS)}
    <ul class="pd-list">
      <li><b>Economía.</b> Los cortes obligan a la industria a operar por debajo de su capacidad y a recurrir a generadores
      de diésel. Según IEEFA, una mejor planificación reduciría los pagos por capacidad ociosa.<sup>1,2</sup></li>
      <li><b>Salud.</b> En agosto de 2026, Prothom Alo reportó la suspensión de servicios de
      rayos X, electrocardiograma y laboratorio durante los cortes en 14 hospitales públicos.<sup>3</sup></li>
      <li><b>Vida diaria.</b> Algunas localidades reportaron entre 8 y 12 horas sin electricidad al día; el calor sin
      ventilación afecta sobre todo a niños, adultos mayores y pacientes.<sup>3</sup></li>
    </ul>
    <p class="nb-caption">Fuentes: <sup>1</sup> Banco Mundial, <i>Enterprise Surveys: Bangladesh 2022</i>.
    <sup>2</sup> IEEFA, <i>Fixing Bangladesh's Power Sector</i> (diciembre de 2024). <sup>3</sup> Prothom Alo,
    <i>Power crisis disrupts healthcare across the country</i> (31 de agosto de 2026). Las cifras de demanda y de horas
    con cortes provienen del dataset de PGCB usado en este proyecto.</p>

    <h2 class="pd-h2"><small>La solución</small>Como llegar a un mejor pronóstico</h2>
    {cards("step", SOLUTION_STEPS)}
    """)

    new_period = load_new_data(train_median)
    new_true = new_period[TARGET]
    new_svr = metrics(new_true, new_period["prediction"])
    new_persistence = metrics(new_true, new_period["prediction_persistence"])
    within_5 = (np.abs(new_true - new_period["prediction"]) / new_true <= 0.05).mean()
    st.html('<h2 class="pd-h2"><small>La prueba</small>Validación con datos que el modelo no había visto</h2>')
    st.markdown(
        f"El modelo se entrenó con datos hasta junio de 2025. Después se evaluó con **{len(new_period):,} horas "
        f"nuevas**<sup>4</sup>, de {month_year(new_period['datetime'].min())} a {month_year(new_period['datetime'].max())}, "
        "comparando cada pronóstico con la demanda registrada.",
        unsafe_allow_html=True,
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("Error medio", f"{new_svr['mape']:.1f}%", help=f"MAPE. Equivale a {new_svr['mae']:,.0f} MW por hora en promedio.")
    k2.metric("Horas con error < 5%", f"{within_5:.0%}")
    k3.metric("Menos error que «igual que ayer»", f"{1 - new_svr['rmse'] / new_persistence['rmse']:.0%}",
              help="Reducción del RMSE frente a la persistencia de 24 h, que repite la demanda del día anterior.")
    st.html(
        '<p class="nb-caption">Fuente: <sup>4</sup> Shekh, M. I., &amp; Rafi, R. H. (2026). <i>Hourly Electricity '
        "Generation, Demand, Load Shedding, and Fuel Mix Dataset for Bangladesh (2015–2026)</i>. Mendeley Data, V1 "
        f'(CC BY 4.0). <a href="{MENDELEY_URL}" target="_blank" rel="noreferrer">{MENDELEY_URL.removeprefix("https://")}</a>. '
        "Se usan solo las horas posteriores al periodo de estudio, que ningún modelo vio. El clima de esas horas "
        "proviene de Open-Meteo.</p>"
    )

    peak_day = new_period.set_index("datetime")[TARGET].resample("D").mean().idxmax()
    peak = new_period[new_period["datetime"].dt.normalize() == peak_day]
    peak_hour = peak.loc[peak[TARGET].idxmax()]
    st.markdown(
        f"**Un caso crítico.** El {peak_day.day} de {month_year(peak_day)} fue el día de mayor demanda del periodo nuevo: "
        f"a las {peak_hour['datetime']:%H:%M} se requirieron **{peak_hour[TARGET]:,.0f} MW**. Un día antes, el modelo "
        f"pronosticó **{peak_hour['prediction']:,.0f} MW** para esa hora; repetir la demanda del día anterior habría "
        f"dado {peak_hour['prediction_persistence']:,.0f} MW."
    )
    series_chart(peak.set_index("datetime")[[TARGET, "prediction", "prediction_persistence"]], "MW")
    st.html(f'<p class="nb-caption">Demanda real, pronóstico del modelo y persistencia de 24 h, hora por hora, el '
            f'{peak_day:%d/%m/%Y}.</p>')

    st.html('<h2 class="pd-h2"><small>Simular Nuevos Escenarios</small>Prueba el modelo con horas reales</h2>')
    st.markdown("Elige horas al azar y compara el pronóstico con lo que realmente ocurrió. Ningún caso está "
                "elegido a mano.")
    source = st.segmented_control(
        "Origen de los datos", ["Nuevos datos (jun 2025 – mar 2026)", "Datos Test (nov 2024 – jun 2025)"],
        default="Nuevos datos (jun 2025 – mar 2026)",
    )
    case_source = test if source == "Periodo de prueba (nov 2024 – jun 2025)" else new_period
    case_a, case_b = st.columns([2, 1])
    with case_a:
        n_cases = st.slider("Número de horas", 1, 20, 5)
    with case_b:
        if st.button("Generar nuevos casos", type="primary") or "case_seed" not in st.session_state:
            st.session_state.case_seed = int(np.random.default_rng().integers(1_000_000))
    cases = case_source.sample(min(n_cases, len(case_source)), random_state=st.session_state.case_seed).sort_values("datetime")
    case_error = (cases[TARGET] - cases["prediction"]).abs() / cases[TARGET] * 100

    def verdict(error):
        return "Preciso" if error <= 3 else "Aproximado" if error <= 7 else "Impreciso"
    
    table(
        ["Fecha y hora", "Temperatura", "Demanda real", "Pronóstico", "Error", "Resultado"],
        [(f"{DAYS[r['datetime'].dayofweek]} {r['datetime']:%d/%m/%Y · %H:%M}", f"{r['temperature_2m']:.0f} °C",
          f"{r[TARGET]:,.0f} MW", f"{r['prediction']:,.0f} MW", f"{case_error[i]:.1f}%", verdict(case_error[i]))
         for i, r in cases.iterrows()],
        note="Preciso: error de hasta 3% · Aproximado: hasta 7% · Impreciso: más de 7%.",
    )
    st.markdown(f"En esta muestra, el pronóstico quedó a menos de 3% de la demanda real en "
                f"**{int((case_error <= 3).sum())} de {len(cases)}** horas.")

    PRODUCT_MODELS = {"SVR RBF (modelo de esta página)": "prediction", "Stacking": "prediction_stacking",
                      "Voting hard": "prediction_voting_hard", "Voting ponderado": "prediction_voting",
                      "SVR lineal": "prediction_svr_linear", "Regresión polinomial": "prediction_poly",
                      "Ridge": "prediction_ridge", "Gradient Boosting": "prediction_gb",
                      "Persistencia (igual que ayer)": "prediction_persistence"}
    with st.expander("Comparación con otros modelos"):
        table(["Modelo", "RMSE", "MAE", "MAPE (%)", "R²"],
              sorted([(name, *(round(v, 2) for v in metrics(new_true, new_period[col]).values()))
                      for name, col in PRODUCT_MODELS.items()], key=lambda row: row[1]),
                                    best=0, note=f"Métricas en MW sobre las {len(new_period):,} horas nuevas. El detalle está en la pestaña "
                           "Reporte técnico.")
        series_chart(new_period.set_index("datetime")[[TARGET, "prediction", "prediction_stacking"]].resample("D").mean(),
                     "MW (promedio diario)")


    st.html('<h2 class="pd-h2"><small>Simulador</small>Evalúa tus propios escenarios</h2>')
    st.markdown("Elige un escenario o ajusta los controles: con la demanda reciente y el clima previsto, ¿cuánta "
                "electricidad se necesitará a esa hora? Si indicas la generación disponible, verás si alcanzaría.")
    # Cada escenario es la mediana de las horas de Train que cumplen su condición; top = (variable, cuantil) se queda
    # solo con las horas de ese cuantil en adelante (p. ej., las más calurosas o las más lluviosas)
    SCENARIO_RULES = {
        "Día típico": {"rule": "Todas las horas de Train", "query": None, "hour": 12, "month": 4},
        "Noche de verano": {"rule": "Junio a las 21:00", "query": "Month == 6 and Hour == 21", "hour": 21, "month": 6},
        "Tarde de calor": {"rule": "Mayo a las 15:00, el 25% de horas con mayor sensación térmica",
                           "query": "Month == 5 and Hour == 15", "hour": 15, "month": 5,
                           "top": ("apparent_temperature", 0.75)},
        "Día lluvioso": {"rule": "Julio de 12:00 a 16:00, el 10% de horas con más lluvia",
                         "query": "Month == 7 and 12 <= Hour <= 16", "hour": 14, "month": 7,
                         "top": ("precipitation", 0.90)},
        "Mañana de invierno": {"rule": "Enero a las 08:00", "query": "Month == 1 and Hour == 8", "hour": 8, "month": 1},
    }
    SCENARIO_DAY = 2  # miércoles, un día laborable, en todos los escenarios

    def scenario_values(spec):
        """Mediana de las horas de Train del escenario, redondeada al paso de cada control."""
        hours = train.query(spec["query"]) if spec["query"] else train
        if "top" in spec:
            column, quantile = spec["top"]
            hours = hours[hours[column] >= hours[column].quantile(quantile)]
        m = hours.median(numeric_only=True)

        def mw(v):
            return int(np.clip(round(v / 50) * 50, 6000, 18000))

        def half(v, low, high):
            return float(np.clip(round(v * 2) / 2, low, high))

        values = (mw(m["demand_lag_24h"]), mw(m["demand_lag_168h"]), half(m["temperature_2m"], 10, 40),
                  half(m["apparent_temperature"], 10, 50), int(round(m["relative_humidity_2m"])),
                  half(m["precipitation"], 0, 20), int(round(m["cloud_cover"])), spec["hour"], spec["month"],
                  SCENARIO_DAY, mw(m["generation_mw"]))
        return values, len(hours)

    scenario_data = {name: scenario_values(spec) for name, spec in SCENARIO_RULES.items()}
    scenarios = {name: values for name, (values, _) in scenario_data.items()}
    sim_keys = ["sim_lag24", "sim_lag168", "sim_temp", "sim_apparent", "sim_humidity", "sim_rain", "sim_cloud",
                "sim_hour", "sim_month", "sim_day", "sim_generation"]

    def apply_scenario():
        chosen = st.session_state.get("sim_scenario")
        if chosen:
            st.session_state.update(dict(zip(sim_keys, scenarios[chosen])))

    for key, value in zip(sim_keys, scenarios["Día típico"]):
        st.session_state.setdefault(key, value)
    st.segmented_control("Escenario", list(scenarios), key="sim_scenario", on_change=apply_scenario,
                         label_visibility="collapsed")
    with st.popover("¿Cómo se calculan los escenarios?"):
        st.markdown(
            "Cada escenario es la **mediana de las horas reales de Train** que cumplen su condición: demanda de ayer y "
            "de hace una semana, clima y generación disponible. La hora y el mes son los de la condición, y el día es "
            "siempre miércoles (laborable). Los valores se redondean al paso de cada control."
        )
        table(["Escenario", "Horas de Train usadas", "Horas", "Demanda ayer", "Sensación térmica", "Lluvia",
               "Generación"],
              [(name, SCENARIO_RULES[name]["rule"], count, f"{v[0]:,} MW", f"{v[3]:.1f} °C", f"{v[5]:.1f} mm",
                f"{v[10]:,} MW") for name, (v, count) in scenario_data.items()],
              note="Solo se usan datos de Train, los mismos con los que se entrenó el modelo. La lluvia es el "
                   "promedio de las 8 capitales, por eso es menor que la de una sola ciudad.")

    col_a, col_b = st.columns(2)
    with col_a:
        lag24 = st.slider("Demanda ayer a la misma hora (MW)", 6000, 18000, step=50, key="sim_lag24")
        lag168 = st.slider("Demanda hace una semana a la misma hora (MW)", 6000, 18000, step=50, key="sim_lag168")
        hour = st.slider("Hora del día", 0, 23, key="sim_hour")
        month = st.slider("Mes", 1, 12, key="sim_month")
        day = st.selectbox("Día de la semana", range(7), format_func=lambda d: DAYS[d].capitalize(), key="sim_day")
        generation = st.slider("Generación disponible prevista (MW)", 6000, 18000, step=50, key="sim_generation")
    with col_b:
        temperature = st.slider("Temperatura (°C)", 10.0, 40.0, step=0.5, key="sim_temp")
        apparent = st.slider("Sensación térmica (°C)", 10.0, 50.0, step=0.5, key="sim_apparent")
        humidity = st.slider("Humedad relativa (%)", 20, 100, key="sim_humidity")
        rain = st.slider("Precipitación (mm)", 0.0, 20.0, step=0.5, key="sim_rain")
        cloud = st.slider("Nubosidad (%)", 0, 100, key="sim_cloud")

    # Fecha de 2024 con el mes y el día de la semana elegidos; add_time_features deriva el resto del calendario
    first = pd.Timestamp(2024, month, 1)
    date = first + pd.Timedelta(days=(day - first.dayofweek) % 7, hours=hour)
    row = add_time_features(pd.DataFrame({"datetime": [date]}))
    row = row.assign(demand_lag_24h=lag24, demand_lag_168h=lag168, temperature_2m=temperature,
                     apparent_temperature=apparent, relative_humidity_2m=humidity, precipitation=rain,
                     cloud_cover=cloud)[FEATURES]
    prediction = float(model.predict(row)[0])
    base_row = voting_artifact["simple"].transform(row)[0]  # SVR RBF, Gradient Boosting, AdaBoost
    sim_models = {
        "Persistencia (igual que ayer)": ("Referencia", float(lag24)),
        "Regresión lineal": ("Lineal", float(linear_artifact["linear"].predict(row)[0])),
        "Ridge": ("Lineal", float(linear_artifact["ridge"].predict(row)[0])),
        "Regresión polinomial": ("Lineal", float(linear_artifact["poly"].predict(row)[0])),
        "SVR lineal": ("SVR", float(load_artifact("svr_linear.joblib")["model"].predict(row)[0])),
        "SVR RBF (modelo de esta página)": ("SVR", prediction),
        "Gradient Boosting": ("Ensamble", float(base_row[1])),
        "AdaBoost": ("Ensamble", float(base_row[2])),
        "Stacking": ("Ensamble", float(stacking.predict(row)[0])),
        "Voting soft": ("Ensamble", float(base_row.mean())),
        "Voting ponderado": ("Ensamble", float(voting_artifact["weighted"].predict(row)[0])),
        "Voting hard": ("Ensamble", float(np.median(base_row))),
    }

    change = prediction / lag24 - 1
    deficit = prediction - generation
    supply = (f"Con {generation:,} MW disponibles, <b>faltarían unos {deficit:,.0f} MW</b>: habría que programar cortes."
              if deficit > 0 else f"Con {generation:,} MW disponibles, la generación cubriría la demanda "
                                  f"(margen de {-deficit:,.0f} MW).")
    st.html(f"""
    <div class="pd-result">
      <span class="nb-callout-label">Demanda pronosticada · {DAYS[day]} a las {hour:02d}:00</span>
      <div class="pd-big">{prediction:,.0f} MW</div>
      <p><b>{change:+.1%} respecto a ayer a la misma hora.</b> {supply}</p>
    </div>
    """)
    st.html(f'<p class="nb-caption">Pronóstico del SVR RBF. Como referencia, el Stacking pronostica '
            f'{sim_models["Stacking"][1]:,.0f} MW y el Voting hard {sim_models["Voting hard"][1]:,.0f} MW. El déficit es '
            'una aproximación: la demanda publicada por PGCB ya incluye el load shedding estimado.</p>')

    with st.expander("Pronóstico de todos los modelos"):
        sim_values = [value for _, value in sim_models.values()]
        table(["Modelo", "Familia", "Pronóstico", "Diferencia con el SVR"],
              [(name, family, f"{value:,.0f} MW", f"{value - prediction:+,.0f} MW")
               for name, (family, value) in sim_models.items()],
              best=list(sim_models).index("SVR RBF (modelo de esta página)"),
              note=f"Pronóstico de cada modelo para este escenario. Rango: {min(sim_values):,.0f}–{max(sim_values):,.0f} MW; "
                   f"mediana: {np.median(sim_values):,.0f} MW.")
        sim_frame = pd.DataFrame({"Modelo": list(sim_models), "Familia": [f for f, _ in sim_models.values()],
                                  "MW": sim_values})
        st.altair_chart(
            alt.Chart(sim_frame).mark_bar().encode(
                x=alt.X("MW:Q", title="Demanda pronosticada (MW)", scale=alt.Scale(zero=False)),
                y=alt.Y("Modelo:N", sort=list(sim_models), title=None),
                color=alt.Color("Familia:N", legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Modelo", "Familia", alt.Tooltip("MW:Q", format=",.0f")],
            ).properties(height=340),
            width="stretch",
        )

    st.html("""
    <h2 class="pd-h2"><small>Alcance</small>Consideraciones importantes</h2>
    <ul class="pd-list">
      <li><b>Necesita el pronóstico del clima.</b> El modelo usa el clima de la hora pronosticada. Aquí se evaluó con el
      clima observado; en operación se usaría un pronóstico meteorológico, con su propio error.</li>
      <li><b>No conoce los feriados.</b> En Eid, Ramadán y otros feriados la demanda cambia de forma brusca, y el modelo
      solo lo nota con un día de retraso.</li>
      <li><b>La demanda publicada es una estimación.</b> En horas con cortes, PGCB calcula la demanda a partir de la
      generación y del load shedding; el modelo aprende esa estimación, no el consumo medido.</li>
      <li><b>Debe reentrenarse periódicamente.</b> La demanda crece año con año; un modelo que no se actualiza tiende a
      subestimarla.</li>
    </ul>
    <p class="pd-text">Encontrarás la metodología completa en la pestaña <b>Reporte técnico</b>.</p>
    """)
