"""Componentes reutilizables de la interfaz: tema, estilos, encabezados, callouts, tablas, tarjetas,
gráficas, índice y referencias. No dependen del modelo ni de los datos del proyecto."""
import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components

# Tokens de notebook-theme-preview.html; los colores base vienen de .streamlit/config.toml
LIGHT = {
    "accent": "#5B3DC8", "accent-soft": "#ECE6FC", "mark": "#E3D9FF", "surface": "#F5F2FA",
    "border": "#E4DFEE", "fg": "#1F1B2D", "muted": "#686179", "tip": "#127F6E", "warn": "#A86A00",
}
DARK = {
    "accent": "#B39DFF", "accent-soft": "#2A2245", "mark": "#43357A", "surface": "#1E1A29",
    "border": "#2F2940", "fg": "#ECE8F5", "muted": "#A39BB7", "tip": "#4FCBB5", "warn": "#E3A94A",
}

_figures = {"count": 0}


def theme_tokens():
    """Tokens de color del tema activo (claro u oscuro)."""
    return DARK if st.context.theme.type == "dark" else LIGHT


REPORT_BLOCK = '[data-testid="stTabPanel"] > [data-testid="stVerticalBlock"]:has(> * h2.nb-h2)'

# Impresión / PDF: el reporte se imprime como diapositivas horizontales (portada y una sección por página), sin la
# interfaz de Streamlit. Los contenedores de Streamlit tienen altura fija con scroll; sin liberarlos, solo se imprimiría
# lo visible en pantalla.
PRINT_CSS = f"""
@media print {{
  @page {{ size: A4 landscape; margin: 12mm 14mm; }}
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
    height: auto !important; overflow: visible !important; position: static !important; }}
  [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], [data-testid="stTabs"] [role="tablist"], [data-testid="stIFrame"],
  [data-testid="stExpander"] summary svg {{ display: none !important; }}
  [data-testid="stMainBlockContainer"] {{ max-width: 100% !important; padding: 0 !important; }}
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: #ffffff !important; }}
  * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  {REPORT_BLOCK} > * {{ display: block !important; break-inside: avoid; }}
  {REPORT_BLOCK} > *:has(.nb-title) {{ break-after: page; }}
  {REPORT_BLOCK} > *:has(h2.nb-h2) {{ break-before: page; }}
  .nb-h2 {{ margin-top: 0 !important; }}
  /* El mapa de pydeck es WebGL y el navegador no lo imprime: se ocultan el mapa y su pie; el mapa de calor siguiente
     tiene los mismos datos */
  {REPORT_BLOCK} > [data-testid="stElementContainer"]:has([data-testid="stDeckGlJsonChart"]),
  {REPORT_BLOCK} > [data-testid="stElementContainer"]:has([data-testid="stDeckGlJsonChart"]) + * {{ display: none !important; }}
}}
"""


def pdf_button(label="Descargar PDF", tokens=None):
    """Botón que abre el diálogo de impresión del navegador para guardar el reporte como PDF.

    st.html no ejecuta JavaScript, así que el botón vive en un componente (iframe) y llama a la ventana principal.
    """
    tokens = tokens or {}
    accent, fg, border = tokens.get("accent", "#5B3DC8"), tokens.get("fg", "#1F1B2D"), tokens.get("border", "#E4DFEE")
    st_components.html(f"""
    <button onclick="window.parent.print()" style="width:100%; padding:0.45rem 0.75rem; border-radius:0.5rem;
      border:1px solid {border}; background:transparent; color:{fg}; font:500 0.95rem 'IBM Plex Sans', sans-serif;
      cursor:pointer;" onmouseover="this.style.borderColor='{accent}'; this.style.color='{accent}'"
      onmouseout="this.style.borderColor='{border}'; this.style.color='{fg}'">{label}</button>
    """, height=44)


def inject_styles(tokens):
    """Inyecta las variables CSS del tema y los estilos de todos los componentes. Llamar una vez por ejecución."""
    css_vars = "".join(f"--nb-{k}: {v};" for k, v in tokens.items())
    _figures["count"] = 0  # Streamlit re-ejecuta el script en cada interacción: las figuras vuelven a numerarse desde 1
    st.html(f"""
<style>
:root {{ {css_vars}
  --nb-font-display: "IBM Plex Mono", "Courier New", monospace; }}
.block-container {{ padding-top: 3.5rem; }}
.nb-title {{ font-family: var(--nb-font-display); font-size: clamp(2rem, 5vw, 2.9rem); font-weight: 600;
  line-height: 1.12; letter-spacing: -0.045em; word-spacing: -0.2em; margin: 0 0 1.1rem; text-wrap: balance; }}
.nb-description {{ font-size: 1.2rem; line-height: 1.5; color: var(--nb-muted); max-width: 52ch; margin: 0 0 1.5rem; }}
.nb-meta {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; padding-top: 1rem; margin: 0;
  border-top: 1px solid var(--nb-border); color: var(--nb-muted); font-size: 0.8rem; font-family: var(--nb-font-display); }}
.nb-tags {{ display: flex; flex-wrap: wrap; gap: 0.4rem; list-style: none; padding: 0 !important; margin: 1rem 0 0 !important; }}
.nb-tags li {{ font-family: var(--nb-font-display); font-size: 0.75rem; font-weight: 500; padding: 0.15rem 0.6rem;
  border-radius: 999px; background: var(--nb-accent-soft); color: var(--nb-accent); margin: 0; }}
.nb-h2 {{ font-family: var(--nb-font-display); font-size: 1.45rem; font-weight: 600; letter-spacing: -0.035em;
  word-spacing: -0.15em; margin: 2.75rem 0 0.25rem; scroll-margin-top: 4rem; color: var(--nb-fg); }}
.nb-h2 span {{ color: var(--nb-accent); margin-right: 0.6em; }}
.nb-callout {{ --kind: var(--nb-accent); margin: 1rem 0 1.5rem; padding: 0.9rem 1.15rem 0.9rem 1.1rem;
  border-left: 3px solid var(--kind); border-radius: 0 4px 4px 0; background: var(--nb-surface); }}
.nb-callout[data-kind="result"] {{ background: var(--nb-accent-soft); }}
.nb-callout[data-kind="tip"] {{ --kind: var(--nb-tip); }}
.nb-callout[data-kind="warning"] {{ --kind: var(--nb-warn); }}
.nb-callout-label {{ display: block; font-family: var(--nb-font-display); font-size: 0.75rem; font-weight: 600;
  color: var(--kind); margin-bottom: 0.25rem; }}
.nb-callout p {{ margin: 0; }}
mark {{ background: var(--nb-mark); color: inherit; padding: 0.05em 0.25em; border-radius: 2px; }}
.nb-caption {{ font-size: 0.875rem; line-height: 1.5; color: var(--nb-muted); margin: 0.25rem 0 1.5rem; max-width: 60ch; }}
.nb-caption b {{ font-family: var(--nb-font-display); font-size: 0.92em; color: var(--nb-fg); }}
.nb-table {{ border-collapse: collapse; font-size: 0.9rem; font-variant-numeric: tabular-nums; line-height: 1.45;
  border-top: 1.5px solid var(--nb-fg); border-bottom: 1.5px solid var(--nb-fg); width: 100%; margin: 0.5rem 0; }}
.nb-table th {{ text-align: right; font-weight: 600; font-family: var(--nb-font-display); font-size: 0.8rem;
  padding: 0.55rem 0.8rem 0.45rem; border-bottom: 1px solid var(--nb-fg); }}
.nb-table td {{ text-align: right; padding: 0.4rem 0.8rem; border-bottom: 1px solid var(--nb-border); }}
.nb-table th:first-child, .nb-table td:first-child {{ text-align: left; }}
.nb-table tr:last-child td {{ border-bottom: 0; }}
.nb-table tr.best td {{ font-weight: 600; }}
.nb-toc-title {{ font-family: var(--nb-font-display); font-weight: 600; font-size: 0.85rem; margin-bottom: 0.75rem; }}
.nb-toc ol {{ list-style: none; padding: 0; margin: 0; counter-reset: toc; font-family: var(--nb-font-display); font-size: 0.8rem; }}
.nb-toc li {{ counter-increment: toc; padding: 0.3rem 0 0.3rem 0.85rem; border-left: 2px solid var(--nb-border); margin: 0; }}
.nb-toc li:hover {{ border-left-color: var(--nb-accent); }}
.nb-toc a {{ color: var(--nb-muted) !important; text-decoration: none !important; }}
.nb-toc a::before {{ content: counter(toc) ". "; }}
.nb-toc a:hover {{ color: var(--nb-accent) !important; }}
.nb-refs {{ padding-left: 1.2rem !important; margin: 0.5rem 0 0 !important; line-height: 1.6; }}
.nb-refs li {{ margin-bottom: 0.6rem; }}
.nb-refs a {{ color: var(--nb-accent) !important; text-decoration: underline; text-underline-offset: 3px;
  text-decoration-color: color-mix(in srgb, var(--nb-accent) 40%, transparent); }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"] p {{ font-family: var(--nb-font-display); font-size: 0.78rem; color: var(--nb-muted); }}
.pd-hero {{ padding: 0.5rem 0 0.5rem; }}
.pd-kicker {{ font-family: var(--nb-font-display); font-size: 0.8rem; font-weight: 600; color: var(--nb-accent); margin: 0 0 0.6rem; }}
.pd-title {{ font-family: var(--nb-font-display); font-size: clamp(2rem, 5.5vw, 3rem); font-weight: 600; line-height: 1.1;
  letter-spacing: -0.045em; word-spacing: -0.2em; margin: 0 0 1rem; text-wrap: balance; }}
.pd-lead {{ font-size: 1.15rem; line-height: 1.6; color: var(--nb-muted); max-width: 56ch; margin: 0; }}
.pd-h2 {{ font-family: var(--nb-font-display); font-size: 1.4rem; font-weight: 600; letter-spacing: -0.035em;
  word-spacing: -0.15em; margin: 2.5rem 0 0.5rem; color: var(--nb-fg); text-wrap: balance; }}
.pd-h2 small {{ display: block; font-size: 0.72rem; letter-spacing: 0.08em; word-spacing: 0; text-transform: uppercase;
  color: var(--nb-accent); margin-bottom: 0.35rem; }}
.pd-text {{ line-height: 1.6; margin: 0 0 0.75rem; }}
.pd-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.75rem; margin: 1rem 0; }}
.pd-step {{ background: var(--nb-surface); border: 1px solid var(--nb-border); border-radius: 6px; padding: 1rem; }}
.pd-step b {{ display: block; font-family: var(--nb-font-display); font-size: 0.9rem; margin-bottom: 0.35rem; }}
.pd-step span {{ font-size: 0.9rem; line-height: 1.5; color: var(--nb-muted); }}
.pd-result {{ background: var(--nb-accent-soft); border-left: 3px solid var(--nb-accent); border-radius: 0 6px 6px 0;
  padding: 1.1rem 1.3rem; margin: 1.25rem 0 0.5rem; }}
.pd-result p {{ margin: 0.6rem 0 0; line-height: 1.5; }}
.pd-big {{ font-family: var(--nb-font-display); font-size: clamp(2.4rem, 7vw, 3.4rem); font-weight: 600;
  letter-spacing: -0.04em; line-height: 1; color: var(--nb-accent); font-variant-numeric: tabular-nums; }}
.pd-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; margin: 1rem 0 1.25rem; }}
.pd-stat {{ border-top: 3px solid var(--nb-accent); background: var(--nb-surface); border-radius: 0 0 6px 6px; padding: 0.9rem 1rem; }}
.pd-stat > b {{ display: block; font-family: var(--nb-font-display); font-size: 1.6rem; font-weight: 600; letter-spacing: -0.03em;
  color: var(--nb-accent); font-variant-numeric: tabular-nums; margin-bottom: 0.3rem; }}
.pd-stat span {{ font-size: 0.88rem; line-height: 1.5; color: var(--nb-muted); }}
.pd-list {{ padding-left: 1.2rem !important; line-height: 1.6; }}
.pd-list li {{ margin-bottom: 0.5rem; }}
{PRINT_CSS}
</style>
""")


def heading(anchor, sections):
    """Encabezado numerado de sección; sections es la lista [(ancla, título)] del índice."""
    number = [a for a, _ in sections].index(anchor) + 1
    st.html(f'<h2 class="nb-h2" id="{anchor}"><span>{number}</span>{dict(sections)[anchor]}</h2>')


def toc(sections, title):
    """Índice con enlaces a cada sección (para la barra lateral)."""
    items = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in sections)
    st.html(f'<div class="nb-toc-title">{title}</div><nav class="nb-toc"><ol>{items}</ol></nav>')


def presentation_controls(sections, key="slide"):
    """Interruptor de modo presentación con flechas y selector de sección.

    Devuelve el índice de la sección activa, o None si el modo está apagado.
    """
    if not st.toggle("Modo presentación", key=f"{key}_on"):
        return None
    st.session_state.setdefault(key, 0)
    last = len(sections) - 1

    def move(step):
        st.session_state[key] = min(max(st.session_state[key] + step, 0), last)

    back, forward = st.columns(2)
    back.button("Anterior", icon=":material/arrow_back:", key=f"{key}_back", on_click=move, args=(-1,),
                disabled=st.session_state[key] == 0, width="stretch")
    forward.button("Siguiente", icon=":material/arrow_forward:", key=f"{key}_forward", on_click=move, args=(1,),
                   disabled=st.session_state[key] == last, width="stretch")
    st.selectbox("Sección", range(len(sections)), format_func=lambda i: f"{i + 1}. {sections[i][1]}", key=key)
    st.caption(f"Sección {st.session_state[key] + 1} de {len(sections)}")
    return st.session_state[key]


def presentation_styles(sections, index):
    """Muestra solo la sección activa del reporte, como un carrusel.

    Los encabezados (h2.nb-h2 con el ancla como id) son hermanos del contenido que los sigue, así que basta ocultar
    lo que está antes del encabezado activo y desde el encabezado siguiente en adelante. La primera sección conserva
    la portada del reporte.
    """
    block = REPORT_BLOCK
    rules = []
    if index > 0:
        rules.append(f'{block} > *:has(~ * h2#{sections[index][0]})')
    if index < len(sections) - 1:
        following = sections[index + 1][0]
        rules += [f'{block} > *:has(h2#{following})', f'{block} > *:has(h2#{following}) ~ *']
    if rules:
        # Solo en pantalla: al imprimir o guardar como PDF se incluyen todas las secciones
        st.html(f"<style>@media screen {{ {', '.join(rules)} {{ display: none !important; }} }}</style>")


def callout(kind, label, text):
    """Recuadro destacado; kind es "result", "tip" o "warning"."""
    st.html(f'<div class="nb-callout" data-kind="{kind}"><span class="nb-callout-label">{label}</span><p>{text}</p></div>')


def caption(text):
    """Pie de figura con numeración automática."""
    _figures["count"] += 1
    st.html(f'<p class="nb-caption"><b>Figura {_figures["count"]}.</b> {text}</p>')


def table(header, rows, best=None, note=None):
    def cell(value):
        return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else value

    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(
        f'<tr class="{"best" if i == best else ""}">' + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>"
        for i, row in enumerate(rows)
    )
    st.html(f'<div style="overflow-x:auto"><table class="nb-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
    if note:
        st.html(f'<p class="nb-caption">{note}</p>')


def cards(kind, items):
    """Tarjetas (título, texto) en rejilla; kind es "stat" (cifra destacada) o "step" (paso numerado)."""
    body = "".join(f'<div class="pd-{kind}"><b>{title}</b><span>{text}</span></div>' for title, text in items)
    return f'<div class="pd-{kind}s">{body}</div>'


def references(items):
    """Lista numerada de referencias; items son diccionarios con "text" y, opcionalmente, "url" y "label"."""
    def item(ref):
        link = (f' <a href="{ref["url"]}" target="_blank" rel="noreferrer">{ref.get("label", ref["url"])}</a>'
                if ref.get("url") else "")
        return f"<li>{ref['text']}{link}</li>"

    st.html(f'<ol class="nb-refs">{"".join(item(ref) for ref in items)}</ol>')


def series_chart(frame, y_title, labels, height=320):
    """Serie temporal (labels: columna → nombre; "Real" es la serie observada): el valor real en línea gruesa, los pronósticos en líneas finas y la persistencia punteada."""
    frame = frame.rename(columns=labels)
    present = [s for s in labels.values() if s in frame.columns]
    long = frame.reset_index().melt(id_vars="datetime", var_name="Serie", value_name="MW")
    base = alt.Chart(long).encode(
        x=alt.X("datetime:T", title=None),
        y=alt.Y("MW:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color("Serie:N", sort=present, scale=alt.Scale(domain=present),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("datetime:T", title="Fecha"), "Serie:N", alt.Tooltip("MW:Q", format=",.0f")],
    )
    real = base.transform_filter(alt.datum.Serie == "Real").mark_line(strokeWidth=2.4)
    pred = base.transform_filter(alt.datum.Serie != "Real").mark_line(strokeWidth=1.3).encode(
        strokeDash=alt.condition(alt.datum.Serie == "Persistencia 24 h", alt.value([4, 3]), alt.value([1, 0])))
    st.altair_chart(alt.layer(real, pred).properties(height=height).interactive(bind_y=False), width="stretch")


def bar_chart(frame, x, y, x_title, y_title, height=240, zero=True):
    chart = alt.Chart(frame).mark_bar().encode(
        x=alt.X(f"{x}:O", title=x_title, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=zero)),
        tooltip=[x, alt.Tooltip(f"{y}:Q", format=",.1f")],
    )
    st.altair_chart(chart.properties(height=height), width="stretch")


def fold_chart(folds, height=280):
    """RMSE de validación por fold: barras agrupadas por modelo."""
    long = (pd.DataFrame(folds, index=[f"Fold {i}" for i in range(1, 6)]).reset_index(names="Fold")
            .melt(id_vars="Fold", var_name="Modelo", value_name="RMSE"))
    st.altair_chart(
        alt.Chart(long).mark_bar().encode(
            x=alt.X("Fold:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="Modelo:N",
            y=alt.Y("RMSE:Q", title="RMSE de validación (MW)"),
            color=alt.Color("Modelo:N", sort=list(folds), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Fold", "Modelo", alt.Tooltip("RMSE:Q", format=",.1f")],
        ).properties(height=height),
        width="stretch",
    )
