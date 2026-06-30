#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  QUANT DASHBOARD  ·  Análisis, Optimización, Predicción y Riesgo
================================================================================
Herramienta cuantitativa para tomadores de decisiones. Universo personalizable.
Cuatro módulos:
  1. Análisis cuantitativo (retorno/riesgo por activo, correlaciones)
  2. Optimización de portafolio (frontera eficiente de Markowitz)
  3. Predicción 1/3/5 años (proyección probabilística) + backtest ML walk-forward
  4. Riesgo del portafolio (VaR, CVaR, drawdown)

Datos reales vía Yahoo Finance. NO es asesoría de inversión. Las proyecciones
asumen estacionariedad del proceso de retornos (supuesto fuerte y frecuentemente
violado). El backtest mide skill out-of-sample sin maquillaje.

Deploy:   requirements.txt -> streamlit, yfinance, pandas, numpy, plotly,
          scikit-learn, scipy
Local:    python3 -m streamlit run streamlit_app.py
================================================================================
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor

DIAS = 252

# ------------------------------------------------------------------ Página + tema
st.set_page_config(page_title="Quant Dashboard", page_icon="📊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#f5f6f8; --surface:#ffffff; --surface2:#f0f2f5; --border:#e4e7ec;
  --text:#161a23; --text2:#5b6472; --text3:#98a0ac;
  --accent:#6366f1; --accent2:#0ea5e9; --green:#10b981; --red:#ef4444; --amber:#f59e0b;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 560px at 85% -12%, rgba(99,102,241,.08), transparent 60%),
    radial-gradient(800px 460px at -5% 0%, rgba(14,165,233,.06), transparent 55%),
    #f5f6f8;
}
[data-testid="stHeader"]{background:rgba(0,0,0,0);}
[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid var(--border);}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:var(--text);}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;color:var(--text);letter-spacing:-.02em;}
.block-container{padding-top:2.4rem;max-width:1440px;}

[data-testid="stMetric"]{
  background:#ffffff;border:1px solid var(--border);border-radius:16px;padding:1.1rem 1.3rem;
  box-shadow:0 1px 3px rgba(16,24,40,.06),0 1px 2px rgba(16,24,40,.04);transition:.2s;
}
[data-testid="stMetric"]:hover{border-color:rgba(99,102,241,.45);
  box-shadow:0 6px 20px rgba(99,102,241,.10);}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif;font-size:1.55rem;
  font-weight:600;color:var(--text);}
[data-testid="stMetricLabel"]{color:var(--text3);font-size:.78rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.05em;}
[data-testid="stMetricDelta"]{font-family:'JetBrains Mono',monospace;font-size:.82rem;}

.stTabs [data-baseweb="tab-list"]{gap:4px;background:#eef0f4;border:1px solid var(--border);
  border-radius:14px;padding:5px;}
.stTabs [data-baseweb="tab"]{font-size:.9rem;font-weight:600;color:var(--text2);
  padding:9px 18px;border-radius:10px;}
.stTabs [data-baseweb="tab"]:hover{color:var(--text);}
.stTabs [aria-selected="true"]{background:var(--accent);color:#fff !important;
  box-shadow:0 4px 14px rgba(99,102,241,.35);}
.stTabs [data-baseweb="tab-highlight"]{background:transparent;}

.stTextArea textarea,.stTextInput input,.stNumberInput input{
  background:#ffffff !important;border:1px solid var(--border) !important;
  border-radius:10px !important;color:var(--text) !important;
  font-family:'JetBrains Mono',monospace !important;}

.mono{font-family:'JetBrains Mono',monospace;}
.tag{display:inline-flex;align-items:center;background:rgba(99,102,241,.10);
  border:1px solid rgba(99,102,241,.28);border-radius:8px;padding:3px 10px;
  font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#4f46e5;margin:3px 3px 0 0;}
.warn{background:linear-gradient(180deg,rgba(245,158,11,.10),rgba(245,158,11,.04));
  border:1px solid rgba(245,158,11,.3);border-radius:12px;padding:.9rem 1.1rem;
  font-size:.85rem;color:#b45309;line-height:1.55;}
.hdr{font-family:'JetBrains Mono',monospace;font-size:.74rem;color:var(--text3);
  text-transform:uppercase;letter-spacing:.12em;margin:.2rem 0 .5rem;display:block;}
.bigtitle{font-family:'Space Grotesk',sans-serif;font-size:2.4rem;font-weight:700;
  background:linear-gradient(135deg,#161a23 25%,#6366f1 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin:0 0 .2rem;letter-spacing:-.03em;}
.subtitle{color:var(--text2);font-size:.98rem;margin:0 0 1.2rem;}
hr{border-color:var(--border);}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

# Plotly dark template
PLOT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", color="#5b6472", size=11))
GRID = "rgba(16,24,40,.08)"
ACCENT = "#6366f1"; GREEN = "#10b981"; RED = "#ef4444"; AMBER = "#f59e0b"; PURP = "#8b5cf6"


# ------------------------------------------------------------------ Data
@st.cache_data(ttl=1800, show_spinner=False)
def bajar(tickers, anos):
    fin = datetime.today(); ini = fin - timedelta(days=int(365.25 * anos))
    d = yf_download(tickers, ini, fin)
    return d


def yf_download(tickers, ini, fin):
    import yfinance as yf
    data = yf.download(tickers, start=ini.strftime("%Y-%m-%d"),
                       end=fin.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"]
    else:
        px = data[["Close"]]; px.columns = tickers if isinstance(tickers, list) else [tickers]
    return px.dropna(how="all").ffill()


# ------------------------------------------------------------------ Métricas
def max_drawdown(valor):
    peak = np.maximum.accumulate(valor)
    dd = (valor - peak) / peak
    return dd.min()


def metricas_serie(rets, rf_anual, bench_rets=None):
    mu = rets.mean() * DIAS
    sigma = rets.std() * np.sqrt(DIAS)
    sharpe = (mu - rf_anual) / sigma if sigma > 0 else np.nan
    downside = rets[rets < 0].std() * np.sqrt(DIAS)
    sortino = (mu - rf_anual) / downside if downside > 0 else np.nan
    valor = (1 + rets).cumprod()
    mdd = max_drawdown(valor.values)
    beta = np.nan
    if bench_rets is not None:
        df = pd.concat([rets, bench_rets], axis=1).dropna()
        if len(df) > 5 and df.iloc[:, 1].var() > 0:
            beta = df.cov().iloc[0, 1] / df.iloc[:, 1].var()
    return dict(ret=mu, vol=sigma, sharpe=sharpe, sortino=sortino, mdd=mdd, beta=beta)


# ------------------------------------------------------------------ Markowitz
def frontera(retornos, rf_anual, n_port=8000, semilla=7):
    rng = np.random.default_rng(semilla)
    activos = retornos.columns.tolist(); n = len(activos)
    mu = retornos.mean().values * DIAS
    cov = retornos.cov().values * DIAS
    W = rng.dirichlet(np.ones(n), size=n_port)
    rp = W @ mu
    vp = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    sp = (rp - rf_anual) / vp
    return W, rp, vp, sp, activos


# ------------------------------------------------------------------ ML features
def construir_features(precio):
    df = pd.DataFrame(index=precio.index)
    r = precio.pct_change()
    df["ret1"] = r
    df["ret5"] = precio.pct_change(5)
    df["ret21"] = precio.pct_change(21)
    df["mom63"] = precio.pct_change(63)
    df["vol21"] = r.rolling(21).std()
    df["sma_ratio"] = precio / precio.rolling(50).mean() - 1
    # RSI 14
    delta = precio.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    return df


def backtest_ml(precio, horizonte_dias, min_train=252):
    """Walk-forward: reentrena mensual, predice retorno forward. Devuelve métricas OOS."""
    feats = construir_features(precio)
    target = precio.shift(-horizonte_dias) / precio - 1.0
    data = feats.copy(); data["y"] = target
    data = data.dropna()
    if len(data) < min_train + horizonte_dias + 40:
        return None
    X = data.drop(columns="y").values; y = data["y"].values
    preds, reales, naive = [], [], []
    paso = 21
    for i in range(min_train, len(data) - 1, paso):
        Xtr, ytr = X[:i], y[:i]
        Xte = X[i:i + paso]; yte = y[i:i + paso]
        m = GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                      learning_rate=0.05, subsample=0.8, random_state=0)
        m.fit(Xtr, ytr)
        preds.extend(m.predict(Xte)); reales.extend(yte); naive.extend([ytr.mean()] * len(yte))
    preds, reales, naive = map(np.array, (preds, reales, naive))
    ss_res = np.sum((reales - preds) ** 2)
    ss_tot = np.sum((reales - naive) ** 2)
    r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan          # vs naive (media)
    hit = np.mean(np.sign(preds) == np.sign(reales)) * 100
    mae = np.mean(np.abs(reales - preds))
    return dict(r2=r2_oos, hit=hit, mae=mae, n=len(preds),
                preds=preds, reales=reales)


# ------------------------------------------------------------------ Monte Carlo proyección
def proyeccion(mu, sigma, anos_lista, n_sims=10000, semilla=11):
    rng = np.random.default_rng(semilla)
    out = {}
    for a in anos_lista:
        dias = int(a * DIAS)
        # log-normal: simular suma de retornos log diarios
        mu_d = mu / DIAS; sig_d = sigma / np.sqrt(DIAS)
        shocks = rng.normal(mu_d - 0.5 * sig_d ** 2, sig_d, size=(n_sims, dias))
        growth = np.exp(shocks.sum(axis=1))
        out[a] = dict(p5=np.percentile(growth, 5), p50=np.percentile(growth, 50),
                      p95=np.percentile(growth, 95), prob_pos=(growth > 1).mean() * 100,
                      esperado=growth.mean(), muestra=growth)
    return out


# ================================================================== SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("<span class='hdr'>Mis empresas y grupos</span>", unsafe_allow_html=True)
    st.caption("Escribe tus tickers y asígnales un grupo (sector, país, lo que quieras). "
               "El botón ➕ agrega filas.")
    _default = pd.DataFrame({
        "Ticker": ["WALMEX.MX", "GFNORTEO.MX", "GMEXICOB.MX", "AMXB.MX",
                   "VOO", "AAPL", "MSFT"],
        "Grupo": ["Consumo MX", "Banca MX", "Industria MX", "Telecom MX",
                  "USA Índice", "USA Tech", "USA Tech"],
    })
    holdings = st.data_editor(
        _default, num_rows="dynamic", use_container_width=True, key="holdings",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", required=True),
            "Grupo": st.column_config.TextColumn("Grupo", required=True,
                     help="Etiqueta para agrupar (ej. Banca MX, USA Tech)."),
        })
    holdings = holdings.dropna(subset=["Ticker"]).copy()
    holdings["Ticker"] = holdings["Ticker"].astype(str).str.strip().str.upper()
    holdings["Grupo"] = (holdings["Grupo"].astype(str).str.strip()
                         .replace({"": "Sin grupo", "NAN": "Sin grupo"}))
    holdings = holdings[holdings["Ticker"] != ""]
    grupos_disp = sorted(holdings["Grupo"].unique())
    grupos_sel = st.multiselect("Grupos a analizar", grupos_disp, default=grupos_disp,
                                help="Filtra el dashboard a los grupos que elijas.")
    st.divider()
    bench = st.text_input("Benchmark", "^MXX", help="^MXX = IPC, ^GSPC = S&P 500")
    anos_hist = st.slider("Años de historia", 2, 15, 7)
    rf = st.number_input("Tasa libre de riesgo anual (%)", 0.0, 25.0, 9.0, 0.5) / 100
    h_ml = st.select_slider("Horizonte del modelo ML (meses)", [1, 3, 6, 12], 3)
    st.caption("Proyección de portafolio fija a 1, 3 y 5 años.")

holdings_f = holdings[holdings["Grupo"].isin(grupos_sel)]
tickers = holdings_f["Ticker"].drop_duplicates().tolist()
grupo_de = dict(zip(holdings_f["Ticker"], holdings_f["Grupo"]))
if len(tickers) < 2:
    st.info("Selecciona al menos 2 empresas (revisa tickers y grupos en la barra lateral).")
    st.stop()

# ================================================================== HEADER
st.markdown("<div class='bigtitle'>Quant Dashboard</div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Análisis · Optimización · Predicción · Riesgo — "
            "datos reales, métricas sin maquillaje.</p>", unsafe_allow_html=True)
st.markdown("<span class='hdr'>Universo por grupo</span>", unsafe_allow_html=True)
_html = ""
for _g in grupos_sel:
    _ts = [t for t in tickers if grupo_de.get(t) == _g]
    if not _ts:
        continue
    _html += (f"<span class='tag' style='background:rgba(14,165,233,.10);"
              f"border-color:rgba(14,165,233,.30);color:#0369a1;font-weight:600'>{_g}</span> ")
    _html += " ".join(f"<span class='tag'>{t}</span>" for t in _ts) + "<br>"
_html += (f"<span class='tag' style='color:#b45309;background:rgba(245,158,11,.10);"
          f"border-color:rgba(245,158,11,.30)'>bench: {bench}</span>")
st.markdown(_html, unsafe_allow_html=True)

with st.spinner("Cargando series y calculando..."):
    precios = bajar(tickers, anos_hist)
    bench_px = bajar(bench, anos_hist)

if precios.empty:
    st.error("No se pudieron descargar precios. Revisa los tickers o tu conexión."); st.stop()

validos = [t for t in tickers if t in precios.columns and precios[t].notna().sum() > 120]
fallidos = [t for t in tickers if t not in validos]
if fallidos:
    st.markdown(f"<div class='warn'>⚠️ Sin datos suficientes (omitidos): "
                f"{', '.join(fallidos)}</div>", unsafe_allow_html=True)
if len(validos) < 2:
    st.error("Se necesitan al menos 2 tickers válidos."); st.stop()

precios = precios[validos].dropna()
retornos = precios.pct_change().dropna()
bench_ret = None
if not bench_px.empty:
    bser = bench_px.iloc[:, 0] if isinstance(bench_px, pd.DataFrame) else bench_px
    bench_ret = bser.pct_change().dropna()
    bench_ret.name = "bench"

tab1, tab2, tab3, tab4 = st.tabs(["① Análisis", "② Optimización", "③ Predicción", "④ Riesgo"])

# ================================================================== TAB 1
with tab1:
    st.markdown("<span class='hdr'>Métricas por activo · anualizadas</span>", unsafe_allow_html=True)
    filas = []
    for t in validos:
        m = metricas_serie(retornos[t], rf, bench_ret)
        filas.append([t, m["ret"] * 100, m["vol"] * 100, m["sharpe"], m["sortino"],
                      m["mdd"] * 100, m["beta"]])
    dfm = pd.DataFrame(filas, columns=["Ticker", "Retorno %", "Vol %", "Sharpe",
                                       "Sortino", "Max DD %", "Beta"]).set_index("Ticker")
    st.dataframe(dfm.style.format({"Retorno %": "{:+.1f}", "Vol %": "{:.1f}",
                 "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max DD %": "{:.1f}",
                 "Beta": "{:.2f}"}).background_gradient(subset=["Sharpe"], cmap="RdYlGn"),
                 use_container_width=True)

    # ---- Métricas agregadas por GRUPO (cartera equiponderada de cada grupo) ----
    grupos_act = [g for g in grupos_sel if any(grupo_de.get(t) == g for t in validos)]
    if len(grupos_act) >= 1:
        st.markdown("<span class='hdr'>Métricas por grupo · cartera equiponderada del grupo</span>",
                    unsafe_allow_html=True)
        filas_g = []
        for g in grupos_act:
            ts = [t for t in validos if grupo_de.get(t) == g]
            rg = retornos[ts].mean(axis=1)
            mg = metricas_serie(rg, rf, bench_ret)
            filas_g.append([g, len(ts), mg["ret"]*100, mg["vol"]*100, mg["sharpe"], mg["mdd"]*100])
        dfg = pd.DataFrame(filas_g, columns=["Grupo", "# emp.", "Retorno %", "Vol %",
                           "Sharpe", "Max DD %"]).set_index("Grupo")
        gc1, gc2 = st.columns([3, 2])
        with gc1:
            st.dataframe(dfg.style.format({"Retorno %": "{:+.1f}", "Vol %": "{:.1f}",
                         "Sharpe": "{:.2f}", "Max DD %": "{:.1f}"})
                         .background_gradient(subset=["Sharpe"], cmap="RdYlGn"),
                         use_container_width=True)
        with gc2:
            figg = go.Figure(go.Bar(x=dfg["Retorno %"], y=dfg.index, orientation="h",
                             marker_color=ACCENT, text=dfg["Retorno %"].round(1),
                             texttemplate="%{text}%", textposition="outside"))
            figg.update_layout(height=max(180, 60*len(dfg)), margin=dict(t=6, b=6, l=6, r=20),
                               **PLOT, xaxis=dict(title="Retorno anual %", gridcolor=GRID),
                               yaxis=dict(gridcolor=GRID))
            st.plotly_chart(figg, use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("<span class='hdr'>Retorno acumulado · base 100</span>", unsafe_allow_html=True)
        fig = go.Figure()
        for t in validos:
            base = (1 + retornos[t]).cumprod() * 100
            fig.add_trace(go.Scatter(x=base.index, y=base, name=t, mode="lines",
                                     line=dict(width=1.4)))
        fig.update_layout(height=380, margin=dict(t=6, b=6, l=6, r=6), **PLOT,
                          xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
                          legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("<span class='hdr'>Matriz de correlación</span>", unsafe_allow_html=True)
        corr = retornos.corr()
        figc = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                         colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                         text=np.round(corr.values, 2), texttemplate="%{text}",
                         textfont=dict(size=9)))
        figc.update_layout(height=380, margin=dict(t=6, b=6, l=6, r=6), **PLOT)
        st.plotly_chart(figc, use_container_width=True)

# ================================================================== TAB 2
with tab2:
    st.markdown("<span class='hdr'>Frontera eficiente · Markowitz (8,000 portafolios)</span>",
                unsafe_allow_html=True)
    W, rp, vp, sp, activos = frontera(retornos, rf)
    i_sharpe = int(np.argmax(sp)); i_minvar = int(np.argmin(vp))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vp * 100, y=rp * 100, mode="markers",
                  marker=dict(size=4, color=sp, colorscale="Viridis",
                  showscale=True, colorbar=dict(title="Sharpe")), name="Portafolios"))
    fig.add_trace(go.Scatter(x=[vp[i_sharpe] * 100], y=[rp[i_sharpe] * 100],
                  mode="markers", marker=dict(size=16, color=GREEN, symbol="star"),
                  name="Máx Sharpe"))
    fig.add_trace(go.Scatter(x=[vp[i_minvar] * 100], y=[rp[i_minvar] * 100],
                  mode="markers", marker=dict(size=14, color=ACCENT, symbol="diamond"),
                  name="Mín Varianza"))
    fig.update_layout(height=440, margin=dict(t=6, b=6, l=6, r=6), **PLOT,
                      xaxis=dict(title="Volatilidad anual %", gridcolor=GRID),
                      yaxis=dict(title="Retorno anual %", gridcolor=GRID),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    cc1, cc2 = st.columns(2)
    for col, idx, nombre, col_acc in [(cc1, i_sharpe, "Máximo Sharpe", GREEN),
                                       (cc2, i_minvar, "Mínima Varianza", ACCENT)]:
        with col:
            st.markdown(f"<span class='hdr' style='color:{col_acc}'>{nombre}</span>",
                        unsafe_allow_html=True)
            a, b, c = st.columns(3)
            a.metric("Retorno", f"{rp[idx]*100:.1f}%")
            b.metric("Vol", f"{vp[idx]*100:.1f}%")
            c.metric("Sharpe", f"{sp[idx]:.2f}")
            pesos = pd.DataFrame({"Ticker": activos, "Peso %": W[idx] * 100})
            pesos = pesos[pesos["Peso %"] > 0.5].sort_values("Peso %", ascending=False)
            st.dataframe(pesos.style.format({"Peso %": "{:.1f}"}),
                         use_container_width=True, hide_index=True)
    st.session_state["w_sharpe"] = W[i_sharpe]
    st.session_state["activos_opt"] = activos

# ================================================================== TAB 3
with tab3:
    st.markdown("<span class='hdr'>Proyección probabilística del portafolio · 1 / 3 / 5 años</span>",
                unsafe_allow_html=True)
    modo = st.radio("Portafolio a proyectar", ["Máx Sharpe (óptimo)", "Equiponderado"],
                    horizontal=True)
    if modo.startswith("Máx") and "w_sharpe" in st.session_state:
        w = st.session_state["w_sharpe"]
    else:
        w = np.repeat(1 / len(validos), len(validos))
    ret_port = (retornos[validos] * w).sum(axis=1)
    mu_p = ret_port.mean() * DIAS; sig_p = ret_port.std() * np.sqrt(DIAS)

    a, b, c = st.columns(3)
    a.metric("Retorno esperado anual", f"{mu_p*100:+.1f}%")
    b.metric("Volatilidad anual", f"{sig_p*100:.1f}%")
    c.metric("Sharpe", f"{(mu_p-rf)/sig_p:.2f}" if sig_p > 0 else "—")

    proy = proyeccion(mu_p, sig_p, [1, 3, 5])
    filas = []
    for yr in [1, 3, 5]:
        p = proy[yr]
        filas.append([f"{yr} año(s)", (p["p50"]-1)*100, (p["p5"]-1)*100,
                      (p["p95"]-1)*100, p["prob_pos"]])
    dfp = pd.DataFrame(filas, columns=["Horizonte", "Mediana %", "Pesimista p5 %",
                       "Optimista p95 %", "Prob. > 0 %"]).set_index("Horizonte")
    st.dataframe(dfp.style.format({"Mediana %": "{:+.1f}", "Pesimista p5 %": "{:+.1f}",
                 "Optimista p95 %": "{:+.1f}", "Prob. > 0 %": "{:.0f}"}),
                 use_container_width=True)

    # Fan chart
    fig = go.Figure()
    eje = np.arange(0, int(5 * DIAS) + 1) / DIAS
    rng = np.random.default_rng(3)
    mu_d = mu_p / DIAS; sig_d = sig_p / np.sqrt(DIAS)
    sims = np.exp(np.cumsum(rng.normal(mu_d - 0.5*sig_d**2, sig_d,
                  size=(4000, len(eje)-1)), axis=1))
    sims = np.hstack([np.ones((4000, 1)), sims]) * 100
    for lo, hi, al in [(5, 95, .12), (25, 75, .22)]:
        fig.add_trace(go.Scatter(x=eje, y=np.percentile(sims, hi, axis=0),
                      mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=eje, y=np.percentile(sims, lo, axis=0), mode="lines",
                      line=dict(width=0), fill="tonexty",
                      fillcolor=f"rgba(88,166,255,{al})", name=f"p{lo}-p{hi}"))
    fig.add_trace(go.Scatter(x=eje, y=np.percentile(sims, 50, axis=0), mode="lines",
                  line=dict(color=ACCENT, width=2), name="Mediana"))
    fig.update_layout(height=360, margin=dict(t=6, b=6, l=6, r=6), **PLOT,
                      xaxis=dict(title="Años", gridcolor=GRID),
                      yaxis=dict(title="Índice (base 100)", gridcolor=GRID),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class='warn'>⚠️ La proyección asume que μ={mu_p*100:.1f}% y
    σ={sig_p*100:.1f}% (estimados del histórico) se mantienen. Es un supuesto fuerte:
    regímenes de mercado, crisis y cambios estructurales rompen esta hipótesis. Úsala
    como rango de escenarios, no como pronóstico puntual.</div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<span class='hdr'>Backtest del modelo ML · ¿hay skill predictiva?</span>",
                unsafe_allow_html=True)
    activo_ml = st.selectbox("Activo a modelar", validos)
    with st.spinner("Entrenando walk-forward..."):
        res = backtest_ml(precios[activo_ml], int(h_ml * 21))
    if res is None:
        st.info("Historia insuficiente para el backtest. Sube los años de historia.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² out-of-sample", f"{res['r2']:+.3f}",
                  help="vs predicción naïve (media). >0 = el modelo aporta. Suele rondar 0.")
        m2.metric("Hit rate (dirección)", f"{res['hit']:.1f}%",
                  help="% de veces que acierta el signo. 50% = volado.")
        m3.metric("MAE", f"{res['mae']*100:.2f}%")
        m4.metric("Predicciones OOS", f"{res['n']}")
        figb = go.Figure()
        figb.add_trace(go.Scatter(x=res["reales"]*100, y=res["preds"]*100, mode="markers",
                       marker=dict(size=5, color=ACCENT, opacity=.5), name="OOS"))
        lim = max(abs(res["reales"]).max(), abs(res["preds"]).max()) * 100
        figb.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                       line=dict(color=AMBER, dash="dash", width=1), name="Perfecto"))
        figb.update_layout(height=320, margin=dict(t=6, b=6, l=6, r=6), **PLOT,
                           xaxis=dict(title=f"Retorno real {h_ml}m %", gridcolor=GRID),
                           yaxis=dict(title="Retorno predicho %", gridcolor=GRID),
                           legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(figb, use_container_width=True)
        veredicto = ("El modelo NO supera a la media: el R² OOS es ≤ 0. Esto es lo "
                     "normal y confirma lo difícil que es predecir retornos."
                     if (np.isnan(res["r2"]) or res["r2"] <= 0) else
                     "El R² OOS es positivo: el modelo aporta algo de información. "
                     "Trátalo con cautela y vigila estabilidad temporal.")
        st.markdown(f"<div class='warn'>📌 {veredicto}</div>", unsafe_allow_html=True)

# ================================================================== TAB 4
with tab4:
    st.markdown("<span class='hdr'>Riesgo del portafolio (Máx Sharpe)</span>",
                unsafe_allow_html=True)
    w = st.session_state.get("w_sharpe", np.repeat(1/len(validos), len(validos)))
    ret_port = (retornos[validos] * w).sum(axis=1)

    conf = st.select_slider("Nivel de confianza", [90, 95, 99], 95)
    a = (100 - conf) / 100
    var_h = np.percentile(ret_port, a * 100)
    cvar_h = ret_port[ret_port <= var_h].mean()
    valor = (1 + ret_port).cumprod()
    mdd = max_drawdown(valor.values)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"VaR {conf}% (diario)", f"{var_h*100:.2f}%",
              help="Pérdida diaria que solo se supera (100-conf)% del tiempo.")
    k2.metric(f"CVaR {conf}% (diario)", f"{cvar_h*100:.2f}%",
              help="Pérdida promedio en la cola más allá del VaR.")
    k3.metric("Máx Drawdown", f"{mdd*100:.1f}%")
    k4.metric("Vol anual", f"{ret_port.std()*np.sqrt(DIAS)*100:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<span class='hdr'>Curva de drawdown</span>", unsafe_allow_html=True)
        peak = np.maximum.accumulate(valor)
        dd = (valor - peak) / peak * 100
        fig = go.Figure(go.Scatter(x=dd.index, y=dd, mode="lines", fill="tozeroy",
                        line=dict(color=RED, width=1), fillcolor="rgba(248,81,73,.15)"))
        fig.update_layout(height=320, margin=dict(t=6, b=6, l=6, r=6), **PLOT,
                          xaxis=dict(gridcolor=GRID), yaxis=dict(title="DD %", gridcolor=GRID))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("<span class='hdr'>Distribución de retornos diarios</span>",
                    unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=ret_port*100, nbinsx=80, marker_color=ACCENT,
                        opacity=.8))
        fig.add_vline(x=var_h*100, line=dict(color=AMBER, width=2, dash="dash"),
                      annotation_text=f"VaR {conf}%")
        fig.update_layout(height=320, margin=dict(t=6, b=6, l=6, r=6), **PLOT,
                          xaxis=dict(title="Retorno diario %", gridcolor=GRID),
                          yaxis=dict(title="Frec.", gridcolor=GRID))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='warn'>Herramienta cuantitativa de análisis, <b>no es asesoría "
            "de inversión</b>. Datos vía Yahoo Finance (posible retraso). Las métricas "
            "se calculan sobre la ventana histórica seleccionada; cambiar la ventana "
            "cambia los resultados. El desempeño pasado no garantiza resultados futuros."
            "</div>", unsafe_allow_html=True)
