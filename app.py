"""
Dashboard Sistem Peringatan Dini Anomali Pompa Pelarut (Proposal Tesis - Irfan Riady)
Dataset Kaggle yang sesuai: "Pump Sensor Data" (nphantawee/pump-sensor-data) -> sensor.csv
Jalankan: streamlit run app.py
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Early Warning Pompa Pelarut", page_icon="⚙️", layout="wide")

STATUS_MAP = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 2}
LABELS = {0: "Normal", 1: "Peringatan Dini", 2: "Abnormal"}
COLORS = {"Normal": "#1f9d63", "Peringatan Dini": "#f0a30a", "Abnormal": "#c9424f"}
MODELS = ["Isolation Forest", "PCA-Autoencoder"]


# ----------------------------------------------------------------- DATA
def synthetic() -> pd.DataFrame:
    """Data demo dengan skema sama seperti Kaggle Pump Sensor Data."""
    rng = np.random.default_rng(42)
    n, k = 60000, 12
    t = np.arange(n)
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    base, amp = rng.uniform(20, 80, k), rng.uniform(1, 4, k)
    data = (base + amp * np.sin(2 * np.pi * t[:, None] / 1440 + rng.uniform(0, 6, k))
            + rng.normal(0, 0.8, (n, k)))
    status = np.array(["NORMAL"] * n, dtype=object)
    for f in (14000, 31000, 50000):
        ramp = np.clip((t - (f - 4320)) / 4320, 0, 1) ** 2 * (t <= f)
        for j in (0, 2, 5, 7):
            data[:, j] += ramp * (8 + j)
        status[f:f + 60] = "BROKEN"
        status[f + 60:f + 360] = "RECOVERING"
        data[f:f + 360, :4] -= 15
    df = pd.DataFrame(data, columns=[f"sensor_{i:02d}" for i in range(k)])
    df.insert(0, "timestamp", idx)
    df["machine_status"] = status
    return df


@st.cache_resource(show_spinner="Mengunduh dataset dari Kaggle (kagglehub)...")
def kaggle_path():
    """Unduh dataset via kagglehub dan kembalikan path CSV-nya."""
    import kagglehub
    folder = kagglehub.dataset_download("nphantawee/pump-sensor-data")
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)
    raise FileNotFoundError("File CSV tidak ditemukan di " + folder)


@st.cache_data(show_spinner="Membaca data...")
def get_raw(path, file_bytes):
    if file_bytes is not None:
        comp = "zip" if file_bytes[:2] == b"PK" else "infer"
        return pd.read_csv(io.BytesIO(file_bytes), compression=comp)
    if path:
        return pd.read_csv(path)
    return synthetic()


@st.cache_data(show_spinner="Praproses data...")
def get_prepared(path, file_bytes, step):
    df = get_raw(path, file_bytes)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    tcol = next(c for c in df.columns if c.lower() in ("timestamp", "time", "datetime", "date"))
    scol = next(c for c in df.columns if "status" in c.lower())
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol).drop_duplicates(tcol).set_index(tcol)
    status = df.pop(scol).astype(str).str.upper().map(STATUS_MAP).fillna(0).astype(int)
    X = df.select_dtypes("number")
    missing = (X.isna().mean() * 100).sort_values(ascending=False)
    X = X.dropna(axis=1, how="all").ffill().bfill()
    X = X.loc[:, X.std() > 0]
    Xr = X.resample(f"{step}min").mean().ffill().bfill()
    sr = status.resample(f"{step}min").max().reindex(Xr.index).ffill().fillna(0).astype(int)
    Xr["status_code"] = sr
    return Xr, missing


def segments(mask: pd.Series):
    m = mask.values.astype(int)
    d = np.diff(np.r_[0, m, 0])
    s, e = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return [(mask.index[a], mask.index[b]) for a, b in zip(s, e)]


def make_labels(df, horizon_h):
    y = pd.Series(0, index=df.index)
    for s, _ in segments(df.status_code == 2):
        y[(df.index >= s - pd.Timedelta(hours=horizon_h)) & (df.index < s)] = 1
    y[df.status_code >= 1] = 2
    return y


def shade(fig, df):
    step = df.index[1] - df.index[0]
    for s, e in segments(df.status_code == 1):
        fig.add_vrect(x0=s, x1=e + step, fillcolor="#f0a30a", opacity=0.15, line_width=0)
    for s, e in segments(df.status_code == 2):
        fig.add_vrect(x0=s, x1=e + step, fillcolor="#c9424f", opacity=0.45, line_width=0)
    return fig


def thin(df, n=6000):
    return df.iloc[::max(1, len(df) // n)]


# --------------------------------------------------------------- MODELS
@st.cache_data(show_spinner="Melatih model tak terawasi...")
def fit_unsup(X, y):
    train = X[y == 0]
    sc = StandardScaler().fit(train)
    Z, Zt = sc.transform(X), sc.transform(train)
    iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1).fit(Zt)
    pca = PCA(n_components=0.95, svd_solver="full").fit(Zt)
    recon = pca.inverse_transform(pca.transform(Z))
    return pd.DataFrame({"Isolation Forest": -iso.score_samples(Z),
                         "PCA-Autoencoder": ((Z - recon) ** 2).mean(axis=1)}, index=X.index)


@st.cache_data(show_spinner="Melatih Random Forest...")
def fit_rf(X, y, tr, te, n_est):
    rf = RandomForestClassifier(n_estimators=n_est, class_weight="balanced_subsample",
                                min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(X.iloc[tr], y[tr])
    return rf.classes_, rf.predict_proba(X.iloc[te]), pd.Series(rf.feature_importances_, index=X.columns)


def split_idx(y, mode, test=0.3):
    idx = np.arange(len(y))
    if mode.startswith("Kronologis"):
        c = int(len(y) * (1 - test))
        return idx[:c], idx[c:]
    strat = y if pd.Series(y).value_counts().min() >= 2 else None
    return train_test_split(idx, test_size=test, stratify=strat, random_state=42)


def sustained(flag: pd.Series, k: int):
    return flag.astype(int).rolling(k, min_periods=k).sum() >= k


def episodes(alert: pd.Series, y: pd.Series):
    edges = alert & ~alert.shift(fill_value=False)
    return int((edges & (y == 0)).sum())


# -------------------------------------------------------------- SIDEBAR
st.sidebar.title("⚙️ Early Warning Pompa")
st.sidebar.caption("Proposal Tesis · Irfan Riady · Univ. Pakuan")
menu = st.sidebar.radio("Menu", ["🏠 Beranda", "📊 Ringkasan Data", "📈 Eksplorasi Sensor",
                                 "🚨 Deteksi Anomali", "🌲 Klasifikasi Status",
                                 "⚖️ Perbandingan Model", "⏱️ Peringatan Dini"])
st.sidebar.divider()
src = st.sidebar.radio("Sumber data", ["Kaggle (otomatis via kagglehub)", "Unggah sensor.csv",
                                       "Data demo (sintetis)"])
file_bytes, path, label = None, None, "Data demo (sintetis)"
if src.startswith("Kaggle"):
    try:
        path = kaggle_path()
        label = "Kaggle: nphantawee/pump-sensor-data"
    except Exception as e:
        st.sidebar.error(f"Gagal mengunduh dari Kaggle: {e}")
        if os.path.exists("sensor.csv"):
            path, label = "sensor.csv", "sensor.csv (lokal)"
        else:
            st.sidebar.warning("Memakai data demo.")
elif src.startswith("Unggah"):
    up = st.sidebar.file_uploader("Unggah sensor.csv atau archive.zip", type=["csv", "zip"])
    if up:
        file_bytes, label = up.getvalue(), "Unggahan: " + up.name
    elif os.path.exists("sensor.csv"):
        path, label = "sensor.csv", "sensor.csv (lokal)"
    else:
        st.sidebar.warning("Belum ada file, memakai data demo.")
step = st.sidebar.selectbox("Resolusi waktu (menit)", [1, 5, 10, 30, 60], index=2)
horizon = st.sidebar.slider("Horizon pra-kegagalan (jam)", 6, 504, 72, 6,
                            help="Baris dalam rentang ini sebelum BROKEN dilabeli 'Peringatan Dini'.")
st.sidebar.caption(f"Sumber aktif: {label}")

df, missing = get_prepared(path, file_bytes, step)
y = make_labels(df, horizon)
X = df.drop(columns="status_code")
sensors = list(X.columns)
ylab = y.map(LABELS)


# ---------------------------------------------------------------- PAGES
if menu == "🏠 Beranda":
    st.title("Sistem Peringatan Dini Anomali Pompa Pelarut")
    st.markdown("""
Dashboard ini mendukung proposal tesis **Sistem Peringatan Dini Berbasis AI/ML untuk Mendeteksi
Anomali dan Kondisi Abnormal Pompa Pelarut**.

**Dataset Kaggle yang sesuai:** [Pump Sensor Data (nphantawee)](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)
berisi ±52 sensor pompa industri, interval 1 menit, dengan status `NORMAL`, `RECOVERING`, dan `BROKEN`.
Data diunduh otomatis lewat `kagglehub`; alternatifnya unggah `sensor.csv` secara manual dari sidebar.

**Pemetaan ke tesis**
- Variabel keluaran: *Normal / Peringatan Dini / Abnormal* dibentuk dari status asli + horizon pra-kegagalan.
- Model: Isolation Forest, PCA-Autoencoder (pengganti ringan LSTM-Autoencoder), Random Forest.
- Metrik: akurasi, presisi, sensitivitas (recall), F1, ROC-AUC, dan *lead time* peringatan.

**Catatan penting**
- Data LEADS (tekanan, suhu seal, arus motor JTB) bersifat internal dan tidak ada di Kaggle; dataset ini hanya proxy multisensor.
- Angka literatur (mis. peringatan 3–14 hari, sinyal palsu <4%) tidak dihasilkan aplikasi ini; hasil di sini murni dari data yang dimuat.
- LSTM-Autoencoder penuh memerlukan TensorFlow dan tidak disertakan agar aplikasi tetap ringan.
""")

elif menu == "📊 Ringkasan Data":
    st.header("Ringkasan Data")
    c = st.columns(4)
    c[0].metric("Baris (setelah resample)", f"{len(df):,}")
    c[1].metric("Jumlah sensor", len(sensors))
    c[2].metric("Periode", f"{(df.index[-1] - df.index[0]).days} hari")
    c[3].metric("Kejadian BROKEN", len(segments(df.status_code == 2)))
    a, b = st.columns(2)
    cnt = ylab.value_counts().reset_index()
    cnt.columns = ["Status", "Jumlah"]
    a.plotly_chart(px.pie(cnt, names="Status", values="Jumlah", hole=0.5, color="Status",
                          color_discrete_map=COLORS, title="Distribusi status"), use_container_width=True)
    m = missing.head(15).reset_index()
    m.columns = ["Sensor", "Missing (%)"]
    b.plotly_chart(px.bar(m, x="Sensor", y="Missing (%)", title="Data hilang per sensor (15 teratas)"),
                   use_container_width=True)
    tl = thin(pd.DataFrame({"Kode": y}))
    fig = px.line(tl, y="Kode", line_shape="hv", title="Linimasa status (0 Normal, 1 Pra-kegagalan, 2 Abnormal)")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Pratinjau data"):
        st.dataframe(df.head(200))

elif menu == "📈 Eksplorasi Sensor":
    st.header("Eksplorasi Sensor")
    sel = st.multiselect("Pilih sensor", sensors, default=sensors[:3])
    z = st.checkbox("Standarisasi (z-score)", value=True)
    if sel:
        d = thin(X[sel])
        if z:
            d = (d - d.mean()) / d.std()
        fig = go.Figure([go.Scatter(x=d.index, y=d[s], name=s, mode="lines") for s in sel])
        shade(fig, df)
        fig.update_layout(title="Deret waktu (merah = BROKEN, oranye = RECOVERING)", height=420)
        st.plotly_chart(fig, use_container_width=True)
        a, b = st.columns(2)
        one = a.selectbox("Sensor untuk boxplot", sel)
        bx = pd.DataFrame({"Status": ylab, one: X[one]})
        a.plotly_chart(px.box(bx, x="Status", y=one, color="Status", color_discrete_map=COLORS,
                              title=f"{one} per status"), use_container_width=True)
        top = X.var().sort_values(ascending=False).head(15).index
        b.plotly_chart(px.imshow(X[top].corr(), zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                 title="Korelasi (15 sensor varians tertinggi)"), use_container_width=True)

elif menu == "🚨 Deteksi Anomali":
    st.header("Deteksi Anomali (dilatih hanya pada data normal)")
    if (y == 0).sum() < 50:
        st.error("Data normal terlalu sedikit."); st.stop()
    scores = fit_unsup(X, y.values)
    c = st.columns(3)
    model = c[0].selectbox("Model", MODELS)
    q = c[1].slider("Persentil ambang (data normal)", 90.0, 99.9, 99.0, 0.1)
    k = c[2].slider("Alert bila anomali berturut-turut", 1, 12, 3)
    sc = scores[model]
    thr = np.percentile(sc[y == 0], q)
    flag = sc > thr
    alert = sustained(flag, k)
    fig = go.Figure(go.Scatter(x=thin(sc).index, y=thin(sc), name="Skor anomali", line=dict(width=1)))
    fig.add_hline(y=thr, line_dash="dash", line_color="red", annotation_text=f"Ambang {thr:.3f}")
    shade(fig, df)
    fig.update_layout(title=f"Skor anomali - {model}", height=420)
    st.plotly_chart(fig, use_container_width=True)
    m = st.columns(4)
    m[0].metric("Alert pada data normal", f"{alert[y == 0].mean() * 100:.2f}%")
    m[1].metric("Alert pada pra-kegagalan", f"{alert[y == 1].mean() * 100:.1f}%" if (y == 1).any() else "-")
    m[2].metric("Alert pada abnormal", f"{alert[y == 2].mean() * 100:.1f}%" if (y == 2).any() else "-")
    m[3].metric("Episode alert (normal)", episodes(alert, y))
    st.plotly_chart(px.histogram(pd.DataFrame({"Skor": sc, "Status": ylab}), x="Skor", color="Status",
                                 nbins=80, barmode="overlay", opacity=0.6, color_discrete_map=COLORS,
                                 title="Distribusi skor per status"), use_container_width=True)

elif menu == "🌲 Klasifikasi Status":
    st.header("Klasifikasi Status: Random Forest")
    c = st.columns(2)
    mode = c[0].selectbox("Pembagian data", ["Acak berstrata", "Kronologis (70/30)"])
    n_est = c[1].slider("Jumlah pohon", 50, 500, 200, 50)
    if mode.startswith("Acak"):
        st.caption("⚠️ Pembagian acak pada deret waktu cenderung terlalu optimistis (kebocoran temporal). Gunakan kronologis untuk evaluasi yang lebih jujur.")
    tr, te = split_idx(y.values, mode)
    classes, proba, imp = fit_rf(X, y.values, tr, te, n_est)
    pred = classes[proba.argmax(1)]
    yt = y.values[te]
    m = st.columns(3)
    m[0].metric("Akurasi", f"{accuracy_score(yt, pred) * 100:.1f}%")
    m[1].metric("F1-macro", f"{f1_score(yt, pred, average='macro', zero_division=0):.3f}")
    m[2].metric("Data uji", f"{len(te):,}")
    names = [LABELS[i] for i in (0, 1, 2)]
    cm = confusion_matrix(yt, pred, labels=[0, 1, 2])
    a, b = st.columns(2)
    a.plotly_chart(px.imshow(cm, x=names, y=names, text_auto=True, color_continuous_scale="Viridis",
                             labels=dict(x="Prediksi", y="Aktual"), title="Confusion matrix (data uji)"),
                   use_container_width=True)
    top = imp.sort_values().tail(15)
    b.plotly_chart(px.bar(top, orientation="h", title="Kepentingan fitur (15 teratas)",
                          labels={"value": "Importance", "index": "Sensor"}).update_layout(showlegend=False),
                   use_container_width=True)
    rep = pd.DataFrame(classification_report(yt, pred, labels=[0, 1, 2], target_names=names,
                                             output_dict=True, zero_division=0)).T
    st.dataframe(rep.round(3))

elif menu == "⚖️ Perbandingan Model":
    st.header("Perbandingan Model (deteksi abnormal vs normal)")
    c = st.columns(3)
    mode = c[0].selectbox("Pembagian data", ["Kronologis (70/30)", "Acak berstrata"])
    q = c[1].slider("Persentil ambang tak terawasi", 90.0, 99.9, 99.0, 0.1)
    n_est = c[2].slider("Jumlah pohon RF", 50, 500, 200, 50)
    scores = fit_unsup(X, y.values)
    tr, te = split_idx(y.values, mode)
    yb = (y.values[te] > 0).astype(int)
    if yb.sum() == 0:
        st.warning("Data uji tidak memiliki kejadian abnormal; ubah pembagian data."); st.stop()
    rows = {}
    for mname in MODELS:
        s = scores[mname]
        thr = np.percentile(s[y == 0], q)
        st_te = s.values[te]
        rows[mname] = (st_te > thr).astype(int), st_te
    classes, proba, _ = fit_rf(X, y.values, tr, te, n_est)
    p_ab = 1 - proba[:, list(classes).index(0)]
    rows["Random Forest"] = (p_ab >= 0.5).astype(int), p_ab
    out = []
    for n_, (pr, sc_) in rows.items():
        try:
            auc = roc_auc_score(yb, sc_)
        except ValueError:
            auc = np.nan
        out.append({"Model": n_, "Presisi": precision_score(yb, pr, zero_division=0),
                    "Sensitivitas": recall_score(yb, pr, zero_division=0),
                    "F1": f1_score(yb, pr, zero_division=0), "ROC-AUC": auc})
    res = pd.DataFrame(out).set_index("Model")
    st.dataframe(res.style.format("{:.3f}"))
    long = res.reset_index().melt("Model", var_name="Metrik", value_name="Nilai")
    st.plotly_chart(px.bar(long, x="Metrik", y="Nilai", color="Model", barmode="group",
                           title="Perbandingan kinerja pada data uji"), use_container_width=True)
    st.caption("Model tak terawasi dilatih pada seluruh data normal (termasuk bagian uji), sehingga sedikit lebih optimistis dibanding RF yang dievaluasi pada data uji murni.")

elif menu == "⏱️ Peringatan Dini":
    st.header("Peringatan Dini: Lead Time vs Ambang Konvensional")
    fails = segments(df.status_code == 2)
    if not fails:
        st.info("Tidak ada kejadian BROKEN pada data ini."); st.stop()
    scores = fit_unsup(X, y.values)
    c = st.columns(5)
    model = c[0].selectbox("Model", MODELS)
    q = c[1].slider("Persentil ambang", 90.0, 99.9, 99.0, 0.1)
    k = c[2].slider("Berturut-turut (k)", 1, 12, 3)
    look = c[3].slider("Jendela mundur (hari)", 1, 21, 7)
    base_sensor = c[4].selectbox("Sensor ambang tetap", sensors)
    sigma = st.slider("Ambang konvensional (± σ data normal)", 2.0, 6.0, 3.0, 0.5)
    sc = scores[model]
    alert_ml = sustained(sc > np.percentile(sc[y == 0], q), k)
    nm = X.loc[y == 0, base_sensor]
    zb = ((X[base_sensor] - nm.mean()) / nm.std()).abs()
    alert_cv = sustained(zb > sigma, k)

    def lead(alert, s):
        w = alert[(alert.index >= s - pd.Timedelta(days=look)) & (alert.index < s)]
        hit = w[w]
        return (s - hit.index[0]).total_seconds() / 3600 if len(hit) else 0.0

    rows = [{"Kejadian": f"#{i + 1} ({s:%Y-%m-%d %H:%M})", "start": s,
             f"{model} (jam)": lead(alert_ml, s), "Ambang tetap (jam)": lead(alert_cv, s)}
            for i, (s, _) in enumerate(fails)]
    t = pd.DataFrame(rows)
    cols = [f"{model} (jam)", "Ambang tetap (jam)"]
    m = st.columns(4)
    m[0].metric(f"Rata-rata lead time {model}", f"{t[cols[0]].mean():.1f} jam")
    m[1].metric("Rata-rata lead time ambang tetap", f"{t[cols[1]].mean():.1f} jam")
    span = (df.index[-1] - df.index[0]).days / 30
    m[2].metric(f"Episode alert palsu/30 hari ({model})", f"{episodes(alert_ml, y) / max(span, 1):.1f}")
    m[3].metric("Episode alert palsu/30 hari (tetap)", f"{episodes(alert_cv, y) / max(span, 1):.1f}")
    long = t.melt("Kejadian", value_vars=cols, var_name="Metode", value_name="Lead time (jam)")
    st.plotly_chart(px.bar(long, x="Kejadian", y="Lead time (jam)", color="Metode", barmode="group",
                           title="Lead time peringatan sebelum BROKEN (0 = tidak terdeteksi)"),
                    use_container_width=True)
    pick = st.selectbox("Lihat detail kejadian", t["Kejadian"])
    s = t.loc[t["Kejadian"] == pick, "start"].iloc[0]
    w0, w1 = s - pd.Timedelta(days=look), s + pd.Timedelta(hours=12)
    seg = sc[(sc.index >= w0) & (sc.index <= w1)]
    fig = go.Figure(go.Scatter(x=seg.index, y=seg, name="Skor anomali"))
    fig.add_hline(y=np.percentile(sc[y == 0], q), line_dash="dash", line_color="red")
    fig.add_vline(x=s, line_color="black", annotation_text="BROKEN")
    st.plotly_chart(fig, use_container_width=True)
