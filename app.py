import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DNZ ALGO V2.3", page_icon="🚀", layout="wide")

BINANCE_BASES = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
]
TWELVE_DATA_BASE = "https://api.twelvedata.com"

DEFAULT_CRYPTO = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT",
    "LTCUSDT","SUIUSDT","APTUSDT","NEARUSDT","ATOMUSDT",
    "ARBUSDT","OPUSDT","UNIUSDT","FILUSDT","INJUSDT",
    "TRXUSDT","ETCUSDT","AAVEUSDT","ALGOUSDT","ICPUSDT"
]

def binance_klines(symbol, interval, limit=300):
    last_error = None
    for base in BINANCE_BASES:
        try:
            r = requests.get(
                base + "/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=12,
            )
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) < 50:
                raise ValueError("Binance geçerli veri döndürmedi.")
            x = pd.DataFrame(data, columns=[
                "time","open","high","low","close","volume",
                "close_time","qv","trades","tb","tq","ignore"
            ])
            for c in ["open","high","low","close","volume"]:
                x[c] = pd.to_numeric(x[c], errors="coerce")
            x = x.dropna(subset=["open","high","low","close","volume"])
            return x
        except Exception as e:
            last_error = e
    raise ValueError(f"Binance bağlantısı başarısız: {last_error}")

def xauusd_klines(interval, api_key, limit=300):
    if not api_key:
        raise ValueError("XAU/USD için Twelve Data API Key gerekli.")
    r = requests.get(
        TWELVE_DATA_BASE + "/time_series",
        params={
            "symbol": "XAU/USD",
            "interval": interval,
            "outputsize": min(limit, 5000),
            "apikey": api_key,
            "timezone": "UTC",
        },
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") == "error":
        raise ValueError(data.get("message", "Twelve Data API hatası."))
    values = data.get("values", [])
    if not values:
        raise ValueError("XAU/USD verisi alınamadı.")
    x = pd.DataFrame(values)
    for c in ["open","high","low","close","volume"]:
        if c not in x.columns:
            x[c] = 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.sort_values("datetime").reset_index(drop=True)
    x["time"] = pd.to_datetime(x["datetime"], utc=True)
    return x[["time","open","high","low","close","volume"]].dropna(subset=["close"])

def get_klines(symbol, interval, api_key="", limit=300):
    if symbol == "XAU/USD":
        return xauusd_klines("5min" if interval == "5m" else "15min", api_key, limit)
    return binance_klines(symbol, interval, limit)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def enrich(x):
    x = x.copy()
    x["ema20"] = ema(x.close, 20)
    x["ema50"] = ema(x.close, 50)
    x["ema200"] = ema(x.close, 200)
    x["rsi"] = rsi(x.close)
    prev = x.close.shift(1)
    tr = pd.concat([
        x.high - x.low,
        (x.high - prev).abs(),
        (x.low - prev).abs()
    ], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["volma"] = x.volume.rolling(20).mean()
    x["momentum"] = x.close.pct_change(5) * 100
    return x

def analyse(symbol, min_score=65, rr=2.0, atr_mult=1.5, api_key=""):
    fast = enrich(get_klines(symbol, "5m", api_key))
    trend = enrich(get_klines(symbol, "15m", api_key))

    if len(fast) < 210 or len(trend) < 210:
        raise ValueError("Yeterli mum verisi yok.")

    a = fast.iloc[-2]
    b = trend.iloc[-2]

    long_score = 0
    short_score = 0
    long_reasons = []
    short_reasons = []

    if b.close > b.ema50 > b.ema200:
        long_score += 30
        long_reasons.append("15m güçlü yükseliş trendi")
    elif b.close < b.ema50 < b.ema200:
        short_score += 30
        short_reasons.append("15m güçlü düşüş trendi")

    if a.close > a.ema20 > a.ema50:
        long_score += 20
        long_reasons.append("5m EMA uyumu")
    elif a.close < a.ema20 < a.ema50:
        short_score += 20
        short_reasons.append("5m EMA düşüş uyumu")

    if 52 <= a.rsi <= 68:
        long_score += 15
        long_reasons.append("RSI long bölgesinde")
    elif 32 <= a.rsi <= 48:
        short_score += 15
        short_reasons.append("RSI short bölgesinde")

    volratio = float(a.volume / a.volma) if pd.notna(a.volma) and a.volma > 0 else 0.0
    if volratio >= 1.2:
        if a.close >= a.open:
            long_score += 15
            long_reasons.append("yüksek hacim + pozitif mum")
        else:
            short_score += 15
            short_reasons.append("yüksek hacim + negatif mum")

    if a.momentum > 0.15:
        long_score += 10
        long_reasons.append("pozitif momentum")
    elif a.momentum < -0.15:
        short_score += 10
        short_reasons.append("negatif momentum")

    prev = fast.iloc[-3]

    if prev.close <= prev.ema20 and a.close > a.ema20:
        long_score += 10
        long_reasons.append("EMA20 reclaim")
    elif prev.close >= prev.ema20 and a.close < a.ema20:
        short_score += 10
        short_reasons.append("EMA20 breakdown")

    atr = float(a.atr)
    entry = float(a.close)

    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("ATR hesaplanamadı.")

    if long_score >= short_score:
        direction = "LONG"
        raw_score = long_score
        reasons = long_reasons
        stop = entry - atr_mult * atr
        target = entry + rr * (entry - stop)
    else:
        direction = "SHORT"
        raw_score = short_score
        reasons = short_reasons
        stop = entry + atr_mult * atr
        target = entry - rr * (stop - entry)

    opposing = short_score if direction == "LONG" else long_score
    edge = raw_score - opposing

    confidence = min(
        100,
        max(0, 50 + edge * 0.9 + min(volratio, 2.0) * 5)
    )

    if raw_score < min_score or edge < 15:
        signal = "WAIT"
    else:
        signal = direction

    return {
        "symbol": symbol,
        "price": entry,
        "score": round(float(raw_score), 1),
        "confidence": round(float(confidence), 1),
        "signal": signal,
        "stop": round(float(stop), 8),
        "target": round(float(target), 8),
        "risk_reward": rr,
        "volume_ratio": round(volratio, 2),
        "reason": ", ".join(reasons) if reasons else "Yeterli onay yok",
        "long_score": long_score,
        "short_score": short_score,
    }

def scan(symbols, min_score, rr, atr_mult, api_key=""):
    results = []
    errors = []

    for symbol in symbols:
        try:
            results.append(
                analyse(symbol, min_score, rr, atr_mult, api_key)
            )
        except Exception as e:
            errors.append(f"{symbol}: {e}")

    if not results:
        return pd.DataFrame(), errors

    df = pd.DataFrame(results)

    signal_order = {
        "LONG": 2,
        "SHORT": 2,
        "WAIT": 1
    }

    df["_rank"] = df["signal"].map(signal_order).fillna(0)

    df = df.sort_values(
        ["_rank", "confidence", "score"],
        ascending=False
    ).drop(columns=["_rank"])

    return df.reset_index(drop=True), errors

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}
.sub {
    color:#9aa4b2;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 DNZ ALGO V2.3")

st.markdown(
    '<p class="sub">Live Market Scanner • Crypto + Gold • '
    'PAPER / RESEARCH MODE</p>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Ayarlar")

    min_score = st.slider(
        "Minimum DNZ Score",
        0, 100, 65
    )

    rr = st.number_input(
        "Risk / Reward",
        0.5, 10.0, 2.0, 0.5
    )

    atr_mult = st.number_input(
        "ATR Stop Çarpanı",
        0.5, 5.0, 1.5, 0.5
    )

    refresh = st.selectbox(
        "Otomatik tarama (sn)",
        [30, 60, 120, 300],
        index=0
    )

    top_n = st.slider(
        "TOP fırsat",
        1, 10, 5
    )

    st.subheader("📡 Varlıklar")

    crypto_text = st.text_area(
        "Coin listesi (virgülle)",
        ",".join(DEFAULT_CRYPTO),
        height=160,
    )

    crypto = [
        x.strip().upper()
        for x in crypto_text.split(",")
        if x.strip()
    ]

    gold_enabled = st.checkbox(
        "🥇 XAU/USD",
        value=False
    )

    twelve_key = st.text_input(
        "Twelve Data API Key",
        type="password",
        help="Sadece XAU/USD için gerekir."
    )

    st.caption("Kripto: Binance Spot public API")
    st.caption("Gerçek emir göndermez. Araştırma/PAPER modudur.")

symbols = crypto + (["XAU/USD"] if gold_enabled else [])

@st.fragment(run_every=f"{refresh}s")
def live_dashboard():

    started = time.strftime("%H:%M:%S")

    with st.spinner("Piyasa taranıyor..."):
        df, errors = scan(
            symbols,
            min_score,
            rr,
            atr_mult,
            twelve_key
        )

    if df.empty:
        st.error(
            "Hiç veri alınamadı. Hata listesini kontrol et."
        )

        if errors:
            with st.expander(f"⚠️ {len(errors)} hata"):
                st.write("\n".join(errors))

        return

    longs = df[
        df["signal"] == "LONG"
    ].head(top_n)

    shorts = df[
        df["signal"] == "SHORT"
    ].head(top_n)

    active = pd.concat(
        [longs, shorts]
    ).drop_duplicates("symbol").head(top_n)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🟢 Sistem", "AKTİF")
    c2.metric("📡 Taranan", len(symbols))
    c3.metric(
        "🟢 LONG / 🔴 SHORT",
        f"{len(longs)} / {len(shorts)}"
    )
    c4.metric("🕐 Son tarama", started)

    st.divider()

    st.subheader("🏆 EN GÜÇLÜ FIRSATLAR")

    if active.empty:

        st.info(
            "Şu anda minimum skor ve yön farkı "
            "kriterini geçen sinyal yok."
        )

    else:

        show = active[
            [
                "symbol",
                "price",
                "score",
                "confidence",
                "signal",
                "stop",
                "target",
                "risk_reward",
                "volume_ratio"
            ]
        ].copy()

        show.columns = [
            "Varlık",
            "Fiyat",
            "DNZ Score",
            "Confidence",
            "Sinyal",
            "Stop",
            "Target",
            "R/R",
            "Vol. Oranı"
        ]

        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("📊 Tüm Varlıklar")

    all_show = df[
        [
            "symbol",
            "price",
            "score",
            "confidence",
            "signal",
            "stop",
            "target",
            "risk_reward",
            "volume_ratio"
        ]
    ].head(max(top_n, 10)).copy()

    all_show.columns = [
        "Varlık",
        "Fiyat",
        "DNZ Score",
        "Confidence",
        "Sinyal",
        "Stop",
        "Target",
        "R/R",
        "Vol. Oranı"
    ]

    st.dataframe(
        all_show,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🧠 Sinyal Detayı")

    for _, row in active.iterrows():

        icon = (
            "🟢"
            if row["signal"] == "LONG"
            else "🔴"
        )

        with st.expander(
            f"{icon} {row['symbol']} • "
            f"{row['signal']} • "
            f"Score {row['score']} • "
            f"Confidence {row['confidence']}"
        ):

            st.write(row["reason"])

            st.write(
                f"Fiyat: `{row['price']}` | "
                f"Stop: `{row['stop']}` | "
                f"Target: `{row['target']}` | "
                f"R/R: `{row['risk_reward']}`"
            )

            st.write(
                f"Long Score: `{row['long_score']}` | "
                f"Short Score: `{row['short_score']}`"
            )

    if errors:

        with st.expander(
            f"⚠️ {len(errors)} varlık taranamadı"
        ):
            st.write("\n".join(errors))

    st.download_button(
        "📥 Sonuçları CSV indir",
        df.to_csv(index=False).encode("utf-8"),
        "dnz_live_scan_v23.csv",
        "text/csv",
    )

live_dashboard()

st.caption(
    "DNZ ALGO V2.3 • Binance Spot • "
    "PAPER / RESEARCH MODE • "
    "Yatırım tavsiyesi değildir."
)