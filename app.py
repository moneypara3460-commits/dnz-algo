import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="DNZ ALGO V2.2",
    page_icon="🚀",
    layout="wide",
)

BINANCE_BASE = "https://api.binance.com"
TWELVE_DATA_BASE = "https://api.twelvedata.com"

DEFAULT_CRYPTO = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT",
    "LTCUSDT","SUIUSDT","APTUSDT","NEARUSDT","ATOMUSDT",
    "ARBUSDT","OPUSDT","UNIUSDT","FILUSDT","INJUSDT",
    "TRXUSDT","ETCUSDT","AAVEUSDT","ALGOUSDT","ICPUSDT"
]

def binance_klines(symbol, interval, limit=300):
    r = requests.get(
        BINANCE_BASE + "/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    r.raise_for_status()
    x = pd.DataFrame(r.json(), columns=[
        "time","open","high","low","close","volume",
        "close_time","qv","trades","tb","tq","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        x[c] = pd.to_numeric(x[c])
    return x

def xauusd_klines(interval, api_key, limit=300):
    if not api_key:
        raise ValueError("XAU/USD için Twelve Data API anahtarı gerekli.")

    r = requests.get(
        TWELVE_DATA_BASE + "/time_series",
        params={
            "symbol": "XAU/USD",
            "interval": interval,
            "outputsize": min(limit, 5000),
            "apikey": api_key,
            "timezone": "UTC",
        },
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()

    if data.get("status") == "error":
        raise ValueError(data.get("message", "Twelve Data API hatası."))

    values = data.get("values", [])
    if not values:
        raise ValueError("XAU/USD için veri alınamadı.")

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
        return xauusd_klines(interval, api_key, limit)
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
    return 100 - (100 / (1 + rs))

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
    return x

def analyse(symbol, min_score=65, rr=2.0, atr_mult=1.5, api_key=""):
    f = enrich(get_klines(symbol, "5min" if symbol == "XAU/USD" else "5m", api_key))
    t = enrich(get_klines(symbol, "15min" if symbol == "XAU/USD" else "15m", api_key))

    if len(f) < 210 or len(t) < 210:
        raise ValueError("Yeterli mum verisi bulunamadı.")

    a, b = f.iloc[-1], t.iloc[-1]

    score = 0
    reasons = []

    if b.close > b.ema50 > b.ema200:
        score += 25
        reasons.append("15m UP trend")
    elif b.close < b.ema50 < b.ema200:
        score -= 25
        reasons.append("15m DOWN trend")

    if a.close > a.ema20 > a.ema50:
        score += 20
        reasons.append("5m EMA alignment")
    elif a.close < a.ema20 < a.ema50:
        score -= 20
        reasons.append("5m weakness")

    if 45 <= a.rsi <= 65:
        score += 15
        reasons.append("RSI healthy")
    elif a.rsi > 75:
        score -= 10
        reasons.append("RSI overheated")

    volma = f.volume.rolling(20).mean().iloc[-1]
    volratio = float(a.volume / volma) if pd.notna(volma) and volma > 0 else 0.0

    if volratio >= 1.2:
        score += 20
        reasons.append("volume expansion")

    if a.low <= a.ema20 and a.close > a.ema20:
        score += 20
        reasons.append("EMA20 reclaim")

    dnz = max(0, min(100, score))
    confidence = max(
        0,
        min(
            100,
            dnz * 0.65
            + (15 if volratio >= 1.2 else 0)
            + (10 if 45 <= b.rsi <= 65 else 0)
            + 10,
        ),
    )

    entry = float(a.close)
    atr = float(a.atr)
    stop = entry - atr_mult * atr
    target = entry + rr * (entry - stop)

    signal = "LONG" if dnz >= min_score and b.close > b.ema50 else "WAIT"

    return {
        "symbol": symbol,
        "price": entry,
        "score": round(dnz, 1),
        "confidence": round(confidence, 1),
        "signal": signal,
        "stop": round(stop, 8),
        "target": round(target, 8),
        "risk_reward": rr,
        "volume_ratio": round(volratio, 2),
        "reason": ", ".join(reasons) or "No strong confirmation",
    }

def scan(symbols, min_score, rr, atr_mult, api_key=""):
    results, errors = [], []
    for symbol in symbols:
        try:
            results.append(analyse(symbol, min_score, rr, atr_mult, api_key))
        except Exception as e:
            errors.append(f"{symbol}: {e}")

    results.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
    return pd.DataFrame(results), errors

# ---------- UI ----------
st.markdown("""
<style>
.main-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0;}
.sub {color:#9aa4b2; margin-top:0;}
.card {padding:16px 18px; border-radius:14px; border:1px solid rgba(128,128,128,.25);}
</style>
""", unsafe_allow_html=True)

st.title("🚀 DNZ ALGO V2.2")
st.markdown(
    '<p class="sub">Live Market Scanner • Crypto + Gold • PAPER / RESEARCH MODE</p>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Ayarlar")
    min_score = st.slider("Minimum DNZ Score", 0, 100, 65)
    rr = st.number_input(
        "Risk / Reward", min_value=0.5, max_value=10.0,
        value=2.0, step=0.5
    )
    atr_mult = st.number_input(
        "ATR Stop Çarpanı", min_value=0.5, max_value=5.0,
        value=1.5, step=0.5
    )
    refresh = st.selectbox("Otomatik tarama", [30, 60, 120, 300], index=0)
    top_n = st.slider("TOP fırsat", 1, 10, 5)

    st.subheader("📡 Varlıklar")
    crypto_text = st.text_area(
        "Coin listesi (virgülle)",
        ",".join(DEFAULT_CRYPTO),
        height=160,
    )
    crypto = [x.strip().upper() for x in crypto_text.split(",") if x.strip()]

    gold_enabled = st.checkbox("🥇 XAU/USD (Altın / Dolar)", value=True)

    st.subheader("🔑 Altın veri bağlantısı")
    twelve_key = st.text_input(
        "Twelve Data API Key",
        type="password",
        help="XAU/USD 5m ve 15m verisi için gereklidir."
    )
    st.caption("Kripto verileri Binance Spot API'den alınır.")
    st.caption("⚠️ Bu sürüm gerçek emir göndermez.")

symbols = crypto + (["XAU/USD"] if gold_enabled else [])

@st.fragment(run_every=f"{refresh}s")
def live_dashboard():
    started = time.strftime("%H:%M:%S")

    with st.spinner("Piyasa taranıyor..."):
        df, errors = scan(symbols, min_score, rr, atr_mult, twelve_key)

    longs = df[df["signal"] == "LONG"].head(top_n) if not df.empty else df
    top = df.head(top_n) if not df.empty else df

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Sistem", "AKTİF")
    c2.metric("📡 Taranan", len(symbols))
    c3.metric("🏆 LONG", len(longs))
    c4.metric("🕐 Son tarama", started)

    st.divider()

    st.subheader("🏆 TOP FIRSATLAR")
    if longs.empty:
        st.info("Şu anda minimum skor kriterini geçen LONG fırsatı yok.")
    else:
        show = longs[[
            "symbol","price","score","confidence",
            "signal","stop","target","risk_reward","volume_ratio"
        ]].copy()
        show.columns = [
            "Varlık","Fiyat","DNZ Score","Confidence",
            "Sinyal","Stop","Target","R/R","Vol. Oranı"
        ]
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("📊 Tüm Varlıklar")
    if top.empty:
        st.info("Henüz sonuç yok.")
    else:
        all_show = top[[
            "symbol","price","score","confidence",
            "signal","stop","target","risk_reward","volume_ratio"
        ]].copy()
        all_show.columns = [
            "Varlık","Fiyat","DNZ Score","Confidence",
            "Sinyal","Stop","Target","R/R","Vol. Oranı"
        ]
        st.dataframe(all_show, use_container_width=True, hide_index=True)

    st.subheader("🧠 Sinyal Detayı")
    if not longs.empty:
        for _, row in longs.iterrows():
            with st.expander(
                f"{row['symbol']} • Score {row['score']} • {row['signal']}"
            ):
                st.write(row["reason"])
                st.write(
                    f"Fiyat: `{row['price']}` | Stop: `{row['stop']}` | "
                    f"Target: `{row['target']}` | R/R: `{row['risk_reward']}`"
                )

    if errors:
        with st.expander(f"⚠️ {len(errors)} varlık taranamadı"):
            st.write("\n".join(errors))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Sonuçları CSV indir",
        csv,
        "dnz_live_scan.csv",
        "text/csv",
    )

live_dashboard()

st.caption(
    "DNZ ALGO V2.2 • Kripto: Binance Spot API • Altın: Twelve Data XAU/USD • "
    "Yatırım tavsiyesi değildir."
)
