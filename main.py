# main.py
import os
import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import tensorflow as tf
from tensorflow.keras import layers, models

# -------- CONFIG ----------------------------------------------------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

ASSETS = ["BTC/USD", "ETH/USD", "XAU/USD"]
H1_COUNT = 300   # number of 1h candles to fetch for live features
H4_COUNT = 200   # number of 4h candles for trend filter
TRAIN_COUNT = 1500  # number of candles for training

MODEL_RF_PATH = "model_rf.joblib"
SCALER_PATH = "scaler.joblib"
LSTM_PATH = "lstm_model"  # tensorflow saved model dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trading-bot")

# aiogram
dp = Dispatcher()
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)

user_settings = {}  # {user_id: {"asset":..., "muted":False}}

# NLP
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ML placeholders
rf_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
scaler = StandardScaler()
lstm_model = None
ml_ready = False

# ---------------- utilities (data fetch) ----------------------------------
async def get_twelvedata(asset: str, interval: str = "1h", count: int = 500):
    """Get OHLCV from TwelveData. Returns DataFrame or None on error."""
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": asset, "interval": interval, "outputsize": count, "apikey": TWELVEDATA_API_KEY}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=30) as resp:
                data = await resp.json()
    except Exception as e:
        logger.exception("HTTP error fetching TwelveData")
        return None

    if not data or "values" not in data:
        logger.warning("TwelveData returned empty or error: %s", data)
        return None

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    # attempt numeric cast for available columns
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ---------------- indicators ---------------------------------------------
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def compute_obv(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        # fallback zeros
        return pd.Series([0.0]*len(df), index=df.index)
    sign = np.sign(df["close"].diff()).fillna(0)
    obv = (sign * df["volume"]).cumsum().fillna(0)
    return obv

def compute_bollinger(df: pd.DataFrame, n: int = 20):
    ma = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    upper = ma + 2*std
    lower = ma - 2*std
    return ma, upper, lower

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "close" not in df:
        raise ValueError("DataFrame must contain close")
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["atr"] = compute_atr(df)
    df["obv"] = compute_obv(df)
    ma20, up20, low20 = compute_bollinger(df, 20)
    df["bb_mid"] = ma20
    df["bb_upper"] = up20
    df["bb_lower"] = low20
    df["momentum3"] = df["close"] - df["close"].shift(3)
    df["volatility14"] = df["close"].rolling(14).std()
    df["ret1"] = df["close"].pct_change(1)
    df = df.dropna().reset_index(drop=True)
    return df

# ---------------- labels (thresholded) -----------------------------------
def make_labels(df: pd.DataFrame, horizon: int = 3, thr: float = 0.008):
    # returns {-1,0,1}
    future_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    labels = pd.Series(0, index=df.index)
    labels[future_ret > thr] = 1
    labels[future_ret < -thr] = -1
    return labels

# ---------------- support/resistance (swing) ------------------------------
def support_resistance(df: pd.DataFrame, window:int = 20):
    # simple: rolling high/low
    recent_high = df["high"].rolling(window).max().iloc[-1]
    recent_low = df["low"].rolling(window).min().iloc[-1]
    return float(recent_high), float(recent_low)

def smart_tp_sl(price: float, atr: float, recent_high: float, recent_low: float, direction: str):
    # prefer nearest swing level if sensible, else ATR-based
    if direction == "buy":
        tp = recent_high if recent_high > price else price + 2.0 * atr
        sl = recent_low if recent_low < price else price - 1.0 * atr
    elif direction == "sell":
        tp = recent_low if recent_low < price else price - 2.0 * atr
        sl = recent_high if recent_high > price else price + 1.0 * atr
    else:
        tp = sl = price
    tp_pct = (tp / price - 1.0) * 100
    sl_pct = (1.0 - sl / price) * 100 if direction == "buy" else (sl / price - 1.0) * 100
    return round(float(tp), 6), round(float(sl), 6), round(float(tp_pct), 3), round(float(abs(sl_pct)), 3)

# ---------------- news filtering -----------------------------------------
def news_is_strong(article: dict):
    """Check if article title/description contains strong keywords (ETF, Fed, CPI, FOMC, regulation...)"""
    txt = (article.get("title") or "") + " " + (article.get("description") or "")
    text = txt.lower()
    strong = ["etf", "fomc", "fed", "interest rate", "cpi", "inflation", "regulat", "ban", "suspend", "approval", "policy", "sec", "bank", "rate hike", "rate cut"]
    return any(k in text for k in strong)

async def get_news_sentiment_strong(asset: str):
    """Collect only strong news and compute average VADER compound. Returns -1..1"""
    query = "bitcoin" if "BTC" in asset else "gold" if "XAU" in asset else "ethereum"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en&pageSize=10"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=20) as resp:
                data = await resp.json()
    except Exception:
        return 0.0
    if not data or "articles" not in data:
        return 0.0
    strong_texts = []
    for art in data["articles"]:
        if news_is_strong(art):
            txt = (art.get("title") or "") + " " + (art.get("description") or "")
            strong_texts.append(txt)
            if len(strong_texts) >= 5: break
    if not strong_texts:
        return 0.0
    scores = [sia.polarity_scores(t)["compound"] for t in strong_texts]
    return float(np.mean(scores))

# ---------------- LSTM model helpers -------------------------------------
def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.15),
        layers.Dense(3, activation="softmax")  # classes [-1,0,1]
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ---------------- Training routine --------------------------------------
async def train_models(assets=ASSETS, retrain_lstm=True):
    """Train RF/GBM and LSTM ensemble on combined assets.
       Saves models to disk to avoid retrain on every start."""
    global rf_model, scaler, lstm_model, ml_ready

    # try to load existing
    if os.path.exists(MODEL_RF_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(LSTM_PATH):
        logger.info("Loading models from disk...")
        rf_model = joblib.load(MODEL_RF_PATH)
        scaler = joblib.load(SCALER_PATH)
        lstm_model = tf.keras.models.load_model(LSTM_PATH)
        ml_ready = True
        logger.info("Models loaded.")
        return

    logger.info("Training models from scratch (this can take a while)...")
    frames = []
    for asset in assets:
        df = await get_twelvedata(asset, interval="1h", count=TRAIN_COUNT)
        if df is None or len(df) < 200:
            logger.warning("Skipping %s, not enough data", asset)
            continue
        df = add_indicators(df)
        labels = make_labels(df, horizon=3, thr=0.008)
        df = df.iloc[: len(labels)]
        df["label"] = labels.values
        df["asset"] = asset
        frames.append(df)
    if not frames:
        logger.error("No training data")
        return
    df_all = pd.concat(frames, ignore_index=True).dropna()
    feature_cols = ["ema10", "ema50", "rsi", "macd", "atr", "obv", "bb_upper", "bb_lower", "momentum3", "volatility14", "ret1"]
    X = df_all[feature_cols].values
    y = df_all["label"].values.astype(int)  # -1,0,1

    # map to 0..2 for classifier (suitable for sklearn)
    class_map = {-1:0, 0:1, 1:2}
    y_sklearn = np.array([class_map[int(v)] for v in y])

    X_scaled = scaler.fit_transform(X)
    # train RF / GBM (we use GradientBoostingClassifier for multiclass)
    rf_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    rf_model.fit(X_scaled, y_sklearn)
    joblib.dump(rf_model, MODEL_RF_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Trained and saved RF/GBM model and scaler.")

    # Prepare sequences for LSTM: group by asset and create sliding windows
    if retrain_lstm:
        seq_len = 24  # 24 hours
        X_seq = []
        y_seq = []
        for asset in assets:
            df_asset = df_all[df_all["asset"] == asset].reset_index(drop=True)
            arr = df_asset[feature_cols].values
            labels_asset = df_asset["label"].values
            for i in range(len(arr) - seq_len - 3):
                X_seq.append(arr[i:i+seq_len])
                # label at i+seq_len -> future movement
                y_seq.append(labels_asset[i+seq_len])
        if len(X_seq) >= 50:
            X_seq = np.array(X_seq)
            # map labels -1,0,1 to 0,1,2
            y_seq = np.array([class_map[int(v)] for v in y_seq])
            # build LSTM
            lstm_model = build_lstm(X_seq.shape[1:])
            # train/val split
            Xtr, Xval, ytr, yval = train_test_split(X_seq, y_seq, test_size=0.15, random_state=42, shuffle=True)
            lstm_model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=8, batch_size=64, verbose=1)
            lstm_model.save(LSTM_PATH)
            logger.info("Trained and saved LSTM model.")
        else:
            logger.warning("Not enough sequence samples for LSTM; skipping LSTM training.")
    ml_ready = True
    logger.info("All models ready.")

# ---------------- prediction ensemble ------------------------------------
def predict_ensemble_row(row):
    """Return direction and confidence [0..1]"""
    if not ml_ready:
        return "neutral", 0.5
    feature_cols = ["ema10", "ema50", "rsi", "macd", "atr", "obv", "bb_upper", "bb_lower", "momentum3", "volatility14", "ret1"]
    X = row[feature_cols].values.reshape(1, -1)
    Xs = scaler.transform(X)
    # RF/GBM prediction -> probabilities for classes 0,1,2 -> map back to -1,0,1
    probs_rf = rf_model.predict_proba(Xs)[0]  # shape (3,)
    # map indices: 0->-1, 1->0, 2->1
    prob_map_rf = {-1: probs_rf[0], 0: probs_rf[1], 1: probs_rf[2]}

    # LSTM
    prob_map_lstm = {-1:0.0, 0:0.0, 1:0.0}
    if lstm_model is not None:
        # LSTM expects sequence; we'll feed last seq_len values (if present)
        # For simplicity, if no sequence available, skip LSTM contribution
        # This function will be used on row from df; user should ensure sequence presence when calling from send_signal
        try:
            # attempt to get last seq from global df? We'll allow caller to pass a row that contains 'seq' if needed.
            pass
        except Exception:
            pass

    # For simplicity, we'll combine only rf_model + (if lstm_model available later: include)
    # Choose best class by rf first
    best_idx = int(np.argmax(probs_rf))
    mapping = {0:-1, 1:0, 2:1}
    best_class = mapping[best_idx]
    confidence = prob_map_rf[best_class]
    # If LSTM available, you'd average probs (omitted here for per-row simple)
    # return textual direction and confidence
    if best_class == 1:
        return "buy", float(confidence)
    elif best_class == -1:
        return "sell", float(confidence)
    else:
        return "neutral", float(confidence)

# ---------------- combine ml + rule + news --------------------------------
def rule_signal_from_row(row):
    # returns "buy"/"sell"/"neutral" and normalized confidence 0..1
    ema_sig = 1 if row["ema10"] > row["ema50"] else -1
    rsi_sig = 1 if row["rsi"] < 30 else -1 if row["rsi"] > 70 else 0
    macd_sig = 1 if row["macd"] > 0 else -1
    votes = ema_sig + rsi_sig + macd_sig
    if votes > 0:
        return "buy", min(1.0, votes/3.0)
    if votes < 0:
        return "sell", min(1.0, abs(votes)/3.0)
    return "neutral", 0.33

def combine(ml_dir, ml_conf, rule_dir, rule_conf, news_score):
    w_ml = 0.5
    w_rule = 0.3
    w_news = 0.2
    def dir_num(d, conf):
        if d == "buy":
            return conf
        if d == "sell":
            return -conf
        return 0.0
    val = w_ml * dir_num(ml_dir, ml_conf) + w_rule * dir_num(rule_dir, rule_conf) + w_news * np.clip(news_score, -1.0, 1.0)
    thresh = 0.4
    if val >= thresh:
        return "buy", val
    if val <= -thresh:
        return "sell", abs(val)
    return "neutral", abs(val)

# ---------------- send signal (combines H1 + H4) --------------------------
async def send_signal(user_id: int, asset: str):
    try:
        # get H1 and H4 data
        df_h1 = await get_twelvedata(asset, interval="1h", count=H1_COUNT)
        df_h4 = await get_twelvedata(asset, interval="4h", count=H4_COUNT)
        if df_h1 is None or df_h4 is None or len(df_h1) < 60 or len(df_h4) < 40:
            await bot.send_message(user_id, f"⚠️ Недостаточно данных для {asset}")
            return
        df_h1 = add_indicators(df_h1)
        df_h4 = add_indicators(df_h4)

        # rule signals on both TF
        rule_h1_dir, rule_h1_conf = rule_signal_from_row(df_h1.iloc[-1])
        rule_h4_dir, rule_h4_conf = rule_signal_from_row(df_h4.iloc[-1])
        # require filter: direction must be same on H4 and H1 or H4 neutral
        if rule_h4_dir != "neutral" and rule_h1_dir != rule_h4_dir:
            # filter out - trend mismatch
            await bot.send_message(user_id, f"⚠️ H1/H4 trend mismatch for {asset} — пропускаем сигнал.")
            return

        # ML on H1 latest
        ml_dir, ml_conf = predict_ensemble_row(df_h1.iloc[-1])

        # LSTM ensemble (if saved model exists): optionally we could compute seq prob and average — omitted for brevity

        # news strong sentiment
        news_score = await get_news_sentiment_strong(asset)  # -1..1

        # combine
        final_dir, final_conf = combine(ml_dir, ml_conf, rule_h1_dir, rule_h1_conf, news_score)

        # require minimal confidence to send
        if final_dir == "neutral" or final_conf < 0.55:
            await bot.send_message(user_id, f"⚠️ По {asset} нет уверенного сигнала (score {final_conf:.2f}).")
            return

        price = float(df_h1["close"].iloc[-1])
        atr = float(df_h1["atr"].iloc[-1]) if "atr" in df_h1.columns else price * 0.01
        recent_high, recent_low = support_resistance(df_h1, window=20)
        tp_price, sl_price, tp_pct, sl_pct = smart_tp_sl(price, atr, recent_high, recent_low, final_dir)

        # message
        msg = (
            f"📢 Сигнал — <b>{asset}</b>\n"
            f"Направление: <b>{final_dir.upper()}</b>\n"
            f"Цена входа: <b>{price}</b>\n"
            f"🟢 TP: {tp_price} (+{tp_pct}%)\n"
            f"🔴 SL: {sl_price} (-{sl_pct}%)\n"
            f"📊 Надёжность (оценка): <b>{round(final_conf*100,2)}%</b>\n"
            f"📰 Новости (strong): {'позитив' if news_score>0.05 else 'негатив' if news_score<-0.05 else 'нейтрально'}\n\n"
            f"🧾 Подробно: ML={ml_dir}({ml_conf:.2f}), Rule(H1)={rule_h1_dir}({rule_h1_conf:.2f}), Rule(H4)={rule_h4_dir}({rule_h4_conf:.2f})"
        )
        muted = user_settings.get(user_id, {}).get("muted", False)
        await bot.send_message(user_id, msg, disable_notification=muted)
    except Exception as e:
        logger.exception("Error in send_signal")
        try:
            await bot.send_message(user_id, f"⚠️ Ошибка при генерации сигнала: {e}")
        except Exception:
            pass

# ---------------- handlers and loop --------------------------------------
def get_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton("🔄 Получить сигнал")],
            [KeyboardButton("BTC/USD"), KeyboardButton("ETH/USD"), KeyboardButton("XAU/USD")],
            [KeyboardButton("🔕 Mute"), KeyboardButton("🔔 Unmute")],
            [KeyboardButton("🕒 Расписание")]
        ],
        resize_keyboard=True
    )

@dp.message(CommandStart())
async def on_start(msg: types.Message):
    user_settings[msg.from_user.id] = {"asset":"BTC/USD","muted":False}
    await msg.answer("🤖 Бот запущен. Выберите актив и запросите сигнал.", reply_markup=get_keyboard())

@dp.message()
async def on_message(msg: types.Message):
    uid = msg.from_user.id
    text = (msg.text or "").strip()
    if uid not in user_settings:
        user_settings[uid] = {"asset":"BTC/USD","muted":False}
    lower = text.lower()
    # robust mute
    if "mute" in lower or text.startswith("🔕") or text.startswith("🔇"):
        user_settings[uid]["muted"] = True
        await msg.answer("🔕 Автосигналы отключены.")
        return
    if "unmute" in lower or text.startswith("🔔") or text.startswith("🔊"):
        user_settings[uid]["muted"] = False
        await msg.answer("🔔 Автосигналы включены.")
        return
    if text == "🔄 Получить сигнал":
        await send_signal(uid, user_settings[uid]["asset"])
        return
    if text in ASSETS:
        user_settings[uid]["asset"] = text
        await msg.answer(f"✅ Актив установлен: {text}")
        return
    if text == "🕒 Расписание":
        await msg.answer("Авто-проверка каждые 15 минут.")
        return
    # else ignore

async def auto_loop():
    while True:
        try:
            for uid, s in list(user_settings.items()):
                if not s.get("muted", False):
                    await send_signal(uid, s["asset"])
        except Exception:
            logger.exception("auto_loop exception")
        await asyncio.sleep(900)

async def main():
    await train_models()  # may take time on first run
    asyncio.create_task(auto_loop())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
