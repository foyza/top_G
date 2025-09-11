import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.enums import ParseMode
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# === CONFIG ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

ASSETS = ["BTC/USD", "ETH/USD", "XAU/USD"]

logging.basicConfig(level=logging.INFO)

dp = Dispatcher()
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)

user_settings = {}  # {uid: {"asset": ..., "muted": False}}

# === ML MODELS ===
gb_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
scaler = StandardScaler()
lstm_model = None
ml_trained = False

# === NLP ===
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# === UI ===
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Получить сигнал")],
            [KeyboardButton(text="BTC/USD"), KeyboardButton(text="ETH/USD"), KeyboardButton(text="XAU/USD")],
            [KeyboardButton(text="🔕 Mute"), KeyboardButton(text="🔔 Unmute")]
        ],
        resize_keyboard=True
    )

# === DATA ===
async def get_twelvedata(asset, interval="1h", count=300):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": asset, "interval": interval, "outputsize": count, "apikey": TWELVEDATA_API_KEY}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            if "values" not in data:
                return None
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            return df

async def get_news_sentiment(asset):
    query_map = {"BTC/USD": "bitcoin ETF regulation", "ETH/USD": "ethereum upgrade regulation", "XAU/USD": "gold inflation FED"}
    query = query_map.get(asset, "crypto")
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            data = await r.json()
            if "articles" not in data:
                return 0
            scores = []
            for art in data["articles"][:5]:
                text = art.get("title", "") + " " + art.get("description", "")
                if any(word in text.lower() for word in ["etf", "sec", "federal reserve", "inflation", "ban", "approve"]):
                    scores.append(sia.polarity_scores(text)["compound"])
            return float(np.mean(scores)) if scores else 0

# === INDICATORS ===
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    return ema12 - ema26

def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def compute_obv(df):
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    return obv

def compute_bollinger(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def add_indicators(df):
    df = df.copy()
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["atr"] = compute_atr(df)
    df["obv"] = compute_obv(df)
    df["bb_upper"], df["bb_lower"] = compute_bollinger(df["close"])
    return df.dropna()

# === ML TRAINING ===
async def train_models(asset="BTC/USD"):
    global ml_trained, gb_model, scaler, lstm_model
    df = await get_twelvedata(asset, count=500)
    if df is None: return
    df = add_indicators(df)
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)

    features = df[["ema10", "ema50", "rsi", "macd", "atr", "obv"]].iloc[:-3]
    labels = df["target"].iloc[:-3]
    X = scaler.fit_transform(features)
    y = labels
    gb_model.fit(X, y)

    X_seq = features.values.reshape((features.shape[0], 1, features.shape[1]))
    lstm_model = Sequential([
        LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    lstm_model.fit(X_seq, y, epochs=5, batch_size=16, verbose=0)

    ml_trained = True
    logging.info("✅ ML модели обучены")

def ml_predict(latest_row):
    if not ml_trained:
        return "neutral", 50
    X = np.array([[latest_row["ema10"], latest_row["ema50"], latest_row["rsi"], latest_row["macd"], latest_row["atr"], latest_row["obv"]]])
    X_scaled = scaler.transform(X)
    prob_gb = gb_model.predict_proba(X_scaled)[0][1]
    prob_lstm = float(lstm_model.predict(X_scaled.reshape((1, 1, X.shape[1])), verbose=0)[0][0])
    prob = (prob_gb + prob_lstm) / 2
    if prob > 0.6:
        return "buy", int(prob*100)
    elif prob < 0.4:
        return "sell", int((1-prob)*100)
    return "neutral", 50

# === SIGNAL ===
async def send_signal(uid, asset):
    df_h1 = await get_twelvedata(asset, interval="1h")
    df_h4 = await get_twelvedata(asset, interval="4h")
    if df_h1 is None or df_h4 is None:
        await bot.send_message(uid, f"⚠️ Нет данных для {asset}")
        return
    df_h1 = add_indicators(df_h1)
    df_h4 = add_indicators(df_h4)

    dir_ml, acc_ml = ml_predict(df_h1.iloc[-1])
    dir_h4, acc_h4 = ml_predict(df_h4.iloc[-1])

    if dir_ml != dir_h4:  # фильтр по старшему ТФ
        direction, accuracy = "neutral", 50
    else:
        direction, accuracy = dir_ml, acc_ml

    news_score = await get_news_sentiment(asset)
    if news_score > 0.2 and direction != "sell":
        direction = "buy"
        accuracy = min(100, accuracy + 10)
    elif news_score < -0.2 and direction != "buy":
        direction = "sell"
        accuracy = min(100, accuracy + 10)

    price = df_h1["close"].iloc[-1]
    atr = df_h1["atr"].iloc[-1]
    if direction == "buy":
        tp_price = round(price + 2 * atr, 2)
        sl_price = round(price - 1.5 * atr, 2)
    elif direction == "sell":
        tp_price = round(price - 2 * atr, 2)
        sl_price = round(price + 1.5 * atr, 2)
    else:
        tp_price, sl_price = price, price

    msg = f"📢 Сигнал для <b>{asset}</b>\nНаправление: <b>{direction.upper()}</b>\nЦена: {price}\n🟢 TP: {tp_price}\n🔴 SL: {sl_price}\n📊 Точность: {accuracy}%\n📰 Новости: {'позитив' if news_score>0 else 'негатив' if news_score<0 else 'нейтрально'}"
    muted = user_settings.get(uid, {}).get("muted", False)
    await bot.send_message(uid, msg, disable_notification=muted)

# === HANDLERS ===
@dp.message(CommandStart())
async def start(message: types.Message):
    user_settings[message.from_user.id] = {"asset": "BTC/USD", "muted": False}
    await message.answer("🚀 Trading Bot запущен!", reply_markup=get_main_keyboard())

@dp.message()
async def handle_buttons(message: types.Message):
    uid = message.from_user.id
    text = message.text
    if uid not in user_settings:
        user_settings[uid] = {"asset": "BTC/USD", "muted": False}
    if text == "🔄 Получить сигнал":
        await send_signal(uid, user_settings[uid]["asset"])
    elif text in ASSETS:
        user_settings[uid]["asset"] = text
        await message.answer(f"✅ Актив установлен: {text}")
    elif text == "🔕 Mute":
        user_settings[uid]["muted"] = True
        await message.answer("🔕 Уведомления отключены")
    elif text == "🔔 Unmute":
        user_settings[uid]["muted"] = False
        await message.answer("🔔 Уведомления включены")

# === AUTO LOOP ===
async def auto_signal_loop():
    while True:
        for uid, settings in user_settings.items():
            await send_signal(uid, settings["asset"])
        await asyncio.sleep(900)

async def main():
    await train_models("BTC/USD")
    loop = asyncio.get_event_loop()
    loop.create_task(auto_signal_loop())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
