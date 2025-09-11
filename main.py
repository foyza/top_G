import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from dotenv import load_dotenv
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# === CONFIG ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

ASSETS = ['BTC/USD', 'XAU/USD', 'ETH/USD']

logging.basicConfig(level=logging.INFO)

dp = Dispatcher()
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)

user_settings = {}  # {uid: {"asset": ... , "muted": False}}

# === ML MODEL ===
gb_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
lstm_model = None
scaler = StandardScaler()
ml_trained = False

# === NLP ===
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# === UI ===
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Получить сигнал")],
            [KeyboardButton(text="BTC/USD"), KeyboardButton(text="XAU/USD"), KeyboardButton(text="ETH/USD")],
            [KeyboardButton(text="🔕 Mute"), KeyboardButton(text="🔔 Unmute")]
        ],
        resize_keyboard=True
    )

# === DATA ===
async def get_twelvedata(asset, interval="1h", count=150):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": asset,
        "interval": interval,
        "outputsize": count,
        "apikey": TWELVEDATA_API_KEY,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            if "values" not in data:
                return None
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime")
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            return df

async def get_news_sentiment(asset):
    # Фильтруем только сильные новости: ETF, регуляторы, макростатистика
    query_map = {
        "BTC": "bitcoin (ETF OR regulation OR macro)",
        "XAU": "gold (ETF OR regulation OR macro)",
        "ETH": "ethereum (ETF OR regulation OR macro)"
    }
    query = query_map.get(asset.split("/")[0], asset)
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            data = await r.json()
            if "articles" not in data:
                return 0
            scores = []
            for art in data["articles"][:5]:
                text = art.get("title", "") + " " + art.get("description", "")
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
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_obv(df):
    if "volume" not in df.columns:
        return pd.Series(np.zeros(len(df)))
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    return obv

def compute_bollinger(df, period=20):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = sma + 2*std
    lower = sma - 2*std
    return upper, lower

def add_indicators(df):
    df = df.copy()
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["atr"] = compute_atr(df)
    df["obv"] = compute_obv(df)
    df["bb_upper"], df["bb_lower"] = compute_bollinger(df)
    df = df.dropna()
    return df

# === ML TRAINING ===
async def train_models(asset="BTC/USD"):
    global ml_trained, gb_model, lstm_model, scaler
    df = await get_twelvedata(asset, count=500)
    if df is None: return
    df = add_indicators(df)
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    features = ["ema10","ema50","rsi","macd","atr","obv","bb_upper","bb_lower"]
    X = df[features].iloc[:-3]
    y = df["target"].iloc[:-3]
    X_scaled = scaler.fit_transform(X)
    gb_model.fit(X_scaled, y)
    
    # Простая LSTM
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, input_shape=(X_scaled.shape[1],1)))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
    lstm_model.fit(X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1],1)), y, epochs=5, batch_size=16, verbose=0)
    
    ml_trained = True
    logging.info("✅ ML + LSTM модели обучены")

def ml_predict(latest_row):
    if not ml_trained:
        return "neutral", 50
    features = ["ema10","ema50","rsi","macd","atr","obv","bb_upper","bb_lower"]
    X = np.array([[latest_row[f] for f in features]])
    X_scaled = scaler.transform(X)
    gb_prob = gb_model.predict_proba(X_scaled)[0][1]
    lstm_prob = lstm_model.predict(X_scaled.reshape((1,X_scaled.shape[1],1)), verbose=0)[0][0]
    avg_prob = (gb_prob + lstm_prob)/2
    if avg_prob > 0.55:
        return "buy", int(avg_prob*100)
    elif avg_prob < 0.45:
        return "sell", int((1-avg_prob)*100)
    return "neutral", 50

# === SIGNAL ===
async def send_signal(uid, asset):
    df = await get_twelvedata(asset)
    if df is None or len(df) < 50:
        await bot.send_message(uid, f"⚠️ Нет данных для {asset}")
        return
    df = add_indicators(df)
    dir_ml, acc_ml = ml_predict(df.iloc[-1])
    news_score = await get_news_sentiment(asset)

    # Комбинируем ML и новости
    direction = dir_ml
    accuracy = acc_ml
    if news_score > 0.2 and direction != "sell":
        direction = "buy"
        accuracy = min(100, accuracy + 10)
    elif news_score < -0.2 and direction != "buy":
        direction = "sell"
        accuracy = min(100, accuracy + 10)

    price = df["close"].iloc[-1]
    # Умные TP/SL через ATR
    atr = df["atr"].iloc[-1]
    tp_price = round(price + atr*2 if direction=="buy" else price - atr*2,2)
    sl_price = round(price - atr if direction=="buy" else price + atr,2)

    msg = f"📢 Сигнал для <b>{asset}</b>\nНаправление: <b>{direction.upper()}</b>\nЦена: {price}\n🟢 TP: {tp_price}\n🔴 SL: {sl_price}\n📊 Точность: {accuracy}%\n📰 Новости: {'позитив' if news_score>0 else 'негатив' if news_score<0 else 'нейтрально'}"
    muted = user_settings.get(uid, {}).get("muted", False)
    await bot.send_message(uid, msg, disable_notification=muted)

# === HANDLERS ===
@dp.message(CommandStart())
async def start(message: types.Message):
    user_settings[message.from_user.id] = {"asset": "BTC/USD", "muted": False}
    await message.answer("Escape the matrix", reply_markup=get_main_keyboard())

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
