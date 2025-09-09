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
model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
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
async def get_twelvedata(asset, interval="1h", count=50):
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
    query = "bitcoin" if "BTC" in asset else "gold" if "XAU" in asset else "ethereum"
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

def add_indicators(df):
    df = df.copy()
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df = df.dropna()
    return df

# === ML TRAINING ===
async def train_model(asset="BTC/USD"):
    global ml_trained, model, scaler
    df = await get_twelvedata(asset, count=500)
    if df is None: return
    df = add_indicators(df)
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    features = df[["ema10", "ema50", "rsi", "macd"]].iloc[:-3]
    labels = df["target"].iloc[:-3]
    X = scaler.fit_transform(features)
    y = labels
    model.fit(X, y)
    ml_trained = True
    logging.info("✅ ML модель обучена")

def ml_predict(latest_row):
    if not ml_trained:
        return "neutral", 50
    X = np.array([[latest_row["ema10"], latest_row["ema50"], latest_row["rsi"], latest_row["macd"]]])
    X = scaler.transform(X)
    prob = model.predict_proba(X)[0]
    if prob[1] > 0.55:
        return "buy", int(prob[1]*100)
    elif prob[0] > 0.55:
        return "sell", int(prob[0]*100)
    return "neutral", 50

# === SIGNAL ===
async def send_signal(uid, asset):
    df = await get_twelvedata(asset)
    if df is None or len(df) < 50:
        await bot.send_message(uid, f"⚠️ Нет данных для {asset}")
        return
    df = add_indicators(df)
    dir_rule, acc_rule = ("buy", 50)  # можно добавить rule-based позже
    dir_ml, acc_ml = ml_predict(df.iloc[-1])
    news_score = await get_news_sentiment(asset)

    # Комбинируем ML и новости
    direction = dir_ml
    accuracy = acc_ml
    if news_score > 0.15 and direction != "sell":
        direction = "buy"
        accuracy = min(100, accuracy + 10)
    elif news_score < -0.15 and direction != "buy":
        direction = "sell"
        accuracy = min(100, accuracy + 10)

    price = df["close"].iloc[-1]
    tp_pct, sl_pct = 2.0, 1.0
    tp_price = round(price*(1+tp_pct/100),2) if direction=="buy" else round(price*(1-tp_pct/100),2)
    sl_price = round(price*(1-sl_pct/100),2) if direction=="buy" else round(price*(1+sl_pct/100),2)

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
        await asyncio.sleep(900)  # каждые 15 минут

async def main():
    await train_model("BTC/USD")
    loop = asyncio.get_event_loop()
    loop.create_task(auto_signal_loop())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
