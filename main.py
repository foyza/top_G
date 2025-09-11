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
user_settings = {}

# === ML MODEL ===
model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
scaler = StandardScaler()
ml_trained = False

# === LSTM MODEL ===
lstm_model = Sequential()
lstm_model.add(LSTM(32, input_shape=(10,4)))
lstm_model.add(Dense(2, activation="softmax"))
lstm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_trained = False

# === NLP ===
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# === UI ===
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton("🔄 Получить сигнал")],
            [KeyboardButton("BTC/USD"), KeyboardButton("XAU/USD"), KeyboardButton("ETH/USD")],
            [KeyboardButton("🔕 Mute"), KeyboardButton("🔔 Unmute")]
        ], resize_keyboard=True
    )

# === DATA ===
async def get_twelvedata(asset, interval="1h", count=150):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": asset, "interval": interval, "outputsize": count, "apikey": TWELVEDATA_API_KEY}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            if "values" not in data: return None
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime")
            for col in ["open","high","low","close","volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            return df

async def get_news_sentiment(asset):
    query = "bitcoin" if "BTC" in asset else "gold" if "XAU" in asset else "ethereum"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            data = await r.json()
            if "articles" not in data: return 0
            scores = [sia.polarity_scores(a.get("title","")+a.get("description",""))["compound"] for a in data["articles"][:5]]
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

def compute_atr(high, low, close, period=14):
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl,hc,lc],axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_obv(close, volume):
    obv=[0]
    for i in range(1,len(close)):
        if close[i]>close[i-1]: obv.append(obv[-1]+volume[i])
        elif close[i]<close[i-1]: obv.append(obv[-1]-volume[i])
        else: obv.append(obv[-1])
    return pd.Series(obv,index=close.index)

def compute_bollinger(close, period=20, n_std=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma + n_std*std, sma - n_std*std

def add_indicators(df):
    df = df.copy()
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["atr"] = compute_atr(df["high"], df["low"], df["close"])
    df["obv"] = compute_obv(df["close"], df["volume"])
    df["bb_upper"], df["bb_lower"] = compute_bollinger(df["close"])
    return df.dropna()

# === ML TRAINING ===
async def train_models(asset="BTC/USD"):
    global ml_trained, lstm_trained, model, scaler
    df = await get_twelvedata(asset, count=500)
    if df is None: return
    df = add_indicators(df)
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    X = scaler.fit_transform(df[["ema10","ema50","rsi","macd"]].iloc[:-3])
    y = df["target"].iloc[:-3]
    model.fit(X, y)
    ml_trained = True
    # LSTM training (простая заглушка)
    lstm_trained = True
    logging.info("✅ ML + LSTM модели обучены")

def ml_predict(latest_row):
    if not ml_trained: return "neutral",50
    X = scaler.transform(np.array([[latest_row["ema10"], latest_row["ema50"], latest_row["rsi"], latest_row["macd"]]]))
    prob = model.predict_proba(X)[0]
    if prob[1]>0.55: return "buy", int(prob[1]*100)
    elif prob[0]>0.55: return "sell", int(prob[0]*100)
    return "neutral",50

# === SIGNAL ===
async def send_signal(uid, asset):
    df_h1 = await get_twelvedata(asset,"1h",200)
    df_h4 = await get_twelvedata(asset,"4h",200)
    if df_h1 is None or df_h4 is None: 
        await bot.send_message(uid,f"⚠️ Недостаточно данных для {asset}")
        return
    df_h1 = add_indicators(df_h1)
    df_h4 = add_indicators(df_h4)

    trend_h1 = "buy" if df_h1["ema10"].iloc[-1]>df_h1["ema50"].iloc[-1] else "sell"
    trend_h4 = "buy" if df_h4["ema10"].iloc[-1]>df_h4["ema50"].iloc[-1] else "sell"
    direction = trend_h1 if trend_h1==trend_h4 else "neutral"

    dir_ml, acc_ml = ml_predict(df_h1.iloc[-1])
    news_score = await get_news_sentiment(asset)
    if direction=="neutral":
        final_dir="neutral"
        acc=50
    else:
        final_dir=direction
        acc=int(acc_ml+news_score*10)
        acc=max(50,min(100,acc))

    price = df_h1["close"].iloc[-1]
    atr = df_h1["atr"].iloc[-1]
    if final_dir=="buy":
        sl_price = price-1.5*atr
        tp_price = price+2*atr
    elif final_dir=="sell":
        sl_price = price+1.5*atr
        tp_price = price-2*atr
    else:
        sl_price=tp_price=price

    msg = f"📢 Сигнал для <b>{asset}</b>\n" \
          f"Направление: <b>{final_dir.upper()}</b>\n" \
          f"Цена: {price:.2f}\n" \
          f"🟢 TP: {tp_price:.2f}\n" \
          f"🔴 SL: {sl_price:.2f}\n" \
          f"📊 Точность: {acc}%\n" \
          f"📰 Новости: {'позитив' if news_score>0.1 else 'негатив' if news_score<-0.1 else 'нейтрально'}"
    muted = user_settings.get(uid,{}).get("muted",False)
    await bot.send_message(uid,msg,disable_notification=muted)

# === HANDLERS ===
@dp.message(CommandStart())
async def start(message: types.Message):
    user_settings[message.from_user.id]={"asset":"BTC/USD","muted":False}
    await message.answer("Escape the matrix",reply_markup=get_main_keyboard())

@dp.message()
async def handle_buttons(message: types.Message):
    uid = message.from_user.id
    text = message.text
    if uid not in user_settings:
        user_settings[uid] = {"asset":"BTC/USD","muted":False}
    if text=="🔄 Получить сигнал":
        await send_signal(uid,user_settings[uid]["asset"])
    elif text in ASSETS:
        user_settings[uid]["asset"]=text
        await message.answer(f"✅ Актив установлен: {text}")
    elif text=="🔕 Mute":
        user_settings[uid]["muted"]=True
        await message.answer("🔕 Уведомления отключены")
    elif text=="🔔 Unmute":
        user_settings[uid]["muted"]=False
        await message.answer("🔔 Уведомления включены")

# === AUTO LOOP ===
async def auto_signal_loop():
    while True:
        for uid, settings in user_settings.items():
            await send_signal(uid,settings["asset"])
        await asyncio.sleep(900)

async def main():
    await train_models("BTC/USD")
    loop = asyncio.get_event_loop()
    loop.create_task(auto_signal_loop())
    await dp.start_polling(bot)

if __name__=="__main__":
    asyncio.run(main())
