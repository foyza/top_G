import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
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
user_settings = {}  # {uid: {"asset":..., "muted":False}}

# === ML ===
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
lgb_model = None
scaler = StandardScaler()
ml_trained = False

# === NLP ===
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# === UI ===
def get_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton("🔄 Получить сигнал")],
            [KeyboardButton("BTC/USD"), KeyboardButton("XAU/USD"), KeyboardButton("ETH/USD")],
            [KeyboardButton("🔕 Mute"), KeyboardButton("🔔 Unmute")],
            [KeyboardButton("🕒 Расписание")]
        ],
        resize_keyboard=True
    )

# === DATA ===
async def get_twelvedata(asset, interval="1h", count=1000):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": asset, "interval": interval, "outputsize": count, "apikey": TWELVEDATA_API_KEY}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
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
            scores = [sia.polarity_scores(a["title"] + " " + (a.get("description") or ""))["compound"] for a in data["articles"][:5]]
            return np.mean(scores) if scores else 0

# === INDICATORS ===
def compute_indicators(df):
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - 100 / (1 + up.rolling(14).mean() / down.rolling(14).mean())
    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["momentum"] = df["close"] - df["close"].shift(3)
    df["volatility"] = df["close"].rolling(14).std()
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = ma20 + 2*std20
    df["bb_lower"] = ma20 - 2*std20
    return df.dropna()

# === TRAINING ===
async def train_ml():
    global ml_trained, rf_model, lgb_model, scaler
    all_data=[]
    for asset in ASSETS:
        df = await get_twelvedata(asset, count=1000)
        if df is None: continue
        df = compute_indicators(df)
        df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
        df["asset"] = asset
        all_data.append(df)
    df_all = pd.concat(all_data).dropna()
    features=["ema10","ema50","rsi","macd","momentum","volatility","obv","bb_upper","bb_lower"]
    X = scaler.fit_transform(df_all[features])
    y = df_all["target"]
    rf_model.fit(X, y)
    lgb_model = lgb.LGBMClassifier(n_estimators=500)
    lgb_model.fit(X, y)
    ml_trained=True
    logging.info("✅ ML обучен")

def ml_predict(row):
    if not ml_trained: return "neutral",50
    features=["ema10","ema50","rsi","macd","momentum","volatility","obv","bb_upper","bb_lower"]
    X = scaler.transform(np.array([row[features]]))
    prob_rf = rf_model.predict_proba(X)[0]
    prob_lgb = lgb_model.predict_proba(X)[0]
    buy_prob = (prob_rf[1]+prob_lgb[1])/2
    sell_prob = (prob_rf[0]+prob_lgb[0])/2
    if buy_prob>0.55: return "buy", int(buy_prob*100)
    elif sell_prob>0.55: return "sell", int(sell_prob*100)
    return "neutral",50

# === SIGNAL ===
async def send_signal(user_id, asset):
    df = await get_twelvedata(asset, count=50)
    if df is None or len(df)<50: 
        await bot.send_message(user_id,f"⚠️ Нет данных для {asset}"); return
    df=compute_indicators(df)
    dir_ml, acc_ml = ml_predict(df.iloc[-1])
    news_score = await get_news_sentiment(asset)
    direction = dir_ml
    accuracy = int(acc_ml*0.7 + abs(news_score)*0.3*100)
    price = df["close"].iloc[-1]
    tp, sl = 2.0, 1.0
    tp_price = round(price*(1+tp/100),2) if direction=="buy" else round(price*(1-tp/100),2)
    sl_price = round(price*(1-sl/100),2) if direction=="buy" else round(price*(1+sl/100),2)
    news_txt="позитив" if news_score>0 else "негатив" if news_score<0 else "нейтрально"
    msg=(
        f"📢 <b>{asset}</b>\nНаправление: <b>{direction.upper()}</b>\nЦена: {price}\n"
        f"🟢 TP: {tp_price} (+{tp}%)\n🔴 SL: {sl_price} (-{sl}%)\n📊 Точность: {accuracy}%\n📰 Новости: {news_txt}"
    )
    muted=user_settings.get(user_id,{}).get("muted",False)
    await bot.send_message(user_id,msg,disable_notification=muted)

# === HANDLERS ===
@dp.message(CommandStart())
async def start(msg: types.Message):
    user_settings[msg.from_user.id]={"asset":"BTC/USD","muted":False}
    await msg.answer("Добро пожаловать!",reply_markup=get_keyboard())

@dp.message()
async def handle(msg: types.Message):
    uid=msg.from_user.id; t=msg.text
    if uid not in user_settings: user_settings[uid]={"asset":"BTC/USD","muted":False}
    if t=="🔄 Получить сигнал": await send_signal(uid,user_settings[uid]["asset"])
    elif t in ASSETS: user_settings[uid]["asset"]=t; await msg.answer(f"✅ Актив: {t}")
    elif t=="🔕 Mute": user_settings[uid]["muted"]=True; await msg.answer("🔕 Уведомления отключены")
    elif t=="🔔 Unmute": user_settings[uid]["muted"]=False; await msg.answer("🔔 Уведомления включены")
    elif t=="🕒 Расписание": await msg.answer("Сигналы каждые 15 минут")

# === AUTO LOOP ===
async def auto_loop():
    while True:
        for uid,s in user_settings.items(): await send_signal(uid,s["asset"])
        await asyncio.sleep(900)

async def main():
    await train_ml()
    asyncio.create_task(auto_loop())
    await dp.start_polling(bot)

if __name__=="__main__":
    asyncio.run(main())
