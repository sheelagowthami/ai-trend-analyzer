import nltk
import streamlit as st
from pytrends.request import TrendReq
from textblob import TextBlob
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI Trend Analyzer", layout="centered")

st.title("📈 AI Trend + Smart Prediction Analyzer")

keyword = st.text_input("Enter keyword (e.g., AI, IPL, Jobs):")

if keyword:

    pytrends = TrendReq(hl='en-US', tz=330)
    pytrends.build_payload([keyword], timeframe='today 5-y')
    data = pytrends.interest_over_time()

    if not data.empty:

        df = data.reset_index()

        # 📊 Original Trend
        st.subheader("📊 Original Trend")
        st.line_chart(data[keyword])

        # 🔥 Smoothing
        df['Smoothed'] = df[keyword].rolling(window=7).mean()

        st.subheader("📉 Smoothed Trend")
        st.line_chart(df.set_index('date')['Smoothed'])

        # 🔥 Momentum (last 7 days)
        recent = df['Smoothed'].dropna().tail(7)

        if len(recent) > 1:
            momentum = recent.iloc[-1] - recent.iloc[0]
        else:
            momentum = 0

        # 🔮 Prediction
        future_days = 30
        last_value = df['Smoothed'].iloc[-1]

        future_values = []
        current = last_value

        for i in range(future_days):
            current = current + (momentum / 7)
            future_values.append(max(0, current))

        future_dates = pd.date_range(
            start=df['date'].iloc[-1],
            periods=future_days + 1
        )[1:]

        future_df = pd.DataFrame({
            'date': future_dates,
            'Prediction': future_values
        }).set_index('date')

        st.subheader("🔮 Future Prediction (Next 30 Days)")
        st.line_chart(future_df)

        # 📈 Combined
        combined = pd.concat([
            df.set_index('date')['Smoothed'],
            future_df['Prediction']
        ])

        st.subheader("📈 Combined Trend (Past + Future)")
        st.line_chart(combined)

        # 😊 Sentiment
        blob = TextBlob(keyword)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            sentiment = "😊 Positive"
        elif polarity < 0:
            sentiment = "😡 Negative"
        else:
            sentiment = "😐 Neutral"

        # 📈 Trend Direction
        if momentum > 0:
            trend_direction = "📈 Rising"
        elif momentum < 0:
            trend_direction = "📉 Falling"
        else:
            trend_direction = "➡️ Stable"

        # 🔥 Smart Confidence
        std_dev = df['Smoothed'].std()
        momentum_strength = abs(momentum)

        score = momentum_strength / (std_dev + 1)

        if score > 1.5:
            confidence = "High"
        elif score > 0.7:
            confidence = "Medium"
        else:
            confidence = "Low"

        # 📊 Growth
        first = df[keyword].iloc[0]
        last = df[keyword].iloc[-1]
        growth = ((last - first) / first) * 100 if first != 0 else 0

        # 📌 Insights
        st.subheader("📌 Insights")

        st.write(f"**Growth (5 years):** {growth:.2f}%")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Trend Direction:** {trend_direction}")
        st.write(f"**Prediction Confidence:** {confidence}")

        # 💡 Explanation
        if confidence == "High":
            st.success("Stable trend → prediction is more reliable.")
        elif confidence == "Medium":
            st.warning("Moderate variation → prediction may vary.")
        else:
            st.error("Highly volatile trend → prediction is uncertain.")

    else:
        st.error("No data found. Try another keyword.")
