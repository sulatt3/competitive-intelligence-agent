
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from difflib import SequenceMatcher
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import google.generativeai as genai
from newsapi import NewsApiClient
from pytrends.request import TrendReq

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-pro")

st.set_page_config(page_title="Competitive Intelligence Agent", layout="wide")

st.markdown("""
<style>
.main { padding: 2rem; }
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    height: 3em;
    border-radius: 5px;
    font-weight: bold;
}
.metric-card {
    background-color: #f8f9fa;
    border: 2px solid #e9ecef;
    border-radius: 10px;
    padding: 25px;
    text-align: center;
}
.metric-label {
    font-size: 14px;
    color: #6c757d;
    margin-bottom: 10px;
}
.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #212529;
}
</style>
""", unsafe_allow_html=True)

def fetch_news(company: str) -> List[Dict[str, Any]]:
    try:
        newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
        from_date = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
        articles = newsapi.get_everything(q=company, language="en", sort_by="publishedAt", from_param=from_date, page_size=100)
        return [{"source": "News", "title": a["title"], "url": a["url"], 
                 "date": a["publishedAt"][:10] if a["publishedAt"] else "Unknown",
                 "snippet": a["description"] or a["content"] or "",
                 "full_content": a["content"] or ""} 
                for a in articles.get("articles", [])]
    except Exception as e:
        st.warning(f"NewsAPI error: {e}")
        return []

# Replace ONLY the fetch_trends function with this debugged version:

def fetch_trends(company: str) -> List[Dict[str, Any]]:
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload([company], timeframe="today 3-m", geo="")
        df = pytrends.interest_over_time()
        
        # Debug logging
        st.write(f"DEBUG - Trends DataFrame empty: {df.empty}")
        st.write(f"DEBUG - Columns: {df.columns.tolist() if not df.empty else 'None'}")
        st.write(f"DEBUG - Company '{company}' in columns: {company in df.columns if not df.empty else False}")
        
        if df.empty or company not in df.columns:
            return [{"source": "Google Trends", "current_interest_level": None}]
        
        interest = df[company]
        mean, std = interest.mean(), interest.std()
        spikes = interest[interest > mean + 1.5 * std]
        
        result = {
            "source": "Google Trends",
            "current_interest_level": int(interest.iloc[-1]),
            "peak_90d": int(interest.max()),
            "average_90d": int(mean),
            "spike_dates": [idx.strftime("%Y-%m-%d") for idx in spikes.index],
            "trend_direction": "rising" if interest.iloc[-7:].mean() > mean else "stable/declining"
        }
        
        st.write(f"DEBUG - Trends result: {result}")
        return [result]
        
    except Exception as e:
        st.warning(f"Google Trends error: {e}")
        st.write(f"DEBUG - Full error: {str(e)}")
        return [{"source": "Google Trends", "current_interest_level": None}]

def collect_all_data(company: str, use_news: bool, use_trends: bool) -> List[Dict[str, Any]]:
    fetchers = []
    if use_news:
        fetchers.append(fetch_news)
    if use_trends:
        fetchers.append(fetch_trends)
    all_data = []
    if fetchers:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(f, company): f for f in fetchers}
            for future in as_completed(futures):
                data = future.result()
                all_data.extend(data if isinstance(data, list) else [data])
    return all_data

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def deduplicate_by_title(items: List[Dict[str, Any]], threshold: float = 0.85) -> List[Dict[str, Any]]:
    if not items:
        return []
    unique, seen = [], []
    for item in items:
        title = item.get("title", "")
        if title and not any(similarity(title, s) > threshold for s in seen):
            unique.append(item)
            seen.append(title)
    return unique

def score_relevance(item: Dict[str, Any], company: str) -> float:
    score, company_lower = 0.0, company.lower()
    title = item.get("title", "").lower()
    if company_lower in title:
        score += 40
    combined = f"{item.get('snippet', '').lower()} {item.get('full_content', '').lower()}"
    if company_lower in combined:
        score += 20
    score += min(combined.count(company_lower) * 5, 20)
    date_str = item.get("date", "")
    if date_str != "Unknown":
        try:
            days_old = (datetime.now() - datetime.strptime(date_str, "%Y-%m-%d")).days
            score += 15 if days_old <= 7 else (10 if days_old <= 14 else (5 if days_old <= 30 else 0))
        except:
            pass
    return min(score, 100)

def rank_and_filter(items: List[Dict[str, Any]], company: str, top_n: int = 60) -> List[Dict[str, Any]]:
    for item in items:
        item["relevance_score"] = score_relevance(item, company)
    return sorted(items, key=lambda x: x.get("relevance_score", 0), reverse=True)[:top_n]

def process_data(raw_data: List[Dict[str, Any]], company: str, top_n: int = 60) -> List[Dict[str, Any]]:
    return rank_and_filter(deduplicate_by_title(raw_data), company, top_n=top_n)

def analyze_sentiment(text: str) -> str:
    pos = ["launch", "growth", "success", "innovation", "partnership", "funding", "expansion", "breakthrough"]
    neg = ["loss", "decline", "controversy", "lawsuit", "criticism", "failure", "layoff", "scandal"]
    t = text.lower()
    p, n = sum(1 for w in pos if w in t), sum(1 for w in neg if w in t)
    return "positive" if p > n else ("negative" if n > p else "neutral")

def check_data_quality(processed_data, company):
    news = [i for i in processed_data if i.get("source") == "News"]
    if len(news) < 5:
        return {"sufficient": False, "warning": f"# Data Quality Warning: {company}\n\nOnly {len(news)} news articles found.\n\n**Try:** Full legal name or enable more sources."}
    high = [i for i in news if i.get("relevance_score", 0) > 50]
    if len(high) < 3:
        return {"sufficient": False, "warning": f"# Data Quality Warning\n\nLow relevance data. Verify company name."}
    return {"sufficient": True}

def build_synthesis_prompt(processed_data: List[Dict[str, Any]], company: str) -> str:
    news = [i for i in processed_data if i.get("source") == "News"]
    trends = [i for i in processed_data if i.get("source") == "Google Trends"]
    news_text = "\n".join([f"[{i}] {item['title']} ({item['date']})" for i, item in enumerate(news, 1)])
    trends_text = ""
    if trends and trends[0].get("current_interest_level"):
        t = trends[0]
        trends_text = f"\nTrends: {t['current_interest_level']}/100, {t['trend_direction']}, Spikes: {', '.join(t.get('spike_dates', []))}"
    return f"""You are an expert competitive intelligence analyst. Analyze {company} and create a comprehensive brief with these sections:

# Competitive Intelligence Brief: {company}
**Generated:** {datetime.now().strftime("%Y-%m-%d")}

## Executive Summary
High-level overview of competitive position and recent activities

## Recent Key Moves
Strategic moves and developments in last 30 days with dates

## Product Launches & Features
New products, features, updates

## Funding, Partnerships & Hiring
Funding rounds, partnerships, acquisitions, key hires

## Google Trends Analysis
Analyze search interest, spikes, trend direction

## Competitive Threats & Opportunities
What threats does {company} pose? What opportunities/weaknesses exist?

## Strategic Recommendations
3-5 actionable recommendations for competitors

## Timeline of Key Events
Chronological timeline of important events

## Sources
Articles analyzed and date range

Be specific, cite dates, focus on facts. 1000-1500 words.

DATA:
{news_text}
{trends_text}"""

def generate_competitive_brief(company: str, use_news: bool, use_trends: bool, num_articles: int) -> tuple:
    raw = collect_all_data(company, use_news, use_trends)
    if not raw:
        return "# Error\n\nNo data sources enabled.", None
    processed = process_data(raw, company, top_n=num_articles)
    quality = check_data_quality(processed, company)
    if not quality["sufficient"]:
        return quality["warning"], None
    response = model.generate_content(build_synthesis_prompt(processed, company))
    return response.text, processed

def create_timeline_viz(processed_data):
    news = [i for i in processed_data if i.get("source") == "News" and i.get("date") != "Unknown"]
    if not news:
        return None
    def classify(title):
        t = title.lower()
        if any(w in t for w in ["launch", "release", "unveil", "introduce"]):
            return "Product Launch"
        if any(w in t for w in ["funding", "raises", "investment", "series"]):
            return "Funding"
        if any(w in t for w in ["partner", "acquisition", "merger", "deal"]):
            return "Partnership"
        if any(w in t for w in ["hire", "appoint", "ceo", "executive"]):
            return "Leadership"
        return "General News"
    for i in news:
        i["event_type"] = classify(i.get("title", ""))
    df = pd.DataFrame(news)
    df["date"] = pd.to_datetime(df["date"])
    fig = px.scatter(df, x="date", y="relevance_score", color="event_type", size="relevance_score",
                     hover_data=["title"], title="Timeline of Key Events (Color-Coded by Type)",
                     color_discrete_map={"Product Launch": "#00CC66", "Funding": "#0066CC", 
                                        "Partnership": "#9933FF", "Leadership": "#FF6600", "General News": "#999999"})
    fig.update_layout(height=450, hovermode="closest")
    return fig

def create_sentiment_viz(processed_data):
    news = [i for i in processed_data if i.get("source") == "News" and i.get("date") != "Unknown"]
    if not news:
        return None
    for i in news:
        i["sentiment"] = analyze_sentiment(i.get("title", "") + " " + i.get("snippet", ""))
    df = pd.DataFrame(news)
    df["date"] = pd.to_datetime(df["date"])
    counts = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig = px.bar(counts, x="date", y="count", color="sentiment", title="Sentiment Analysis Over Time",
                 color_discrete_map={"positive": "#28a745", "negative": "#dc3545", "neutral": "#6c757d"})
    fig.update_layout(height=450)
    return fig

def create_competitive_matrix(data):
    news = len([d for d in data if d.get("source") == "News"])
    scores = [d.get("relevance_score", 0) for d in data if "relevance_score" in d]
    avg = sum(scores) / len(scores) if scores else 0
    trends = [d for d in data if d.get("source") == "Google Trends"]
    interest = trends[0].get("current_interest_level") if trends and trends[0].get("current_interest_level") else 50
    fig = go.Figure(go.Scatter(x=[news], y=[avg], mode="markers+text", text=["Analyzed Company"],
                               textposition="top center",
                               marker=dict(size=interest, color="#0066CC", opacity=0.6, line=dict(width=2, color="white"))))
    fig.update_layout(title="Market Presence Analysis", xaxis_title="News Volume (Last 30 Days)", 
                     yaxis_title="Average Relevance Score", height=450)
    return fig

if "current_report" not in st.session_state:
    st.session_state.current_report = None
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

st.title("Competitive Intelligence Agent")
st.markdown("**AI-powered competitive analysis from news and search trends**")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    company_name = st.text_input("Company Name", placeholder="e.g., Anthropic, Perplexity, OpenAI")
    
    st.subheader("Data Sources")
    use_news = st.checkbox("News API", value=True, help="Recent articles (last 30 days)")
    use_trends = st.checkbox("Google Trends", value=True, help="Search interest data")
    
    st.subheader("Analysis Options")
    num_articles = st.slider(
        "Top articles to analyze",
        min_value=30,
        max_value=100,
        value=60,
        step=10,
        help="More articles = more comprehensive (recommended: 60-80)"
    )
    
    enable_comparison = st.checkbox(
        "Multi-company comparison",
        value=False,
        disabled=True,
        help="Coming in V2 - compare multiple companies side-by-side"
    )
    
    generate_btn = st.button("Generate Report", type="primary")
    st.caption("Built with Gemini 2.5 Pro")

tab1, tab2 = st.tabs(["Report & Visualizations", "Past Reports"])

with tab1:
    if generate_btn:
        if company_name:
            if not (use_news or use_trends):
                st.error("Enable at least one data source")
            else:
                with st.spinner(f"Analyzing {company_name}... analyzing {num_articles} articles"):
                    prog = st.progress(0)
                    prog.progress(33)
                    report, data = generate_competitive_brief(company_name, use_news, use_trends, num_articles)
                    prog.progress(100)
                    st.session_state.current_report = report
                    st.session_state.current_company = company_name
                    st.session_state.processed_data = data
                    prog.empty()
                st.success(f"Report generated for {company_name}")
        else:
            st.error("Enter a company name")
    
    if st.session_state.current_report and st.session_state.processed_data:
        st.download_button("Download Report", st.session_state.current_report,
                          f"{st.session_state.current_company}_{datetime.now().strftime('%Y%m%d')}.md",
                          mime="text/markdown")
        
        st.markdown("---")
        st.subheader("Key Metrics")
        
        data = st.session_state.processed_data
        news_ct = len([d for d in data if d.get("source") == "News"])
        scores = [d.get("relevance_score", 0) for d in data if "relevance_score" in d]
        avg_rel = int(sum(scores) / len(scores)) if scores else 0
        trends = [d for d in data if d.get("source") == "Google Trends"]
        search_int = trends[0].get("current_interest_level") if trends and trends[0].get("current_interest_level") else None
        
        sources_used = []
        if news_ct > 0:
            sources_used.append("News")
        if search_int is not None:
            sources_used.append("Trends")
        sources_str = "+".join(sources_used) if sources_used else "None"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Articles Analyzed</div>
                <div class="metric-value">{news_ct}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            search_display = f"{search_int}/100" if search_int else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Search Interest</div>
                <div class="metric-value">{search_display}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Relevance</div>
                <div class="metric-value">{avg_rel}/100</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sources Used</div>
                <div class="metric-value">{sources_str}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Visual Intelligence")
        
        viz_tabs = st.tabs(["Timeline", "Sentiment", "Market Presence"])
        
        with viz_tabs[0]:
            fig = create_timeline_viz(st.session_state.processed_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timeline data available")
        
        with viz_tabs[1]:
            fig = create_sentiment_viz(st.session_state.processed_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available")
        
        with viz_tabs[2]:
            fig = create_competitive_matrix(st.session_state.processed_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No market data available")
        
        st.markdown("---")
        st.markdown(st.session_state.current_report)
    
    else:
        st.info("Enter a company name in the sidebar and click Generate Report")
        
        st.markdown("### How it works:")
        st.markdown("""
        1. **Data Collection**: Fetches 100+ news articles and Google Trends data
        2. **Smart Filtering**: Deduplicates and ranks articles by relevance (0-100 score)
        3. **AI Synthesis**: Gemini 2.5 Pro analyzes data and generates strategic insights
        4. **Visualizations**: Interactive charts reveal patterns and trends
        5. **Results**: Professional competitive intelligence report in 60 seconds
        """)
        
        st.markdown("### Data Sources:")
        st.markdown("""
        - **News API**: Recent articles, press releases, announcements (last 30 days, ~100 articles)
        - **Google Trends**: Search interest patterns, spike detection, trend direction (90-day history)
        """)
        
        st.markdown("### Analysis Includes:")
        st.markdown("""
        - Executive summary with competitive positioning assessment
        - Recent strategic moves and announcements (with specific dates)
        - Product launches and feature releases
        - Funding rounds, partnerships, and key executive hires
        - Google Trends analysis with event correlation
        - Competitive threats and market opportunities
        - Strategic recommendations for competitors (3-5 actionable items)
        - Chronological timeline of key events
        """)
        
        st.markdown("### Interactive Visualizations:")
        st.markdown("""
        - **Timeline**: Events plotted over time, color-coded by type (product launches, funding, partnerships, leadership changes)
        - **Sentiment Analysis**: Positive/negative/neutral sentiment trends based on article language
        - **Market Presence**: News volume and content relevance positioning
        """)

with tab2:
    st.header("Past Reports")
    st.info("V2 Feature: Searchable report history, comparison mode, and trend tracking")
