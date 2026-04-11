from __future__ import annotations

from typing import Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TITLE = "美股平衡波動率優化系統"

# ===== Strategy parameters =====
MARKET_BULL_TH = 0.008
MARKET_BEAR_TH = -0.008
STOCK_BULL_TH = 0.010
STOCK_BEAR_TH = -0.010

MATRIX_DAY1 = pd.DataFrame(
    [
        [0.20, 0.25, 0.35],
        [0.25, 0.40, 0.60],
        [0.35, 0.60, 0.85],
    ],
    index=["Bull", "Neutral", "Bear"],
    columns=["Bull", "Neutral", "Bear"],
)

MARKET_TICKERS = ["QQQ", "SPY"]
INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "5d"
DAILY_PERIOD = "6mo"
ET_TZ = "America/New_York"
TARGET_TIME = "10:00"


# ===== Helpers =====
def normalize_ticker_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def validate_allocation_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    required = {"ticker", "target_dollar"}
    if not required.issubset(out.columns):
        raise ValueError("Allocation 檔必須包含 ticker 與 target_dollar 欄位。")

    out = out[["ticker", "target_dollar"]].copy()
    out["ticker"] = normalize_ticker_series(out["ticker"])
    out["target_dollar"] = pd.to_numeric(out["target_dollar"], errors="coerce")
    out = out.dropna(subset=["ticker", "target_dollar"])
    out["target_dollar"] = out["target_dollar"].round(0)
    out = out[out["target_dollar"] > 0].reset_index(drop=True)

    if out.empty:
        raise ValueError("Allocation 檔清理後沒有有效資料。")
    return out


def validate_execution_log_df(df: pd.DataFrame) -> pd.DataFrame:
    log = df.copy()
    log.columns = [str(c).strip().lower() for c in log.columns]
    if "ticker" in log.columns:
        log["ticker"] = normalize_ticker_series(log["ticker"])
    if "date" in log.columns:
        log["date"] = pd.to_datetime(log["date"], errors="coerce")
    if "filled_price" in log.columns:
        log["filled_price"] = pd.to_numeric(log["filled_price"], errors="coerce")
    return log


def classify_market_state(gap) -> str:
    gap = pd.to_numeric(gap, errors="coerce")

    if pd.isna(gap):
        return "Neutral"
    if gap > MARKET_BULL_TH:
        return "Bull"
    if gap < MARKET_BEAR_TH:
        return "Bear"
    return "Neutral"


def classify_stock_state(gap: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [gap > STOCK_BULL_TH, gap < STOCK_BEAR_TH],
            ["Bull", "Bear"],
            default="Neutral",
        ),
        index=gap.index,
    )


def get_k1(market_state: pd.Series, stock_state: pd.Series) -> pd.Series:
    row_map = {"Bull": 0, "Neutral": 1, "Bear": 2}
    col_map = {"Bull": 0, "Neutral": 1, "Bear": 2}
    arr = MATRIX_DAY1.to_numpy()
    rows = market_state.map(row_map).to_numpy()
    cols = stock_state.map(col_map).to_numpy()
    return pd.Series(arr[rows, cols], index=market_state.index, dtype=float)


def infer_day_state_from_log(
    tickers: pd.Series, execution_log: Optional[pd.DataFrame]
) -> pd.Series:
    tickers = normalize_ticker_series(tickers)
    if execution_log is None or execution_log.empty:
        return pd.Series(["Day1"] * len(tickers), index=tickers.index)

    log = execution_log.copy()
    log.columns = [str(c).strip().lower() for c in log.columns]
    if not {"ticker", "day", "filled_price"}.issubset(log.columns):
        return pd.Series(["Day1"] * len(tickers), index=tickers.index)

    log["ticker"] = normalize_ticker_series(log["ticker"])
    if "date" in log.columns:
        log["date"] = pd.to_datetime(log["date"], errors="coerce")
        log = log.sort_values(["ticker", "date"])
    latest = log.drop_duplicates("ticker", keep="last").set_index("ticker")

    result = []
    for t in tickers:
        if t not in latest.index:
            result.append("Day1")
            continue

        last_day = str(latest.at[t, "day"]).strip()
        filled = latest.at[t, "filled_price"]

        if pd.notna(filled):
            result.append("Day1")
        else:
            if last_day == "Day1":
                result.append("Day2")
            elif last_day == "Day2":
                result.append("Day3")
            else:
                result.append("Day3")

    return pd.Series(result, index=tickers.index)


def calc_true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def calc_smoothed_atr(daily_df: pd.DataFrame) -> float:
    if daily_df is None or daily_df.empty or len(daily_df) < 20:
        return np.nan
    tr = calc_true_range(daily_df)
    atr14 = tr.rolling(14).mean()
    return float(atr14.tail(5).median())


def flatten_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    cols0 = list(df.columns.get_level_values(0))
    cols1 = list(df.columns.get_level_values(1))

    price_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

    # 情況1：第一層就是價格欄位
    if any(c in price_cols for c in cols0):
        tickers = set(cols1)
        if len(tickers) == 1:
            ticker = list(tickers)[0]
            df = df.xs(ticker, axis=1, level=1)
            return df

    # 情況2：第二層才是價格欄位
    if any(c in price_cols for c in cols1):
        df = df.swaplevel(axis=1)
        tickers = set(df.columns.get_level_values(1))
        if len(tickers) == 1:
            ticker = list(tickers)[0]
            df = df.xs(ticker, axis=1, level=1)
            return df

    return df


def get_prev_close_and_atr(ticker, period="3mo", interval="1d"):
    import pandas as pd
    import yfinance as yf

    required_cols = ["Open", "High", "Low", "Close"]

    try:
        hist = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        print(f"[DEBUG] ticker={ticker}")
        print(f"[DEBUG] columns={list(hist.columns) if hasattr(hist, 'columns') else None}")
        print(f"[DEBUG] shape={hist.shape if hasattr(hist, 'shape') else None}")

        if hist is None or hist.empty:
            print(f"[WARN] {ticker}: empty history")
            return None, None

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0] if isinstance(c, tuple) else c for c in hist.columns]

        missing = [c for c in required_cols if c not in hist.columns]
        if missing:
            print(f"[WARN] {ticker}: missing {missing}, columns={list(hist.columns)}")
            return None, None

        hist = hist.dropna(subset=required_cols)
        if hist.empty:
            print(f"[WARN] {ticker}: empty after dropna")
            return None, None

        prev_close = hist["Close"].iloc[-1]

        tr = pd.concat([
            hist["High"] - hist["Low"],
            (hist["High"] - hist["Close"].shift(1)).abs(),
            (hist["Low"] - hist["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean().iloc[-1]

        return prev_close, atr

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None, None

def get_intraday_price_10am(ticker: str) -> float:
    intraday = yf.download(
        ticker,
        period=INTRADAY_PERIOD,
        interval=INTRADAY_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
        prepost=False,
    )

    intraday = flatten_if_needed(intraday)

    if "Close" not in intraday.columns:
        raise ValueError(f"{ticker} intraday 缺少 Close 欄位，實際欄位={list(intraday.columns)}")

    intraday = intraday.dropna(subset=["Close"])
    if intraday.empty:
        return np.nan

    idx = intraday.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(ET_TZ)
    else:
        idx = idx.tz_convert(ET_TZ)

    intraday = intraday.copy()
    intraday.index = idx
    intraday = intraday.sort_index()

    # 以美東時間的「今天」為準，不是用資料裡最新那天
    today_et = datetime.now(ZoneInfo(ET_TZ)).date()

    # 只取今天資料
    day_df = intraday[intraday.index.date == today_et]
    if day_df.empty:
        return np.nan

    target = pd.Timestamp(f"{today_et} {TARGET_TIME}", tz=ET_TZ)
    eligible = day_df[day_df.index <= target]
    if eligible.empty:
        return np.nan

    return float(eligible["Close"].iloc[-1])

def fetch_security_metrics(ticker: str) -> dict:
    close_y, atr = get_prev_close_and_atr(ticker)
    p_1000 = get_intraday_price_10am(ticker)

    # 🔴 關鍵修正（防呆）
    close_y = pd.to_numeric(close_y, errors="coerce")
    p_1000 = pd.to_numeric(p_1000, errors="coerce")

    gap = np.nan if pd.isna(p_1000) or pd.isna(close_y) or close_y == 0 else (p_1000 - close_y) / close_y

    return {
        "ticker": ticker,
        "close_y": close_y,
        "ATR": atr,
        "P_10:00": p_1000,
        "gap": gap,
    }


def fetch_market_detail() -> tuple[float, pd.DataFrame]:
    rows = []

    for t in MARKET_TICKERS:
        try:
            row = fetch_security_metrics(t)
            rows.append(row)
        except Exception as e:
            print(f"[ERROR] market ticker {t}: {e}")

    mdf = pd.DataFrame(rows)

    if mdf.empty or "gap" not in mdf.columns:
        return np.nan, mdf

    mdf["gap"] = pd.to_numeric(mdf["gap"], errors="coerce")

    if mdf["gap"].isna().all():
        return np.nan, mdf

    market_gap = float(np.clip(mdf["gap"].mean(), -0.03, 0.03))
    return market_gap, mdf


def build_comment(row: pd.Series) -> str:
    if row["order_type"] == "MARKET":
        return (
            f"{row['ticker']} | {row['day']} | Day3 保證成交，使用 MARKET order。"
            f" 前日收盤={row['close_y']:.2f}，10:00價格={row['P_10:00']:.2f}"
        )

    return (
        f"{row['ticker']} | {row['day']} | Market={row['market_state']} / Stock={row['stock_state']} | "
        f"k={row['k']:.2f} | ATR={row['ATR']:.2f} | ATR_pct={row['ATR_pct']:.2%} | "
        f"limit={row['close_y']:.2f} × (1 - {row['k']:.2f} × {row['ATR_pct']:.4f}) = {row['limit_price']:.2f}"
    )


def create_today_log(orders_df: pd.DataFrame) -> pd.DataFrame:
    log = pd.DataFrame()
    log["date"] = [pd.Timestamp.now(tz=ET_TZ).date().isoformat()] * len(orders_df)
    log["ticker"] = orders_df["ticker"]
    log["day"] = orders_df["day"]
    log["market_state"] = orders_df["market_state"]
    log["stock_state"] = orders_df["stock_state"]
    log["k"] = orders_df["k"]
    log["limit_price"] = orders_df["limit_price"]
    log["filled_price"] = np.nan
    log["shares"] = orders_df["shares"]
    log["close_y"] = orders_df["close_y"]
    log["ATR"] = orders_df["ATR"]
    log["entry_distance"] = np.nan
    return log


# ===== UI =====
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Excel allocation → shares → market/stock state → ATR matrix → order table → execution log")

with st.sidebar:
    st.header("上傳檔案")
    allocation_file = st.file_uploader(
        "Allocation 檔（CSV / Excel，需有 ticker / target_dollar）",
        type=["csv", "xlsx", "xls"],
    )
    execution_log_file = st.file_uploader(
        "Execution Log（可選）",
        type=["csv", "xlsx", "xls"],
    )
    run_btn = st.button("開始執行", type="primary")

st.markdown("### 策略摘要")
c1, c2, c3 = st.columns(3)
with c1:
    st.info("shares = round(target_dollar / yesterday_close, 2)")
with c2:
    st.info("Market: QQQ / SPY 10:00 gap 平均，Bull > 0.8%，Bear < -0.8%")
with c3:
    st.info("Stock: 10:00 gap，Bull > 1.0%，Bear < -1.0%；Day3 用 MARKET order")

st.markdown("#### Day1 ATR Matrix")
st.dataframe(MATRIX_DAY1, use_container_width=True)

if run_btn:
    progress = st.progress(0, text="準備開始…")
    checkpoint = st.empty()

    try:
        # 1
        checkpoint.info("Step 1/8：讀取 allocation")
        if allocation_file is None:
            raise ValueError("請先上傳 allocation 檔案。")
        if allocation_file.name.lower().endswith(".csv"):
            alloc_raw = pd.read_csv(allocation_file)
        else:
            alloc_raw = pd.read_excel(allocation_file)
        allocation = validate_allocation_df(alloc_raw)
        progress.progress(10, text="Allocation 完成")

        # 2
        checkpoint.info("Step 2/8：讀取 execution log（若有）")
        execution_log = None
        if execution_log_file is not None:
            if execution_log_file.name.lower().endswith(".csv"):
                execution_log = pd.read_csv(execution_log_file)
            else:
                execution_log = pd.read_excel(execution_log_file)
            execution_log = validate_execution_log_df(execution_log)
        progress.progress(20, text="Execution log 完成")

        # 3
        checkpoint.info("Step 3/8：抓取市場資料（QQQ / SPY）")
        market_gap, market_detail = fetch_market_detail()
        market_state_scalar = classify_market_state(market_gap)
        progress.progress(35, text="市場資料完成")

        if market_detail["P_10:00"].isna().all():
            st.warning("今天沒有有效的 10:00 盤中資料，可能尚未開盤、未到 10:00，或今天休市。")
            st.stop()

        # 4
        checkpoint.info("Step 4/8：抓取個股資料")
        rows = []
        detail = st.progress(0, text="抓取個股資料中…")
        total = len(allocation)
        for i, ticker in enumerate(allocation["ticker"], start=1):
            try:
                rows.append(fetch_security_metrics(ticker))
            except Exception as e:
                rows.append({
                    "ticker": ticker,
                    "close_y": np.nan,
                    "ATR": np.nan,
                    "P_10:00": np.nan,
                    "gap": np.nan,
                    "error": str(e),
                })
            detail.progress(int(i / total * 100), text=f"{ticker} ({i}/{total})")
        sec_df = pd.DataFrame(rows)
        progress.progress(60, text="個股資料完成")

        # 5
        checkpoint.info("Step 5/8：計算 shares 與 day state")
        df = allocation.merge(sec_df, on="ticker", how="left")
        df["shares"] = np.round(df["target_dollar"] / df["close_y"], 2)
        df["day"] = infer_day_state_from_log(df["ticker"], execution_log)
        progress.progress(72, text="shares / day state 完成")

        # 6
        checkpoint.info("Step 6/8：判斷 regime 並計算 k")
        df["market_gap"] = market_gap
        df["market_state"] = market_state_scalar
        df["stock_gap"] = df["gap"]
        df["stock_state"] = classify_stock_state(df["stock_gap"])
        df["k1"] = get_k1(df["market_state"], df["stock_state"])
        df["k"] = np.where(df["day"] == "Day2", df["k1"] * 0.6, df["k1"])
        df["k"] = np.where(df["day"] == "Day3", np.nan, df["k"])
        progress.progress(84, text="regime / matrix 完成")

        # 7
        checkpoint.info("Step 7/8：輸出訂單表")
        df["ATR_pct"] = df["ATR"] / df["close_y"]
        df["limit_price"] = df["close_y"] * (1 - df["k"] * df["ATR_pct"])
        df["limit_price"] = np.where(df["day"] == "Day3", np.nan, df["limit_price"])
        df["limit_price"] = np.round(df["limit_price"], 2)
        df["order_type"] = np.where(df["day"] == "Day3", "MARKET", "LIMIT")
        df["comment"] = df.apply(build_comment, axis=1)
        progress.progress(94, text="訂單表完成")

        # 8
        checkpoint.info("Step 8/8：產生 execution log 草稿")
        order_cols = [
            "ticker", "target_dollar", "shares", "day", "order_type", "limit_price",
            "market_gap", "market_state", "stock_gap", "stock_state",
            "k", "close_y", "ATR", "ATR_pct", "P_10:00", "comment"
        ]
        orders = df[order_cols].copy()
        today_log = create_today_log(df)
        progress.progress(100, text="完成")
        checkpoint.success("執行完成")

        st.markdown("## 市場資料")
        st.dataframe(market_detail[["ticker", "close_y", "P_10:00", "gap"]], use_container_width=True)
        st.metric("Market gap", f"{market_gap:.2%}")
        st.metric("Market state", market_state_scalar)

        st.markdown("## 訂單表")
        st.dataframe(orders, use_container_width=True)

        st.markdown("## 今日 execution log 草稿")
        st.dataframe(today_log, use_container_width=True)

        st.download_button(
            "下載 order_table.csv",
            orders.to_csv(index=False).encode("utf-8-sig"),
            file_name="order_table.csv",
            mime="text/csv",
        )
        st.download_button(
            "下載 execution_log_today.csv",
            today_log.to_csv(index=False).encode("utf-8-sig"),
            file_name="execution_log_today.csv",
            mime="text/csv",
        )

    except Exception as e:
        progress.progress(100, text="失敗")
        checkpoint.error(f"系統錯誤：{e}")
        st.exception(e)
