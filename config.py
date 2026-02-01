# config.py
# 這份檔案是所有可調整參數的唯一入口。部署到 Zeabur 後，你只需要改這裡即可。

# =========================
# 系統設定
# =========================
TIMEZONE = "Asia/Taipei"   # 排程依據的時區
MAX_OUTPUT = 20            # 最終清單最大數量

# 是否只使用「已收盤」K 棒（建議 True；避免最後一根未收盤造成抖動）
USE_CLOSED_CANDLES_ONLY = True

# 排程時間（Asia/Taipei）
# 建議用「整點後 1~3 分鐘」讓 Binance K 棒確定收盤、API 資料更穩定
SCHEDULE_TIMES = ["00:02", "04:02", "08:02", "12:02", "16:02", "20:02"]

# =========================
# 回溯查詢（Telegram /yymmdd hh [ $SYMBOL ]）
# =========================
# 說明：使用者輸入的 hh 代表「該小時收盤（例如 16 表示 15:00~16:00 這根 1H K 的收盤）」。
# 系統會以 BTCUSDT 的可用資料作為對齊基準；若該時間點資料缺漏，會回報實際使用到的最近收盤時間。
RETRO_TIME_TOLERANCE_MINUTES = 10  # 允許「實際可用收盤時間」與使用者指定時間的落差（分鐘）

# =========================
# Binance / 抓取設定
# =========================
OHLCV_TIMEFRAME = "1h"
OHLCV_LIMIT = 1500             # 每個 symbol 抓取幾根 K（1h 下約 62.5 天）
API_CONCURRENCY = 6            # 同時抓取幾個 symbol（越大越容易觸發限流/崩潰）
API_REQUEST_SLEEP = 0.03       # 每次 API 請求前 sleep（秒），降低瞬間爆量

# 重試 + 指數退避
API_MAX_RETRIES = 5
API_BACKOFF_BASE = 0.8
API_BACKOFF_CAP = 10.0
API_RETRY_JITTER = 0.1

# Universe TopN（用 24h quoteVolume 預篩）
# 設為 None 代表不預篩（會大幅增加 API 量與風險）
UNIVERSE_TOP_N_BY_QUOTE_VOL = 300

# =========================
# Universe 排除名單（剔除特定指數與非加密貨幣）
# =========================
EXCLUDE_SYMBOLS = [
    'BTCUSDT', 'USDCUSDT', 'BTCDOMUSDT', 'ALLUSDT',
    'XAUUSDT', 'XAGUSDT', 'TSLAUSDT', 'EURUSDT', 'GBPUSDT'
]
MIN_HISTORY_DAYS = 30     # 至少要有 30 天資料（1h 下 720 根）

# =========================
# 噪音門檻 (Noise Gate)
# =========================
ATR_PERCENTILE_THRESHOLD = 0.95  # 排除 ATR% > P95（極端波動）
VOL_MEDIAN_DAYS = 7
VOL_PERCENTILE_THRESHOLD = 0.10  # 排除成交量 < P10（流動性差）

# =========================
# RS 模組 (BTC Benchmark)
# =========================
RS_WINDOW_SHORT = 72    # 3 Days (hours)
RS_WINDOW_MID = 168     # 7 Days
RS_WINDOW_LONG = 336    # 14 Days

# ✅ 避免「最後一根尖刺」：取最後 N 根的 median 作為該區間代表值
RS_SMOOTH_WINDOW = 12   # 12 小時

RS_WEIGHTS = {
    '14D': 0.40,
    '7D': 0.40,
    '3D': 0.20
}
RS_HARD_THRESHOLD = 0.75  # RS_Rank 必須 > P75（除了 Turning）

# =========================
# Setup 模組
# =========================
# VCP
VCP_IMPULSE_WINDOW = 72  # 3D
VCP_IMPULSE_RANK = 0.80  # P80
VCP_LOOKBACK_MIN = 24
VCP_LOOKBACK_MAX = 168
VCP_LOOKBACK_STEP = 12
VCP_HISTORY_BARS = 24 * 60  # 用 60 天資料做自身歷史分位
VCP_WIDTH_HISTORY_RANK = 0.25  # 寬度需在自身歷史 P25 以下
VCP_POS_THRESHOLD = 0.66      # 收盤價在區間 > 2/3

# PowerPlay
PP_HIGH_WINDOW = 240          # 高點基準窗口（1h）
PP_LOOKBACK_BREAKOUT = 48
PP_FLAG_LOOKBACK = 72
PP_HISTORY_BARS = 24 * 60
PP_TR_HISTORY_RANK = 0.90     # Breakout TR > P90
PP_CLOSE_POS_RANK = 0.75      # Breakout K 收盤在上方
PP_FLAG_WIDTH_RANK = 0.25     # 旗形寬度 < P25

# =========================
# 評分權重
# =========================
SCORE_WEIGHTS = {
    'RS': 0.6,
    'Setup': 0.2,
    'Trend': 0.2
}
