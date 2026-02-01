# Crypto Screener (Binance USDT-M Perpetual) + Telegram Bot

這是一套「非自動下單」的幣圈強勢幣篩選器：  
- 目標：在 Binance USDT-M 永續合約中，找出「強勢 + 高度上漲潛力」的候選幣  
- 核心：RS（相對強勢） + Trend Gate + Setup（VCP / PowerPlay）  
- 部署：可在 Zeabur 長期跑排程，並推送結果到 Telegram  

---

## 1) 功能概覽

### A) 排程掃描（推播）
- 依照 `config.SCHEDULE_TIMES` 固定時間掃描市場
- 推送 Top N 候選清單到 Telegram（若設定 `TELEGRAM_CHAT_ID`）

### B) 即時掃描（/now）
- 手動在 Telegram 輸入 `/now` 立即掃描一次
- 結果回傳到該聊天室

### C) 回溯查詢（/yymmdd hh [ $SYMBOL ]）
- 不儲存過去資料，直接用 Binance API 回抓指定時間點以前的 K 棒，回推判斷
- 支援：
  - `/yymmdd hh`：回到該時間點收盤時的「全市場掃描結果」
  - `/yymmdd hh $SYMBOL`：回到該時間點收盤時，判斷單一幣是否入選；若未入選會逐條列出條件 pass/fail

---

## 2) 策略結構（高層邏輯）

### 2.1 Universe / 基礎過濾
- 僅篩選 Binance USDT-M 永續合約（PERPETUAL）
- 排除 `config.EXCLUDE_SYMBOLS`（指數、非幣、干擾標的）
- 至少需有 `config.MIN_HISTORY_DAYS` 天的 1H 資料（用來穩定計算 RS / trend / setup）
- Universe TopN 預篩：用「當前」24h quoteVolume 取前 N（避免 API 爆量）

> 注意：回溯掃描因 Binance 沒有「歷史 24h quoteVolume」，所以 TopN 預篩仍使用當前 quoteVolume。

### 2.2 Noise Gate（噪音門檻）
- 排除極端波動：ATR% > Pxx
- 排除流動性差：rolling median quoteVolume < Pxx

### 2.3 RS 模組（以 BTC 為 benchmark）
- 計算 3D/7D/14D 的相對報酬（log return 差）
- 用「最後一段時間的一致性（median smoothing）」避免最後一根尖刺扭曲
- 另外加入 RS line 與 RS MA20 + slope 做 trend confirm

### 2.4 Trend Gate（1H + 4H）
- 1H：close > EMA50 > EMA200
- 4H：close > EMA20 > EMA45 且 EMA50 slope 正向
- 4H 僅使用已收盤 bar（resample 後 drop last）

### 2.5 Setup 模組（VCP / PowerPlay）
- VCP：Impulse 確認 + 寬度收斂 + 位置高位 + 不跌破長期均線
- PowerPlay：近端 breakout + breakout TR 分位 + close position + 旗形寬度收斂 + close > EMA20

### 2.6 Bucket / Scoring
- Bucket：Leader / PowerPlay / Turning
- Score：RS + Setup quality + Trend score（percentile）

---

## 3) Telegram 指令

## Telegram（必填）
- `TELEGRAM_TOKEN`：Bot token（必填）
- `TELEGRAM_CHAT_ID`：排程推播的 chat id（建議設定）
  - 不設定也可以用 `/now` 手動跑；排程則會跳過推播

> 取得 Chat ID：對 Bot 發 `/start` 會回傳你的 chat id。

### 回溯查詢（不儲存資料，直接用 Binance API 抓「指定時間點以前」的 1H K 棒）
- **全市場回溯掃描**：`/yymmdd hh`
  例：`/260103 16` 代表「2026-01-03 16:00（GMT+8）收盤」當下的掃描結果。
- **單幣回溯診斷**：`/yymmdd hh $SYMBOL`
  例：`/260103 16 $CHZ` 會回報 **CHZUSDT** 當下是否入選；若未入選，會逐條列出每個篩選條件是否通過。

> 時間解讀：`hh` 代表「該小時收盤」，例如 16 表示 15:00~16:00 這根 1H K 的收盤。

**重要限制（避免誤會）**
- Binance 無法直接提供「歷史 24h quoteVolume/ticker」：因此回溯掃描的 Universe 預篩（Top N by quoteVolume）仍使用「當前」資料。
  - 若你要最大化歷史一致性：可把 `UNIVERSE_TOP_N_BY_QUOTE_VOL=None`，但會大幅增加 API 量與延遲（也更容易觸發限流）。
- 若指定時間點資料缺漏（例如幣種尚未上市、或 BTC 該時段資料缺），系統會回報實際使用到的最近可用收盤時間（避免 silent except）。

### Binance（選填，但建議）
- `BINANCE_API_KEY`
- `BINANCE_SECRET_KEY`

> 沒有 key 也能跑 public endpoint，但更容易觸發速率限制。

---

## 4) 可調參數（只改 config.py）

重點參數：
- `SCHEDULE_TIMES`：排程推播時間（建議整點後 1~3 分鐘）
- `API_CONCURRENCY`：同時抓取 K 棒數（越大越容易觸發限流/崩潰）
- `API_REQUEST_SLEEP`：每次請求前 sleep（避免瞬間爆量）
- `API_MAX_RETRIES / API_BACKOFF_*`：重試與退避策略
- `UNIVERSE_TOP_N_BY_QUOTE_VOL`：TopN 預篩（最有效的「避免崩潰/封鎖」手段）
- `OHLCV_LIMIT`：每個 symbol 抓幾根 1H K（預設 1500 約 62.5 天）

回溯查詢參數：
- `RETRO_TIME_TOLERANCE_MINUTES`：若指定時間點資料缺漏，允許用最近可用收盤時間替代的最大落差（分鐘）

---

## 5) Zeabur 部署（建議）

環境變數：
- `TELEGRAM_TOKEN`（必填）
- `TELEGRAM_CHAT_ID`（建議）
- `BINANCE_API_KEY`（選填）
- `BINANCE_SECRET_KEY`（選填）

部署建議：
- `API_CONCURRENCY=4~6`
- `UNIVERSE_TOP_N_BY_QUOTE_VOL=250~350`
- 排程不要太密（建議 4H 或更慢）

---

## 6) 部署注意事項（對應你的 1~6 點）

### 1) 避免 Zeabur 崩潰
- 降低 `API_CONCURRENCY`（例如 4~6）
- 開啟 `UNIVERSE_TOP_N_BY_QUOTE_VOL`（例如 250~350）
- `OHLCV_LIMIT` 不要大到超過需要（預設 1500 已足夠）

### 2) 避免 Binance API 被封鎖
- `enableRateLimit=True` + `API_REQUEST_SLEEP` + 指數退避重試
- Universe 預篩 top N（最有效）
- 不要把排程頻率設太密（建議 4H 一次或更慢）

### 3) 避免 silent except
- 所有例外都會 log；Telegram send 若失敗會 log warning（不會吞掉）

### 4) 留意排程時間
- 建議用整點後 1~3 分鐘（如 `00:02`），並啟用 `USE_CLOSED_CANDLES_ONLY=True`
- 同時 4H resample 也只使用已收盤 bar，避免 gate 抖動

### 5) 確保使用 config.py 調整
- 所有新增參數都集中在 `config.py`

### 6) requirements.txt 簡約 & 最新版本
- 只保留必要套件，使用 `>=` 允許最新版本解決相依衝突

---

## 7) 常見問題

### Q: 為什麼回溯掃描結果和當時的真實市場可能有些差異？
- Universe 預篩使用的是「當前」quoteVolume（Binance 不提供歷史 24h tickers）
- 部分幣種可能當時未上市或 K 棒缺漏；系統會提示並改用最近可用收盤時間對齊

### Q: /yymmdd hh $SYMBOL 為什麼也會比較慢？
- 為了能產生「逐條條件診斷」且 RS/percentile 必須在一個 universe 上計算，仍需要跑完整 pipeline（但你可降低 TopN / concurrency）
