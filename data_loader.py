import os
import asyncio
import logging
import pandas as pd
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
import pytz
import config

logger = logging.getLogger(__name__)

def normalize_symbol(symbol: str) -> str:
    """把 ccxt symbol 轉成清理後格式（例如 ETH/USDT:USDT -> ETHUSDT）。"""
    if not symbol:
        return symbol
    s = symbol.replace('/USDT:USDT', 'USDT').replace(':USDT', 'USDT').replace('/', '')
    return s

class DataLoader:
    def __init__(self):
        # 支援多種常見環境變數命名
        api_key = (
            os.getenv('BINANCE_API_KEY')
            or os.getenv('BINANCE_KEY')
            or os.getenv('BINANCE_APIKEY')
        )
        secret_key = (
            os.getenv('BINANCE_SECRET_KEY')
            or os.getenv('BINANCE_API_SECRET')
            or os.getenv('BINANCE_SECRET')
        )

        exchange_config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',     # USDT-M 永續
            }
        }

        if api_key and secret_key:
            exchange_config['apiKey'] = api_key
            exchange_config['secret'] = secret_key

        self.exchange = ccxt.binance(exchange_config)

    async def _apply_ticker_prefilter(self, symbols):
        """用當前 24h quoteVolume 先取 Top N，降低 API 壓力。"""
        top_n = getattr(config, 'UNIVERSE_TOP_N_BY_QUOTE_VOL', None)
        if not top_n:
            return symbols

        try:
            tickers = await self.exchange.fetch_tickers(symbols)
            vols = []
            for sym, t in tickers.items():
                try:
                    qv = float(t.get('quoteVolume') or 0.0)
                    vols.append((sym, qv))
                except Exception:
                    continue

            vols.sort(key=lambda x: x[1], reverse=True)
            selected = [sym for sym, _ in vols[:int(top_n)]]
            logger.info(f"Universe prefilter: top {len(selected)} / {len(symbols)} by quoteVolume")
            return selected
        except Exception as e:
            logger.warning(f"Prefilter failed, fallback to full universe: {e}", exc_info=True)
            return symbols

    async def fetch_markets(self, return_map: bool = False):
        """取得 USDT-M 永續合約 symbol 列表（ccxt格式），並排除不要的標的。"""
        try:
            markets = await self.exchange.load_markets()

            symbols = []
            clean_to_symbol = {}

            for sym, m in markets.items():
                # 只要 swap 永續
                if not m.get('swap'):
                    continue
                if m.get('linear') is not True:
                    continue
                if m.get('settle') != 'USDT':
                    continue

                # contractType == PERPETUAL（部分 market 欄位在 info）
                info = m.get('info') or {}
                if info.get('contractType') and info.get('contractType') != 'PERPETUAL':
                    continue

                clean = normalize_symbol(sym)
                if clean in config.EXCLUDE_SYMBOLS:
                    continue

                symbols.append(sym)
                clean_to_symbol[clean] = sym

            return (symbols, clean_to_symbol) if return_map else symbols

        except Exception as e:
            logger.error(f"fetch_markets failed: {e}", exc_info=True)
            return ([], {}) if return_map else []

    async def fetch_ohlcv(self, symbol, semaphore, timeframe=None, limit=None):
        """抓取 OHLCV（加上重試/退避），回傳 DatetimeIndex 的 DataFrame。"""
        timeframe = timeframe or config.OHLCV_TIMEFRAME
        limit = limit or config.OHLCV_LIMIT

        async with semaphore:
            for attempt in range(1, config.API_MAX_RETRIES + 1):
                try:
                    await asyncio.sleep(config.API_REQUEST_SLEEP)

                    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    if not ohlcv:
                        return None

                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp').sort_index()

                    if config.USE_CLOSED_CANDLES_ONLY and len(df) > 0:
                        # 丟掉最後一根：避免未收盤抖動
                        df = df.iloc[:-1]

                    # 基本長度檢查（1h 下 30 天 = 720 根）
                    if len(df) < config.MIN_HISTORY_DAYS * 24:
                        return None

                    return df

                except Exception as e:
                    sleep_s = min(config.API_BACKOFF_CAP, config.API_BACKOFF_BASE * (2 ** (attempt - 1)))
                    sleep_s += (config.API_RETRY_JITTER * attempt)
                    logger.warning(
                        f"Fetch {symbol} failed (attempt {attempt}/{config.API_MAX_RETRIES}): {e} | backoff={sleep_s:.2f}s"
                    )
                    if attempt >= config.API_MAX_RETRIES:
                        logger.error(f"Giving up {symbol} after {attempt} attempts", exc_info=True)
                        return None
                    await asyncio.sleep(sleep_s)

            return None

    async def fetch_ohlcv_asof(self, symbol, semaphore, end_open_utc: datetime, timeframe=None, limit=None):
        """抓取「指定時間點以前」的 OHLCV（加上重試/退避）。
        - end_open_utc：欲使用的最後一根 1H K 的「開盤時間」(UTC, tz-aware)
        回傳：df（DatetimeIndex, naive UTC），資料已切到 <= end_open_utc。
        """
        timeframe = timeframe or config.OHLCV_TIMEFRAME
        limit = limit or config.OHLCV_LIMIT

        if end_open_utc.tzinfo is None:
            raise ValueError("end_open_utc must be timezone-aware UTC datetime")

        # 目前只支援 1h（因為策略計算以 1h 為底）
        if timeframe != "1h":
            raise ValueError("fetch_ohlcv_asof currently supports timeframe='1h' only")

        hour_ms = 60 * 60 * 1000
        end_open_ms = int(end_open_utc.timestamp() * 1000)
        since_ms = end_open_ms - (limit * hour_ms)

        async with semaphore:
            for attempt in range(1, config.API_MAX_RETRIES + 1):
                try:
                    await asyncio.sleep(config.API_REQUEST_SLEEP)

                    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
                    if not ohlcv:
                        return None

                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp').sort_index()

                    # 切到指定 end_open（注意：K 棒 timestamp 為 open time）
                    end_open_naive = pd.to_datetime(end_open_ms, unit='ms')  # naive UTC
                    df = df.loc[df.index <= end_open_naive]

                    if df.empty:
                        return None

                    # 基本長度檢查（1h 下 30 天 = 720 根）
                    if len(df) < config.MIN_HISTORY_DAYS * 24:
                        return None

                    return df

                except Exception as e:
                    sleep_s = min(config.API_BACKOFF_CAP, config.API_BACKOFF_BASE * (2 ** (attempt - 1)))
                    sleep_s += (config.API_RETRY_JITTER * attempt)
                    logger.warning(
                        f"Fetch(asof) {symbol} failed (attempt {attempt}/{config.API_MAX_RETRIES}): {e} | backoff={sleep_s:.2f}s"
                    )
                    if attempt >= config.API_MAX_RETRIES:
                        logger.error(f"Giving up(asof) {symbol} after {attempt} attempts", exc_info=True)
                        return None
                    await asyncio.sleep(sleep_s)

            return None

    async def get_all_data_asof(self, asof_close_local: datetime, force_clean_symbols=None):
        """回溯查詢：抓取「指定收盤時間」(local tz) 當下的全市場資料（不儲存歷史）。
        - asof_close_local：使用者輸入的 hh 視為該小時收盤時間（例如 16 表示 15:00~16:00 的 1H K 收盤）
        - force_clean_symbols：可選，強制包含某些 clean symbol（例如 ['CHZUSDT']），避免被 universe prefilter 排除
        回傳：(data_map, btc_df, meta)
        """
        tz = pytz.timezone(config.TIMEZONE)

        if asof_close_local.tzinfo is None:
            asof_close_local = tz.localize(asof_close_local)

        asof_close_utc = asof_close_local.astimezone(pytz.UTC)
        end_open_utc = asof_close_utc - timedelta(hours=1)

        try:
            # 取得市場清單 + clean 對照
            symbols, clean_to_symbol = await self.fetch_markets(return_map=True)
            if not symbols:
                raise RuntimeError("No markets returned from Binance.")

            # 先套用 ticker prefilter（注意：這是「當前」24h quoteVolume，無法回溯歷史）
            symbols = await self._apply_ticker_prefilter(symbols)

            # 確保 BTCUSDT 有進來（作 benchmark）
            if 'BTCUSDT' not in clean_to_symbol:
                clean_to_symbol['BTCUSDT'] = 'BTC/USDT:USDT'
            if clean_to_symbol['BTCUSDT'] not in symbols:
                symbols.append(clean_to_symbol['BTCUSDT'])

            # 強制包含指定 symbol（若存在於 markets）
            force_clean_symbols = force_clean_symbols or []
            for clean in force_clean_symbols:
                if clean in clean_to_symbol:
                    sym = clean_to_symbol[clean]
                    if sym not in symbols:
                        symbols.append(sym)

            logger.info(
                f"[ASOF] Fetching OHLCV for {len(symbols)} symbols up to close={asof_close_local.strftime('%Y-%m-%d %H:%M %Z')}"
            )

            sem = asyncio.Semaphore(config.API_CONCURRENCY)
            tasks = [self.fetch_ohlcv_asof(symbol, sem, end_open_utc) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            data_map = {}
            btc_data = None

            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.warning(f"[ASOF] Task failed for {symbol}: {result}")
                    continue
                if isinstance(result, pd.DataFrame) and not result.empty:
                    clean = normalize_symbol(symbol)
                    data_map[clean] = result
                    if clean == 'BTCUSDT':
                        btc_data = result

            if btc_data is None:
                raise RuntimeError("Critical: BTCUSDT data not found for asof query.")

            # 以 BTC 的最後一根 open time 作為實際使用到的 end_open（避免缺資料時亂對齊）
            used_end_open_naive = btc_data.index[-1]  # naive UTC
            used_end_open_utc = pytz.UTC.localize(used_end_open_naive)
            used_close_local = (used_end_open_utc + timedelta(hours=1)).astimezone(tz)

            # 對齊其他 symbol：切到 <= used_end_open
            for k, df in list(data_map.items()):
                data_map[k] = df.loc[df.index <= used_end_open_naive]

            diff_min = abs((used_close_local - asof_close_local).total_seconds()) / 60.0
            meta = {
                "requested_close_local": asof_close_local,
                "used_close_local": used_close_local,
                "diff_minutes": float(diff_min),
            }

            return data_map, btc_data, meta

        finally:
            try:
                await self.exchange.close()
            except Exception:
                logger.debug("Exchange close failed (ignored).", exc_info=True)

    async def get_all_data(self):
        """抓取全市場資料，回傳 data_map（clean_symbol -> df）與 btc_df。"""
        try:
            symbols = await self.fetch_markets()
            if not symbols:
                raise RuntimeError("No markets returned from Binance.")

            symbols = await self._apply_ticker_prefilter(symbols)

            # 確保 BTCUSDT 會被抓到（即便在排除名單也要作 benchmark）
            if not any('BTC/USDT' in s for s in symbols):
                logger.info("Adding BTC/USDT:USDT manually for benchmark")
                symbols.append('BTC/USDT:USDT')

            sem = asyncio.Semaphore(config.API_CONCURRENCY)
            tasks = [self.fetch_ohlcv(symbol, sem) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            data_map = {}
            btc_data = None

            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.warning(f"Task failed for {symbol}: {result}")
                    continue
                if isinstance(result, pd.DataFrame) and not result.empty:
                    clean = normalize_symbol(symbol)
                    data_map[clean] = result
                    if clean == 'BTCUSDT':
                        btc_data = result

            if btc_data is None:
                raise RuntimeError("Critical: BTCUSDT data not found.")

            return data_map, btc_data

        finally:
            try:
                await self.exchange.close()
            except Exception:
                logger.debug("Exchange close failed (ignored).", exc_info=True)
