import pandas as pd
import numpy as np
import logging

import config
from indicators import calculate_ema, calculate_atr, get_slope, true_range

logger = logging.getLogger(__name__)


class CryptoScreener:
    def __init__(self, data_map, btc_df):
        self.data = data_map
        self.btc = btc_df  # DatetimeIndex (naive UTC)
        self.results = []

    def _safe_percentile_rank(self, hist: pd.Series, value: float) -> float:
        """回傳 value 在 hist 內的分位（0~1）。hist 需已去除 NaN。"""
        if hist is None:
            return np.nan
        h = hist.dropna()
        if len(h) == 0:
            return np.nan
        return float((h <= value).mean())

    def _dbg_add(self, dbg, stage: str, ok: bool, detail: str):
        if dbg is None:
            return
        dbg["checks"].append({
            "stage": stage,
            "ok": bool(ok),
            "detail": str(detail),
        })

    def run(self, diagnose_symbol: str | None = None, return_debug: bool = False):
        """
        執行篩選。
        - diagnose_symbol: 若指定（例如 'CHZUSDT'），會回傳該幣逐項條件檢查結果。
        - return_debug: True 則回傳 (results, debug)；否則只回傳 results。
        """
        dbg = None
        diag = diagnose_symbol.upper() if isinstance(diagnose_symbol, str) and diagnose_symbol else None
        if diag:
            dbg = {"symbol": diag, "checks": [], "selected": False, "bucket": None, "score": None}

        logger.info("Starting screening pipeline...")

        # 0) 準備全市場 metrics（Noise Gate 用）
        universe_metrics = []
        processed_data = {}

        # 基本：資料是否存在
        if dbg:
            if diag in self.data:
                df0 = self.data.get(diag)
                if isinstance(df0, pd.DataFrame) and not df0.empty:
                    self._dbg_add(dbg, "data_loaded", True, f"bars={len(df0)}, last_open_utc={df0.index[-1]}")
                else:
                    self._dbg_add(dbg, "data_loaded", False, "data exists but empty/invalid")
            else:
                self._dbg_add(dbg, "data_loaded", False, "symbol not present in fetched universe (may be delisted / not in USDT-M swap)")

        for symbol, df in self.data.items():
            if symbol == 'BTCUSDT':
                continue
            if df is None or len(df) < config.MIN_HISTORY_DAYS * 24:
                if dbg and symbol == diag:
                    self._dbg_add(dbg, "min_history", False, f"need>={config.MIN_HISTORY_DAYS*24} bars (1h), got={0 if df is None else len(df)}")
                continue

            df = df.copy()

            # 指標：ATR / 成交額
            df['atr'] = calculate_atr(df)
            df['atr_pct'] = df['atr'] / df['close']
            df['vol_value'] = df['volume'] * df['close']

            last_row = df.iloc[-1]
            universe_metrics.append({
                'symbol': symbol,
                'atr_pct': last_row.get('atr_pct', np.nan),
                'med_vol': df['vol_value'].rolling(24 * config.VOL_MEDIAN_DAYS).median().iloc[-1],
            })
            processed_data[symbol] = df

            if dbg and symbol == diag:
                self._dbg_add(dbg, "min_history", True, f"bars={len(df)}")

        u_df = pd.DataFrame(universe_metrics)
        if u_df.empty:
            logger.warning("Universe metrics empty. Nothing to screen.")
            if dbg:
                self._dbg_add(dbg, "universe", False, "no symbols passed basic history check")
            return ([], dbg) if return_debug else []

        # 1) Noise Gate
        atr_threshold = u_df['atr_pct'].quantile(config.ATR_PERCENTILE_THRESHOLD)
        vol_threshold = u_df['med_vol'].quantile(config.VOL_PERCENTILE_THRESHOLD)

        valid_symbols = u_df[
            (u_df['atr_pct'] <= atr_threshold) &
            (u_df['med_vol'] >= vol_threshold)
        ]['symbol'].tolist()

        logger.info(f"Noise Gate passed: {len(valid_symbols)} / {len(u_df)}")

        if dbg and diag in processed_data:
            atr_pct = float(u_df.loc[u_df['symbol'] == diag, 'atr_pct'].iloc[0]) if (u_df['symbol'] == diag).any() else np.nan
            med_vol = float(u_df.loc[u_df['symbol'] == diag, 'med_vol'].iloc[0]) if (u_df['symbol'] == diag).any() else np.nan
            ok_atr = (not np.isnan(atr_pct)) and (atr_pct <= float(atr_threshold))
            ok_vol = (not np.isnan(med_vol)) and (med_vol >= float(vol_threshold))
            self._dbg_add(dbg, "noise_gate_atr", ok_atr, f"atr_pct={atr_pct:.6f}, threshold(P{int(config.ATR_PERCENTILE_THRESHOLD*100)})={float(atr_threshold):.6f}")
            self._dbg_add(dbg, "noise_gate_liquidity", ok_vol, f"med_quote_vol(rolling)={med_vol:.2f}, threshold(P{int(config.VOL_PERCENTILE_THRESHOLD*100)})={float(vol_threshold):.2f}")
            self._dbg_add(dbg, "noise_gate_overall", (diag in valid_symbols), f"passed={diag in valid_symbols}")

        # 2) RS 計算（對齊 BTC）
        rs_scores = []
        for symbol in valid_symbols:
            df = processed_data[symbol]

            # 對齊 BTC（避免缺值）
            btc_aligned = self.btc.reindex(df.index).ffill()
            if btc_aligned is None or btc_aligned.empty:
                if dbg and symbol == diag:
                    self._dbg_add(dbg, "rs_align_btc", False, "btc alignment empty")
                continue

            df['log_close'] = np.log(df['close'])
            btc_log = np.log(btc_aligned['close'])

            score = 0.0
            per_period_vals = {}
            for period_name, period_len in [
                ('3D', config.RS_WINDOW_SHORT),
                ('7D', config.RS_WINDOW_MID),
                ('14D', config.RS_WINDOW_LONG),
            ]:
                coin_ret = df['log_close'].diff(period_len)
                btc_ret = btc_log.diff(period_len)
                rel_roc = coin_ret - btc_ret

                # ✅ 時間一致性：避免只取最後一根 bar 造成尖刺影響
                tail = rel_roc.tail(config.RS_SMOOTH_WINDOW).dropna()
                val = float(tail.median()) if len(tail) else 0.0
                per_period_vals[period_name] = val

                score += val * float(config.RS_WEIGHTS.get(period_name, 0.0))

            # RS Trend（用 RS line + EMA + slope）
            rs_line = df['close'] / btc_aligned['close']
            rs_ma20 = calculate_ema(rs_line, 20)
            rs_slope = get_slope(rs_ma20, window=48)

            rs_line_last = float(rs_line.iloc[-1])
            rs_ma_last = float(rs_ma20.iloc[-1]) if not np.isnan(rs_ma20.iloc[-1]) else rs_line_last
            slope_last = float(rs_slope.iloc[-1]) if not np.isnan(rs_slope.iloc[-1]) else 0.0

            rs_trend_ok = (rs_line_last > rs_ma_last) and (slope_last > 0)

            rs_scores.append({
                'symbol': symbol,
                'rs_score': score,
                'rs_trend_ok': rs_trend_ok,
                'rs_slope': slope_last,
            })

            if dbg and symbol == diag:
                self._dbg_add(dbg, "rs_score_components", True, f"3D={per_period_vals['3D']:.6f}, 7D={per_period_vals['7D']:.6f}, 14D={per_period_vals['14D']:.6f}")
                self._dbg_add(dbg, "rs_trend", rs_trend_ok, f"rs_line={rs_line_last:.6f}, rs_ma20={rs_ma_last:.6f}, rs_ma20_slope={slope_last:.6f}")

        if not rs_scores:
            logger.warning("RS scores empty after Noise Gate.")
            if dbg:
                self._dbg_add(dbg, "rs", False, "no symbols after noise gate / btc alignment issues")
            return ([], dbg) if return_debug else []

        rs_df = pd.DataFrame(rs_scores)
        rs_df['rs_rank'] = rs_df['rs_score'].rank(pct=True) * 100

        if dbg:
            if (rs_df['symbol'] == diag).any():
                rr = float(rs_df.loc[rs_df['symbol'] == diag, 'rs_rank'].iloc[0])
                self._dbg_add(dbg, "rs_rank", rr >= (config.RS_HARD_THRESHOLD * 100), f"rs_rank={rr:.1f}, hard_threshold={config.RS_HARD_THRESHOLD*100:.1f}")
            else:
                self._dbg_add(dbg, "rs_rank", False, "symbol not present after noise gate (did not pass, or data missing)")

        # 3) 主迴圈：Trend Gate + Setup
        candidates = []

        for _, row in rs_df.iterrows():
            symbol = row['symbol']
            df = processed_data.get(symbol)
            if df is None or df.empty:
                continue

            rs_rank = float(row['rs_rank'])
            close_now = float(df['close'].iloc[-1])

            # --- Trend Gate (1H) ---
            ema50_1h = calculate_ema(df['close'], 50)
            ema200_1h = calculate_ema(df['close'], 200)
            ema20_1h = calculate_ema(df['close'], 20)

            trend_gate_1h = (close_now > float(ema50_1h.iloc[-1])) and (float(ema50_1h.iloc[-1]) > float(ema200_1h.iloc[-1]))

            # --- Trend Gate (4H) ---
            df_4h = df.resample('4h').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()

            # ✅ 只用「已收盤」4H bar：永遠丟掉最後一根 resample bar（避免未收盤抖動）
            if len(df_4h) > 0:
                df_4h = df_4h.iloc[:-1]

            if len(df_4h) < 55:
                if dbg and symbol == diag:
                    self._dbg_add(dbg, "trend_gate_4h", False, f"need>=55 4h bars, got={len(df_4h)}")
                continue

            ema20_4h = calculate_ema(df_4h['close'], 20)
            ema45_4h = calculate_ema(df_4h['close'], 45)
            ema50_4h = calculate_ema(df_4h['close'], 50)

            ema50_slope_pos = float(ema50_4h.iloc[-1]) > float(ema50_4h.iloc[-3])

            trend_gate_4h = (
                float(df_4h['close'].iloc[-1]) > float(ema20_4h.iloc[-1]) and
                float(ema20_4h.iloc[-1]) > float(ema45_4h.iloc[-1]) and
                ema50_slope_pos
            )

            if dbg and symbol == diag:
                self._dbg_add(
                    dbg,
                    "trend_gate_1h",
                    trend_gate_1h,
                    f"close={close_now:.6f}, ema50={float(ema50_1h.iloc[-1]):.6f}, ema200={float(ema200_1h.iloc[-1]):.6f}"
                )
                self._dbg_add(
                    dbg,
                    "trend_gate_4h",
                    trend_gate_4h,
                    f"close4h={float(df_4h['close'].iloc[-1]):.6f}, ema20={float(ema20_4h.iloc[-1]):.6f}, ema45={float(ema45_4h.iloc[-1]):.6f}, ema50_slope_pos={ema50_slope_pos}"
                )

            if not (trend_gate_4h and trend_gate_1h):
                continue

            # Trend raw（用 EMA50 近似斜率強度，之後做 percentile score）
            trend_raw = (float(ema50_4h.iloc[-1]) / float(ema50_4h.iloc[-3])) - 1.0

            # --- Setup Module ---
            setup_vcp = False
            setup_pp = False
            vcp_best_width_rank = 1.0
            pp_best_flag_rank = 1.0

            # ===== VCP: Impulse 確認（使用 config 參數）
            ret_3d = np.log(df['close']).diff(config.VCP_IMPULSE_WINDOW)
            current_ret = float(ret_3d.iloc[-1]) if not np.isnan(ret_3d.iloc[-1]) else 0.0
            hist_rets = ret_3d.dropna().tail(config.VCP_HISTORY_BARS)
            impulse_rank = self._safe_percentile_rank(hist_rets, current_ret)
            impulse_ok = (not np.isnan(impulse_rank)) and (impulse_rank >= config.VCP_IMPULSE_RANK)

            if dbg and symbol == diag:
                self._dbg_add(dbg, "vcp_impulse", impulse_ok, f"ret_3d={current_ret:.6f}, impulse_rank={impulse_rank if not np.isnan(impulse_rank) else 'nan'}, threshold={config.VCP_IMPULSE_RANK}")

            # ===== VCP: 寬度收斂 + 位置
            vcp_detail = "not evaluated"
            if impulse_ok:
                lookbacks = list(range(config.VCP_LOOKBACK_MIN, config.VCP_LOOKBACK_MAX + 1, config.VCP_LOOKBACK_STEP))
                for L in lookbacks:
                    if len(df) < L + 50:
                        continue

                    window_high = df['high'].rolling(L).max()
                    window_low = df['low'].rolling(L).min()

                    hh = window_high.iloc[-1]
                    ll = window_low.iloc[-1]
                    if np.isnan(hh) or np.isnan(ll) or hh == ll:
                        continue

                    atr_last = float(df['atr'].iloc[-1])
                    if atr_last <= 0 or np.isnan(atr_last):
                        continue

                    width = float((hh - ll) / atr_last)
                    pos = float((close_now - ll) / (hh - ll))

                    past_widths = (df['high'].rolling(L).max() - df['low'].rolling(L).min()) / df['atr']
                    past_widths = past_widths.dropna().tail(config.VCP_HISTORY_BARS)

                    width_rank = self._safe_percentile_rank(past_widths, width)  # 0~1，越小代表越收斂
                    if np.isnan(width_rank):
                        continue

                    if (width_rank < config.VCP_WIDTH_HISTORY_RANK) and (pos > config.VCP_POS_THRESHOLD):
                        # 額外：避免跌破長期均線（原有邏輯保留）
                        if float(df['close'].iloc[-L:].min()) > float(ema200_1h.iloc[-1]):
                            setup_vcp = True
                            vcp_best_width_rank = min(vcp_best_width_rank, width_rank)
                            vcp_detail = f"L={L}, width_rank={width_rank:.3f}(<{config.VCP_WIDTH_HISTORY_RANK}), pos={pos:.3f}(>{config.VCP_POS_THRESHOLD})"
            if dbg and symbol == diag:
                self._dbg_add(dbg, "vcp_setup", setup_vcp, vcp_detail)

            # ===== PowerPlay: Breakout + TR rank + Close position rank + 旗形收斂
            pp_detail = "not evaluated"
            if len(df) >= config.PP_HIGH_WINDOW + 5:
                tr = true_range(df)
                hh_base = df['high'].rolling(config.PP_HIGH_WINDOW).max().shift(1)
                breakout_mask = df['close'] > hh_base

                lb = config.PP_LOOKBACK_BREAKOUT
                recent = breakout_mask.tail(lb)
                if recent.any():
                    # 用最近一次 breakout 作為主要 breakout bar
                    breakout_ts = recent[recent].index[-1]
                    b = df.loc[breakout_ts]

                    breakout_tr = float(tr.loc[breakout_ts]) if breakout_ts in tr.index else np.nan
                    tr_hist = tr.dropna().tail(config.PP_HISTORY_BARS)
                    tr_rank = self._safe_percentile_rank(tr_hist, breakout_tr)

                    # Close position in candle
                    rng = float(b['high'] - b['low'])
                    close_pos = float((b['close'] - b['low']) / rng) if rng > 0 else 0.0

                    tr_ok = (not np.isnan(tr_rank)) and (tr_rank >= config.PP_TR_HISTORY_RANK)
                    close_ok = close_pos >= config.PP_CLOSE_POS_RANK

                    if tr_ok and close_ok:
                        flag_L = config.PP_FLAG_LOOKBACK
                        if len(df) >= flag_L + 10:
                            atr_last = float(df['atr'].iloc[-1])
                            if atr_last > 0 and not np.isnan(atr_last):
                                flag_width = float((df['high'].iloc[-flag_L:].max() - df['low'].iloc[-flag_L:].min()) / atr_last)
                                past_flag_widths = (df['high'].rolling(flag_L).max() - df['low'].rolling(flag_L).min()) / df['atr']
                                past_flag_widths = past_flag_widths.dropna().tail(config.PP_HISTORY_BARS)

                                flag_rank = self._safe_percentile_rank(past_flag_widths, flag_width)  # 越小越收斂
                                if not np.isnan(flag_rank):
                                    if (flag_rank < config.PP_FLAG_WIDTH_RANK) and (close_now > float(ema20_1h.iloc[-1])):
                                        setup_pp = True
                                        pp_best_flag_rank = min(pp_best_flag_rank, flag_rank)
                                        pp_detail = f"breakout_ts={breakout_ts}, tr_rank={tr_rank:.3f}(>={config.PP_TR_HISTORY_RANK}), close_pos={close_pos:.3f}(>={config.PP_CLOSE_POS_RANK}), flag_rank={flag_rank:.3f}(<{config.PP_FLAG_WIDTH_RANK})"
                                    else:
                                        pp_detail = f"flag_rank={flag_rank:.3f} (need < {config.PP_FLAG_WIDTH_RANK}) or close<=ema20"
                                else:
                                    pp_detail = "flag_rank nan"
                        else:
                            pp_detail = f"need flag window bars, got={len(df)}"
                    else:
                        pp_detail = f"breakout_ts={breakout_ts}, tr_rank={tr_rank if not np.isnan(tr_rank) else 'nan'}, close_pos={close_pos:.3f} (need tr_rank>={config.PP_TR_HISTORY_RANK} AND close_pos>={config.PP_CLOSE_POS_RANK})"
                else:
                    pp_detail = f"no breakout in last {lb} bars"

            if dbg and symbol == diag:
                self._dbg_add(dbg, "pp_setup", setup_pp, pp_detail)

            if not (setup_vcp or setup_pp):
                if dbg and symbol == diag:
                    self._dbg_add(dbg, "setup_overall", False, "neither VCP nor PowerPlay matched")
                continue

            if dbg and symbol == diag:
                self._dbg_add(dbg, "setup_overall", True, f"matched={'PP' if setup_pp else 'VCP'}")

            # Bucket（原邏輯保留）
            bucket = "None"
            if rs_rank > 85 and bool(row['rs_trend_ok']) and (setup_vcp or setup_pp):
                bucket = "Leader"
            elif rs_rank > 80 and setup_pp:
                bucket = "PowerPlay"
            elif rs_rank > 60 and bool(row['rs_trend_ok']):
                bucket = "Turning"

            if bucket == "None":
                if rs_rank < config.RS_HARD_THRESHOLD * 100:
                    if dbg and symbol == diag:
                        self._dbg_add(dbg, "bucket", False, f"bucket=None and rs_rank={rs_rank:.1f} < hard_threshold={config.RS_HARD_THRESHOLD*100:.1f}")
                    continue

            if dbg and symbol == diag:
                self._dbg_add(dbg, "bucket", True, f"bucket={bucket}, rs_rank={rs_rank:.1f}, rs_trend_ok={bool(row['rs_trend_ok'])}")

            # Setup Quality（修正：PowerPlay 不再因為 best_width_rank=1.0 而被低估）
            vcp_q = (1.0 - vcp_best_width_rank) * 100 if setup_vcp else 0.0
            pp_q = (1.0 - pp_best_flag_rank) * 100 if setup_pp else 0.0
            setup_q_score = max(vcp_q, pp_q)

            candidates.append({
                'symbol': symbol,
                'bucket': bucket,
                'rs_rank': round(rs_rank, 1),
                'price': close_now,
                'setup': 'PP' if setup_pp else 'VCP',
                'setup_q': float(setup_q_score),
                'trend_raw': float(trend_raw),
            })

        if not candidates:
            if dbg:
                self._dbg_add(dbg, "final", False, "no candidates passed trend + setup gates")
            return ([], dbg) if return_debug else []

        cdf = pd.DataFrame(candidates)

        # Trend score：用 trend_raw 做 percentile，避免硬塞 magic number
        cdf['trend_score'] = cdf['trend_raw'].rank(pct=True) * 100

        # Total score
        cdf['score'] = (
            config.SCORE_WEIGHTS['RS'] * cdf['rs_rank'] +
            config.SCORE_WEIGHTS['Setup'] * cdf['setup_q'] +
            config.SCORE_WEIGHTS['Trend'] * cdf['trend_score']
        ).round(1)

        # 排序 + top N
        cdf = cdf.sort_values('score', ascending=False).head(config.MAX_OUTPUT)

        results = cdf[['symbol', 'bucket', 'rs_rank', 'score', 'price', 'setup']].to_dict('records')

        if dbg:
            in_res = any(r['symbol'] == diag for r in results)
            if in_res:
                r0 = next(r for r in results if r['symbol'] == diag)
                dbg["selected"] = True
                dbg["bucket"] = r0.get("bucket")
                dbg["score"] = r0.get("score")
                self._dbg_add(dbg, "final", True, f"SELECTED: bucket={dbg['bucket']} score={dbg['score']}")
            else:
                self._dbg_add(dbg, "final", False, "NOT selected (did not pass one or more conditions)")

        return (results, dbg) if return_debug else results
