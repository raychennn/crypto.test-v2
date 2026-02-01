import os
import asyncio
import logging
import pytz
import re
from datetime import datetime, timedelta
from io import BytesIO

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from data_loader import DataLoader
from screener import CryptoScreener
import config

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

application = None

RETRO_RE = re.compile(r"^/(\d{6})\s+(\d{1,2})(?:\s+\$?([A-Za-z0-9]+))?\s*$")


def _to_clean_symbol(sym: str) -> str:
    s = sym.strip().upper().lstrip('$')
    if not s:
        return s
    # 允許使用者輸入 CHZ 或 CHZUSDT
    if not s.endswith('USDT'):
        s = s + 'USDT'
    return s


def _parse_retro(text: str):
    """
    解析 /yymmdd hh [$SYMBOL]
    回傳： (requested_close_local_dt, target_clean_symbol_or_None, error_or_None)
    """
    m = RETRO_RE.match(text.strip())
    if not m:
        return None, None, "格式錯誤。用法：/yymmdd hh 或 /yymmdd hh $SYMBOL，例如 /260103 16 或 /260103 16 $CHZ"

    yymmdd, hh, sym = m.group(1), m.group(2), m.group(3)

    try:
        yy = int(yymmdd[0:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])
        hour = int(hh)
        if not (0 <= hour <= 23):
            return None, None, "hh 必須是 0~23。"
        year = 2000 + yy
        tz = pytz.timezone(config.TIMEZONE)
        requested_close_local = tz.localize(datetime(year, mm, dd, hour, 0, 0))
    except Exception as e:
        return None, None, f"日期時間解析失敗：{e}"

    target_clean = _to_clean_symbol(sym) if sym else None
    return requested_close_local, target_clean, None


def _format_results(results, title_time: str):
    if not results:
        return f"=== Screening Result ({title_time}) ===\nNo assets passed the strict criteria this round."

    msg = f"=== Screening Result ({title_time}) ===\n"
    msg += f"Top {len(results)} Candidates\n\n"
    for r in results:
        icon = "🚀" if r['bucket'] == 'Leader' else "⚡" if r['bucket'] == 'PowerPlay' else "🔄"
        msg += f"{icon} {r['symbol']} (RS:{r['rs_rank']})\n"
        msg += f"   Type: {r['setup']} | Score: {r['score']} | Price: {r['price']}\n"
    return msg


def _format_diag(dbg: dict):
    """把 screener 回傳的 debug dict 轉成可讀訊息。"""
    sym = dbg.get("symbol", "")
    lines = [f"=== Diagnostic ({sym}) ==="]
    selected = dbg.get("selected", False)
    if selected:
        lines.append(f"✅ PASS | bucket={dbg.get('bucket')} | score={dbg.get('score')}")
    else:
        lines.append("❌ NOT SELECTED")
    lines.append("")

    for chk in dbg.get("checks", []):
        mark = "✅" if chk.get("ok") else "❌"
        stage = chk.get("stage")
        detail = chk.get("detail")
        lines.append(f"{mark} {stage}: {detail}")

    return "\n".join(lines)


async def run_screener_logic(context_chat_id=None):
    """原本即時掃描（現在 /now 會用這個）。"""
    target_chat_id = context_chat_id if context_chat_id else CHAT_ID

    if not target_chat_id:
        logger.warning("No Chat ID provided, skipping message send.")
        return

    loader = DataLoader()
    try:
        if context_chat_id:
            try:
                await application.bot.send_message(chat_id=target_chat_id, text="🔍 Scanning market (throttled mode)...")
            except Exception as e:
                logger.warning(f"Failed to send start message: {e}", exc_info=True)

        data_map, btc_data = await loader.get_all_data()
        screener = CryptoScreener(data_map, btc_data)
        results = screener.run()

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        msg = _format_results(results, now_str)
        await application.bot.send_message(chat_id=target_chat_id, text=msg)

        if results:
            txt_content = ",".join([f"BINANCE:{r['symbol']}" for r in results])
            file_obj = BytesIO(txt_content.encode())
            file_obj.name = f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            await application.bot.send_document(chat_id=target_chat_id, document=file_obj)

    except Exception as e:
        logger.error(f"Error in job: {e}", exc_info=True)
        try:
            await application.bot.send_message(chat_id=target_chat_id, text=f"⚠️ Error: {str(e)}")
        except Exception as se:
            logger.warning(f"Failed to send error message: {se}", exc_info=True)


async def run_screener_asof(update: Update, requested_close_local: datetime, target_clean: str | None):
    """回溯查詢：/yymmdd hh [ $SYMBOL ]"""
    chat_id = update.effective_chat.id
    tz = pytz.timezone(config.TIMEZONE)
    now_local = datetime.now(tz)

    # 避免未來時間
    if requested_close_local > now_local + timedelta(minutes=1):
        await update.message.reply_text(
            f"⚠️ 你指定的是未來時間：{requested_close_local.strftime('%Y-%m-%d %H:%M %Z')}\n"
            f"目前時間：{now_local.strftime('%Y-%m-%d %H:%M %Z')}"
        )
        return

    # 提示開始
    try:
        tip = f"⏳ Retro scan: {requested_close_local.strftime('%Y-%m-%d %H:%M %Z')}"
        if target_clean:
            tip += f" | symbol={target_clean}"
        await update.message.reply_text(tip)
    except Exception as e:
        logger.warning(f"Failed to send start tip: {e}", exc_info=True)

    loader = DataLoader()
    try:
        force = [target_clean] if target_clean else None
        data_map, btc_data, meta = await loader.get_all_data_asof(requested_close_local, force_clean_symbols=force)

        used_close = meta.get("used_close_local")
        diff_min = float(meta.get("diff_minutes", 0.0))
        title_time = used_close.strftime('%Y-%m-%d %H:%M %Z') if used_close else requested_close_local.strftime('%Y-%m-%d %H:%M %Z')

        # 若對齊時間和指定時間差太多，提示使用者（避免 silent）
        if used_close and diff_min > float(getattr(config, "RETRO_TIME_TOLERANCE_MINUTES", 10)):
            await application.bot.send_message(
                chat_id=chat_id,
                text=(
                    "⚠️ 指定時間點資料可能缺漏，已改用最近可用收盤時間。\n"
                    f"Requested: {requested_close_local.strftime('%Y-%m-%d %H:%M %Z')}\n"
                    f"Used:      {used_close.strftime('%Y-%m-%d %H:%M %Z')}\n"
                    f"Diff: {diff_min:.1f} min"
                )
            )

        screener = CryptoScreener(data_map, btc_data)

        if target_clean:
            results, dbg = screener.run(diagnose_symbol=target_clean, return_debug=True)

            # 1) 先回覆是否入選
            in_list = any(r["symbol"] == target_clean for r in results)
            if in_list:
                r0 = next(r for r in results if r["symbol"] == target_clean)
                await application.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"✅ {target_clean} PASS @ {title_time}\n"
                        f"bucket={r0['bucket']} | RS={r0['rs_rank']} | score={r0['score']} | setup={r0['setup']} | price={r0['price']}"
                    )
                )
            else:
                await application.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ {target_clean} NOT SELECTED @ {title_time}\n以下是逐項條件檢查："
                )

            # 2) 再回傳逐項條件（不吞錯）
            await application.bot.send_message(chat_id=chat_id, text=_format_diag(dbg))

        else:
            results = screener.run()
            await application.bot.send_message(chat_id=chat_id, text=_format_results(results, title_time))

            # watchlist 檔案
            if results:
                txt_content = ",".join([f"BINANCE:{r['symbol']}" for r in results])
                file_obj = BytesIO(txt_content.encode())
                file_obj.name = f"watchlist_asof_{used_close.strftime('%Y%m%d_%H%M') if used_close else datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                await application.bot.send_document(chat_id=chat_id, document=file_obj)

    except Exception as e:
        logger.error(f"Error in retro scan: {e}", exc_info=True)
        try:
            await application.bot.send_message(chat_id=chat_id, text=f"⚠️ Retro scan error: {str(e)}")
        except Exception as se:
            logger.warning(f"Failed to send retro error message: {se}", exc_info=True)


async def cmd_retro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    requested_close_local, target_clean, err = _parse_retro(text)
    if err:
        await update.message.reply_text(err)
        return
    # 若使用者指定的 symbol 在排除名單，直接回報（避免誤判）
    if target_clean and target_clean in getattr(config, "EXCLUDE_SYMBOLS", []):
        await update.message.reply_text(
            f"⚠️ {target_clean} 在 EXCLUDE_SYMBOLS 被排除（多為指數/非加密貨幣），不做回溯評估。"
        )
        return

    await run_screener_asof(update, requested_close_local, target_clean)


async def scheduled_job():
    logger.info("Running scheduled screening...")
    await run_screener_logic()


async def cmd_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Request received. Starting scan...")
    await run_screener_logic(context_chat_id=update.effective_chat.id)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"Bot is running! Your Chat ID is: {chat_id}\n"
        "Use /now to screen instantly.\n"
        "Use /yymmdd hh or /yymmdd hh $SYMBOL to run retro scans (e.g., /260103 16 or /260103 16 $CHZ)."
    )


async def post_init(app: Application):
    logger.info("Setting up scheduler in post_init...")
    scheduler = AsyncIOScheduler(timezone=pytz.timezone(config.TIMEZONE))

    times = getattr(config, 'SCHEDULE_TIMES', None) or ["00:02", "04:02", "08:02", "12:02", "16:02", "20:02"]
    for t in times:
        try:
            h, m = t.split(":")
            scheduler.add_job(scheduled_job, 'cron', hour=int(h), minute=int(m))
        except Exception as e:
            logger.warning(f"Invalid schedule time '{t}': {e}", exc_info=True)

    scheduler.start()
    logger.info(f"Scheduler started with {len(times)} jobs: {times}")


def main():
    global application

    if not TOKEN:
        raise RuntimeError("Missing TELEGRAM_TOKEN in environment variables.")

    application = Application.builder().token(TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("now", cmd_now))

    # Retro handler：捕捉 /260103 16 或 /260103 16 $CHZ 這類訊息
    application.add_handler(MessageHandler(filters.Regex(RETRO_RE), cmd_retro))

    logger.info("Bot is starting polling...")
    application.run_polling()


if __name__ == "__main__":
    main()
