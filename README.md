# CURRY

CURRY is a Python 3.12+ MetaTrader 5 automation project. **Paper trading is the default.**
Ten example strategy rules are included; none is a claim of profitability.
Backtest it against suitable historical data and account for execution, spread, slippage,
and commission before drawing conclusions.

## Architecture

- `config.py`: environment-backed settings, supported instruments/timeframes, validation,
  and the live-trading confirmation guard.
- `main.py`: starts MT5, polls closed bars and quotes, coordinates risk checks and execution.
- `market_data.py`: MT5-only terminal connection, quotes, and historical bars; missing data
  raises an error instead of being replaced with fabricated prices.
- `indicators.py`: EMA(20/50/200), RSI(14), MACD(12/26/9), and ATR(14) calculations using pandas/numpy.
- `strategy.py`: ten selectable closed-bar strategies with common ATR-based SL and minimum 1:2 target.
- `analysis/structure.py`: confirmed N-bar fractal swings, HH/HL/LH/LL labels, and per-bar trend states.
- `analysis/zones.py`: ATR-clustered support/resistance and displacement-based supply/demand zones.
- `analysis/smc.py`: causal BOS/CHoCH, FVG, order-block, equal-level, and liquidity-sweep events.
- `filters/news.py`: cached Forex Factory XML calendar, currency-pair mapping, high-impact entry blackout, and fail-closed behavior.
- `filters/sessions.py`: DST-aware Asian, London, and New York session windows and strategy allow-lists.
- `risk_manager.py`: weekend, spread, daily-loss, open-position filters and lot sizing
  using MT5's account-currency profit calculator.
- `trade_executor.py`: simulated paper positions or live MT5 market orders with broker-side
  stop loss and take profit; paper exits and trade history are recorded locally.
- `backtester.py`: sequential OHLC replay using supplied historical bars, next-bar entries,
  and conservative stop-first handling when both exit levels are reached in one bar.
- `logger.py`: application logs and append-only CSV trade history in `logs/`.
- `dashboard.py`: terminal snapshot of account and bot status.

## Project structure

```text
CURRY/
├── main.py
├── config.py
├── strategy.py
├── indicators.py
├── analysis/
│   ├── structure.py
│   ├── zones.py
│   └── smc.py
├── filters/
│   ├── news.py
│   ├── sessions.py
│   └── sessions.yaml
├── risk_manager.py
├── trade_executor.py
├── market_data.py
├── backtester.py
├── logger.py
├── dashboard.py
├── requirements.txt
├── tests/
└── logs/
```

Market-structure indices are zero-based row positions in the supplied bars.
Fractal pivots are not emitted until the configured right-side bars have closed
(`known_at = pivot_index + n`). Supply/demand zone geometry becomes knowable at the
displacement close; `status_known_at` separately identifies when its status last
changed. Support/resistance touch counts include confirmed swing points only.

`analysis/smc.py` uses these event rules: BOS/CHoCH require a close-to-close
cross of the last confirmed swing in or against the prior non-RANGE trend;
wick-only breaks do not count. A three-candle FVG requires candle 1's high below
candle 3's low (bullish), or candle 1's low above candle 3's high (bearish), and
must be at least 0.3 prior ATR wide. FVGs become PARTIAL on entry and FILLED at
the opposite edge. An order block is the nearest prior opposite-colored candle
before a BOS displacement whose body is at least 1.0 prior ATR; a close beyond
the block's far edge mitigates it. Equal levels cluster confirmed same-side
swings within 0.1 ATR. A sweep requires a wick through a previously known swing
or equal-level, a sweep-bar close at/beyond that level, and a close back inside
on one of the next 1-3 bars. FVG and order-block detectors emit a new dataclass
snapshot at each state transition; each snapshot's `known_at` is the bar when
that particular state became observable.

## Economic news filter

The Forex Factory weekly XML feed is refreshed and cached for 15 minutes by default.
Calendar timestamps have no timezone in the feed, so CURRY interprets them in
`America/New_York` by default; set `CURRY_NEWS_TIMEZONE` to the timezone configured
for your Forex Factory calendar. The Windows requirements include `tzdata` for IANA
timezone support.

HIGH-impact events block new entries for affected currencies from 30 minutes before
until 30 minutes after the event. Medium/low events are tagged but do not block entries.
The parser flags rate decisions, NFP, and CPI by title. The `All` currency affects all
configured symbols. Unknown impacts, malformed feeds, and failed/expired refreshes fail
closed: entries are blocked and an ERROR is written to the console and log. Cached
data is not used after a failed refresh.

`CURRY_NEWS_BLOCK_BEFORE_MINUTES`, `CURRY_NEWS_BLOCK_AFTER_MINUTES`,
`CURRY_NEWS_CACHE_SECONDS`, `CURRY_NEWS_TIMEOUT_SECONDS`, `CURRY_NEWS_TIMEZONE`, and
`CURRY_NEWS_CALENDAR_URL` configure the filter. Existing affected positions default to
`CURRY_NEWS_OPEN_TRADE_ACTION=move_to_breakeven`; set it to `none` to disable position
changes. Breakeven is applied only when it tightens the stop and the current quote is
past entry; live MT5 modifications also respect the broker's stop/freeze distance.

## Setup (Windows / VS Code)

1. Install Python 3.12+ and MetaTrader 5. Install dependencies:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Install and open the MT5 desktop terminal. Sign in there, or supply credentials
   through environment variables (`MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`). Never
   put credentials in source control.
3. Ensure your broker exposes the configured symbols. `XUAUSD` is kept exactly as
   requested; brokers commonly use different instrument names/suffixes, so override
   `CURRY_SYMBOLS` with the exact MT5 Market Watch names when necessary.
4. Run `python main.py`. With no environment changes, the bot runs in **paper** mode.
   Paper mode still requires a connected MT5 terminal for real quotes, account
   information and broker-calculated sizing; it never creates MT5 orders.

Settings can be overridden with environment variables, for example:

```powershell
$env:CURRY_TIMEFRAME = "M15"
$env:CURRY_STRATEGY = "breakout_retest"
$env:CURRY_SYMBOLS = "EURUSD,GBPUSD,USDJPY"
$env:CURRY_MAX_SPREAD_POINTS = "30"
python main.py
```

Supported strategies and identifiers for `CURRY_STRATEGY`:

1. `ma_crossover`: EMA 50/200 cross; H1/H4 are typically used.
2. `support_resistance`: rolling 20-bar range rejection; M15/H1/H4.
3. `breakout_retest`: recent 20-bar range break and retest; M15/H1.
4. `rsi`: RSI(14) exits oversold/overbought with candle-direction confirmation.
5. `fibonacci`: rejection near 38.2%, 50%, or 61.8% of a recent 50-bar swing.
6. `liquidity_sweep`: sweep and rejection of clustered highs/lows.
7. `fair_value_gap`: three-candle imbalance revisited with directional confirmation.
8. `london_breakout`: confirmed close outside the same-day 00:00-07:00 UTC range,
   during 07:00-11:00 UTC; M5/M15.
9. `macd`: MACD(12/26/9) crossover confirmed by EMA 50/200 trend.
10. `supply_demand`: retest and rejection of a detected base preceding an impulse.

The default is `ma_crossover`. Only the selected strategy submits signals; strategies
are not combined or voted together. These rules are mechanical approximations of the
named concepts, not discretionary market-structure analysis. The shared account risk
is configurable from 0.5% to 1% per trade (default 0.5%), and target reward/risk cannot
be configured below 1:2. No anti-martingale or revenge-trading recovery logic is used.

Supported timeframes: `M5`, `M15`, `M30`, `H1`, `H4`. The default symbol list is
EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, XUAUSD, GBPJPY.
`filters/sessions.yaml` controls the allowed local trading sessions for each strategy.
The session windows are Tokyo 09:00-18:00, London 08:00-17:00, and New York
08:00-17:00 in their respective IANA timezones. ZoneInfo applies DST transitions
automatically; the London/New York overlap is computed from both local windows.
The YAML lists may be edited per strategy; an empty list disables that strategy's
entries. Forex-weekend protection remains active independently of session selection.

### Live trading safety

Live trading is not enabled by default. To intentionally enable it, set both:

```powershell
$env:CURRY_TRADING_MODE = "live"
$env:CURRY_LIVE_TRADING_CONFIRM = "I_ACCEPT_LIVE_RISK"
python main.py
```

Review all risk settings, verify the connected account and broker symbol specifications,
and test paper mode first. Live mode sends market orders; broker-side SL/TP are included
in each request. A rejected or ambiguous MT5 order is logged as an error rather than
reported as a successful trade.

## Backtesting

`MarketData.fetch_history(symbol, start, end)` retrieves actual MT5 history, and
`backtester.run_backtest(symbol, bars, strategy, spread_price=...)` replays it. `bars`
must contain real OHLC bars with a datetime index. Spread is an explicit input; the
backtester does not manufacture market data, model commissions, or guarantee that past
results predict future performance.
