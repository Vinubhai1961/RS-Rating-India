//@version=6
indicator('RS-Rating', shorttitle='RS-Rating', overlay=true, max_bars_back=500)
//import TradingView/ta/12

// ────────────────────── MARKET SELECTION ──────────────────────
market = input.string("USA", "Market", options=["USA", "India"], group="Market Selection")
indiaBenchmark = input.symbol("NSE:CNX500", "India Benchmark", group="Market Selection")
benchmark = market == "India" ? indiaBenchmark : "SPY"

// ────────────────────── THRESHOLDS (AUTO SWITCHED) ──────────────────────
// Auto-generated RS Rating thresholds - do not edit manually
// Last updated: 2026-06-25 02:57 UTC

// RS thresholds
usa98 = input.float(215.30, "USA 98th → RS ≥", group="USA Thresholds")
usa89 = input.float(120.11, "USA 89th → RS ≥", group="USA Thresholds")
usa69 = input.float(100.08, "USA 69th → RS ≥", group="USA Thresholds")
usa49 = input.float(92.78, "USA 49th → RS ≥", group="USA Thresholds")
usa29 = input.float(88.46, "USA 29th → RS ≥", group="USA Thresholds")
usa09 = input.float(72.21, "USA 9th → RS ≥", group="USA Thresholds")
usa01 = input.float(39.76, "USA 1th → RS ≥", group="USA Thresholds")

// 1M_RS thresholds
usa1m98 = input.float(135.94, "USA1M 98th → RS ≥", group="USA1M Thresholds")
usa1m89 = input.float(111.94, "USA1M 89th → RS ≥", group="USA1M Thresholds")
usa1m69 = input.float(103.56, "USA1M 69th → RS ≥", group="USA1M Thresholds")
usa1m49 = input.float(101.70, "USA1M 49th → RS ≥", group="USA1M Thresholds")
usa1m29 = input.float(99.66, "USA1M 29th → RS ≥", group="USA1M Thresholds")
usa1m09 = input.float(88.45, "USA1M 9th → RS ≥", group="USA1M Thresholds")
usa1m01 = input.float(63.69, "USA1M 1th → RS ≥", group="USA1M Thresholds")

// 3M_RS thresholds
usa3m98 = input.float(174.49, "USA3M 98th → RS ≥", group="USA3M Thresholds")
usa3m89 = input.float(114.95, "USA3M 89th → RS ≥", group="USA3M Thresholds")
usa3m69 = input.float(99.69, "USA3M 69th → RS ≥", group="USA3M Thresholds")
usa3m49 = input.float(93.42, "USA3M 49th → RS ≥", group="USA3M Thresholds")
usa3m29 = input.float(89.46, "USA3M 29th → RS ≥", group="USA3M Thresholds")
usa3m09 = input.float(77.20, "USA3M 9th → RS ≥", group="USA3M Thresholds")
usa3m01 = input.float(50.85, "USA3M 1th → RS ≥", group="USA3M Thresholds")

// 6M_RS thresholds
usa6m98 = input.float(202.30, "USA6M 98th → RS ≥", group="USA6M Thresholds")
usa6m89 = input.float(123.05, "USA6M 89th → RS ≥", group="USA6M Thresholds")
usa6m69 = input.float(102.81, "USA6M 69th → RS ≥", group="USA6M Thresholds")
usa6m49 = input.float(95.71, "USA6M 49th → RS ≥", group="USA6M Thresholds")
usa6m29 = input.float(91.56, "USA6M 29th → RS ≥", group="USA6M Thresholds")
usa6m09 = input.float(69.81, "USA6M 9th → RS ≥", group="USA6M Thresholds")
usa6m01 = input.float(34.79, "USA6M 1th → RS ≥", group="USA6M Thresholds")

// Auto-generated RS Rating thresholds - do not edit manually
// Last updated: 2026-06-25 01:36 UTC

// RS thresholds
ind98 = input.float(264.05, "IND 98th → RS ≥", group="IND Thresholds")
ind89 = input.float(144.46, "IND 89th → RS ≥", group="IND Thresholds")
ind69 = input.float(112.41, "IND 69th → RS ≥", group="IND Thresholds")
ind49 = input.float(99.80, "IND 49th → RS ≥", group="IND Thresholds")
ind29 = input.float(90.32, "IND 29th → RS ≥", group="IND Thresholds")
ind09 = input.float(76.97, "IND 9th → RS ≥", group="IND Thresholds")
ind01 = input.float(56.17, "IND 1th → RS ≥", group="IND Thresholds")

// 1M_RS thresholds
ind1m98 = input.float(145.99, "IND1M 98th → RS ≥", group="IND1M Thresholds")
ind1m89 = input.float(117.61, "IND1M 89th → RS ≥", group="IND1M Thresholds")
ind1m69 = input.float(105.34, "IND1M 69th → RS ≥", group="IND1M Thresholds")
ind1m49 = input.float(99.76, "IND1M 49th → RS ≥", group="IND1M Thresholds")
ind1m29 = input.float(95.24, "IND1M 29th → RS ≥", group="IND1M Thresholds")
ind1m09 = input.float(87.30, "IND1M 9th → RS ≥", group="IND1M Thresholds")
ind1m01 = input.float(73.22, "IND1M 1th → RS ≥", group="IND1M Thresholds")

// 3M_RS thresholds
ind3m98 = input.float(201.24, "IND3M 98th → RS ≥", group="IND3M Thresholds")
ind3m89 = input.float(139.86, "IND3M 89th → RS ≥", group="IND3M Thresholds")
ind3m69 = input.float(114.73, "IND3M 69th → RS ≥", group="IND3M Thresholds")
ind3m49 = input.float(104.18, "IND3M 49th → RS ≥", group="IND3M Thresholds")
ind3m29 = input.float(95.08, "IND3M 29th → RS ≥", group="IND3M Thresholds")
ind3m09 = input.float(83.15, "IND3M 9th → RS ≥", group="IND3M Thresholds")
ind3m01 = input.float(59.50, "IND3M 1th → RS ≥", group="IND3M Thresholds")

// 6M_RS thresholds
ind6m98 = input.float(253.55, "IND6M 98th → RS ≥", group="IND6M Thresholds")
ind6m89 = input.float(145.99, "IND6M 89th → RS ≥", group="IND6M Thresholds")
ind6m69 = input.float(114.16, "IND6M 69th → RS ≥", group="IND6M Thresholds")
ind6m49 = input.float(99.97, "IND6M 49th → RS ≥", group="IND6M Thresholds")
ind6m29 = input.float(89.11, "IND6M 29th → RS ≥", group="IND6M Thresholds")
ind6m09 = input.float(73.06, "IND6M 9th → RS ≥", group="IND6M Thresholds")
ind6m01 = input.float(48.08, "IND6M 1th → RS ≥", group="IND6M Thresholds")

p98 = market == "India" ? ind98 : usa98
p89 = market == "India" ? ind89 : usa89
p69 = market == "India" ? ind69 : usa69
p49 = market == "India" ? ind49 : usa49
p29 = market == "India" ? ind29 : usa29
p09 = market == "India" ? ind09 : usa09
p01 = market == "India" ? ind01 : usa01

// Short-term threshold maps.
// India uses dedicated 1M/3M/6M threshold sets. USA falls back to main USA thresholds until USA short-term sets are generated.
p1m98 = market == "India" ? ind1m98 : usa1m98
p1m89 = market == "India" ? ind1m89 : usa1m89
p1m69 = market == "India" ? ind1m69 : usa1m69
p1m49 = market == "India" ? ind1m49 : usa1m49
p1m29 = market == "India" ? ind1m29 : usa1m29
p1m09 = market == "India" ? ind1m09 : usa1m09
p1m01 = market == "India" ? ind1m01 : usa1m01

p3m98 = market == "India" ? ind3m98 : usa3m98
p3m89 = market == "India" ? ind3m89 : usa3m89
p3m69 = market == "India" ? ind3m69 : usa3m69
p3m49 = market == "India" ? ind3m49 : usa3m49
p3m29 = market == "India" ? ind3m29 : usa3m29
p3m09 = market == "India" ? ind3m09 : usa3m09
p3m01 = market == "India" ? ind3m01 : usa3m01

p6m98 = market == "India" ? ind6m98 : usa6m98
p6m89 = market == "India" ? ind6m89 : usa6m89
p6m69 = market == "India" ? ind6m69 : usa6m69
p6m49 = market == "India" ? ind6m49 : usa6m49
p6m29 = market == "India" ? ind6m29 : usa6m29
p6m09 = market == "India" ? ind6m09 : usa6m09
p6m01 = market == "India" ? ind6m01 : usa6m01


// ────────────────────── DISPLAY OPTIONS ──────────────────────
showLabelNearLine = input.bool(true,  "Show small RS on line", group="Display")
showTopCenter     = input.bool(true,  "Show RS (USA/INDIA) : XX at top center", group="Display")
showRSLine        = input.bool(true,  "Plot RS Line", group="Display")
show21EMA         = input.bool(true,  "Plot 21 EMA on RS Line", group="Display")
lineColor         = input.color(color.blue, "RS Line Color", group="Display")
emaColor          = input.color(color.orange, "21 EMA Color", group="Display")
topLabelColor     = input.color(color.new(#1E90FF, 0), "Top Label Color", group="Display")

// ────────────────────── CORE CALCULATION ──────────────────────
tf      = "D"

// ────────────────────── RS LOOKBACKS ──────────────────────
// These stay standard, but now they are counted on STOCK+BENCHMARK aligned bars.
// This is closer to your GitHub logic: stock/ref aligned rows first, then lookback.
lb1M  = input.int(21,  "1M Lookback Aligned Bars",  group="RS Lookbacks")
lb3M  = input.int(63,  "3M Lookback Aligned Bars",  group="RS Lookbacks")
//lb6M  = input.int(128, "6M Lookback Aligned Bars",  group="RS Lookbacks")
lb6M  = market == "India" ? 128 : 126
lb9M  = input.int(189, "9M Lookback Aligned Bars",  group="RS Lookbacks")
lb12M = input.int(252, "12M Lookback Aligned Bars", group="RS Lookbacks")

// Stock close
closeD = request.security(syminfo.tickerid, tf, close, barmerge.gaps_off, barmerge.lookahead_off)
dailyTimeD = request.security(syminfo.tickerid, tf, time, barmerge.gaps_off, barmerge.lookahead_off)

// Benchmark close with gaps_on for alignment.
// This prevents Pine from silently carrying benchmark values across missing benchmark dates.
benchD = request.security(benchmark, tf, close, barmerge.gaps_on, barmerge.lookahead_off)

// Alignment row = stock has close and benchmark has real close on same DAILY date.
alignedBar = not na(closeD) and not na(benchD)

// Count only one aligned event per daily bar. This prevents lower-timeframe charts
// from counting the same daily close many times.
newDailyBar  = timeframe.change(tf)
alignedEvent = alignedBar and newDailyBar

// Latest aligned daily values. If current chart bar is not aligned, this uses the most recent aligned row.
curCloseD = ta.valuewhen(alignedEvent, closeD, 0)
curBenchD = ta.valuewhen(alignedEvent, benchD, 0)
curTimeD  = ta.valuewhen(alignedEvent, dailyTimeD, 0)

// Count aligned daily rows, similar to GitHub aligned rows.
alignedRows = ta.cum(alignedEvent ? 1 : 0)
alignedBars = int(alignedRows) - 1

// Lookback values from aligned daily rows.
close1M  = ta.valuewhen(alignedEvent, closeD, lb1M)
bench1M  = ta.valuewhen(alignedEvent, benchD, lb1M)
time1M   = ta.valuewhen(alignedEvent, dailyTimeD, lb1M)

close3M  = ta.valuewhen(alignedEvent, closeD, lb3M)
bench3M  = ta.valuewhen(alignedEvent, benchD, lb3M)
time3M   = ta.valuewhen(alignedEvent, dailyTimeD, lb3M)

close6M  = ta.valuewhen(alignedEvent, closeD, lb6M)
bench6M  = ta.valuewhen(alignedEvent, benchD, lb6M)
time6M   = ta.valuewhen(alignedEvent, dailyTimeD, lb6M)

close9M  = ta.valuewhen(alignedEvent, closeD, lb9M)
bench9M  = ta.valuewhen(alignedEvent, benchD, lb9M)
time9M   = ta.valuewhen(alignedEvent, dailyTimeD, lb9M)

close12M = ta.valuewhen(alignedEvent, closeD, lb12M)
bench12M = ta.valuewhen(alignedEvent, benchD, lb12M)
time12M  = ta.valuewhen(alignedEvent, dailyTimeD, lb12M)

has1M  = alignedRows > lb1M  and not na(close1M)  and not na(bench1M)  and bench1M  > 0 and curBenchD > 0
has3M  = alignedRows > lb3M  and not na(close3M)  and not na(bench3M)  and bench3M  > 0 and curBenchD > 0
has6M  = alignedRows > lb6M  and not na(close6M)  and not na(bench6M)  and bench6M  > 0 and curBenchD > 0
has9M  = alignedRows > lb9M  and not na(close9M)  and not na(bench9M)  and bench9M  > 0 and curBenchD > 0
has12M = alignedRows > lb12M and not na(close12M) and not na(bench12M) and bench12M > 0 and curBenchD > 0

valid = has3M and not na(curCloseD) and not na(curBenchD) and curBenchD > 0

perfS = has3M and has6M and has9M and has12M ?
     0.4 * (curCloseD / close3M) +
     0.2 * (curCloseD / close6M) +
     0.2 * (curCloseD / close9M) +
     0.2 * (curCloseD / close12M)
     : na

perfR = has3M and has6M and has9M and has12M ?
     0.4 * (curBenchD / bench3M) +
     0.2 * (curBenchD / bench6M) +
     0.2 * (curBenchD / bench9M) +
     0.2 * (curBenchD / bench12M)
     : na

rsRaw   = not na(perfS) and not na(perfR) and perfR > 0 ? perfS / perfR * 100 : na
rsScore = not na(rsRaw) ? math.min(rsRaw, 700) : na

// ────────────────────── SHORT TERM RS ──────────────────────
rs1m = has1M ? (curCloseD / close1M) / (curBenchD / bench1M) * 100 : na
rs3m = has3M ? (curCloseD / close3M) / (curBenchD / bench3M) * 100 : na
rs6m = has6M ? (curCloseD / close6M) / (curBenchD / bench6M) * 100 : na

// ────────────────────── RS HIGH DETECTION ──────────────────────
rsHighPrev = ta.highest(rsScore[1], 252)

rsBreakout = not na(rsScore) and not na(rsHighPrev) and rsScore > rsHighPrev
     // and rsRating >= 90

// ────────────────────── RATING ──────────────────────
f_rsRating(score) =>
    var int r = 1
    r := 1
    if score >= p98
        r := 99
    else if score >= p89
        r := 90 + math.round(8 * (score - p89) / (p98 - p89))
    else if score >= p69
        r := 70 + math.round(19 * (score - p69) / (p89 - p69))
    else if score >= p49
        r := 50 + math.round(19 * (score - p49) / (p69 - p49))
    else if score >= p29
        r := 30 + math.round(19 * (score - p29) / (p49 - p29))
    else if score >= p09
        r := 10 + math.round(19 * (score - p09) / (p29 - p09))
    else if score >= p01
        r := 2 + math.round(7 * (score - p01) / (p09 - p01))
    math.min(99, math.max(1, r))

// Same mapping formula, but with a supplied threshold set.
// Used for 1M/3M/6M so short-term RS uses its own percentile distribution.
f_rsRating_custom(score, x98, x89, x69, x49, x29, x09, x01) =>
    var int r = 1
    r := 1
    if score >= x98
        r := 99
    else if score >= x89
        r := 90 + math.round(8 * (score - x89) / (x98 - x89))
    else if score >= x69
        r := 70 + math.round(19 * (score - x69) / (x89 - x69))
    else if score >= x49
        r := 50 + math.round(19 * (score - x49) / (x69 - x49))
    else if score >= x29
        r := 30 + math.round(19 * (score - x29) / (x49 - x29))
    else if score >= x09
        r := 10 + math.round(19 * (score - x09) / (x29 - x09))
    else if score >= x01
        r := 2 + math.round(7 * (score - x01) / (x09 - x01))
    math.min(99, math.max(1, r))

rsRating     = valid and not na(rsScore) ? f_rsRating(rsScore) : na
rsRating_1M  = valid and not na(rs1m)    ? f_rsRating_custom(rs1m, p1m98, p1m89, p1m69, p1m49, p1m29, p1m09, p1m01) : na
rsRating_3M  = valid and not na(rs3m)    ? f_rsRating_custom(rs3m, p3m98, p3m89, p3m69, p3m49, p3m29, p3m09, p3m01) : na
rsRating_6M  = valid and not na(rs6m)    ? f_rsRating_custom(rs6m, p6m98, p6m89, p6m69, p6m49, p6m29, p6m09, p6m01) : na

// ────────────────────── IPO RS (ALIGNED + PROGRESSIVE) ──────────────────────
bars       = alignedBars
isIPOPhase = bars > 5 and bars < 200

// Progressive weights (aligned with main RS: 0.4 / 0.2 / 0.2 / 0.2)
w1 = bars >= 21  ? 0.4 : 0.0   // ~1M (replaces 3M slot)
w2 = bars >= 63  ? 0.3 : 0.0   // ~3M (replaces 6M slot)
w3 = bars >= 126 ? 0.2 : 0.0   // ~6M (replaces 9M slot)
w4 = bars >= 189 ? 0.1 : 0.0   // ~9M (replaces 12M slot)

// Normalize active weights
sumW = w1 + w2 + w3 + w4

// Stock performance from aligned rows
perfS_IPO =
     sumW > 0 ?
     ((w1 > 0 and has1M ? w1 * (curCloseD / close1M) : 0) +
      (w2 > 0 and has3M ? w2 * (curCloseD / close3M) : 0) +
      (w3 > 0 and has6M ? w3 * (curCloseD / close6M) : 0) +
      (w4 > 0 and has9M ? w4 * (curCloseD / close9M) : 0)) / sumW
     : na

// Benchmark performance from aligned rows
perfR_IPO =
     sumW > 0 ?
     ((w1 > 0 and has1M ? w1 * (curBenchD / bench1M) : 0) +
      (w2 > 0 and has3M ? w2 * (curBenchD / bench3M) : 0) +
      (w3 > 0 and has6M ? w3 * (curBenchD / bench6M) : 0) +
      (w4 > 0 and has9M ? w4 * (curBenchD / bench9M) : 0)) / sumW
     : na

// Relative Strength
rsRaw_IPO =
     isIPOPhase and not na(perfS_IPO) and not na(perfR_IPO) and perfR_IPO > 0
     ? (perfS_IPO / perfR_IPO) * 100
     : na

// Cap to match main RS scale
IPO_CAP     = 700.0
rsScore_IPO = isIPOPhase and not na(rsRaw_IPO) ? math.min(rsRaw_IPO, IPO_CAP) : na

// Convert to 1–99 using sqrt scaling (smooth early behavior)
//f_rsRating_IPO(score) =>
    //na(score) ? na : math.round(math.min(99, math.max(1, 1 + 98 * math.sqrt(score / IPO_CAP))))
f_rsRating_IPO(score) => na(score) ? na : f_rsRating(score)
rsRating_IPO = f_rsRating_IPO(rsScore_IPO)

finalRS = bars < 200 ? rsRating_IPO : rsRating
finalRSScore = bars < 200 ? rsScore_IPO : rsScore

// ────────────────────── SMART RS LINE ──────────────────────
priceLow   = ta.lowest(low, 50)
priceRange = ta.highest(high, 100) - priceLow
baseOffset = priceLow * 0.92
rsLineY    = baseOffset + (finalRSScore / 700) * priceRange * 0.6

plot(showRSLine ? rsLineY : na, "RS Line", color=lineColor, linewidth=2)

// EMA
rsEMA = ta.ema(rsLineY, 21)
plot(show21EMA ? rsEMA : na, "21 EMA", color=emaColor, linewidth=2)

// RS vs EMA
rsDistPct = not na(rsLineY) and not na(rsEMA) and rsEMA != 0
     ? ((rsLineY - rsEMA) / rsEMA) * 100
     : na

rsEMAText =
     not na(rsDistPct)
     ? (rsDistPct >= 0
         ? "RS > EMA ✅ (" + str.tostring(rsDistPct, "#.##") + "%)"
         : "RS < EMA ❌ (" + str.tostring(rsDistPct, "#.##") + "%)")
     : "RS vs EMA: n/a"

// ────────────────────── LABELS ──────────────────────
//if showLabelNearLine and barstate.islast and not na(rsRating)
    //label.new(bar_index, rsLineY, str.tostring(rsRating),
if showLabelNearLine and barstate.islast and not na(finalRS)
    label.new(bar_index, rsLineY, str.tostring(finalRS),
         style=label.style_label_left, color=color.new(color.black, 80),
         textcolor=topLabelColor, size=size.small)

if rsBreakout and barstate.islast
    label.new(bar_index, rsLineY, "RS High",
         style=label.style_label_up,
         color=color.green,
         textcolor=color.white,
         size=size.small)

// ────────────────────── STAGE-2 (WEEKLY TREND CHECK) ──────────────────────
// === Stage 2: 10WK SMA vs 30WK SMA ===
// (Enhanced & Consistent with rest of your script)
useLookahead = input.bool(true, "Include Realtime in Weekly Data?")
lookaheadMode = useLookahead ? barmerge.lookahead_on : barmerge.lookahead_off
weekly_sma10 = request.security(syminfo.tickerid, "1W", ta.sma(close, 10), lookahead=lookaheadMode)
weekly_sma30 = request.security(syminfo.tickerid, "1W", ta.sma(close, 30), lookahead=lookaheadMode)

stage2_up = not na(weekly_sma10) and not na(weekly_sma30) and weekly_sma10 > weekly_sma30

stage2_pct = not na(weekly_sma10) and not na(weekly_sma30) and weekly_sma30 != 0 ?
     ((weekly_sma10 - weekly_sma30) / weekly_sma30) * 100 : na

stage2Text = "Stage-2 (10W-30W): " + 
     (stage2_up ? "✅ UP" : "❌ DOWN") + 
     (not na(stage2_pct) ? " | " + str.tostring(stage2_pct, "#.##") + "%" : " | n/a")

stage2Color = stage2_up ? color.green : color.red

// === Weekly 10 SMA Distance % ===
weekly_close = request.security(syminfo.tickerid, "1W", close, lookahead=lookaheadMode)
distFromWSMA10 = ((weekly_close - weekly_sma10) / weekly_sma10) * 100
// ────────────────────── DEBUG ──────────────────────
showDebug = input.bool(true, "Show Debug Label", group="Debug")

if showDebug and barstate.islast
    label.new(
         bar_index,
         high,
         "\nCurrent Date=" + str.format("{0,date,yyyy-MM-dd}", curTimeD) +
         "Market=" + market +
         "\nBenchmark=" + benchmark +
         "\nAligned Rows=" + str.tostring(alignedRows, "#") +
         "\nChart Bars=" + str.tostring(bar_index) +
         "\nBench Close=" + str.tostring(curBenchD, "#.##") +
         "\nValid=" + str.tostring(valid) +
         

         "\n\n1M Date: " + str.format("{0,date,yyyy-MM-dd}", time1M) +
         " → " + str.format("{0,date,yyyy-MM-dd}", curTimeD) +
         "\n1M Stock: " + str.tostring(close1M, "#.##") +
         " → " + str.tostring(curCloseD, "#.##") +
         "\n1M Bench: " + str.tostring(bench1M, "#.##") +
         " → " + str.tostring(curBenchD, "#.##") +
         "\n1M Raw=" + str.tostring(rs1m, "#.##") +
         " | 1M RS=" + str.tostring(rsRating_1M) +

         "\n\n3M Date: " + str.format("{0,date,yyyy-MM-dd}", time3M) +
         " → " + str.format("{0,date,yyyy-MM-dd}", curTimeD) +
         "\n3M Stock: " + str.tostring(close3M, "#.##") +
         " → " + str.tostring(curCloseD, "#.##") +
         "\n3M Bench: " + str.tostring(bench3M, "#.##") +
         " → " + str.tostring(curBenchD, "#.##") +
         "\n3M Raw=" + str.tostring(rs3m, "#.##") +
         " | 3M RS=" + str.tostring(rsRating_3M) +

         "\n\n6M Date: " + str.format("{0,date,yyyy-MM-dd}", time6M) +
         " → " + str.format("{0,date,yyyy-MM-dd}", curTimeD) +
         "\n6M Stock: " + str.tostring(close6M, "#.##") +
         " → " + str.tostring(curCloseD, "#.##") +
         "\n6M Bench: " + str.tostring(bench6M, "#.##") +
         " → " + str.tostring(curBenchD, "#.##") +
         "\n6M Raw=" + str.tostring(rs6m, "#.##") +
         " | 6M RS=" + str.tostring(rsRating_6M) +

         "\n\n9M Date: " + str.format("{0,date,yyyy-MM-dd}", time9M) +
         "\n12M Date: " + str.format("{0,date,yyyy-MM-dd}", time12M) +
         "\nRS=" + str.tostring(rsRaw, "#.##") +
         "\nRS Score Main=" + str.tostring(rsScore, "#.##") +
         "\n6M Lookback=" + str.tostring(lb6M) + " | Date=" + str.format("{0,date,yyyy-MM-dd}", time6M) +
         "\n9M Lookback=" + str.tostring(lb9M) + " | Date=" + str.format("{0,date,yyyy-MM-dd}", time9M) +
         "\n12M Lookback=" + str.tostring(lb12M) + " | Date=" + str.format("{0,date,yyyy-MM-dd}", time12M) +
         "\nFinal RS=" + str.tostring(finalRS),
         style=label.style_label_down,
         color=color.yellow,
         textcolor=color.black
    )

// ────────────────────── TABLE ──────────────────────
var table t = table.new(position.top_right, 1, 1, border_width = 15)

if showTopCenter and barstate.islast //and not na(rsRating)
    sectorTxt   = na(syminfo.sector)   ? "Sector: n/a"   : "Sector: " + syminfo.sector
    industryTxt = na(syminfo.industry) ? "Industry: n/a" : "Industry: " + syminfo.industry

    //baseLine = "RS (" + market + ") : " + str.tostring(rsRating)
    baseLine = bars < 200 ? "IPO RS (" + market + ") : " + str.tostring(finalRS) : "RS (" + market + ") : " + str.tostring(finalRS)
    
    ipoLine  = not na(rsRating_IPO) ? "\nIPO RS: " + str.tostring(rsRating_IPO) : ""

    rsMultiLine =
         not na(rsRating_1M) and not na(rsRating_3M) and not na(rsRating_6M)
         ? "\n1M: " + str.tostring(rsRating_1M) +
           " | 3M: " + str.tostring(rsRating_3M) +
           " | 6M: " + str.tostring(rsRating_6M)
         : ""

    //displayText = baseLine + ipoLine + rsMultiLine + "\n" + rsEMAText + "\n" + sectorTxt + "\n" + industryTxt
    displayText = baseLine + ipoLine + rsMultiLine + "\n" + rsEMAText + "\n" + stage2Text + "\n" + sectorTxt + "\n" + industryTxt +"\n"+ "Dist 10SMA (W) %: " + str.tostring(distFromWSMA10, "#.##") + "%"
    table.cell(
         t, 0, 0, displayText,
         text_color  = topLabelColor,
         bgcolor     = color.new(color.white, 100),
         text_halign = text.align_left,
         text_valign = text.align_center
    )
