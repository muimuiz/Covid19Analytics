include("CommonDefs.jl")
module Sentinel

using Logging
using CSV
using DataFrames
using Dates
using Statistics: mean, stdm
using Printf
using Plots
using Plots.PlotMeasures

using ..CommonDefs

@info "========"
@info "モジュール"
@insp @__MODULE__

@info "--------"
@info "作業ディレクトリ"
@insp pwd()

@info "========"
@info "定数"

@info "--------"
@info "解析データ"

const CONFIRMED_CSV_FILEPATHS = Dict(
    :tokyo => "tokyo_confirmed5_latest.csv",
)
@insp CONFIRMED_CSV_FILEPATHS

const SENTINEL_CSV_FILEPATHS = Dict(
    :tokyo => "東京定点患者報告数.csv",
)
@insp SENTINEL_CSV_FILEPATHS

@info "========"
@info "関数定義"
@info "  主要データは module 内の全域変数により共有される"
@info "  （module をクラスの単一インスタンスのように扱う）"

@info "--------"
@info "データ読み込み用関数"

Confirmed_DF = nothing

# 感染確認者数 csv データ読み込み
function load_confirmed_csv(filepath)
    @info "CSV ファイル $filepath を読み込み、必要な列を選別して Confirmed_DF を設定"
    csv_df = CSV.read(filepath, DataFrame; missingstring="")
    confirmed_df = DataFrame()
    @assert has_name(csv_df, "t")
    @assert has_name(csv_df, "dtime")
    @assert has_name(csv_df, "x")
    @assert has_name(csv_df, "zs")
    @assert has_name(csv_df, "lzs")
    @assert has_name(csv_df, "Δlzs")
    confirmed_df.tvalue    = csv_df.t
    confirmed_df.datetime  = csv_df.dtime
    confirmed_df.confirmed = csv_df.x
    confirmed_df.smoothed  = csv_df.zs
    confirmed_df.lsmoothed = csv_df.lzs
    confirmed_df.lgrowth   = csv_df.Δlzs
    global Confirmed_DF = confirmed_df
    @insp nrow(Confirmed_DF), names(Confirmed_DF)
end
@insp load_confirmed_csv

Sentinel_DF = nothing

# 定点感染報告者数 csv データ読み込み
function load_sentinel_csv(filepath)
    @info "CSV ファイル $filepath を読み込み、必要な列を選別して Sentinel_DF を設定"
    csv_df = CSV.read(filepath, DataFrame; missingstring="")
    sentinel_df = DataFrame()
    @assert has_name(csv_df, "date_start")
    @assert has_name(csv_df, "date_end")
    @assert has_name(csv_df, "count")
    sentinel_df.date_start   = csv_df.date_start
    sentinel_df.date_end     = csv_df.date_end
    sentinel_df.tvalue_start = date_to_value.(sentinel_df.date_start)
    sentinel_df.tvalue_end   = date_to_value.(sentinel_df.date_end + Day(1))
    sentinel_df.count        = csv_df.count
    sentinel_df.lcount       = log.(sentinel_df.count)
    global Sentinel_DF = sentinel_df
    @insp nrow(Sentinel_DF), names(Sentinel_DF)
end
@insp load_sentinel_csv

@info "--------"
@info "メイン計算ルーチン"

RegionSymbol = nothing
LogRatio     = nothing
LogRatioSD   = nothing

function run(region_symbol)
    @info "========"
    @info "週別の定点報告者数を日別で公表されていた感染確認者数にフィットさせる"
    @info "--------"
    @info "データ読み込み"
    load_confirmed_csv(CONFIRMED_CSV_FILEPATHS[region_symbol])
    load_sentinel_csv(SENTINEL_CSV_FILEPATHS[region_symbol])
    @info "--------"
    @info "フィッティング"
    log_ratio, log_ratio_sd = (function()
        lrs = Float64[]
        for srow in eachrow(Sentinel_DF)
            cs = filter(
                crow -> srow.tvalue_start ≤ crow.tvalue < srow.tvalue_end && is_regular(crow.confirmed),
                Confirmed_DF
            ).confirmed
            if length(cs) ≥ Dates.value(srow.date_end - srow.date_start)
                # 週に渡る日別データがそろっているときのみ
                push!(lrs, srow.lcount - log(mean(cs)))
            end
        end
        @assert length(lrs) ≥ 1
        @insp length(lrs)
        lrm  = mean(lrs)
        lrsd = stdm(lrs, lrm)
        return lrm, lrsd
    end)()
    @insp log_ratio, log_ratio_sd
    @insp exp(log_ratio), exp(-log_ratio)
    @info "--------"
    @info "全域変数設定"
    global RegionSymbol = region_symbol
    global LogRatio     = log_ratio
    global LogRatioSD   = log_ratio_sd
end
@insp run

@info "========"
@info ""
@info "Sentinel.run(region_symbol) として実行する"
@info "region_symbol は現在 :tokyo のみ"
@info ""
@info "========"

@info "========"
@info "プロット用定数"

@info "--------"
@info "プロット時間範囲"

const PLOT_DATE_START = Date("2022-08-31")
const PLOT_DATE_END   = Date("2023-09-02")
@insp PLOT_DATE_START, PLOT_DATE_END

@info "--------"
@info "プロットとサブプロットの合併型定義"

UPlot = Union{Plots.Plot,Plots.Subplot}
@insp UPlot

@info "--------"
@info "描画色"

const COLORS = [
    HSL(240.0, 1.0, 0.55), # 1 blue
    HSL(180.0, 1.0, 0.40), # 2 cyan
    HSL(120.0, 1.0, 0.30), # 3 green
    HSL( 60.0, 1.0, 0.35), # 4 yellow
    HSL( 40.0, 1.0, 0.40), # 5
    HSL( 20.0, 1.0, 0.50), # 6
    HSL(  0.0, 1.0, 0.55), # 7 red
    HSL(320.0, 1.0, 0.40), # 8
    HSL(280.0, 1.0, 0.40), # 9
]
@insp length(COLORS)

const C = Dict(
    :x  => COLORS[1],  :xm => COLORS[3],  :r  => COLORS[5],
    :ξ  => COLORS[4],  :ξm => COLORS[6],
    :η  => COLORS[7],  :ηs => COLORS[9],
    :ζ  => COLORS[1],  :ζs => COLORS[3],
    :w  => COLORS[1],  :nh => COLORS[5],
    :sc => COLORS[2],  :zs => COLORS[8],
)
@insp length(C)

@info "========"
@info "プロット用関数定義"

function _rl_pos(l::Tuple, r; logscale=false)
    if logscale
        ll = log.(l)
        a = exp((1.0-r) * ll[1] + r * ll[2])
    else
        a = (1.0-r) * l[1] + r * l[2]
    end
    return a
end
@insp _rl_pos
@inline rx(p::UPlot, r; logscale=false) = _rl_pos(Plots.xlims(p), r; logscale=logscale)
@inline ry(p::UPlot, r; logscale=false) = _rl_pos(Plots.ylims(p), r; logscale=logscale)
@insp rx
@insp ry

function x_axis_time!(
    p::UPlot;
    recent = nothing,
    date_start = PLOT_DATE_START,
    date_end   = PLOT_DATE_END,
)
    if !isnothing(recent) date_start = max(date_start, date_end - Day(recent)) end 
    @insp date_start, date_end
    tvalue_start = dtime_to_value(DateTime(date_start))
    tvalue_end   = dtime_to_value(DateTime(date_end))
    xlims!(p, tvalue_start, tvalue_end)
    if isa(p, Plots.Plot)
        days = date_end - date_start
        if     days < Day(14)
            r_ticks = date_start:Day(1):date_end # 毎日
        elseif days < Day(3 * 30)
            r_ticks = Dates.tonext(date_start, Dates.Sunday):Day(7):date_end # 毎週日曜
        else
            r_ticks = Dates.tonext((d) -> (day(d) == 1), date_start):Month(1):date_end # 毎月 1 日
        end
        @assert length(r_ticks) ≤ 30
        ticks_dict = Dict{DateTime, String}()
        year_prev = nothing
        for (i, date) in enumerate(r_ticks)
            if year_prev ≠ year(date)
                ticks_dict[DateTime(date)] = Dates.format(date, "yyyy\u2010mm\u2010dd")
            else
                ticks_dict[DateTime(date)] = Dates.format(date, "mm\u2010dd")
            end
            year_prev = year(date)
        end
        ts     = Float64[]
        labels = String[]
        for d in sort(collect(keys(ticks_dict)))
            t = dtime_to_value(DateTime(d) + Hour(12))
            if tvalue_start ≤ t ≤ tvalue_end
                push!(ts, t)
                push!(labels, ticks_dict[d])
            end
        end
        xticks!(p, ts, labels; xtickfontrotation=90.0)
    end
    return p
end
@insp x_axis_time!

@info "--------"
@info "定点報告者数と感染確認者数のプロット"

function p_log_sentinel_and_confirmed(;
    recent = 240,
    date_start = PLOT_DATE_START,
    date_end   = PLOT_DATE_END,
)
    region_name = Dict(:tokyo => "東京")[RegionSymbol]
    p = plot(
        size=(640, 600),
        framestyle=:box,
        bottom_margin=8px, left_margin=8px, top_margin=8px, right_margin=8px,
        title="$(region_name) COVID-19 定点感染報告者数および過去の感染確認者数",
        legend=:topleft,
        fontfamily="Meiryo",
    )
    # x 軸設定
    x_axis_time!(p; date_start=date_start, date_end=date_end, recent=recent)
    x_axis_lims = collect(Plots.xlims(p))
    t_start, t_end = x_axis_lims
    @insp t_start, t_end
    # データ
    cdf = filter(row -> t_start ≤ row.tvalue ≤ t_end, Confirmed_DF)
    sdf = filter(row -> t_start ≤ row.tvalue_end && row.tvalue_start ≤ t_end, Sentinel_DF)
    # y 軸設定
    # 最小・最大値をデータの範囲から決定
    function minmax(v_t)
        v_T = v_t[is_regular.(v_t)]
        lmin = minimum(v_T) - 1.5
        lmax = maximum(v_T) + 1.5
        return lmin, lmax
    end
    y_cmin, y_cmax = minmax(cdf.lsmoothed)
    y_smin, y_smax = minmax(sdf.lcount) .- LogRatio
    y_min = min(y_cmin, y_smin)
    y_max = max(y_cmax, y_smax)
    if y_max - y_min < log(100.0)
        y_mid = (y_max + y_min) / 2.0
        y_max = y_mid + log(10.0)
        y_min = y_mid - log(10.0)
    end
    @insp y_min, y_max
    ylims!(p, y_min, y_max)
    ylabel!(p, "感染確認者数（対数目盛り）[人/日]")
    # y 軸刻みの設定
    y_bticks = filter(
        v -> y_min ≤ log(v) ≤ y_max,
        [b for i ∈ 0:6 for b ∈ [1,2,4,6,8] .* 10^i]
    )
    yticks!(p, log.(y_bticks), map(v -> @sprintf("%6d", v), y_bticks))
    # グリッド線と上辺をプロットの要素として重ね書き（twinx() のバクにより消されるため）
    map(y -> plot!(p, x_axis_lims, [y]; label=:none, lc=:gray90), log.(y_bticks))
    plot!(p, x_axis_lims, [y_max]; label=:none, lc=:black)
    # 凡例ラベル（順序を保証するため空のプロットでまとめて登録する）
    plot!(p, [], []; label="日別感染確認者数（公表日）", la=0.2, lc=C[:x], m=:circle, mc=C[:x])
    plot!(p, [], []; label="トレンド（曜日・休日補正、LOESS 平滑化）", lc=C[:zs])
    scatter!(p, [], []; label="定点感染報告者数（右目盛り）", lc=C[:ξm], m=:square, color=C[:ξm], edgecolor=C[:ξm])
    # データ
    t_ct   = cdf.tvalue
    lx_ct  = log.(cdf.confirmed)
    lzs_ct = cdf.lsmoothed
    ts_st  = sdf.tvalue_start
    te_st  = sdf.tvalue_end
    t_st   = (ts_st .+ te_st) / 2.0
    lc_st  = sdf.lcount
    # プロット
    plot!(p, t_ct, lx_ct; label=:none, la=0.2, lc=C[:x], m=:circle, mc=C[:x])
    plot!(p, t_ct, lzs_ct; label=:none, lc=C[:zs])
    scatter!(p, t_st, lc_st .- LogRatio;
        m=:square,
        xerror=(t_st .- ts_st, te_st .- t_st),
        yerror=([log(1.1)], [log(1.1)]),
        label=:none, color=C[:ξm], edgecolor=C[:ξm]
    )
    # アノテーション
    data_source = Dict(
        :tokyo => "東京都福祉保健局 新型コロナウイルス感染症モニタリング分析資料",
    )[RegionSymbol]
    annotate!(p, rx(p, 0.02), ry(p, 0.05),
        text(
            """
            データソース：$(data_source).
            2023-05-08 までの日別データは、厚労省 新規陽性者数の推移（日別）より.
            データが重なる範囲においてフィッティングし、定点データの値を $(@sprintf("%.1f", exp(-LogRatio))) 倍した.
            縦軸エラーバーは、定点データの誤差目安に合わせて一律 ±10% の幅を示している.
            """,
            font("Meiryo", 6), RGB(0.3,0.3,0.3), :left
        )
    )
    # 右 y 軸
    p2 = twinx(p)
    y2_min = y_min + LogRatio
    y2_max = y_max + LogRatio
    @insp y2_min, y2_max
    ylims!(p2, y2_min, y2_max)
    ylabel!(p2, "定点医療機関当たり患者報告数（対数目盛り）[人/定点/週]")
    y2_bticks = filter(
        v -> y2_min ≤ log(v) ≤ y2_max,
        [b for i ∈ -4:3 for b ∈ [1,2,4,6,8] .* 10.0^i]
    )
    @insp y2_bticks
    yticks!(p2, log.(y2_bticks), map(v -> @sprintf("%g", v), y2_bticks))
    return p
end
@insp p_log_sentinel_and_confirmed

@info "--------"
@info "プロット用メイン関数定義"

# グローバル変数
P = nothing

function generate(region_symbol = nothing)
    @info "========"
    @info "プロット生成"
    @info "--------"
    @info "計算結果が生成されていなければ、計算ルーチンを呼び出す"
    if isnothing(RegionSymbol) run(region_symbol) end
    @info "--------"
    @info "Plots 初期化"
    pyplot(
        titlefont=font("Meiryo",9),
        guidefont=font("Meiryo",8),
        tickfont=font("Meiryo",8)
    )
    p = Dict{Symbol,Plots.Plot}()
    @info "--------"
    @info "プロット生成"
    @info "log_sentinel_and_confirmed"
    p[:log_sentinel_and_confirmed] = p_log_sentinel_and_confirmed(
        recent = 210, date_end = Sentinel_DF.date_end[end] + Day(7)
    )
    @info "--------"
    @info "全域変数設定"
    global P = p
    return P
end
@insp generate

@info "========"
@info ""
@info "Sentinel.generate(:tokyo) として実行する"
@info "メイン計算ルーチン run が実行されていなければ、内部で呼び出す"
@info ""
@info "========"

end #module

##