include("CommonDefs.jl")
module Variants_Pred

using Logging
using CSV
using JLD2
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

const INPUT_JLD2_FILEPATHS = Dict(
    :tokyo => "variants_latest_tokyo.jld2",
    :osaka => "variants_latest_osaka.jld2",
)
@insp INPUT_JLD2_FILEPATHS

const CONFIRMED_CSV_FILEPATHS = Dict(
    :tokyo => "tokyo_confirmed5_latest.csv",
)
@insp CONFIRMED_CSV_FILEPATHS

@info "--------"
@info "対象となる変異株名（区分は度々変更されるため区分の世代別）"

const VARIANT_NAMES_v_g = [
    ["BA.2", "BA.2.75", "BA.5", "BF.7", "BN.1", "BQ.1", "BQ.1.1", "XBB_old"],
    ["BA.2", "BA.2.75", "BA.5", "BF.7", "BN.1", "BQ.1", "BQ.1.1", "XBB.1.5", "XBB+1.9.1"],
    ["BA.2", "BA.2.75", "BA.5", "BF.7", "BN.1", "BQ.1", "BQ.1.1", "XBB.1.5", "XBB.1.9.1", "XBB"],
    ["BA.2", "BA.2.75", "BA.5", "BF.7", "BN.1", "BQ.1", "BQ.1.1", "XBB.1.5", "XBB.1.9.1", "XBB.1.16", "XBB"],
]
const VARIANT_NAMES_NUM_OF_GENERATIONS = length(VARIANT_NAMES_v_g)
const VARIANT_NAMES_LATEST_v = VARIANT_NAMES_v_g[VARIANT_NAMES_NUM_OF_GENERATIONS]
@insp VARIANT_NAMES_NUM_OF_GENERATIONS
@insp VARIANT_NAMES_LATEST_v

@info "基準変異株名"
const BASE_VARIANT_NAME = "BA.5"
@assert all(BASE_VARIANT_NAME .∈ VARIANT_NAMES_v_g)
@insp BASE_VARIANT_NAME

@info "--------"
@info "フィッティング時刻範囲"

const FIT_DATE_START = Date("2023-01-22")
const FIT_DATE_END   = Date("2023-02-11")
@insp FIT_DATE_START, FIT_DATE_END
const FIT_TVALUE_START = date_to_value(FIT_DATE_START; noon=false)
const FIT_TVALUE_END   = date_to_value(FIT_DATE_END;   noon=false)
@insp FIT_TVALUE_START, FIT_TVALUE_END

@info "--------"
@info "予測期間"

const PREDICTION_DATE_START = Date("2023-01-01")
const PREDICTION_DATE_END   = Date("2023-07-01")
@insp PREDICTION_DATE_START, PREDICTION_DATE_END
const PREDICTION_DATES_pt   = PREDICTION_DATE_START:Day(1):PREDICTION_DATE_END
const PREDICTION_TVALUES_pt = date_to_value.(PREDICTION_DATES_pt; noon=false)
@insp PREDICTION_DATES_pt
@insp length(PREDICTION_TVALUES_pt)

@info "========"
@info "関数定義"
@info "  主要データは module 内の全域変数により共有される"
@info "  （module をクラスの単一インスタンスのように扱う）"

@info "--------"
@info "データ読み込み用関数"

Variants_DF = nothing
LogitReg_DF = nothing

# 変異株回帰 jld2 データ読み込み
function load_jld2(filepath)
    @info "JLD2 ファイル $(filepath) を読み込み、全域変数 Variants_DF, LogitReg_DF を設定"
    jld2_data = load(filepath)
    global Variants_DF = jld2_data["Variants_DF"]
    global LogitReg_DF = jld2_data["LogitReg_DF"]
    @insp nrow(Variants_DF)
    @insp nrow(LogitReg_DF), names(LogitReg_DF)
end
@insp load_jld2

Confirmed_DF = nothing

# 感染確認者数 csv データ読み込み
function load_confirmed_csv(filepath)
    @info "CSV ファイル $filepath を読み込み、必要な列を選別して confirmed_df を設定"
    csv_df = CSV.read(filepath, DataFrame; missingstring="")
    confirmed_df = DataFrame()
    @assert has_name(csv_df, "t")
    @assert has_name(csv_df, "dtime")
    @assert has_name(csv_df, "Δlzs")
    confirmed_df.tvalue   = csv_df.t
    confirmed_df.datetime = csv_df.dtime
    confirmed_df.lgrowth  = csv_df.Δlzs
    global Confirmed_DF = confirmed_df
    @insp nrow(Confirmed_DF), names(Confirmed_DF)
end
@insp load_confirmed_csv

@info "--------"
@info "対数増加率算出関数"

# パラメーターから求めた対数増加率（定数分の不定性がある）
# 数学的関係について docs/varlogit.md 参照
function estimate_lgrowth(t, αs, βs)
    odds = exp.(αs .+ βs * t)
    return sum(βs .* odds) / (1.0 + sum(odds))
end
@insp estimate_lgrowth

@info "--------"
@info "メイン計算ルーチン"

# run() が設定する module 内グローバル変数
RegionSymbol              = nothing # 領域シンボル
ConfirmedTValues_ct       = nothing # 実データ時刻リスト
ConfirmedLGrowths_ct      = nothing # 実データ対数増加率
BaseLGrowthEstimated      = nothing # 基準株対数増加率推定値
BaseLGrowthSD             = nothing # 基準株対数増加率推定値標準偏差
PredictedLGrowths_pt      = nothing # 対数増加率時間推移予測
VariantTValues_vt         = nothing # 変異株別実データ時刻リスト（区分世代共通）
VariantProportions_vt_v_g = nothing # 区分世代別変異株別割合実データ
PredictedAmounts_pt_v     = nothing # 変異株別相対量時間推移予測
PredictedProportions_pt_v = nothing # 変異株別割合時間推移予測

function run(region_symbol)
    @info "========"
    @info "変異株回帰データから感染力の変化に基づく今後の対数増加率を予測し、"
    @info "感染確認者数対数増加率の実データと比較するプロットを作成"
    @info "--------"
    @info "データ読み込み"
    load_jld2(INPUT_JLD2_FILEPATHS[region_symbol])
    load_confirmed_csv(CONFIRMED_CSV_FILEPATHS[region_symbol])
    @info "実データの対数増加率"
    confirmed_tvalues_ct, confirmed_lgrowths_ct = (function()
        df = filter(row -> !ismissing(row.lgrowth), Confirmed_DF)
        return df.tvalue, df.lgrowth
    end)()
    @insp length(confirmed_tvalues_ct)
    @info "実データをフィットさせる範囲に制限した時刻と対数増加率"
    fit_tvalues_ft, fit_lgrowths_ft = ((tv_ct, lg_ct) ->
    begin
        # ビットベクトルによる添字制限
        ft_ct = (FIT_TVALUE_START .≤ tv_ct .≤ FIT_TVALUE_END) .& is_regular.(lg_ct)
        return tv_ct[ft_ct], lg_ct[ft_ct]
    end
    )(confirmed_tvalues_ct, confirmed_lgrowths_ct)
    @insp length(fit_tvalues_ft)
    @assert length(fit_tvalues_ft) > 0
    @info "--------"
    @info "基準株が一致する回帰パラメーターを取り出す"
    α_v, β_v = (() ->
    begin
        α_v = []; sizehint!(α_v, length(VARIANT_NAMES_LATEST_v))
        β_v = []; sizehint!(β_v, length(VARIANT_NAMES_LATEST_v))
        for vn ∈ VARIANT_NAMES_LATEST_v
            if vn == BASE_VARIANT_NAME
                push!(α_v, 0.0)
                push!(β_v, 0.0)
            else
                df = filter(row -> (row.base_variant == BASE_VARIANT_NAME) && (row.variant == vn), LogitReg_DF)
                @assert nrow(df) == 1
                push!(α_v, df.α_j[1])
                push!(β_v, df.β_j[1])
            end
        end
        return α_v, β_v
    end
    )()
    @insp length(α_v), length(β_v)
    @info "回帰パラメーターから求めたフィッテイング範囲の相対対数増加率と基準株対数増加率の推定"
    base_lgrowth_estimated, base_lgrowth_sd = ((α_v, β_v, lg_ft, tv_ft) ->
    begin
        elg_ft = map(t -> estimate_lgrowth(t, α_v, β_v), tv_ft)
        dlg_ft = lg_ft .- elg_ft
        blg_est = mean(dlg_ft)
        blg_sd  = stdm(dlg_ft, blg_est)
        return blg_est, blg_sd
    end
    )(α_v, β_v, fit_lgrowths_ft, fit_tvalues_ft)
    @insp base_lgrowth_estimated, base_lgrowth_sd
    @info "--------"
    @info "変異株の割合の変化にともなう感染力の変化に基づく対数増加率推定"
    predicted_lgrowths_pt = map(t -> base_lgrowth_estimated + estimate_lgrowth(t, α_v, β_v), PREDICTION_TVALUES_pt)
    @insp length(predicted_lgrowths_pt)
    @info "-------"
    @info "変異株別実データ時刻（区分世代共通）"
    variant_tvalues_vt = (() ->
    begin
        tvs_vt = date_to_value.(Variants_DF.date_start)
        tve_vt = date_to_value.(Variants_DF.date_end .+ Day(1))
        return (tvs_vt .+ tve_vt) / 2.0 # ビンの中央値
    end
    )()
    @insp length(variant_tvalues_vt)
    @info "--------"
    @info "区分世代別各変異株の割合実データ"
    variant_proportions_vt_v_g = (() ->
    begin
        p_vt_v_g = []
        for g ∈ 1:VARIANT_NAMES_NUM_OF_GENERATIONS
            n_vt_v = [Variants_DF[:,vn] for vn ∈ VARIANT_NAMES_v_g[g]]
            s_vt   = reduce(.+, n_vt_v)
            p_vt_v = [n_vt_v[v] ./ s_vt for v ∈ 1:length(VARIANT_NAMES_v_g[g])]
            push!(p_vt_v_g, p_vt_v)
        end
        return p_vt_v_g
    end
    )()
    @insp length.(variant_proportions_vt_v_g)
    @info "--------"
    @info "変異株ごとの相対量・割合の時間推移推定"
    predicted_amounts_pt_v, predicted_proportions_pt_v = ((blg_est) ->
    begin
        n_pt(α, β) = [exp(blg_est + α + β * t) for t ∈ PREDICTION_TVALUES_pt]
        n_pt_v = [n_pt(α_v[v], β_v[v]) for v ∈ 1:length(VARIANT_NAMES_LATEST_v)]
        s_pt   = reduce(.+, n_pt_v)
        p_pt_v = [n_pt_v[v] ./ s_pt for v ∈ 1:length(VARIANT_NAMES_LATEST_v)]
        return n_pt_v, p_pt_v
    end
    )(base_lgrowth_estimated)
    @insp length(predicted_amounts_pt_v), length(predicted_proportions_pt_v)
    @info "--------"
    @info "全域変数設定"
    global RegionSymbol              = region_symbol
    global ConfirmedTValues_ct       = confirmed_tvalues_ct
    global ConfirmedLGrowths_ct      = confirmed_lgrowths_ct
    global BaseLGrowthEstimated      = base_lgrowth_estimated
    global BaseLGrowthSD             = base_lgrowth_sd
    global PredictedLGrowths_pt      = predicted_lgrowths_pt
    global VariantTValues_vt         = variant_tvalues_vt
    global VariantProportions_vt_v_g = variant_proportions_vt_v_g
    global PredictedAmounts_pt_v     = predicted_amounts_pt_v
    global PredictedProportions_pt_v = predicted_proportions_pt_v
end
@insp run

@info "========"
@info ""
@info "Variants_Pred.run(region_symbol) として実行する"
@info "region_symbol はシンボル :tokyo または :osaka"
@info "module 内全域変数が設定される"
@info ""
@info "========"

@info "========"
@info "プロット用定数"

@info "--------"
@info "プロット時間範囲"

const PLOT_DATE_START = Date("2022-12-31")
const PLOT_DATE_END   = Date("2023-07-02")
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

@inline RGB256(r, g, b) = RGB(r/255, g/255, b/255)
@insp RGB256
const VARIANT_COLORS_vn = Dict(
    "BA.2"      => RGB256(236, 126,  42),
    "BA.5"      => RGB256(156, 196, 230),
    "BF.7"      => RGB256(255, 192,   0),
    "BN.1"      => RGB256(196,  90,  16),
    "BQ.1.1"    => RGB256( 46, 116, 182),
    "BA.2.75"   => RGB256(255, 104, 214),
    "BQ.1"      => RGB256(168, 208, 142),
    "XBB"       => RGB256(132, 152, 176),
    "XBB.1.5"   => RGB256(112,  44, 160),
    "XBB.1.9.1" => RGB256( 84, 132,  52),
    "XBB.1.16"  => RGB256(128,  96,   0),
    "XBB+1.9.1" => RGB256( 98,  88, 106),
)
@insp length(VARIANT_COLORS_vn)

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

@info "積み上げ面グラフ描画関数（ribbon を利用する）"
function stacked_area!(p::UPlot, x, ys; lc, fcs)
    @assert length(ys) == length(fcs)
    nj = length(ys)
    z  = fill(0.0, length(ys[1]))
    ays = accumulate(.+, ys)
    for j ∈ 1:nj
        plot!(p, x, ays[j]; ribbon=(ys[j], z), lc=lc, fc=fcs[j], label=:none)
    end
    return p
end
@insp stacked_area!

@info "--------"
@info "確認感染者数データと変異株回帰パラメーターからの対数増加率比較プロット関数"

function p_log_growth_comparison(
    date_start = PLOT_DATE_START, date_end = PLOT_DATE_END,
)
    region_name = Dict(:tokyo => "東京", :osaka => "大阪")[RegionSymbol]
    p = plot(
        size=(640, 600),
        framestyle=:box,
        bottom_margin=8px, left_margin=8px, top_margin=8px, right_margin=8px,
        title="感染確認者数の対数増加率と変異株比から推定される対数増加率の比較（$(region_name)）",
        legend=:topleft,
        fontfamily="Meiryo",
    )
    # x 軸設定
    x_axis_time!(p; date_start=date_start, date_end=date_end)
    x_axis_lims = collect(Plots.xlims(p))
    plot!(p, collect(Plots.xlims(p)), [0.0], label=:none, color=:black)
    # y 軸設定
    # 最小・最大値をデータの範囲から決定
    y_min, y_max = ((v_t) -> begin
        v_T = v_t[is_regular.(v_t)]
        lmin = min(0.0, minimum(v_T)) - 0.05
        lmax = max(0.0, maximum(v_T)) + 0.05
        return lmin, lmax
    end)(PredictedLGrowths_pt)
    @insp y_min, y_max
    # y 軸刻みの設定
    y_ticks = ((lmin, lmax) -> begin
        modfu(x) = begin ix = floor(x); (x - ix, ix) end
        fp, ip = modfu(log10(lmax - lmin))
        step = ((fp < log10(2.0)) ? 0.2 : (fp < log10(5.0)) ? 0.5 : 1.0) * 10^ip
        return filter(v -> lmin ≤ v ≤ lmax, (-10.0 * step):step:(10.0 * step))
    end)(y_min, y_max)
    @insp y_ticks
    ylims!(p, y_min, y_max)
    yticks!(p, y_ticks, map(v -> replace(@sprintf("%g", v), "-" => "－"), y_ticks))
    ylabel!(p, "感染者 対数増加率 [/日]")
    # グリッド線と上辺をプロットの要素として重ね書き（twinx() のバクにより消されるため）
    map(y -> if y ≠ 0 plot!(p, x_axis_lims, [y]; label=:none, lc=:gray90) end, y_ticks)
    plot!(p, x_axis_lims, [0.0],   label=:none, color=:black)
    plot!(p, x_axis_lims, [y_max]; label=:none, color=:black)
    # プロット
    plot!(
        p, ConfirmedTValues_ct, ConfirmedLGrowths_ct;
        label="感染確認者数にもとづく実際の対数増加率",
        la=0.8, lc=COLORS[3], lw=2
    )
    plot!(
        p, PREDICTION_TVALUES_pt, PredictedLGrowths_pt;
        label="変異株の感染力の違いとして推定される対数増加率",
        la=0.8, lc=COLORS[9],
    )
    # アノテーション
    annotate!(p, rx(p, 0.45), ry(p, 0.08),
        text(
            """
            データソース: 変異株比の元データは東京都資料より.
            検出数が小さいいくつかの変異株の区分を除く.
            感染確認者数は厚労省「新規陽性者数の推移（日別）」より.
            変異株比からの対数増加率推移の推定は定数分の任意性がある.
            基準とした $(BASE_VARIANT_NAME) の増加率を $(@sprintf("%.3f", BaseLGrowthEstimated)) とした場合.
            """,
            font("Meiryo",7), RGB(0.3,0.3,0.3), :left
        )
    )
    # 右 y 軸
    wtoy(w) = log(w) / 7.0
    ytow(y) = exp(7.0 * y)
    y2_bmin, y2_bmax = ytow.([y_min, y_max])
    @insp y2_bmin, y2_bmax
    y2_bticks = (function(bmin, bmax)
        bticks = [b for i ∈ -1:1 for b ∈ [1., 1.2, 1.5, 2., 3., 4., 5., 6., 7., 8., 9.] .* 10.0^i]
        return filter(t -> bmin ≤ t ≤ bmax, bticks)
    end)(y2_bmin, y2_bmax)
    p2 = twinx(p)
    ylims!(p2, y_min, y_max)
    yticks!(p2, wtoy.(y2_bticks), map(v -> (@sprintf("%g", v) * " "^4)[1:5], y2_bticks))
    ylabel!(p2, "週あたり拡大率（対数目盛り）")
    return p
end
@insp p_log_growth_comparison

@info "--------"
@info "変異株回帰パラメーターから推定される変異株の割合の積み上げ面プロット関数"

function p_stacked_area_variant_proportions(
    date_start = PLOT_DATE_START, date_end = PLOT_DATE_END,
)
    region_name = Dict(:tokyo => "東京", :osaka => "大阪")[RegionSymbol]
    p = plot(
        size=(640, 600),
        framestyle=:box,
        bottom_margin=8px, left_margin=8px, top_margin=8px, right_margin=8px,
        title="変異株比から推定される変異株の割合の推移（$(region_name)）",
        legend=:topleft,
        fontfamily="Meiryo",
    )
    # データ
    vn_vr     = reverse(VARIANT_NAMES_LATEST_v)
    p_pt_vr   = reverse(PredictedProportions_pt_v)
    a_pt_vr   = accumulate(.+, p_pt_vr)
    p_vt_vr_g = reverse.(VariantProportions_vt_v_g)
    a_vt_vr_g = accumulate.(.+, p_vt_vr_g)
    # x 軸設定
    x_axis_time!(p; date_start=date_start, date_end=date_end)
    # y 軸設定
    ylims!(p, 0.0, 1.0)
    yticks!(p, 0.0:0.2:1.0, map(pc -> @sprintf("%d", pc), 0:20:100))
    ylabel!(p, "変異株割合 [%]")
    # プロット
    stacked_area!(
        p, PREDICTION_TVALUES_pt, p_pt_vr;
        lc=:black, fcs=[VARIANT_COLORS_vn[vn] for vn ∈ vn_vr],
    )
    for g ∈ 1:VARIANT_NAMES_NUM_OF_GENERATIONS
        for vr ∈ 1:(length(VARIANT_NAMES_v_g[g])-1)
            plot!(p, VariantTValues_vt, a_vt_vr_g[g][vr]; lc=:black, la=0.5, ls=:dot, label=:none)
            scatter!(p, VariantTValues_vt, a_vt_vr_g[g][vr]; c=:gray, alpha=0.3, label=:none)
        end
    end
    # ラベル
    ptl = div(1 * length(PREDICTION_TVALUES_pt), 4) # 整数除算
    ptc = div(2 * length(PREDICTION_TVALUES_pt), 4)
    ptr = div(3 * length(PREDICTION_TVALUES_pt), 4)
    @insp ptl, ptc, ptr
    for (vr, var) ∈ enumerate(vn_vr)
        a = argmax([p_pt_vr[vr][ptl], p_pt_vr[vr][ptc], p_pt_vr[vr][ptr]])
        pta = [ptl, ptc, ptr][a]
        lx = a / 4.0
        ly = a_pt_vr[vr][pta] - p_pt_vr[vr][pta] / 2.0
        annotate!(p, rx(p, lx), ry(p, ly), text(var, font("Meiryo",8), :black, :center))
    end
    # アノテーション
    annotate!(p, rx(p, 0.50), ry(p, 0.04),
        text(
            """
            データソース: 変異株比の元データは東京都資料より.
            「組替体」の区分を除く. 丸印は実際の検出数の割合.
            """,
            font("Meiryo",7), RGB(0.3,0.3,0.3), :left
        )
    )
end

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
    @info "log_growth_comparison"
    p[:log_growth_comparison] = p_log_growth_comparison()
    @info "stack_area_variant_proportions"
    p[:stack_area_variant_proportions] = p_stacked_area_variant_proportions()
    @info "--------"
    @info "全域変数設定"
    global P = p
    return P
end
@insp generate

@info "========"
@info ""
@info "Variants_Pred.generate(:tokyo) または Variants_Pred.generate(:osaka) として実行する"
@info "メイン計算ルーチン run が実行されていなければ、内部で呼び出す"
@info "プロットは全域変数 P とともに、返り値として返される Plots.Plot を値として持つ Dict 型変数"
@info ""
@info "========"

end #module

##