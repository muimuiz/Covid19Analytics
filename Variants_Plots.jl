include("CommonDefs.jl")
module Variants_Plots

using Logging
using JLD2
using DataFrames
using Dates
using Statistics: mean, stdm
using Printf
using LaTeXStrings
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
@info "指定可能な対象地域"

const CAPABLE_REGION_SYMBOLS = [:tokyo, :osaka]
@insp CAPABLE_REGION_SYMBOLS

@info "--------"
@info "解析データ"

const INPUT_JLD2_FILEPATHS = Dict(
    :tokyo => "variants_latest_tokyo.jld2",
    :osaka => "variants_latest_osaka.jld2",
)
@insp INPUT_JLD2_FILEPATHS

@info "--------"
@info "プロットする変異株"

const VARIANT_NAMES_TO_BE_PLOTTED_v = [
    "JN.1",
#=
#    "BA.2",
#    "BA.2.75",
    "BA.5",
    "BF.7",
    "BN.1",
#    "BQ.1",
    "BQ.1.1",
    "XBB",
    "XBB.1.5",
    "XBB.1.9.1",
    "XBB.1.9.2",
    "XBB.1.16",
#    "XBB+XBB.1.9.1+XBB.1.9.2+XBB.1.16",
=#
]
@insp VARIANT_NAMES_TO_BE_PLOTTED_v

@info "--------"
@info "プロット時間範囲"

const PLOT_DATE_START = Date("2023-10-31")
const PLOT_DATE_END   = Date("2024-03-02")
#=
const PLOT_DATE_START = Date("2022-10-31")
const PLOT_DATE_END   = Date("2023-06-02")
=#
@insp PLOT_DATE_START, PLOT_DATE_END

@info "--------"
@info "プロットとサブプロットの合併型定義"

UPlot = Union{Plots.Plot,Plots.Subplot}
@insp UPlot

@info "--------"
@info "画像出力ディレクトリ"

const FIGURE_DIRECTORY = "CurrentFigs/"
@insp FIGURE_DIRECTORY

@info "--------"
@info "描画色"

@inline RGB256(r, g, b) = RGB(r/255, g/255, b/255)
@insp RGB256
const VARIANT_COLORS_vn = Dict(
    "EG.5"    => RGB256(255, 124, 128),
    "JN.1"    => RGB256(155,  45, 163),
    "BA.2.86" => RGB256(188,  20, 120),
    "XBB.2.3" => RGB256(140, 193, 104),
    "others"  => RGB256(128, 128, 128),
#=
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
    "XBB.1.9.2" => RGB256(104, 142, 208),
    "XBB.1.16"  => RGB256(128,  96,   0),
    "XBB.1.16 (transient-free)"  => RGB256(128,  96,   0),
#    "XBB+XBB.1.9.1+XBB.1.9.2+XBB.1.16" => RGB256( 98,  88, 106),
=#
)
@insp length(VARIANT_COLORS_vn)

@info "========"
@info "関数定義"

RegionSymbol = nothing

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

#=
for key in keys(D)
    skey  = string(key)
    sexpr = "const $(skey) = D[:$(skey)]"
    eval(Meta.parse(sexpr))
end
=#

@info "--------"
@info "時刻区間代表値計算用関数"

@inline λ_logistic(t, α, β) = α + β * t
@insp λ_logistic

function λ_star(λs, λe)
    if λs == λe return λs end 
    if λs + λe ≤ 0.0
        if λs > -300.0 || λe > -300.0
            return logit((softplus(λe) - softplus(λs)) / (λe - λs))
        else
            return logsubexp(λe, λs) - log(abs(λe - λs)) # 近似式
            #return log1mexp(-abs(λe - λs)) - log(abs(λe - λs)) + max(λs, λe)
        end
    end
    return -λ_star(-λs, -λe)
end
@insp λ_star

function t_star(ts, te, α, β)
    @assert ts < te
    if 1e-12 < abs(β)
        λ = λ_star(λ_logistic(ts, α, β), λ_logistic(te, α, β))
        t = (λ - α) / β
    else
        t = (ts + te) / 2.0
    end
    @assert ts ≤ t ≤ te
    return t
end
@insp t_star

@info "--------"
@info "プロット用関数"

function _rl_pos(l::Tuple, r; logscale=false)
    if logscale
        ll = log.(l)
        a = exp((1.0-r) * ll[1] + r * ll[2])
    else
        a = (1.0-r) * l[1] + r * l[2]
    end
    return a
end
@info _rl_pos
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
@info "基準株に対する他の株のロジットの時間推移プロット関数"

function p_variant_logit_transitions_against_base_variant(;
    base_variant_name = "EG.5", #"BA.5",
    variant_names_to_be_plotted_v = VARIANT_NAMES_TO_BE_PLOTTED_v,
    date_start = PLOT_DATE_START,
    date_end   = PLOT_DATE_END,
    ymin = nothing,
    ymax = nothing,
    value_annotations = [],
)
    region_name = Dict(:tokyo => "東京", :osaka => "大阪")[RegionSymbol]
    p = plot(
        size=(640, 600),
        framestyle=:box,
        bottom_margin=8px, left_margin=8px, top_margin=8px, right_margin=8px,
        title="$(base_variant_name) に対する他株の検出数比の推移（$(region_name)）",
        legend=:topleft,
        fontfamily="Meiryo",
    )
    # データ
    ts_vt = date_to_value.(Date.(Variants_DF[:,"date_start"]); noon=false)
    te_vt = date_to_value.(Date.(Variants_DF[:,"date_end"]);   noon=false) .+ 1.0
    tc_vt = (ts_vt .+ te_vt) ./ 2.0
    λe_vt_vn = Dict{String, Vector}()
    λl_vt_vn = Dict{String, Vector}()
    λu_vt_vn = Dict{String, Vector}()
    for vn ∈ variant_names_to_be_plotted_v
        if vn == base_variant_name continue end
        if !has_name(Variants_DF, vn) continue end
        λn_for = "$(vn)_$(base_variant_name)_logit"
        λn_rev = "$(base_variant_name)_$(vn)_logit"
        if     has_name(Variants_DF, λn_for)
            λe_vt_vn[vn] =    Variants_DF[:,λn_for]
            λl_vt_vn[vn] =    Variants_DF[:,λn_for * "_cil"]
            λu_vt_vn[vn] =    Variants_DF[:,λn_for * "_ciu"]
        elseif has_name(Variants_DF, λn_rev)
            λe_vt_vn[vn] = .- Variants_DF[:,λn_rev]
            λl_vt_vn[vn] = .- Variants_DF[:,λn_rev * "_cil"]
            λu_vt_vn[vn] = .- Variants_DF[:,λn_rev * "_ciu"]
        else
            @warn "$(vn) と $(base_variant_name) に対するロジットデータがない"
        end
    end
    # x 軸設定
    x_axis_time!(p; date_start=date_start, date_end=date_end)
    x_lims = collect(Plots.xlims(p))
    # y 軸設定
    @insp λe_vt_vn
    ytol = 1.0 #0.2
    if isnothing(ymin) ymin0 = min(0.0, reduce(min, minimum.(skipmissing.(values(λe_vt_vn))))) end
    if isnothing(ymax) ymax0 = max(0.0, reduce(max, maximum.(skipmissing.(values(λe_vt_vn))))) end
    if isnothing(ymin) ymin = (1.0 + ytol) * ymin0 - ytol * ymax0 end
    if isnothing(ymax) ymax = (1.0 + ytol) * ymax0 - ytol * ymin0 end
    @insp ymin, ymax
    ylims!(p, ymin, ymax)
    y_ticks = yticks(p)[1][1]
    ylabel!(p, "$(base_variant_name) 株の検出数に対する他の株の検出数の比の自然対数（ロジット）")
    # グリッド線と上辺をプロットの要素として重ね書き（twinx() のバクにより消されるため）
    map(y -> if y ≠ 0 plot!(p, x_lims, [y]; label=:none, lc=:gray90) end, y_ticks)
    plot!(p, x_lims, [0.0], label=:none, color=:black)
    plot!(p, x_lims, [ymax]; label=:none, lc=:black)    
    # プロット
    for vn ∈ variant_names_to_be_plotted_v
        if vn == base_variant_name continue end
        # 回帰直線
        df = filter(row -> (row.variant == vn) && (row.base_variant == base_variant_name), LogitReg_DF)
        if nrow(df) == 0
            @warn "基準株 $(base_variant_name) に対する変異株 $(vn) の回帰データがない"
            continue
        end
        if nrow(df) ≠ 1
            @warn "基準株 $(base_variant_name) に対する変異株 $(vn) の回帰データが複数ある"
        end
        reg = df[1,:]
        α = reg.α_j
        β = reg.β_j
        t_rt = date_to_value.([date_start, date_end])
        col = VARIANT_COLORS_vn[vn]
        plot!(p,
            t_rt, α .+ β .* t_rt,
            label=:none, color=col, alpha=0.75,
            linestyle=:dot,
        )
        # データ
        λe_vt = λe_vt_vn[vn]
        wt_vt = .!(ismissing.(λe_vt))
        ts_wt = ts_vt[wt_vt]
        te_wt = te_vt[wt_vt]
        tc_wt = t_star.(ts_wt, te_wt, α, β)
        λe_wt = λe_vt[wt_vt]
        λl_wt = replace(λl_vt_vn[vn][wt_vt], -Inf => -1e8)
        λu_wt = replace(λu_vt_vn[vn][wt_vt],  Inf =>  1e8)
        scatter!(p,
            tc_wt, λe_wt,
            xerror=(tc_wt .- ts_wt, te_wt .- tc_wt),
            yerror=(λe_wt .- λl_wt, λu_wt .- λe_wt),
            label=vn, color=col, alpha=1.0,
            markerstrokecolor=col, edgecolor=col,
        )
    end
    # アノテーション
    for (vn, x, y) ∈ value_annotations
        df = filter(row -> (row.variant == vn) && (row.base_variant == base_variant_name), LogitReg_DF)
        if nrow(df) ≠ 1
            @warn "基準株 $(base_variant_name) に対する変異株 $(vn) の回帰データが複数ある"
        end
        reg = df[1,:]
        β = reg.β_j
        annotate!(p, x, y,
            text(
                @sprintf(
                    "%s の %s に対する\n対数増加率：%.3f /日",
                    vn, base_variant_name, β
                ),
                font("Meiryo", 9), VARIANT_COLORS_vn[vn], :left
            )
        )
    end
    data_source = Dict(
        :tokyo => "東京都「新型コロナウイルス感染症モニタリング会議資料」変異株検査",
        :osaka => "大阪府「新型コロナウイルス感染症患者の発生状況について」",
    )[RegionSymbol]
    annotate!(p, rx(p, 0.35), ry(p, 0.05),
        text(
            """
            データソース：$(data_source)
            変異株 $(base_variant_name) に対する他の変異株の検出数の比の対数を示す
            (2 株間の比は感染者数の増減によらずおよそ直線的に推移することが知られている).
            縦軸エラーバーは二項分布を仮定したときの 95% 信頼区間を示す.
            点線は二項分布の最尤推定により求めたパラメーターにもとづく回帰直線.
            """,
            font("Meiryo", 6), RGB(0.3,0.3,0.3), :left
        )
    )
    # 右 y 軸
    #=
    p2 = twinx(p)
    ylims!(p2, ymin, ymax)
    y2_eticks = (function(emin, emax)
        eticks = [10.0^b for b ∈ -9:9]
        return filter(e -> emin ≤ e ≤ emax, eticks)
    end)(exp(ymin), exp(ymax))
    yticks!(p2, log.(y2_eticks), map(v -> (@sprintf("%g", v) * " "^5)[1:5], y2_eticks))
    ylabel!(p2, "$(base_variant_name) 株の検出数に対する他の株の検出数の比（オッズ、対数目盛り）")
    =#
    return p
end
@info p_variant_logit_transitions_against_base_variant

@info "---------"
@info "メイン関数定義"

# グローバル変数
P = nothing

function generate(region_symbol)
    @info "========"
    @info "プロット生成"
    @info "--------"
    @info "地域シンボル名"
    @assert region_symbol ∈ CAPABLE_REGION_SYMBOLS
    global RegionSymbol = region_symbol
    @insp RegionSymbol
    @info "--------"
    @info "データ読み込み"
    load_jld2(INPUT_JLD2_FILEPATHS[RegionSymbol])
    @insp size(Variants_DF), size(LogitReg_DF)
    @info "--------"
    @info "追加アノテーションのパラメーター定義"
    value_annotations_JN_1_against_EG_5 = [
        ("JN.1",   date_to_value(Date("2023-12-10")), -2.0),
    ]
    value_annotations_XBB_against_BA_5 = [
        ("XBB.1.5",   date_to_value(Date("2023-02-01")), -5.5),
#        ("XBB+XBB.1.9.1+XBB.1.9.2+XBB.1.16", date_to_value(Date("2023-03-01")), -3.5)
    ]
    @info "--------"
    @info "Plots 初期化"
    pyplot(
        titlefont=font("Meiryo",9),
        guidefont=font("Meiryo",8),
        tickfont=font("monospace",8)
    )
    P = Dict{Symbol, Plots.Plot}()
    @info "--------"
    @info "プロット生成"
    #-------
    s = :against_EG_5 
    @insp s
    P[s] = p_variant_logit_transitions_against_base_variant(
        value_annotations = value_annotations_JN_1_against_EG_5,
    )
    @info "プロット書き出し"
    f = FIGURE_DIRECTORY * "tokyo_logit_transitions.png"
    @insp f
    savefig(P[s], f)
    #=
    #-------
    s = :against_BA_5 
    @insp s
    P[s] = p_variant_logit_transitions_against_base_variant(
        value_annotations = value_annotations_XBB_against_BA_5,
    )
    @info "プロット書き出し"
    f = FIGURE_DIRECTORY * "tokyo_logit_transitions"
    @insp f
    savefig(P[s], f)
    #-------
    s = :XBB_against_BA_5
    @insp s
    P[s] = p_variant_logit_transitions_against_base_variant(
        variant_names_to_be_plotted_v = ["XBB", "XBB.1.5", "XBB.1.9.1", "XBB.1.16"],
        value_annotations = value_annotations_XBB_against_BA_5,
    )
    #-------
    for bvn ∈ ["BF.7", "BQ.1.1"]
        s = Symbol(replace("against_$(bvn)", "." => "_"))
        @insp s
        P[s] = p_variant_logit_transitions_against_base_variant(
            base_variant_name = bvn,
        )
    end
    #-------
    s = :XBB_against_XBB
    @insp s
    P[s] = p_variant_logit_transitions_against_base_variant(
        base_variant_name = "XBB.1.5",
        variant_names_to_be_plotted_v = ["BA.5", "XBB", "XBB.1.9.1", "XBB.1.9.2", "XBB.1.16", "XBB.1.16 (transient-free)"],
        date_start = Date("2023-02-15"), date_end = Date("2023-06-15"),
        ymin = -5.0, ymax = 3.0,
        value_annotations = [
            ("XBB.1.16", date_to_value(Date("2023-04-01")),  2.5),
            ("XBB.1.16 (transient-free)", date_to_value(Date("2023-04-01")),  1.5),
            ("BA.5",     date_to_value(Date("2023-05-10")), -3.0),
        ],
    )
    =#
    global P = P
    return P
end
@insp generate

@info "========"
@info ""
@info "Variants_Plots.generate(:tokyo) または Variants_Plots.generate(:osaka) として実行する"
@info "プロットは全域変数 P とともに、返り値として返される Plots.Plot を値として持つ Dict 型変数"
@info "一部のプロットは画像用ディレクトリに直ちに出力される"
@info ""
@info "========"

end #module

##