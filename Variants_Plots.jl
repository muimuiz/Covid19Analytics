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
@info @__MODULE__

@info "--------"
@info "作業ディレクトリ"
@info pwd()

@info "========"
@info "定数"

@info "--------"
@info "解析データ"

const input_jld2_filepaths = Dict(
    :tokyo => "variants_latest_tokyo.jld2",
    :osaka => "variants_latest_osaka.jld2",
)
@info "input_jld2_filepaths", input_jld2_filepaths

@info "--------"
@info "プロットする変異株"

const variant_names_to_be_plotted = [
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
    "XBB+1.9.1",
]
@info "variant_names_to_be_plotted", variant_names_to_be_plotted

@info "--------"
@info "プロット時間範囲"

const plot_earliest_start_date = Date("2022-10-31")
const plot_latest_end_date     = Date("2023-06-02")
@info "plot_earliest_start_date", plot_earliest_start_date
@info "plot_latest_end_date", plot_latest_end_date

@info "--------"
@info "画像出力ディレクトリ"

const figdir = "CurrentFigs/"
@info "figdir", figdir

@info "--------"
@info "描画色"

RGB256(r, g, b) = RGB(r/255, g/255, b/255)
const cols = Dict(
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
    "XBB+1.9.1" => RGB256( 98,  88, 106),
)
@info "length(cols)", length(cols)

@info "========"
@info "関数定義"

@info "--------"
@info "データ読み込み用関数"

function load_jld2(filepath)
    jld2_data = load(filepath)
    D = Dict{Symbol,Any}()
    for k ∈ keys(jld2_data)
        D[Symbol(k)] = jld2_data[k]
    end
    variants_df = D[:variants_df]
    logitreg_df = D[:logitreg_df]
    return variants_df, logitreg_df
end
@info load_jld2

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
@info λ_logistic

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
@info λ_star

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
@info t_star

@info "--------"
@info "プロット用関数"

UPlot = Union{Plots.Plot,Plots.Subplot}

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
@info rx
@info ry

function x_axis_time!(
    p::UPlot;
    recent = nothing,
    start_date = plot_earliest_start_date,
    end_date   = plot_latest_end_date,
)
    date_end   = end_date
    date_start = (isnothing(recent)) ? start_date : date_end - Day(recent) 
    t_start = dtime_to_value(DateTime(date_start))
    t_end   = dtime_to_value(DateTime(date_end))
    xlims!(p, t_start, t_end)
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
            if t_start ≤ t ≤ t_end
                push!(ts, t)
                push!(labels, ticks_dict[d])
            end
        end
        xticks!(p, ts, labels; xtickfontrotation=90.0)
    end
    return p
end
@info x_axis_time!

#-------
# 基準株に対する他の株の検出数推移

function p_variant_logit_transitions_against_base_variant(
    region, variants_df, logitreg_df;
    base_variant_name = "BA.5",
    variant_names_to_be_plotted = variant_names_to_be_plotted,
    start_date = plot_earliest_start_date,
    end_date = plot_latest_end_date,
    ymin = nothing,
    ymax = nothing,
    value_annotations = [],
)
    region_name = Dict(:tokyo => "東京", :osaka => "大阪")[region]
    p = plot(
        size=(640, 600),
        framestyle=:box,
        bottom_margin=8px, left_margin=8px, top_margin=8px, right_margin=8px,
        title="$(base_variant_name) に対する他株の検出数比の推移（$(region_name)）",
        legend=:topleft,
        fontfamily="Meiryo"
    )
    # データ
    ts_i = date_to_value.(Date.(variants_df[:,"date_start"]); noon=false)
    te_i = date_to_value.(Date.(variants_df[:,"date_end"]);   noon=false) .+ 1.0
    tm_i = (ts_i .+ te_i) ./ 2.0
    v_is  = Dict{String, Vector}()
    vl_is = Dict{String, Vector}()
    vu_is = Dict{String, Vector}()
    for varname ∈ variant_names_to_be_plotted
        if varname == base_variant_name continue end
        if !has_name(variants_df, varname) continue end
        if varname ∉ variant_names_to_be_plotted continue end
        logitname_for = "$(varname)_$(base_variant_name)_logit"
        logitname_rev = "$(base_variant_name)_$(varname)_logit"
        if     has_name(variants_df, logitname_for)
            v_is[varname]  =    variants_df[:,logitname_for]
            vl_is[varname] =    variants_df[:,logitname_for * "_cil"]
            vu_is[varname] =    variants_df[:,logitname_for * "_ciu"]
        elseif has_name(variants_df, logitname_rev)
            v_is[varname]  = .- variants_df[:,logitname_rev]
            vl_is[varname] = .- variants_df[:,logitname_rev * "_cil"]
            vu_is[varname] = .- variants_df[:,logitname_rev * "_ciu"]
        end
    end
    # x 軸設定
    x_axis_time!(p; start_date=start_date, end_date=end_date)
    x_axis_lims = collect(Plots.xlims(p))
    plot!(p, collect(Plots.xlims(p)), [0.0, 0.0], label=:none, color=:black)
    # y 軸設定
    ytol = 0.2
    if isnothing(ymin) ymin0 = min(0.0, reduce(min, minimum.(skipmissing.(values(v_is))))) end
    if isnothing(ymax) ymax0 = max(0.0, reduce(max, maximum.(skipmissing.(values(v_is))))) end
    if isnothing(ymin) ymin = (1.0 + ytol) * ymin0 - ytol * ymax0 end
    if isnothing(ymax) ymax = (1.0 + ytol) * ymax0 - ytol * ymin0 end
    ylims!(p, ymin, ymax)
    y_ticks = yticks(p)[1][1]
    ylabel!(p, "$(base_variant_name) 株の検出数に対する他の株の検出数の比の自然対数（ロジット）")
    # グリッド線と上辺をプロットの要素として重ね書き（twinx() のバクにより消されるため）
    map(y -> plot!(p, x_axis_lims, [y]; label=:none, lc=:gray90), y_ticks)
    plot!(p, x_axis_lims, [ymax]; label=:none, lc=:black)    
    # プロット
    for var ∈ variant_names_to_be_plotted
        if var == base_variant_name continue end
        if var ∉ variant_names_to_be_plotted continue end
        # 回帰直線
        bi = (logitreg_df.variant .== var) .& (logitreg_df.base_variant .== base_variant_name)
        lrows = logitreg_df[bi,:]
        if nrow(lrows) ≠ 1 continue end
        α   = lrows.α_j[1]
        β   = lrows.β_j[1]
        ts  = date_to_value.([start_date, end_date])
        col = cols[var]
        plot!(p,
            ts, α .+ β .* ts,
            label=:none, color=col, alpha=0.75,
            linestyle=:dot,
        )
        # データ
        v_i  = v_is[var]
        I_i  = Vector{Bool}(.!(ismissing.(v_i)))
        ts_I = ts_i[I_i]
        te_I = te_i[I_i]
        t_I  = t_star.(ts_I, te_I, α, β)
        v_I  = v_i[I_i]
        vl_I = replace(vl_is[var][I_i], -Inf => -1e8)
        vu_I = replace(vu_is[var][I_i],  Inf =>  1e8)
        scatter!(p,
            t_I, v_I,
            xerror=(t_I .- ts_I, te_I .- t_I),
            yerror=(v_I .- vl_I, vu_I .- v_I),
            label=var, color=col, alpha=1.0,
            markerstrokecolor=col, edgecolor=col,
        )
    end
    # アノテーション
    for (var, x, y) ∈ value_annotations
        bi = (logitreg_df.variant .== var) .& (logitreg_df.base_variant .== base_variant_name)
        lrows = logitreg_df[bi,:]
        if nrow(lrows) == 1
            β = lrows.β_j[1]
            annotate!(p, x, y,
                text(
                    @sprintf(
                        "%s の %s に対する\n対数増加率：%.3f /日",
                        var, base_variant_name, β
                    ),
                    font("Meiryo", 9), cols[var], :left
                )
            )
        end
    end
    data_source = Dict(
        :tokyo => "東京都「新型コロナウイルス感染症モニタリング会議資料」変異株検査",
        :osaka => "大阪府「新型コロナウイルス感染症患者の発生状況について」",
    )[region]
    annotate!(p, rx(p, 0.35), ry(p, 0.08),
        text(
            """
            データソース：$(data_source)
            変異株 $(base_variant_name) に対する他の変異株の検出数の比の対数を示す
            (2 株間の比は感染者数の増減によらずおよそ直線的に推移することが知られている).
            縦軸エラーバーは二項分布を仮定したときの 95% 信頼区間を示す.
            点線は二項分布の最尤推定により求めたパラメーターにもとづく回帰直線.
            枝番のない XBB は XBB.1.5 と 1.9.1 を除いた XBB 系統の株を,
            XBB+1.9.1 は XBB.1.5 のみを除いた XBB 系統の株を表す.
            """,
            font("Meiryo", 6), RGB(0.3,0.3,0.3), :left
        )
    )
    # 右 y 軸
    p2 = twinx(p)
    ylims!(p2, ymin, ymax)
    y2_eticks = (function(emin, emax)
        eticks = [10.0^b for b ∈ -9:9]
        return filter(e -> emin ≤ e ≤ emax, eticks)
    end)(exp(ymin), exp(ymax))
    yticks!(p2, log.(y2_eticks), map(v -> (@sprintf("%g", v) * " "^5)[1:5], y2_eticks))
    ylabel!(p2, "$(base_variant_name) 株の検出数に対する他の株の検出数の比（オッズ、対数目盛り）")
    return p
end
@info p_variant_logit_transitions_against_base_variant

@info "---------"
@info "メイン関数定義"

function generate(region)
    variants_df, logitreg_df = load_jld2(input_jld2_filepaths[region])
    pyplot(
        titlefont=font("Meiryo",9),
        guidefont=font("Meiryo",8),
        tickfont=font("monospace",8)
    )
    P = Dict{Symbol, Plots.Plot}()
    value_annotations_XBB_against_BA_5 = [
        ("XBB.1.5",   date_to_value(Date("2023-02-01")), -5.5),
        ("XBB+1.9.1", date_to_value(Date("2023-03-01")), -3.5)
    ]
    #-------
    @info "against_BA_5"
    P[:against_BA_5] = p_variant_logit_transitions_against_base_variant(region, variants_df, logitreg_df;
        value_annotations = value_annotations_XBB_against_BA_5,
    )
    savefig(P[:against_BA_5], figdir * "tokyo_logit_transitions")
    #-------
    @info "XBB_against_BA_5"
    P[:XBB_against_BA_5] = p_variant_logit_transitions_against_base_variant(region, variants_df, logitreg_df;
        variant_names_to_be_plotted = ["XBB", "XBB.1.5", "XBB.1.9.1"],
        value_annotations = value_annotations_XBB_against_BA_5,
    )
    #-------
    for base_variant_name ∈ ["BF.7", "BQ.1.1", "BN.1"]
        symbol = Symbol(replace("against_$(base_variant_name)", "." => "_"))
        @info symbol
        P[symbol] = p_variant_logit_transitions_against_base_variant(region, variants_df, logitreg_df;
            base_variant_name = base_variant_name,
        )
    end
    #-------
    @info "XBB_against_XBB"
    P[:XBB_against_XBB] = p_variant_logit_transitions_against_base_variant(region, variants_df, logitreg_df;
        base_variant_name = "XBB.1.5",
        variant_names_to_be_plotted = ["BA.5", "XBB", "XBB.1.9.1"],
        start_date = Date("2023-02-15"), end_date = Date("2023-05-15"), ymin = -5.0, ymax = 1.5,
    )
    return P
end
@info generate

@info "========"
@info ""
@info "Variants_Plots.generate(:tokyo) または Variants_Plots.generate(:osaka) として実行する"
@info "返り値は Plots.Plot を値として持つ Dict 型変数"
@info "一部のプロットは画像用ディレクトリに出力される"
@info ""
@info "========"

end #module

##