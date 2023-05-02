include("CommonDefs.jl")
module Variants_Pred

using Logging
using CSV
using JLD2
using DataFrames
using Dates
using Statistics: mean
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

const input_jld2_filepaths = Dict(
    :tokyo => "variants_latest_tokyo.jld2",
    :osaka => "variants_latest_osaka.jld2",
)
@insp input_jld2_filepaths

const confirmed_csv_filepaths = Dict(
    :tokyo => "tokyo_confirmed5_latest.csv",
)
@insp confirmed_csv_filepaths

@info "--------"
@info "対象となる変異株名"
const variant_names = [
    "BA.2",
    "BA.2.75",
    "BA.5",
    "BF.7",
    "BN.1",
    "BQ.1",
    "BQ.1.1",
    "XBB",
    "XBB.1.5",
    "XBB.1.9.1",
    #"XBB+1.9.1",
]
@insp variant_names

@info "基準変異株名"
const base_variant_name = "BA.5"
@insp base_variant_name

@info "--------"
@info "フィッティング時刻範囲"

const fit_date_start = Date("2023-01-22")
const fit_date_end   = Date("2023-02-11")
@insp fit_date_start, fit_date_end
const fit_t_start = date_to_value(fit_date_start; noon=false)
const fit_t_end   = date_to_value(fit_date_end;   noon=false) + 1.0
@insp fit_t_start, fit_t_end

@info "--------"
@info "予測期間"

const pred_dates = Date("2023-01-01"):Day(1):Date("2023-07-01")
const pred_ts    = date_to_value.(pred_dates; noon=false)
@assert length(pred_dates) == length(pred_ts)
@insp pred_dates
@insp length(pred_ts)

@info "========"
@info "関数定義"

@info "--------"
@info "データ読み込み用関数"

# 変異株回帰 jld2 データ読み込み
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
@insp load_jld2

# 感染確認者数 csv データ読み込み
function load_confirmed_csv(filepath)
    @info "CSV ファイル $filepath を読み込み、必要な列を選別して DataFrame として返す"
    @info "--------"
    @info "CSV 読込み"
    csv_df = CSV.read(filepath, DataFrame; missingstring="")
    @insp size(csv_df)
    @info "--------"
    @info "出力用データフレーム作成"
    df = DataFrame()
    @info "--------"
    @info "時刻のコピー"
    @assert has_name(csv_df, "t")
    @assert has_name(csv_df, "dtime")
    df.t     = csv_df.t
    df.dtime = csv_df.dtime
    @info "--------"
    @info "対数増加率トレンドのコピー"
    @assert has_name(csv_df, "Δlzs")
    df.Δlzs = csv_df.Δlzs
    @insp size(df)
    @insp names(df)
    return df
end
@insp load_confirmed_csv

# 確認者数データ confirmed_df から対数増加率トレンド値を抽出
function get_t_Δlzs(confirmed_df, t_start, t_end)
    t_i    = Missable{Float64}[] # 時刻
    Δlzs_i = Missable{Float64}[] # 時刻に対応する対数増加率トレンド値
    for row ∈ eachrow(confirmed_df)
        if !is_regular(row.t) || !is_regular(row.Δlzs) continue end
        if !(t_start ≤ row.t < t_end) continue end
        # 与えられた時間範囲内のみ抽出
        push!(t_i,    row.t)
        push!(Δlzs_i, row.Δlzs)
    end
    return t_i, Δlzs_i
end
@insp get_t_Δlzs

@info "--------"
@info "対数増加率算出関数"

# パラメーターから求めた対数増加率（定数分の不定性がある）
# 数学的関係について docs/varlogit.md 参照
function est_Δlz(t, αs, βs)
    eλs = exp.(αs .+ βs .* t)
    return sum(βs .* eλs) / (1.0 + sum(eλs))
end
@insp est_Δlz

@info "--------"
@info "メイン計算ルーチン"

# グローバル変数
confirmed_df = nothing
pred_Δlzs    = nothing
β0           = nothing
var_names    = nothing
pred_pss     = nothing

function run(region)
    @info "========"
    @info "変異株回帰データから感染力の変化に基づく今後の対数増加率を予測し、"
    @info "感染確認者数対数増加率データと比較するプロットを作成"
    @info "--------"
    @info "データ読み込み"
    variants_df, logitreg_df = load_jld2(input_jld2_filepaths[region])
    confirmed_df = load_confirmed_csv(confirmed_csv_filepaths[region])
    @insp size(variants_df), size(logitreg_df)
    @insp size(confirmed_df)
    @info "対数増加率トレンド値を抽出"
    fit_t_i, fit_Δlzs_i = get_t_Δlzs(confirmed_df, fit_t_start, fit_t_end)
    @assert length(fit_t_i) == length(fit_Δlzs_i)
    @insp length(fit_Δlzs_i)
    @info "--------"
    @info "変異株別相対パラメーター"
    o0df = logitreg_df[logitreg_df.base_variant .== base_variant_name,:] # 基準株が一致するもののみ
    ovars = filter(var -> var ≠ base_variant_name, variant_names) # 基準株を除いた変異株リスト
    o1df = DataFrame(vcat([o0df[findfirst(o0df.variant .== var),:] for var ∈ ovars]))
    ovar_names = o1df.variant # 各株 j の名前
    ovar_αs    = o1df.α_j     # 各株 j の回帰パラメーター α_j
    ovar_βs    = o1df.β_j     # 各株 j の回帰パラメーター β_j
    @insp ovar_names
    @insp ovar_αs
    @insp ovar_βs
    @assert length(ovar_names) == length(ovar_αs) == length(ovar_βs)
    n_ovars = length(ovar_names)
    @insp n_ovars
    @info "回帰パラメーターから求めた対数増加率"
    ovar_Δlz_i = map(t -> est_Δlz(t, ovar_αs, ovar_βs), fit_t_i)
    @insp length(ovar_Δlz_i)
    @info "フィッティングによる基準株対数増加率の推定"
    β0 = mean(skipmissing(fit_Δlzs_i .- ovar_Δlz_i))
    @insp β0
    @info "--------"
    @info "変異株の割合の変化にともなう感染力の変化に基づく対数増加率推定"
    pred_Δlzs = map(t -> β0 + est_Δlz(t, ovar_αs, ovar_βs), pred_ts)
    @assert length(pred_ts) == length(pred_Δlzs)
    @insp length(pred_Δlzs)
    @info "--------"
    @info "各変異株の割合の推定"
    var_names = vcat([base_variant_name], ovar_names)
    @insp var_names
    var_αs = vcat([0.0], ovar_αs)
    var_βs = vcat([0.0], ovar_βs)
    n_vars = length(var_names)
    @assert n_vars == n_ovars + 1
    @insp n_vars
    rns_vars = [[exp(var_αs[j] + var_βs[j] * t) for t ∈ pred_ts] for j ∈ 1:n_vars]
    rns_sum  = reduce(.+, rns_vars)
    pred_pss = [rns_vars[j] ./ rns_sum for j ∈ 1:n_vars]
    @assert n_vars == length(pred_pss)
    @insp length(pred_pss)
    @info "--------"
    @info "全域変数設定"
    global confirmed_df = confirmed_df
    global pred_Δlzs    = pred_Δlzs
    global β0           = β0
    global var_names    = var_names
    global pred_pss     = pred_pss
end
@insp run

@info "========"
@info ""
@info "Variants_Pred.run(region) として実行する"
@info "region はシンボル :tokyo または :osaka"
@info "module 内全域変数 pred_Δlzs が設定される"
@info ""
@info "========"

@info "========"
@info "プロット用定数"

@info "--------"
@info "プロット時間範囲"

const plot_earliest_start_date = Date("2022-12-31")
const plot_latest_end_date     = Date("2023-07-02")
@insp plot_earliest_start_date, plot_latest_end_date

@info "--------"
@info "プロットとサブプロットの合併型定義"

UPlot = Union{Plots.Plot,Plots.Subplot}
@insp UPlot

@info "--------"
@info "描画色"

const colors = [
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
@insp length(colors)

@inline RGB256(r, g, b) = RGB(r/255, g/255, b/255)
@insp RGB256
const varcols = Dict(
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
@insp length(varcols)

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
    region, confirmed_df, pred_Δlzs, β0;
    start_date = plot_earliest_start_date,
    end_date = plot_latest_end_date,
    recent = 120,
)
    region_name = Dict(:tokyo => "東京", :osaka => "大阪")[region]
    p = plot(
        size=(640, 600),
        framestyle=:box,
        bottom_margin=8px, left_margin=8px, top_margin=8px, right_margin=8px,
        title="感染確認者数の対数増加率と変異株比から推定される対数増加率の比較（$(region_name)）",
        legend=:topleft,
        fontfamily="Meiryo",
    )
    # データ
    t_t    = confirmed_df.t[(end - recent):end]
    Δlzs_t = confirmed_df.Δlzs[(end - recent):end]
    # x 軸設定
    x_axis_time!(p; start_date=start_date, end_date=end_date)
    x_axis_lims = collect(Plots.xlims(p))
    plot!(p, collect(Plots.xlims(p)), [0.0], label=:none, color=:black)
    # y 軸設定
    # 最小・最大値をデータの範囲から決定
    ymin, ymax = (function(v_t)
        v_T = v_t[is_regular.(v_t)]
        lmin = min(0.0, minimum(v_T)) - 0.03
        lmax = max(0.0, maximum(v_T)) + 0.03
        return lmin, lmax
    end)(Δlzs_t)
    @insp ymin, ymax
    # y 軸刻みの設定
    y_ticks = (function(lmin, lmax)
        modfu(x) = begin ix = floor(x); (x - ix, ix) end
        fp, ip = modfu(log10(lmax - lmin))
        step = ((fp < log10(2.0)) ? 0.2 : (fp < log10(5.0)) ? 0.5 : 1.0) * 10^ip
        return filter(v -> lmin ≤ v ≤ lmax, (-10.0 * step):step:(10.0 * step))
    end)(ymin, ymax)
    @insp y_ticks
    ylims!(p, ymin, ymax)
    yticks!(p, y_ticks, map(v -> replace(@sprintf("%g", v), "-" => "－"), y_ticks))
    ylabel!(p, "感染者 対数増加率 [/日]")
    # グリッド線と上辺をプロットの要素として重ね書き（twinx() のバクにより消されるため）
    map(y -> if y ≠ 0 plot!(p, x_axis_lims, [y]; label=:none, lc=:gray90) end, y_ticks)
    plot!(p, x_axis_lims, [0.0], label=:none, color=:black)
    plot!(p, x_axis_lims, [ymax]; label=:none, lc=:black)
    # プロット
    plot!(
        p, t_t, Δlzs_t;
        label="感染確認者数にもとづく実際の対数増加率",
        la=0.8, lc=colors[3], lw=2
    )
    plot!(
        p, pred_ts, pred_Δlzs;
        label="変異株の感染力の違いとして推定される対数増加率",
        la=0.8, lc=colors[9],
    )
    # アノテーション
    annotate!(p, rx(p, 0.45), ry(p, 0.08),
        text(
            """
            データソース: 変異株比の元データは東京都資料より.
            検出数が小さいいくつかの変異株の区分を除く.
            感染確認者数は厚労省「新規陽性者数の推移（日別）」より.
            変異株比からの対数増加率推移の推定は定数分の任意性がある.
            基準とした $(base_variant_name) の増加率を $(@sprintf("%.3f", β0)) とした場合.
            """,
            font("Meiryo",7), RGB(0.3,0.3,0.3), :left
        )
    )
    # 右 y 軸
    wtoy(w) = log(w) / 7.0
    ytow(y) = exp(7.0 * y)
    y2_bmin, y2_bmax = ytow.([ymin, ymax])
    @insp y2_bmin, y2_bmax
    y2_bticks = (function(bmin, bmax)
        bticks = [b for i ∈ -1:1 for b ∈ [1., 1.2, 1.5, 2., 3., 4., 5., 6., 7., 8., 9.] .* 10.0^i]
        return filter(t -> bmin ≤ t ≤ bmax, bticks)
    end)(y2_bmin, y2_bmax)
    p2 = twinx(p)
    ylims!(p2, ymin, ymax)
    yticks!(p2, wtoy.(y2_bticks), map(v -> (@sprintf("%g", v) * " "^4)[1:5], y2_bticks))
    ylabel!(p2, "週あたり拡大率（対数目盛り）")
    return p
end
@insp p_log_growth_comparison

@info "--------"
@info "変異株回帰パラメーターから推定される変異株の割合の積み上げ面プロット関数"

function p_stacked_area_variant_proportions(
    region, var_names, pred_pss;
    start_date = plot_earliest_start_date,
    end_date = plot_latest_end_date,
)
    region_name = Dict(:tokyo => "東京", :osaka => "大阪")[region]
    p = plot(
        size=(640, 600),
        framestyle=:box,
        bottom_margin=8px, left_margin=8px, top_margin=8px, right_margin=8px,
        title="変異株比から推定される変異株の割合の推移（$(region_name)）",
        legend=:topleft,
        fontfamily="Meiryo",
    )
    # データ
    pc_rpss = map(ps -> 100.0 * ps, reverse(pred_pss))
    ac_rpss = accumulate(.+, pc_rpss)
    @assert length(pc_rpss) == length(ac_rpss)
    # x 軸設定
    x_axis_time!(p; start_date=start_date, end_date=end_date)
    # y 軸設定
    ylims!(p, 0.0, 100.0)
    ylabel!(p, "変異株割合 [%]")
    # プロット
    stacked_area!(
        p, pred_ts, pc_rpss;
        lc=:black, fcs=reverse([varcols[var] for var ∈ reverse(var_names)]),
    )
    # ラベル
    il = length(pc_rpss[1]) ÷ 4
    ir = 3 * il
    @insp il, ir
    for (j, var) ∈ enumerate(reverse(var_names))
        if pc_rpss[j][il] > pc_rpss[j][ir]
            lx = 0.25
            ly = (ac_rpss[j][il] - pc_rpss[j][il] / 2.0) / 100.0
        else
            lx = 0.75
            ly = (ac_rpss[j][ir] - pc_rpss[j][ir] / 2.0) / 100.0
        end
        annotate!(p, rx(p, lx), ry(p, ly), text(var, font("Meiryo",8), :black, :center))
    end
    # アノテーション
    annotate!(p, rx(p, 0.45), ry(p, 0.04),
        text(
            """
            データソース: 変異株比の元データは東京都資料より.
            検出数が小さいいくつかの変異株の区分を除く.
            """,
            font("Meiryo",7), RGB(0.3,0.3,0.3), :left
        )
    )
end

@info "--------"
@info "プロット用メイン関数定義"

# グローバル変数
P = nothing

function generate(region)
    @info "========"
    @info "プロット生成"
    @info "--------"
    @info "計算結果が生成されていなければ、計算ルーチンを呼び出す"
    if isnothing(confirmed_df) || isnothing(pred_Δlzs) || isnothing(β0)
        run(region)
    end
    @info "--------"
    @info "Plots 初期化"
    pyplot(
        titlefont=font("Meiryo",9),
        guidefont=font("Meiryo",8),
        tickfont=font("Meiryo",8)
    )
    P = Dict{Symbol,Plots.Plot}()
    @info "--------"
    @info "プロット生成"
    @info "log_growth_comparison"
    P[:log_growth_comparison] = p_log_growth_comparison(
        region, confirmed_df, pred_Δlzs, β0,
    )
    @info "stack_area_variant_proportions"
    P[:stack_area_variant_proportions] = p_stacked_area_variant_proportions(
        region, var_names, pred_pss,
    )
    @info "--------"
    @info "全域変数設定"
    global P = P
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