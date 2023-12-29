include("CommonDefs.jl")
module Variants

using Logging
using CSV
using JLD2
using DataFrames
using Dates
using Combinatorics: combinations
using Statistics: mean, stdm
using LogExpFunctions: log1pexp, logsubexp
using SpecialFunctions: logabsbinomial
using LinearAlgebra: norm, det, inv
using HypothesisTests: BinomialTest, confint
using GLM
using RCall

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
@info "データファイル"

const INPUT_CSV_FILEPATHS = Dict(
    :tokyo => "./東京変異株.csv",
    :osaka => "./大阪変異株.csv",
)
@insp INPUT_CSV_FILEPATHS
@info "  \"$(INPUT_CSV_FILEPATHS[:tokyo])\" はおそらく感染研に依頼したゲノム解析の結果のデータ"
@info "  都PDF資料からGoogleスプレッドシートに手入力しCSV出力したもの"
@info "  cf. https://www.fukushihoken.metro.tokyo.lg.jp/iryo/kansen/corona_portal/henikabu/screening.html"
@info "      https://www.bousai.metro.tokyo.lg.jp/taisaku/saigai/1021348/index.html"
@info "      https://docs.google.com/spreadsheets/d/1WKMmu_2Ba_IZgsjgL6GhDgR5mdQDG2WxYXVsgCEGl9Q/edit?usp=sharing"
@info "  \"$(INPUT_CSV_FILEPATHS[:osaka])\" も同じく感染研に依頼したゲノム解析の結果のデータ"
@info "  日付は「結果判明日」であり、検体が採取された日はこれより数週前と思われる"
@info "  府PDF資料からGoogleスプレッドシートに手入力しCSV出力したもの"
@info "  cf. https://www.pref.osaka.lg.jp/iryo/osakakansensho/happyo.html"
@info "      https://www.pref.osaka.lg.jp/iryo/osakakansensho/happyo_kako.html"
@info "      https://docs.google.com/spreadsheets/d/1pzzU7rX04yJlh0w_SmoH6YtH3Y7QjV8WUUWZffhHcAo/edit?usp=sharing"

@info "--------"
@info "出力ファイル"

const OUTPUT_JLD2_FILEPATHS = Dict(
    :tokyo => "./variants_latest_tokyo.jld2",
    :osaka => "./variants_latest_osaka.jld2",
)
@insp OUTPUT_JLD2_FILEPATHS

@info "--------"
@info "対象となる変異株名"
const VARIANT_NAMES_v = [
    "EG.5",
    "JN.1",
    "BA.2.86",
    "XBB.2.3",
    "others",
#=
    "BA.2",
    "BA.2.75",
    "BA.5",
    "BF.7",
    "BN.1",
    "BQ.1",
    "BQ.1.1",
    "XBB.1.5",
    "XBB.1.9.1",
    "XBB.1.9.2",
    "XBB.1.16",
    "XBB.1.16 (transient-free)",
    "XBB",
    "XBB+1.5+1.9.1+1.9.2+1.16",
    "XBB+1.9.1+1.9.2+1.16",
=#
]
@insp VARIANT_NAMES_v

@info "比較基準となる変異株名"
const BASE_VARIANT_NAMES_bv = [
    "EG.5",
#=
    "BA.5",
    "BF.7",
    "BN.1",
    "BQ.1.1",
    "XBB.1.5",
=#
]
@insp BASE_VARIANT_NAMES_bv
@assert BASE_VARIANT_NAMES_bv ⊆ VARIANT_NAMES_v

@info "--------"
@info "予測させる時刻範囲"

const PREDICTION_DATE_START = Date("2023-11-01")
const PREDICTION_DATE_END   = Date("2024-03-01")
#=
const PREDICTION_DATE_START = Date("2022-11-01")
const PREDICTION_DATE_END   = Date("2023-07-01")
=#
@insp PREDICTION_DATE_START, PREDICTION_DATE_END
const PREDICTION_DATES_pt   = PREDICTION_DATE_START:Day(1):PREDICTION_DATE_END
const PREDICTION_TVALUES_pt = date_to_value.(PREDICTION_DATES_pt)
@insp PREDICTION_DATES_pt
@insp length(PREDICTION_TVALUES_pt)

@info "========"
@info "関数定義"
@info "  主要データは module 内の全域変数により共有される"
@info "  （module をクラスの単一インスタンスのように扱う）"

@info "--------"
@info "データ読み込み用関数"

Variants_DF = nothing

function load_csv(filepath)
    @info "--------"
    @info "CSV ファイルを読み込み、必要な列を選別して DataFrame として返す"
    @info "--------"
    @info "CSV 読込み"
    @insp filepath
    csv_df = CSV.read(filepath, DataFrame; missingstring="")
    @insp size(csv_df)
    @info "--------"
    @info "出力用データフレーム作成"
    df = DataFrame()
    @info "日付のコピー"
    @assert has_name(csv_df, "date_start")
    @assert has_name(csv_df, "date_end")
    df.date_start = csv_df.date_start
    df.date_end   = csv_df.date_end
    @assert all(df.date_start .≤ df.date_end)
    @info "変異株数のコピー"
    for vn ∈ VARIANT_NAMES_v
        if !has_name(csv_df, vn)
            @warn "CSV $(filepath) は $(vn) のカラムを持たない"
            continue
        end
        df[:,vn] = csv_df[:,vn]
    end
    @insp size(df)
    global Variants_DF = df
end
@insp load_csv

@info "--------"
@info "データ処理関数"

function stat!()
    @info "--------"
    @info "変異株間のオッズその他の統計量を算出する"
    n_rows = nrow(Variants_DF)
    @insp n_rows
    @info "--------"
    @info "変異株間のオッズ (VAR1_VAR2_odds)、信頼区間 (VAR1_VAR2_odds_(cil|ciu))"
    for (vn1, vn2) in combinations(VARIANT_NAMES_v, 2)
        if !has_name(Variants_DF, vn1) || !has_name(Variants_DF, vn2)
            @warn "Variants_DF は $(vn1), $(vn2) の対を持たない"
            continue
        end
        k1_vt = Variants_DF[!,vn1]
        k2_vt = Variants_DF[!,vn2]
        odds_vt = k1_vt ./ k2_vt
        global Variants_DF[!,"$(vn1)_$(vn2)_odds"]  = odds_vt
        global Variants_DF[!,"$(vn1)_$(vn2)_logit"] = log.(odds_vt)
        n_vt  = k1_vt .+ k2_vt
        cl_vt = VectorMissing(Float64, n_rows)
        cu_vt = VectorMissing(Float64, n_rows)
        for vt in 1:n_rows
            if ismissing(k1_vt[vt]) || ismissing(k2_vt[vt]) continue end
            (cl_vt[vt], cu_vt[vt]) = 1.0 ./ (1.0 ./ confint(BinomialTest(k1_vt[vt], n_vt[vt])) .- 1.0)
        end
        global Variants_DF[!,"$(vn1)_$(vn2)_odds_cil"]  = cl_vt
        global Variants_DF[!,"$(vn1)_$(vn2)_odds_ciu"]  = cu_vt
        global Variants_DF[!,"$(vn1)_$(vn2)_logit_cil"] = log.(cl_vt)
        global Variants_DF[!,"$(vn1)_$(vn2)_logit_ciu"] = log.(cu_vt)
    end
end
@insp stat!
@info "全域変数 Variants_DF を直接操作する意味で関数名に \"!\" を付す（Julia の習慣とは異なる）"

@info "--------"
@info "時刻区間代表値計算用関数"

@inline λ_logistic(t, α, β) = α + β * t
@inline p_logistic(t, α, β) = sigmoid(λ_logistic(t, α, β))
@insp λ_logistic
@insp p_logistic

# 区間の両端のロジット値 (λs, λe) から、区間を代表するロジット値を求める
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

# 時間区間 (ts, te) に対する成功確率から引き戻した時刻
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
@info "自前の最適化問題計算用関数"

# 2項分布の対数負値（自己情報量）
function h_binom(n, k, t, α, β)
    @assert 0 ≤ k ≤ n
    λ = λ_logistic(t, α, β)
    return - logabsbinomial(n, k)[1] - k * λ + n * log1pexp(λ)
end
@insp h_binom

# 対数尤度
ℓ_binom(ns, ks, ts, α, β) = reduce(+, h_binom.(ns, ks, ts, α, β))
@insp ℓ_binom

# 対数尤度の勾配
function ∇ℓ_binom(ns, ks, ts, α, β)
    @assert length(ns) == length(ks) == length(ts)
    hs = ns .* p_logistic.(ts, α, β) .- ks
    return [reduce(+, hs), reduce(+, hs .* ts)]
end
@insp ∇ℓ_binom

# 対数尤度のヘッセ行列
function Hesse_ℓ(ns, ks, ts, α, β)
    ps = p_logistic.(ts, α, β)
    qs = 1.0 .- ps
    h0 = reduce(+, ns .* ps .* qs)
    h1 = reduce(+, ns .* ps .* qs .* ts)
    h2 = reduce(+, ns .* ps .* qs .* ts.^2)
    return [h0 h1; h1 h2]
end
@insp Hesse_ℓ

# ニュートン＝ラフソン法による回帰パラメーター推定
function newton_raphson_method(df, α, β)
    # 時間のスケール
    N  = nrow(df)
    mt = mean(df.t_vt)
    σt  = stdm(df.t_vt, mt; corrected=false)
    σt′ = tan(((N - 1) / N) * (π / 2.0)) / sqrt(3.0)
    rt  = σt / σt′
    t′s_vt = (df.ts_vt .- mt) ./ rt
    t′e_vt = (df.te_vt .- mt) ./ rt
    α′ = α + β * mt 
    β′ = β * rt
    # 探索ループ
    #rate = 0.01
    #desc = false
    t′_vt = t_star.(t′s_vt, t′e_vt, α′, β′)
    ℓ′  = ℓ_binom(df.n_vt, df.k1_vt, t′_vt, α′, β′)
    ∇ℓ′ = ∇ℓ_binom(df.n_vt, df.k1_vt, t′_vt, α′, β′)
    slope = norm(∇ℓ′)
    for j in 1:1000
        θ′_prev = [α′, β′]
        H′ = Hesse_ℓ(df.n_vt, df.k1_vt, t′_vt, α′, β′)
        if abs(det(H′)) < 1e-8 @warn "H′, det(H′)", H′, det(H′) end
        α′, β′ = θ′_prev .- (inv(H′) * ∇ℓ′)
        ℓ′_prev = ℓ′
        ℓ′ = ℓ_binom(df.n_vt, df.k1_vt, t′_vt, α′, β′)
        #desc_prev = desc
        #desc = (ℓ′_prev > ℓ′)
        #rate *= (!desc && desc_prev) ? 0.6 : 0.999
        ∇ℓ′ = ∇ℓ_binom(df.n_vt, df.k1_vt, t′_vt, α′, β′)
        slope = norm(∇ℓ′)
        if (slope < 1e-8) || (θ′_prev == [α′, β′]) break end
    end
    β = β′ / rt
    α = α′ - β * mt
    return α, β
end
@insp newton_raphson_method

@info "--------"
@info "ロジスティック回帰用関数"

LogitReg_DF = nothing

function varpair(vn1, vn2)
    @info "$(vn1), $(vn2) 間、対象データの抜き出し"
    df = DataFrame()
    df.ts_vt = date_to_value.(Variants_DF.date_start; noon=false)
    df.te_vt = date_to_value.(Variants_DF.date_end;   noon=false) .+ 1.0
    df.k1_vt = Variants_DF[:,vn1]
    df.k2_vt = Variants_DF[:,vn2]
    df.n_vt  = df.k1_vt .+ df.k2_vt
    dropmissing!(df, :n_vt)
    filter!(row -> row.n_vt ≠ 0, df)
    @insp nrow(df)
    return df
end

# 2株間の回帰パラメーター探索
function logitreg_pair(df; method=:glm_julia)
    @info "--------"
    @info "ロジスティック回帰パラメーター探索 ($(method))"
    @info "--------"
    n_row = nrow(df)
    @assert n_row ≥ 2
    df.tc_vt   = (df.ts_vt .+ df.te_vt) ./ 2.0
    df.p_vt    = df.k1_vt ./ df.n_vt
    df.odds_vt = df.k1_vt ./ df.k2_vt
    df.λ_vt    = log.(df.odds_vt)
    @info "--------"
    @info "初期パラメーター"
    rt_vt = is_regular.(df.λ_vt)
    α, β = linear_regression(df.tc_vt[rt_vt], df.λ_vt[rt_vt])
    @insp α, β
    @assert is_regular(α) && is_regular(β)
    df.t_vt = df.tc_vt
    @info "--------"
    @info "探索ループ"
    β_ci = nothing
    for i in 1:100
        α_prev, β_prev = α, β
        start = [α, β]
        if     method == :glm_julia
            glm_result = glm(@formula(p_vt ~ 1 + t_vt), df, Binomial(), LogitLink(); wts=df.n_vt, start=start)
            α, β = coef(glm_result)
            β_ci = confint(glm_result)[2,:]
        elseif method == :glm_r
            R"""
            glm_result <- glm(cbind(k1_vt,k2_vt) ~ 1 + t_vt, $df, family=binomial(link="logit"), start=$start)
            coefs <- coef(glm_result)
            cis <- confint(glm_result)
            """
            α, β = @rget(coefs)
            β_ci = @rget(cis)[2,:]
        elseif method == :nrm
            α, β = newton_raphson_method(df, α, β)
        end
        if abs(α - α_prev) < 1e-10 && abs(β - β_prev) < 1e-12 break end
        df.t_vt = t_star.(df.ts_vt, df.te_vt, α, β)
    end
    @info "-------"
    @info "近似結果"
    slope = norm(∇ℓ_binom(df.n_vt, df.k1_vt, df.t_vt, α, β))
    @insp α, β, slope, β_ci
    return α, β, slope, β_ci
end
@info logitreg_pair

function logitreg()
    @info "--------"
    @info "基準株に対する他の株のロジスティック回帰"
    df = DataFrame(
        variant      = String[],
        base_variant = String[],
        α_j     = Float64[],
        β_j     = Float64[],
        slope_j = Float64[],
        β_ci_j  = Vector{Float64}[],
        α_r     = Float64[],
        β_r     = Float64[],
        slope_r = Float64[],
        β_ci_r  = Vector{Float64}[],
        α_s     = Float64[],
        β_s     = Float64[],
        slope_s = Float64[],
    )
    for bvn ∈ BASE_VARIANT_NAMES_bv
        if !has_name(Variants_DF, bvn) continue end
        for vn ∈ VARIANT_NAMES_v
            if !has_name(Variants_DF, vn) continue end
            if bvn == vn continue end
            pdf = varpair(vn, bvn)
            if nrow(pdf) < 2 continue end
            α_j, β_j, slope_j, β_ci_j = logitreg_pair(pdf; method=:glm_julia)
            α_r, β_r, slope_r, β_ci_r = logitreg_pair(pdf; method=:glm_r)
            α_s, β_s, slope_s, β_ci_s = logitreg_pair(pdf; method=:nrm)
            push!(df, [
                vn, bvn,
                α_j, β_j, slope_j, β_ci_j,
                α_r, β_r, slope_r, β_ci_r,
                α_s, β_s, slope_s,
            ])
        end
    end
    @info "--------"
    @info "結果のチェック（warn が出力されないとき各方法での結果は一致）"
    slope_tol = 1e-6
    α_tol = 1e-8
    β_tol = 1e-8
    for row ∈ eachrow(df)
        if abs(row.slope_j) > slope_tol @warn "glm_julia は収束していない: $(row.slope_j)" end
        if abs(row.slope_r) > slope_tol @warn "glm_r は収束していない: $(row.slope_r)" end
        if abs(row.slope_s) > slope_tol @warn "nrm は収束していない: $(row.slope_s)" end
        if abs(row.α_j - row.α_r) > α_tol || abs(row.β_j - row.β_r) > β_tol
            @warn "glm_julia と glm_r の結果は一致しない ($(row.α_j), $(row.β_j)), ($(row.α_r), $(row.β_r))"
        end  
        if abs(row.α_j - row.α_s) > α_tol || abs(row.β_j - row.β_s) > β_tol
            @warn "glm_julia と nrm の結果は一致しない ($(row.α_j), $(row.β_j)), ($(row.α_s), $(row.β_s))"
        end
    end
    global LogitReg_DF = df
end
@insp logitreg

@info "--------"
@info "計算結果保存用関数"

function save_jld2(filepath)
    @info "計算結果を JLD2 形式で $(filepath) に出力する"
    jldopen(filepath, "w") do jld2_file
        jld2_file["VARIANT_NAMES_v"]       = VARIANT_NAMES_v
        jld2_file["BASE_VARIANT_NAMES_bv"] = BASE_VARIANT_NAMES_bv
        jld2_file["Variants_DF"]           = Variants_DF
        jld2_file["LogitReg_DF"]           = LogitReg_DF
        @insp keys(jld2_file)
    end
end
@insp save_jld2

@info "--------"
@info "メインルーチン"

RegionSymbol = nothing

function main(region_symbol)
    @info "========"
    @info "メインルーチン"
    @assert region_symbol ∈ CAPABLE_REGION_SYMBOLS
    @info "--------"
    @info "CSV ファイル読み込み"
    load_csv(INPUT_CSV_FILEPATHS[region_symbol])
    @info "読み込んだデータフレーム"
    @insp Variants_DF
    @info "--------"
    @info "データ中の最新の日付"
    date_latest = Variants_DF.date_end[end]
    @insp date_latest
    date_latest_suffix_form = Dates.format(date_latest, "yymmdd")
    @insp date_latest_suffix_form
    @info "--------"
    @info "統計量を算出"
    stat!()
    @insp size(Variants_DF)
    @info "--------"
    @info "パラメーター探索"
    logitreg()
    @insp size(LogitReg_DF)
    @info "--------"
    @info "解析データ書き出し"
    @info "  データをJLD2形式のファイルとして出力する"
    save_jld2(OUTPUT_JLD2_FILEPATHS[region_symbol])
    @info "--------"
    @info "全域変数設定"
    global RegionSymbol = region_symbol
end
@insp main

@info "========"
@info ""
@info "Variants.main(:tokyo) または Variants.main(:osaka) として実行する"
@info ""
@info "========"

end #module

##