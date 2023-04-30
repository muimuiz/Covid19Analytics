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

@info "--------"
@info "指定可能な対象地域"

const capable_regions = [:tokyo, :osaka]
@insp capable_regions

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
    "XBB+1.9.1",
]
@insp variant_names

@info "比較基準となる変異株名"
const base_variant_names = [
    "BA.5",
    "BF.7",
    "BN.1",
    "BQ.1.1",
    "XBB.1.5",
]
@insp base_variant_names
@assert base_variant_names ⊆ variant_names

@info "--------"
@info "予測させる時刻範囲"

const predict_date_range = Date("2022-11-01"):Day(1):Date("2023-07-01")
@insp predict_date_range
const predict_ts = date_to_value.(predict_date_range)
@insp length(predict_ts)

@info "--------"
@info "データファイル"

const csv_filepaths = Dict(
    :tokyo => "./東京変異株.csv",
    :osaka => "./大阪変異株.csv",
)
@insp csv_filepaths
@info "  \"$(csv_filepaths[:tokyo])\" はおそらく感染研に依頼したゲノム解析の結果のデータ"
@info "  都PDF資料からGoogleスプレッドシートに手入力しCSV出力したもの"
@info "  cf. https://www.fukushihoken.metro.tokyo.lg.jp/iryo/kansen/corona_portal/henikabu/screening.html"
@info "      https://www.bousai.metro.tokyo.lg.jp/taisaku/saigai/1021348/index.html"
@info "      https://docs.google.com/spreadsheets/d/1WKMmu_2Ba_IZgsjgL6GhDgR5mdQDG2WxYXVsgCEGl9Q/edit?usp=sharing"
@info "  \"$(csv_filepaths[:osaka])\" も同じく感染研に依頼したゲノム解析の結果のデータ"
@info "  日付は「結果判明日」であり、検体が採取された日はこれより数週前と思われる"
@info "  府PDF資料からGoogleスプレッドシートに手入力しCSV出力したもの"
@info "  cf. https://www.pref.osaka.lg.jp/iryo/osakakansensho/happyo.html"
@info "      https://www.pref.osaka.lg.jp/iryo/osakakansensho/happyo_kako.html"
@info "      https://docs.google.com/spreadsheets/d/1pzzU7rX04yJlh0w_SmoH6YtH3Y7QjV8WUUWZffhHcAo/edit?usp=sharing"

@info "--------"
@info "出力ファイル"

const jld2_filepaths = Dict(
    :tokyo => "./variants_latest_tokyo.jld2",
    :osaka => "./variants_latest_osaka.jld2",
)
@insp jld2_filepaths

@info "========"
@info "関数定義"

@info "--------"
@info "データ読み込み用関数"

function load_csv(region)
    @info "--------"
    @info "CSV ファイルを読み込み、必要な列を選別して DataFrame として返す"
    @info "region", region
    @info "--------"
    @info "CSV 読込み"
    filepath = csv_filepaths[region]
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
    for var ∈ variant_names
        if !has_name(csv_df, var) continue end
        @insp var
        df[:,var] = csv_df[:,var]
    end
    @insp size(df)
    return df
end
@info load_csv

@info "--------"
@info "データ処理関数"

function stat!(variants_df)
    @info "--------"
    @info "変異株間のオッズその他の統計量を算出する"
    n_rows = nrow(variants_df)
    @insp n_rows
    @info "--------"
    @info "変異株間のオッズ (VAR1_VAR2_odds)、信頼区間 (VAR1_VAR2_odds_(cil|ciu))"
    for (var1, var2) in combinations(variant_names, 2)
        if !has_name(variants_df, var1) || !has_name(variants_df, var2) continue end
        @insp var1, var2
        k1s = variants_df[!,var1]
        k2s = variants_df[!,var2]
        odds = k1s ./ k2s
        variants_df[!,"$(var1)_$(var2)_odds"]  = odds
        variants_df[!,"$(var1)_$(var2)_logit"] = log.(odds)
        ns  = k1s .+ k2s
        cls = VectorMissing(Float64, n_rows)
        cus = VectorMissing(Float64, n_rows)
        for i in 1:n_rows
            if ismissing(k1s[i]) || ismissing(k2s[i]) continue end
            (cls[i], cus[i]) = 1.0 ./ (1.0 ./ confint(BinomialTest(k1s[i], ns[i])) .- 1.0)
        end
        variants_df[!,"$(var1)_$(var2)_odds_cil"]  = cls
        variants_df[!,"$(var1)_$(var2)_odds_ciu"]  = cus
        variants_df[!,"$(var1)_$(var2)_logit_cil"] = log.(cls)
        variants_df[!,"$(var1)_$(var2)_logit_ciu"] = log.(cus)
    end
    return variants_df
end
@insp stat!

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
    mt = mean(df.t)
    σt  = stdm(df.t, mt; corrected=false)
    σt′ = tan(((N - 1) / N) * (π / 2.0)) / sqrt(3.0)
    rt  = σt / σt′
    t′ss = (df.ts .- mt) ./ rt
    t′es = (df.te .- mt) ./ rt
    α′ = α + β * mt 
    β′ = β * rt
    # 探索ループ
    #rate = 0.01
    #desc = false
    t′s = t_star.(t′ss, t′es, α′, β′)
    ℓ′  = ℓ_binom(df.n, df.k1, t′s, α′, β′)
    ∇ℓ′ = ∇ℓ_binom(df.n, df.k1, t′s, α′, β′)
    slope = norm(∇ℓ′)
    for j in 1:1000
        θ′_prev = [α′, β′]
        H′ = Hesse_ℓ(df.n, df.k1, t′s, α′, β′)
        if abs(det(H′)) < 1e-8 @warn "H′, det(H′)", H′, det(H′) end
        α′, β′ = θ′_prev .- (inv(H′) * ∇ℓ′)
        ℓ′_prev = ℓ′
        ℓ′ = ℓ_binom(df.n, df.k1, t′s, α′, β′)
        #desc_prev = desc
        #desc = (ℓ′_prev > ℓ′)
        #rate *= (!desc && desc_prev) ? 0.6 : 0.999
        ∇ℓ′ = ∇ℓ_binom(df.n, df.k1, t′s, α′, β′)
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

# 2株間の回帰パラメーター探索
function logitreg_pair(variants_df, varname1, varname2; method=:glm_julia)
    @info "--------"
    @info "$(varname1), $(varname2) 間、ロジスティック回帰パラメーター探索 ($(method))"
    @info "--------"
    @info "対象行の抜き出し"
    df = DataFrame()
    df.ts = date_to_value.(variants_df.date_start; noon=false)
    df.te = date_to_value.(variants_df.date_end;   noon=false) .+ 1.0
    df.k1 = variants_df[:,varname1]
    df.k2 = variants_df[:,varname2]
    df.n  = df.k1 .+ df.k2
    dropmissing!(df, :n)
    filter!(row -> row.n != 0, df)
    N = nrow(df)
    @insp N
    @assert N ≥ 2
    df.t_mid = (df.ts .+ df.te) ./ 2.0
    df.prop  = df.k1 ./ df.n
    df.odds  = df.k1 ./ df.k2
    df.logit = log.(df.odds)
    @info "--------"
    @info "初期パラメーター"
    ir = is_regular.(df.logit)
    α, β = linear_regression(df.t_mid[ir], df.logit[ir])
    @insp α, β
    @assert is_regular(α) && is_regular(β)
    df.t = df.t_mid
    @info "--------"
    @info "探索ループ"
    β_ci = nothing
    for i in 1:100
        α_prev, β_prev = α, β
        start = [α, β]
        if     method == :glm_julia
            glm_result = glm(@formula(prop ~ 1 + t), df, Binomial(), LogitLink(); wts=df.n, start=start)
            α, β = coef(glm_result)
            β_ci = confint(glm_result)[2,:]
        elseif method == :glm_r
            R"""
            glm_result <- glm(cbind(k1,k2) ~ 1 + t, $df, family=binomial(link="logit"), start=$start)
            coefs <- coef(glm_result)
            cis <- confint(glm_result)
            """
            α, β = @rget(coefs)
            β_ci = @rget(cis)[2,:]
        elseif method == :nrm
            α, β = newton_raphson_method(df, α, β)
        end
        if abs(α - α_prev) < 1e-10 && abs(β - β_prev) < 1e-12 break end
        df.t = t_star.(df.ts, df.te, α, β)
    end
    @info "-------"
    @info "近似結果"
    slope = norm(∇ℓ_binom(df.n, df.k1, df.t, α, β))
    @insp α, β, slope, β_ci
    return α, β, slope, β_ci
end
@info logitreg_pair

function logitreg(variants_df)
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
    for base ∈ base_variant_names
        if !has_name(variants_df, base) continue end
        for var ∈ variant_names
            if !has_name(variants_df, var) continue end
            if var == base continue end
            α_j, β_j, slope_j, β_ci_j = logitreg_pair(variants_df, var, base; method=:glm_julia)
            α_r, β_r, slope_r, β_ci_r = logitreg_pair(variants_df, var, base; method=:glm_r)
            α_s, β_s, slope_s, β_ci_s = logitreg_pair(variants_df, var, base; method=:nrm)
            push!(df, [
                var, base,
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
    return df
end
@insp logitreg

@info "--------"
@info "計算結果保存用関数"

function save_jld2(region, variants_df, logitreg_df)
    filepath = jld2_filepaths[region]
    jldopen(filepath, "w") do jld2_file
        jld2_file["variant_names"]      = variant_names
        jld2_file["base_variant_names"] = base_variant_names
        jld2_file["variants_df"]        = variants_df
        jld2_file["logitreg_df"]        = logitreg_df
    end
end
@insp save_jld2

@info "--------"
@info "メインルーチン"

function main(region)
    @assert region ∈ capable_regions
    @info "========"
    @info "メインルーチン"
    @info "--------"
    @info "CSV ファイル読み込み"
    variants_df = load_csv(region)
    @info "読み込んだデータフレーム"
    @insp variants_df
    @info "--------"
    @info "データ中の最新の日付"
    date_latest = variants_df.date_end[end]
    @insp date_latest
    date_latest_suffix_form = Dates.format(date_latest, "yymmdd")
    @insp date_latest_suffix_form
    @info "--------"
    @info "統計量を算出"
    stat!(variants_df)
    @insp size(variants_df)
    @info "--------"
    @info "パラメーター探索"
    logitreg_df = logitreg(variants_df)
    @insp size(logitreg_df)
    @info "--------"
    @info "解析データ書き出し"
    @info "  データをJLD2形式のファイルとして出力する"
    save_jld2(region, variants_df, logitreg_df)
end
@insp main

@info "========"
@info ""
@info "Variants.main(:tokyo) または Variants.main(:osaka) として実行する"
@info ""
@info "========"

end #module

##