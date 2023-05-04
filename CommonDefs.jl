module CommonDefs

using Dates
using DataFrames
using Statistics
using LogExpFunctions

#--------
# nothing & missing

Nothingable{T} = Union{Nothing, T}
Nullable{T}    = Nothingable{T}
export Nothingable, Nullable
@assert Nothingable{Int} == Nullable{Int} == Union{Int, Nothing}

ArrayNothing(D, T, ns...) = Array{Nothingable{T},D}(fill(nothing, ns...))
VectorNothing(T, n1)         = ArrayNothing(1, T, n1)
MatrixNothing(T, n1, n2)     = ArrayNothing(2, T, n1, n2)
Array2Nothing(T, n1, n2)     = ArrayNothing(2, T, n1, n2)
Array3Nothing(T, n1, n2, n3) = ArrayNothing(3, T, n1, n2, n3)
export ArrayNothing, VectorNothing, MatrixNothing, Array2Nothing, Array3Nothing
@assert isa(Array2Nothing(Int, 10, 10), Array{Union{Int, Nothing}, 2})
@assert all(isnothing.(VectorNothing(Int, 10)))

Missable{T} = Union{Missing, T}
export Missable
@assert Missable{Int} == Union{Int, Missing}

ArrayMissing(D, T, ns...) = Array{Missable{T},D}(fill(missing, ns...))
VectorMissing(T, n1)         = ArrayMissing(1, T, n1)
MatrixMissing(T, n1, n2)     = ArrayMissing(2, T, n1, n2)
Array2Missing(T, n1, n2)     = ArrayMissing(2, T, n1, n2)
Array3Missing(T, n1, n2, n3) = ArrayMissing(3, T, n1, n2, n3)
export ArrayMissing, VectorMissing, MatrixMissing, Array2Missing, Array3Missing
@assert isa(Array2Missing(Int, 10, 10), Array{Union{Int, Missing}, 2})
@assert all(ismissing.(VectorMissing(Int, 10)))

@inline is_regular(x) = !ismissing(x) && !isnothing(x) && isfinite(x)
export is_regular

#--------
# nested vectors

NestedVector{T} = Vector{Vector{T}}
export NestedVector
Nest2Vector{T} = NestedVector{T}
Nest3Vector{T} = Vector{Vector{Vector{T}}}
export Nest2Vector, Nest3Vector

#--------
# mutind

function mutind(usvs1::Vector{<:Missable{T}}, usvs2::Vector{<:Missable{T}}) where T <: Any
    is2_v = Dict(v => i for (i, v) in enumerate(usvs2))
    ms_i1 = fill(false, length(usvs1))
    ms_i2 = fill(false, length(usvs2))
    for (i1, v) in enumerate(usvs1)
        if v ∈ keys(is2_v)
            ms_i1[i1] = ms_i2[is2_v[v]] = true
        end
    end
    return BitVector(ms_i1), BitVector(ms_i2)
end
export mutind

function segments(xs::Union{BitVector, Vector{<:Bool}})::Vector{UnitRange{Int}}
    n = length(xs)
    rs = UnitRange{Int}[]
    ie = 1
    while true
        is = findnext(xs, ie)
        if isnothing(is); break; end
        ie = findnext(!, xs, is)
        if isnothing(ie); ie = n + 1; end
        push!(rs, is:(ie-1))
    end
    return rs
end
export segments

#--------
# Δ and movavg

function _Δ(vs::Vector{<:Missable{<:Real}}; spacing, init)
    @assert spacing ≥ 0
    @assert 0 ≤ init ≤ spacing
    n = length(vs)
    @assert spacing < n
    Δs = VectorMissing(Float64, n)
    for i in 1:(n-spacing)
        Δs[i+init] = (vs[i+spacing] - vs[i]) / spacing
    end
    return Δs
end

@inline Δ_forward(vs; spacing=1)  = _Δ(vs; spacing, init=0)
@inline Δ_backward(vs; spacing=1) = _Δ(vs; spacing, init=spacing)
export Δ_forward, Δ_backward

@inline function Δ_center(vt; spacing=2)
    @assert spacing % 2 == 0
    return _Δ(vs; spacing, init=div(spacing, 2))
end
export Δ_center

function _movavg(vs::Vector{<:Missable{<:Real}}, period; init)
    @assert period ≥ 1
    @assert 0 ≤ init ≤ period-1
    n = length(vs)
    @assert period ≤ n
    mas = VectorMissing(Float64, n)
    for i in 1:(n-period+1)
        mas[i+init] = mean(vs[i:i+period-1])
    end
    return mas
end

@inline movavg_forward(vs, period)  = _movavg(vs, period; init=0)
@inline movavg_backward(vs, period) = _movavg(vs, period; init=period-1)
export movavg_forward, movavg_backward

@inline function movavg(vs, period)
    @assert period % 2 == 1
    return _movavg(vs, period; init=div(init, 2))
end
export movavg

#--------
# interpolate

@inline function interpolate1(x; xa, xb, ya, yb)
    r = (x - xa) / (xb - xa)
    return (yb - ya) * r + ya
end
export interpolate1

struct LinConv
    # a + b * x
    a::Float64
    b::Float64
end
export LinConv

function LinConv(x1::Real, y1::Real, x2::Real, y2::Real)
    @assert x1 ≠ x2
    b = (y2 - y1) / (x2 - x1)
    a = y1 - b * x1
    return LinConv(a, b)
end

@inline conv(lc::LinConv, x::Real) = lc.a + lc.b * x
@inline conv_inv(lc::LinConv, y::Real) = (y - lc.a)/ lc.b
export conv, conv_inv

#--------
# numerical differentiation

function diff_center(f, x, h)
    ff = f(x + h)
    fb = f(x - h)
    return (ff - fb) / (2.0 * h)
end
export diff_center

function diff_five_points(f, x, h)
    ff2 = f(x + h + h)
    ff1 = f(x + h)
    fb1 = f(x - h)
    fb2 = f(x - h - h)
    return (- ff2 + 8.0 * ff1 - 8.0 * fb1 + fb2) / (12.0 * h) 
end
export diff_five_points

function diff_seven_points(f, x, h)
    ff3 = f(x + 3.0 * h)
    ff2 = f(x + h + h)
    ff1 = f(x + h)
    fb1 = f(x - h)
    fb2 = f(x - h - h)
    fb3 = f(x - 3.0 * h)
    return (ff3 - 9.0 * ff2 + 45.0 * ff1 - 45.0 * fb1 + 9.0 * fb2 - fb3) / (60.0 * h)
end
export diff_seven_points

#--------
# linear regression

function linear_regression(x::Array{Float64}, y::Array{Float64})
    # normal equation XᵀX p = Xᵀy
    @assert length(x) == length(y)
    n = length(x)
    X = [(j == 1 ? 1.0 : x[i]) for i ∈ 1:n, j ∈ 1:2]
    return (X' * X) \ (X' * y)
end
export linear_regression

# weighted linear regression
function weighted_linear_regression(x::Array{Float64}, y::Array{Float64}, w::Array{Float64})
    # normal equation (WX)ᵀWX p = (WX)ᵀWy
    @assert length(x) == length(y) == length(w)
    n = length(x)
    WX = [(j == 1 ? w[i] : w[i] * x[i]) for i ∈ 1:n, j ∈ 1:2]
    Wy = [w[i] * y[i] for i ∈ 1:n]
    return (WX' * WX) \ (WX' * Wy)
end
export weighted_linear_regression

#--------
# sigmoid / logit / softplus

@inline sigmoid(x) = LogExpFunctions.logistic(x)
export sigmoid

@inline logit(x) = LogExpFunctions.logit(x)
export logit

@inline softplus(x) = LogExpFunctions.log1pexp(x)
export softplus

#--------
# CI_Value

struct CI_Value
    e::Missable{Float64}
    l::Missable{Float64}
    u::Missable{Float64}
    function CI_Value(e, l, u)
        @assert ismissing(l) || isnan(l) || (l ≤ e)
        @assert ismissing(u) || isnan(u) || (e ≤ u)
        new(e, l, u)
    end
end
export CI_Value

@inline est(x::CI_Value) = x.e
@inline cil(x::CI_Value) = x.l
@inline ciu(x::CI_Value) = x.u
export est, cil, ciu

@inline l_error(x::CI_Value) = est(x) - cil(x)
@inline u_error(x::CI_Value) = ciu(x) - est(x)
export l_error, u_error

#--------
# epoch time

const epoch_date    = Date("2021-01-01")
const epoch_dtime   = DateTime(epoch_date)
const epoch_epochms = Dates.datetime2epochms(epoch_dtime)
@info "epoch_date",    epoch_date
@info "epoch_dtime",   epoch_dtime
@info "epoch_epochms", epoch_epochms

const day_ms = 24 * 60 * 60 * 1000
@info "day_ms", day_ms

@inline dtime_to_value(d::DateTime) = (Dates.datetime2epochms(d) - epoch_epochms) / day_ms
@inline value_to_dtime(v)::DateTime = Dates.epochms2datetime(epoch_epochms + round(Int64, day_ms * v))
export dtime_to_value, value_to_dtime
@assert dtime_to_value(epoch_dtime) == 0.0
@assert value_to_dtime(0.0) == epoch_dtime

@inline date_to_value(d::Date; noon=false) = dtime_to_value(DateTime(d)) + ((noon) ? 0.5 : 0.0)
export date_to_value
@assert date_to_value(epoch_date) == 0.0
@assert date_to_value(epoch_date; noon=true) == 0.5

#--------
# DateInterval

struct DateInterval
    ds::Date
    de::Date
    ts::Float64
    te::Float64
    t::Float64
    function DateInterval(ds, de, ts, te, t)
        @assert ds ≤ de
        @assert ts ≤ t ≤ te
        return new(ds, de, ts, te, t)
    end
end
export DateInterval

function DateInterval(date_start, date_end)
    dts = DateTime(date_start)
    dte = DateTime(date_end + Day(1))
    ts  = dtime_to_value(dts)
    te  = dtime_to_value(dte)
    t   = (ts + te) / 2.0
    return DateInterval(date_start, date_end, ts, te, t)
end

@inline DateInterval_single(date) = DateInterval(date, date)
export DateInterval_single

@inline date_start(x::DateInterval) = x.ds
@inline date_end(x::DateInterval)   = x.de
@inline t_start(x::DateInterval)    = x.ts
@inline t_end(x::DateInterval)      = x.te
@inline t(x::DateInterval)          = x.t
export date_start, date_end, t_start, t_end, t

@inline interval_days(x::DateInterval) = x.de - x.ds + Day(1)
@inline t_interval(x::DateInterval)    = x.te - x.ts
@inline t_error(x::DateInterval)       = t_interval(x) / 2.0
export interval_days, t_interval, t_error

#--------
# requires

@inline has_name(df::DataFrame, name) = !isnothing(findfirst(isequal(name), names(df)))
@inline function requires(df::DataFrame; name, type=Any, pred=nothing)
    return has_name(df, name) && isa(df[!,name], type) && (isnothing(pred) || pred(df[!,name]))
end
@inline function requires(d::Dict; name, type=Any, pred=nothing)
    return haskey(d, name) && isa(d[name], type) && (isnothing(pred) || pred(d[name]))
end
export has_name, requires

#--------
# inspect

macro insp(expr)
    str_expr = string(expr) # compile-time stringing of the expression
    return quote # statements to be expanded to the caller
        print("\e[34m[ Insp:\e[0m ") # with escape sequence to specify color
        print($str_expr * " = ") # interpolate the expression string
        println($(esc(expr))) # interpolate the expression
    end
end
export @insp

end #module
