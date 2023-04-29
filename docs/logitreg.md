# 時間区間で集計されたデータに対する正規分布近似を行わないロジスティック回帰の数値的解法についての覚書き

複数の時間区間（ビン）における成功または失敗の結果を持つベルヌーイ試行のデータに対して、成功確率のロジット、すなわち、成功数の失敗数に対するオッズの対数が時間の線形関数で推移するというモデルの下で、正規分布近似を行わず二項分布のもとでパラメーターの最尤推定を数値的に行うロジスティック回帰法を考える。
動機となっている応用例はそれぞれの変異株が異なる対数増加率で指数関数的に推移する COVID-19 の変異株の置き換わりの事態を記述することである。
このモデルの際、各ビン中では成功確率がパラメーターに依存しないという近似のもとで、二項分布の対数尤度からコスト関数を構成し、最小値問題を解決する過程を繰り返すことにより、モデル・パラメーターによる分布パラメーターの依存性を補正しながら最適値を求める。
ただし、この覚書きは、命題の証明の記述が不十分である。

## 標準シグモイド関数とロジット関数

はじめに、この覚書きで頻出する3つの関数について特に関数名を定め、その性質を列挙しておく。
まず、標準シグモイド関数（ロジスティック関数）と、その逆関数であるロジットとを表す記号としてそれぞれ $\mathrm{sigmoid},\ \mathrm{logit}$ を用いる。

__［定義］__
実数値関数である標準シグモイド関数（ロジスティック関数）と、 $(0, 1)$ を定義域とするロジットとを以下のように定める。

$$
\begin{equation*}\begin{split}
\mathrm{sigmoid}\ \lambda
  & = \frac{e^\lambda}{e^\lambda + 1}
    = \frac{1}{1 + e^{-\lambda}},\\
\mathrm{logit}\ p
  & = \log\frac{p}{1 - p}
    = \log p - \log(1 - p). \qquad (0 < p < 1)\\
\end{split}\end{equation*}
$$

__［命題］__
$\mathrm{sigmoid}$ と $\mathrm{logit}$ は互いの逆関数である。

$$
\begin{equation*}\begin{split}
\mathrm{sigmoid}(\mathrm{logit}\ p) & = p, \qquad (0 < p < 1)\\
\mathrm{logit}(\mathrm{sigmoid}\ \lambda) & = \lambda.\\
\end{split}\end{equation*}
$$

また、しばしばソフトプラスと呼ばれる次の関数を $\mathrm{softplus}$ と表すものとする。

__［定義］__
実関数 $\mathrm{softplus}$ を以下のように定める。

$$
\mathrm{softplus}\ \lambda = \log(e^\lambda + 1).
$$

__［命題］__
$\mathrm{sigmoid},\ \mathrm{logit},\ \mathrm{softplus}$ に関して、以下の関係が成り立つ。

$$
\begin{equation*}\begin{split}
\mathrm{sigmoid}\ \lambda & \to 0 \qquad (\lambda \to -\infty),\\
\mathrm{sigmoid}\ 0 & = \frac{1}{2},\\
\mathrm{sigmoid}\ \lambda & \to 1 \qquad (\lambda \to +\infty),\\
\mathrm{sigmoid}(- \lambda) & = 1 - \mathrm{sigmoid}\ \lambda,\\
e^\lambda
  & = \frac{\mathrm{sigmoid}\ \lambda}{1 - \mathrm{sigmoid}\ \lambda},\\
\int \mathrm{sigmoid}\ \lambda\ d\lambda
  & = \mathrm{softplus}\ \lambda + \mathrm{const.}
    = \log(e^\lambda + 1) + \mathrm{const.},\\
\frac{d}{d\lambda}\mathrm{sigmoid}\ \lambda
  & = \frac{e^\lambda}{(e^\lambda + 1)^2}
    = \mathrm{sigmoid}\ \lambda \cdot (1 - \mathrm{sigmoid}\ \lambda),\\
\mathrm{logit}\ p & \to -\infty \qquad (p \to 0^+),\\
\mathrm{logit}\frac{1}{2} & = 0,\\
\mathrm{logit}\ p & \to +\infty \qquad (p \to 1^-),\\
\mathrm{logit}(1 - p) & = - \mathrm{logit}\ p,\\
\log x
  & = \mathrm{logit}\frac{x}{x + 1}, \qquad (x > 0)\\
\int \mathrm{logit}\ p\ dp
  & = p \log p + (1 - p)\log(1 - p) + \mathrm{const.},\\
\frac{d}{dp}\mathrm{logit}\ p
  & = \frac{1}{p\ (1 - p)},\\
\mathrm{softplus}\ \lambda
  & = e^\lambda + O((e^\lambda)^2) \qquad (\lambda \ll 0),\\
\mathrm{softplus}\ \lambda
  & = \lambda + e^{-\lambda} + O((e^{-\lambda})^2) \qquad (\lambda \gg 0),\\
\mathrm{softplus}(- \lambda)
  & = \mathrm{softplus}\ \lambda - \lambda.
\end{split}\end{equation*}
$$

上のように、 $\mathrm{sigmoid}$ の積分が $\mathrm{softplus}$ であり、その逆関数 $\mathrm{logit}$ の積分は（符号を変えた）シャノン・エントロピーとなる。

## モデルとデータ

__［仮定］__
独立変数を1次元の時間 $t$ とみなし、ベルヌーイ試行の成功確率が時間に関するロジスティック関数に従って変化するようなモデルを考える。
すなわち、成功確率のロジット $\lambda$ が、次のようなパラメーター $\alpha, \beta$ を持つ時刻 $t$ の線形関数として表される。

$$
\begin{equation*}\begin{split}
\lambda(t;\ \alpha,\ \beta) & = \alpha + \beta\ t.
\end{split}\end{equation*}
$$

__［例］__
指数関数的な拡大を示す感染症に対して、経時的に無作為な検査を行って感染者を見出す場合、非感染者に対する感染者の数の比（オッズ）の対数（ロジット）は、上のように線形関数として推移すると考えることができる。
感染症に感染力の異なる複数の変異株があり、ゲノム検査等によって3つ以上の変異株を特定する場合も、そのうち2つの変異株に注目し、一方の株に対する他方の株の数の比の対数を考えると同様となる。
ロジットが線形で推移するというモデルが適切とみなせる理論的背景および経験的データに関して、この覚書きでは省略する。

ここでの目標は、与えられたデータから、尤度を最大とするようなパラメーター $\alpha$ と $\beta$ を見出すことである。
すなわち、問題は一般的なロジスティック回帰モデルを用いた最尤法によるパラメーター推定となるが、ここで扱う問題の制約は、上のモデルで示される対象に対する試行の結果が、次に示すようにデータが時間の間隔の区切り（ビン）で集計され、その試行回数と成功回数として与えられている場合を特に考えることである。

__［仮定］__
データ $D$ は、各ビン区別する添字 $i$ をつけて、開始時刻、終了時刻、試行回数、成功回数の組で与えられる。

$$
\begin{equation*}\begin{split}
D & = \left(\ (t_{S1}, t_{E1}, n_1, k_1),\ (t_{S2}, t_{E2}, n_2, k_2), \cdots, (t_{SN}, t_{EN}, n_N, k_N)\ \right)\\
  & \mathrm{where}\quad t_{Si} < t_{Ei}, \quad k_i, n_i \in \mathbb{Z}_{\geq 0},\ 0 \leq k_i \leq n_i, \quad i = 1, 2, \cdots, N.
\end{split}\end{equation*}
$$

ただし、 $N \in \mathbb{Z}_{> 0}$ はビンの総数を表す。
回帰が意味を持つためには、 $N ≥ 2$ である必要がある。

## 確率分布

試行における成功確率の時間的変化がない $(\beta = 0)$ 場合、各ビンのデータは成功確率が一定の通常のベルヌーイ試行の結果であって、成功回数の分布は二項分布に従う。
一方、モデルのように成功確率が時間に関して変化する場合でも、それが確率的に独立に選択される場合には同様であり、ビンの成功数は単一の成功確率をもつベルヌーイ過程のように扱える。
すなわち、

__［定理］__
$n$ 回のベルヌーイ試行に対し、各試行の成功確率自体が確率変数 $P$ であり、確率が $[0, 1]$ を定義域とする累積分布関数 $F$ を用いて、

$$
\begin{equation*}\begin{split}
\mathrm{Pr}(P \leq p)
  & = F(p), \qquad (0 \leq p \leq 1)
\end{split}\end{equation*}
$$

で与えられるとき、 $n$ 試行全体での成功数 $K$ は、 $P$ の期待値 $\langle p\rangle$、

$$
\begin{equation*}\begin{split}
\langle p\rangle = \mathrm{E}[P] & = \int_{0}^{1} dF(p),
\end{split}\end{equation*}
$$

を成功確率とする二項分布に従う確率変数である。

$$
\begin{equation*}\begin{split}
\mathrm{Pr}(K = k)
  & = \mathrm{Bin}(n, k; \langle p\rangle) = \binom{n}{k} \langle p\rangle^k (1 - \langle p\rangle)^{n - k}.
\end{split}\end{equation*}
$$

（証明）略。

__［定義］__
この覚書きでは、ひとつのビンを代表したこの成功確率を $p^\star$ と表すものとする。

$$
\begin{equation*}\begin{split}
p^\star & = \langle p\rangle.
\end{split}\end{equation*}
$$

__［仮定］__
ひとつのビン内において各試行の時刻はビンの時間区間の中から一様分布で選ばれているものと仮定する。

__［系］__
時間の関数である試行の成功確率 $p$ に対して、時間区間 $[t_S, t_E]\;(t_S < t_E)$ において試行が一様分布で独立に選ばれているとき、その区間に対する成功確率 $p^\star$ を考えることができる。

$$
\begin{equation}\begin{split}
p^\star(t_S, t_E; \alpha, \beta)
  & = \frac{1}{t_E - t_S} \int_{t_S}^{t_E} p(t; \alpha, \beta)\ dt\\
  & = \frac{1}{t_E - t_S} \int_{t_S}^{t_E} \mathrm{sigmoid}\ \lambda(t; \alpha, \beta)\ dt\\
  & = \begin{dcases}
    \frac{\mathrm{softplus}\ \lambda_E - \mathrm{softplus}\ \lambda_S}{\lambda_E - \lambda_S}, & \quad (\beta \neq 0)\\
    \mathrm{sigmoid}\ \alpha. & \quad (\beta = 0)
  \end{dcases}
\end{split}\tag{A1}\end{equation}
$$

ただし、 $\lambda_S = \lambda(t_S; \alpha, \beta) = \alpha + \beta\ t_S,\ \lambda_E = \lambda(t_E; \alpha, \beta) = \alpha + \beta\ t_E$ を意味する。

__［定義］__
$p^\star$ に対して、次の関係で結ばれるロジットを $\lambda^\star$ と表す。

$$
\begin{equation*}\begin{split}
p^\star & = \mathrm{sigmoid}\ \lambda^\star,\\
\lambda^\star & = \mathrm{logit}\ p^\star.\\
\end{split}\end{equation*}
$$

また、 $t^\star$ を次の関係で結ばれるものとする。

$$
\begin{equation*}\begin{split}
\lambda^\star & = \alpha + \beta t^\star, \qquad (\beta \neq 0)\\
t^\star
  & = \begin{dcases}
    \frac{\lambda^\star - \alpha}{\beta}
    = \frac{\mathrm{logit}\ p^\star - \alpha}{\beta}, & \quad (\beta \neq 0)\\
    \frac{t_S + t_E}{2}. & \quad (\beta = 0)
  \end{dcases}
\end{split}\tag{A2}\end{equation*}
$$

時間区間 $[t_S, t_E]$ ひとつのビンの成功確率が $p^\star$ で代表されるので、独立な一様分布としてビン中の時刻が選ばれているという仮定のもとで、各ビンのデータが従う分布はそれぞれ対応する単一の時刻 $t^\star$ で定まる二項分布として処理することができる。

__［命題］__
次の大小関係が成り立つ。

$$
\begin{alignat*}{3}
0 < p_S < &\ p^\star && < p_E < 1, & \qquad (\beta > 0)\\
0 < p_E < &\ p^\star && < p_S < 1, & \qquad (\beta < 0)\\
\lambda_S < &\ \lambda^\star && < \lambda_E, & \qquad (\beta > 0)\\
\lambda_E < &\ \lambda^\star && < \lambda_S, & \qquad (\beta < 0)\\
t_S < &\ t^\star && < t_E. & \qquad (\beta \neq 0)
\end{alignat*}
$$

__［仮定］__
$t^\star$ は $\alpha, \beta$ に依存するが、その変化は小さいと考える。
以下でこれを一定とみなして議論する場合、単に $t$ と表し、それから導かれる $p^\star,\ \lambda^\star$ もそれぞれ $p,\ \lambda$ で表す。

__［命題］__
$p$ の $\alpha,\ \beta$ による微分は次のようになる。

$$
\begin{equation*}\begin{split}
\frac{\partial p}{\partial \alpha}
  & = \frac{\partial p}{\partial \lambda} \cdot \frac{\partial \lambda}{\partial \alpha}
    = \frac{\partial}{\partial \lambda} \frac{e^\lambda}{e^\lambda + 1} \cdot \frac{\partial}{\partial \alpha}(\alpha + \beta\ t)\\
  & = \frac{e^\lambda}{{\left(e^\lambda + 1\right)^2}}
    = p\ (1 - p),\\
\frac{\partial p}{\partial \beta}
  & = \frac{\partial p}{\partial \lambda} \cdot \frac{\partial \lambda}{\partial \beta}
    = \frac{\partial}{\partial \lambda} \frac{e^\lambda}{e^\lambda + 1} \cdot \frac{\partial}{\partial \beta}(\alpha + \beta\ t)\\
  & = \frac{e^\lambda}{{\left(e^\lambda+1\right)^2}}\ t
    = p\ (1 - p)\ t.\\
\end{split}\end{equation*}
$$

また、2階微分として次を得る。

$$
\begin{equation*}\begin{split}
\frac{\partial^2 p}{\partial \alpha^2}
  & = p\ (1 - p)\ (1 - 2 p),\\
\frac{\partial^2 p}{\partial \alpha \partial \beta}
  & = p\ (1 - p)\ (1 - 2 p)\ t,\\
\frac{\partial^2 p}{\partial \beta^2}
  & = p\ (1 - p)\ (1 - 2 p)\ t^2.
\end{split}\end{equation*}
$$

## 二項分布の確率の対数

先に見たように、ビン内で試行数 $n$ に対し成功数 $k$ となる確率関数は、成功確率を $p^\star$ とする二項分布 $f(n, k; p^\star) = \mathrm{Bin}(n, k; p^\star)$ となる。

__［定義］__
$f(n, k; p^\star)$ の対数の負値（自己情報量）を $\alpha,\ \beta$ の関数とみて $h$ で表す。

$$
\begin{equation*}\begin{split}
h(\alpha, \beta; t, n, k)
  & = - \log f(n, k; p^\star)\\
  & = - \log \binom{n}{k} - k \log p - (n - k) \log (1 - p)\\
  & = - \log \binom{n}{k} - k\ (\alpha + \beta\ t) + n \log(e^{\alpha + \beta\ t} + 1).
\end{split}\end{equation*}
$$

__［命題］__
表式から明らかなように、パラメーターの空間において $(\alpha, \beta)$ が $\alpha + \beta\ t = \mathrm{const.}$ である方向で $h$ は不変となる。

__［命題］__
$p$ による $h$ の微分は $f$ の対数微分の符号を変えたものであり、次が成り立つ。

$$
\begin{equation*}\begin{split}
\frac{\partial h}{\partial p}
    = - \frac{\partial}{\partial p} \log f
  & = - \frac{1}{f} \cdot \frac{\partial f}{\partial p}
    = \frac{n p - k}{p\ (1 - p)}.
\end{split}\end{equation*}
$$

また、2階微分は次のようになる。

$$
\begin{equation*}\begin{split}
\frac{\partial^2 h}{\partial p^2}
    = - \frac{\partial^2}{\partial p^2} \log f
  & = \frac{n p^2 - 2 k p + k}{p^2\ (1 - p)^2}.
\end{split}\end{equation*}
$$

__［補題］__
$h$ の $\alpha,\ \beta$ による微分として、次の簡明な式を得る。

$$
\begin{alignat*}{3}
\frac{\partial h}{\partial \alpha}
  & = \frac{\partial h}{\partial p} \cdot \frac{\partial p}{\partial \alpha}
  && = n p - k,\\
\frac{\partial h}{\partial \beta}
  & = \frac{\partial h}{\partial p} \cdot \frac{\partial p}{\partial \beta}
  && = (n p - k)\ t.
\end{alignat*}
$$

__［補題］__
2階微分は以下のように表せる。 $k$ に依存しないことに注意。

$$
\begin{alignat*}{3}
\frac{\partial^2 h}{{\partial \alpha}^2}
  & = \frac{\partial^2 h}{{\partial p}^2} \cdot \left(\frac{\partial p}{\partial \alpha}\right)^2 + \frac{\partial h}{\partial p} \cdot \frac{\partial^2 p}{{\partial \alpha}^2}
  && = n p\ (1 - p),\\
\frac{\partial^2 h}{\partial \alpha \partial \beta}
  & = \frac{\partial^2 h}{{\partial p}^2} \cdot \frac{\partial p}{\partial \alpha} \cdot \frac{\partial p}{\partial \beta} + \frac{\partial h}{\partial p} \cdot \frac{\partial^2 p}{\partial \alpha \partial \beta}
  && = n p\ (1 - p)\ t,\\
\frac{\partial^2 h}{{\partial \beta}^2}
  & = \frac{\partial^2 h}{{\partial p}^2} \cdot \left(\frac{\partial p}{\partial \beta}\right)^2 + \frac{\partial h}{\partial p} \cdot \frac{\partial^2 p}{{\partial \beta}^2}
  && = n p\ (1 - p)\ t^2.
\end{alignat*}
$$

__［定理］__
$h$ は $(\alpha, \beta)$ に関し凸関数である。

（略証） $h$ のヘッセ行列は固有値 $0$ と $n p\ (1 - p)\ (t^2 + 1)$ をもつ非負定値行列である。■

__［系］__
以下を満たす $(\alpha, \beta)$ 平面の直線上で $h$ は最小となる。

$$
\begin{equation*}\begin{split}
\alpha + \beta\ t
  & = \mathrm{logit}\frac{k}{n}.\\
\end{split}\end{equation*}
$$

なお、このとき $h$ の最小値 $h_\mathrm{min}$ は以下。

$$
\begin{equation*}\begin{split}
h_\mathrm{min} & = - \log \binom{n}{k} - k\ \log\frac{k}{n ― k} + n \log\frac{n}{n - k}.
\end{split}\end{equation*}
$$

## 対数尤度

__［定義］__
データ全体 $D$ に対するモデルの尤度 $L$ とその対数尤度 $\ell$ を次のように定める。

$$
\begin{equation*}\begin{split}
L & = \prod_i f(n_i, k_i; p_i)
    = \prod_i f_i,\\
\ell
  & = - \log L
    = \sum_i h(n_i, k_i; p_i)
    = \sum_i h_i.\\
\end{split}\end{equation*}
$$

__［定理］__
対数尤度 $\ell$ は凸関数である。

（略証） $\ell$ は凸関数 $h_i$ の和であり、凸関数の和は凸関数であることから導かれる。■

__［命題］__
対数尤度のパラメーター $\alpha,\ \beta$ による微分、2階微分は以下のように表せる。

$$
\begin{equation}\begin{split}
\frac{\partial\ell}{\partial \alpha}
  & = \sum_i \frac{\partial h_i}{\partial \alpha}
    = \sum_i (n_i p_i - k_i),\\
\frac{\partial\ell}{\partial \beta}
  & = \sum_i \frac{\partial h_i}{\partial \beta}
    = \sum_i (n_i p_i - k_i)\ t_i.
\end{split}\tag{B1}\end{equation}
$$

$$
\begin{alignat*}{3}
\frac{\partial^2 \ell}{{\partial \alpha}^2}
  & = \sum_i \frac{\partial^2 h_i}{{\partial \alpha}^2}
  && = \sum_i n_i\ p_i\ (1 - p_i),\\
\frac{\partial^2 \ell}{\partial \alpha \partial \beta}
  & = \sum_i \frac{\partial^2 h_i}{\partial \alpha \partial \beta}
  && = \sum_i n_i\ p_i\ (1 - p_i)\ t_i,\\
\frac{\partial^2 \ell}{{\partial \beta}^2}
  & = \sum_i \frac{\partial^2 h_i}{{\partial \beta}^2}
  && = \sum_i n_i\ p_i\ (1 - p_i)\ {t_i}^2.
\end{alignat*}\tag{B2}
$$

## 数値解法

対数尤度 $\ell$ はパラメーター空間において凸関数であり、パラメーターの全域的な最小値は標準的な最適化問題の数値解法を用いて効率よく求めることが可能である。
ただし、各ビンの時刻を代表するべき $t^\star$ がパラメーターに依存するため、最尤パラメーターを求めるには、現在近似しているパラメーター値から各ビンに対する $t^\star$ を求め、 $t^\star$ が一定であるとの仮定のもとで近似パラメーターの極値を求めるというループを必要精度まで繰り返す。

### $t^\star$ の算出

近似パラメーター $\alpha, \beta$ から $t^\star$ を求めるための関係式 (A1), (A2) は計算過程に指数関数を含み、ある程度大きな $\lambda_S, \lambda_E$ に対してオーバーフローするため、そのまま計算するのには向かない。
実際の計算には以下の関係と近似値を利用する。

__［命題］__
次の対称性が成り立つ。

$$
\begin{equation*}\begin{split}
\mathrm{logit}\frac{\mathrm{softplus}\ \lambda_E - \mathrm{softplus}\ \lambda_S}{\lambda_E - \lambda_S}
  & = -\mathrm{logit}\frac{\mathrm{softplus}(-\lambda_E) - \mathrm{softplus}(-\lambda_S)}{- \lambda_E + \lambda_S} \qquad (\lambda_S \neq \lambda_E)
\end{split}\end{equation*}
$$

よって、 $\lambda^\star$ を求めるために (A!) 式中の $\mathrm{softplus}$ 関数の引数は双方が正とならないようにできる。

また、 $\mathrm{softplus}$ 関数を広い範囲で求めることができるライブラリを活用でき、例えば Julia 言語では、 $\mathtt{LogExpFunctions}$ モジュールの $\mathtt{log1pexp()}$ 関数がこれにあたる。

__［命題］__
次の近似が成り立つ。

$$
\begin{equation*}\begin{split}
\lambda^\star
  & \approx \log(e^{\lambda_E} - e^{\lambda_S}) - \log(\lambda_E - \lambda_S). \qquad (\lambda_S < \lambda_E \ll 0)
\end{split}\end{equation*}
$$

Julia では、右辺第一項のような計算を行うために $\mathtt{LogExpFunctions}$ の関数 $\mathtt{logsubexp()}$ が利用できる。

### 時間のスケール

最も単純に最急降下法を用いる場合は、勾配と更新量 $r_j\ (> 0)$ にもとづき、次のようにパラメーターを変化させる。

$$
\begin{equation*}\begin{split}
\begin{pmatrix}\alpha_{j+1}\\\ \beta_{j+1}\end{pmatrix}
  & = \begin{pmatrix}\alpha_j\\\ \beta_j\end{pmatrix} - r_j \begin{pmatrix}\partial_\alpha \ell_j\\\ \partial_\beta \ell_j\end{pmatrix}.
\end{split}\end{equation*}
$$

一般には、 $r_j$ を適切な正定値行列とし、各ビンの $h_i$ の不変方向が均等かつ互いに十分な角度を持つように変換した方がよいが、予め時間をスケールした方が簡明と思われる。
このために、データ $D$ に関してビンの時間の平均を $0$ 程度とし、標準偏差を、

$$
\begin{equation*}\begin{split}
\frac{1}{\sqrt{3}} \tan\left(\frac{N - 1}{N} \cdot \frac{\pi}{2}\right)
\end{split}\end{equation*}
$$

程度となるよう線形変換する。ただし、 $N$ は $D$ に含まれるビンの数を表す。
これは、各ビンに対応する対数尤度の不変な方向を線形変換の範囲でなるべく均等に配置することを目指したものである。

### 最適化手法

各ビンの $t^\star$ を求め、固定すれば、R 言語などの $\mathtt{glm}$ (genenralized linear model) における標準的なロジスティック回帰を用いてパラメーターの最尤値を探索できる。
$\mathtt{glm}$ のロジスティック回帰ではリンク関数としてロジット関数を、ファミリーとして二項分布を指定することになる。

最急降下法を用いる場合は以下の手順のようになる。

- データの時刻を上述のようにスケールする。
- 近似パラメーターの初期値 $\alpha_0,\ \beta_0$、および、パラメーターの初期更新量 $r_0$ を定める。
- ここから外側のループ
  - 式 (A1), (A2) にもとづき、パラメーター $\alpha_j,\ \beta_j$ からデータの各ビンに対する $p^\star_{ij},\ t^\star_{ij}$ を求める。
  - ここからカウンター $j$ に関する内側のループ
    - 式 (B1) より、 $t^\star_{ij}$ を定数とみなしたときの $\ell$ の勾配 $(\partial_\alpha \ell_j,\ \partial_\beta \ell_j)$ を求める。
  - 勾配が基準に照らして十分に $0$ に近ければ $j$ に関するループを終了。
    - 上式にもとづき、 $\alpha_j,\ \beta_j$ を、 $\alpha_{j+1},\ \beta_{j+1}$ に更新する。
    - 次の更新量 $r_{j+1}$ を適切に定める。
  - $j$ を更新し、ループを繰り返す。
- 求めた最小値のパラメーターが十分に変化していなければ、外側のループを終了。

ここでは、 $t^\star_{ij}$ をパラメーターに依存しないとみなして極値を求める内側のループと、求めたパラメーターから、再び $t^\star_{ij}$ を定める外側のループとの2重の繰り返しを行っている。
両ループとも、発散、振動など力学的な性質により、収束しない可能性もある。

ニュートン法では、近似パラメーター値の更新が式 (B2) で与えられる $\ell_j$ のヘッセ行列の逆行列となる。
すなわち、

$$
\begin{equation*}\begin{split}
\begin{pmatrix}\alpha_{j+1}\\\ \beta_{j+1}\end{pmatrix}
  & = \begin{pmatrix}\alpha_j\\\ \beta_j\end{pmatrix} - \begin{pmatrix}\partial_{\alpha\alpha} \ell_j & \partial_{\alpha\beta} \ell_j\\\ \partial_{\alpha\beta} \ell_j & \partial_{\beta\beta} \ell_j\end{pmatrix}^{-1} \begin{pmatrix}\partial_\alpha \ell_j\\\ \partial_\beta \ell_j\end{pmatrix}.
\end{split}\end{equation*}
$$

$\mathtt{glm}$ を用いる場合は、上アルゴリズムの内側のループを $\mathtt{glm}$ による探索に置き換える。
■

---

<!--
$p_S,\ p_E,\ t_S,\ t_E$ を用いて表せば以下のようになる。

$$
\begin{equation*}\begin{alignat*}{4}
\frac{\partial \hat{p}}{\partial \alpha}
 &  = \hat{p}\ (1 - \hat{p})
 && = \frac{1}{\beta} \cdot \frac{p_E - p_S}{t_E - t_S},\\
\frac{\partial \hat{p}}{\partial \beta}
 &  = \hat{t}\ \hat{p}\ (1 - \hat{p})
 && = \frac{1}{\beta} \left(\frac{t_E\ p_E - t_S\ p_S}{t_E - t_S} - \hat{p}\right).
\end{alignat*}\end{equation*}
$$
-->

<!--
$$
\begin{equation*}\begin{split}
\frac{\partial^2 \hat{p}}{\partial \alpha^2} &
 = \frac{1}{\beta} \cdot \frac{p_E\ (1-p_E) - p_S\ (1-p_S)}{t_E - t_S},\\
\frac{\partial^2 \hat{p}}{\partial \alpha \partial \beta}
 & = \frac{1}{\beta} \cdot \frac{t_E\ p_E\ (1-p_E) - t_S\ p_S\ (1-p_S)}{t_E - t_S} - \frac{1}{\beta^2} \cdot \frac{p_E - p_S}{t_E - t_S}\\
 & = \frac{1}{\beta} \left(\frac{t_E\ p_E\ (1-p_E) - t_S\ p_S\ (1-p_S)}{t_E - t_S} - \frac{\partial \hat{p}}{\partial \alpha}\right),\\
\frac{\partial^2 \hat{p}}{\partial \beta^2}
 & = \frac{1}{\beta} \cdot \frac{{t_E}^2 p_E\ (1-p_E) - {t_S}^2 p_S\ (1-p_S)}{t_E - t_S} - \frac{2}{\beta^2} \left(\frac{t_E\ p_E - t_S\ p_S}{t_E - t_S} - \hat{p}\right)\\
 & = \frac{1}{\beta} \left(\frac{{t_E}^2 p_E\ (1-p_E) - {t_S}^2 p_S\ (1-p_S)}{t_E - t_S} - 2\ \frac{\partial \hat{p}}{\partial \beta}\right).
\end{split}\end{equation*}
$$
-->

<!--
$$
\begin{equation*}\begin{split}
\frac{\partial \phi}{\partial \alpha} &
 = \frac{\partial \phi}{\partial \hat{p}} \cdot \frac{\partial \hat{p}}{\partial \alpha}
 = \frac{k-n\hat{p}}{\hat{p}\ (1-\hat{p})} \cdot \frac{1}{\beta} \cdot \frac{p_E - p_S}{t_E - t_S},\\
\frac{\partial \phi}{\partial \beta} &
 = \frac{\partial \phi}{\partial \hat{p}} \cdot \frac{\partial \hat{p}}{\partial \beta}
 = \frac{k-n\hat{p}}{\hat{p}\ (1-\hat{p})} \cdot \frac{1}{\beta} \left(\frac{t_E\ p_E - t_S\ p_S}{t_E - t_S} - \hat{p}\right).
\end{split}\end{equation*}
$$
-->

<!--
$$
\begin{equation*}\begin{alignat*}{3}
\frac{\partial^2 \phi}{{\partial \alpha}^2}
  & = \frac{\partial^2 \phi}{{\partial q}^2} \left(\frac{\partial q}{\partial \alpha}\right)^2 && + \frac{\partial \phi}{\partial q} \cdot \frac{\partial^2 q}{{\partial \alpha}^2}
  && = (k - n q)^2,\\
\frac{\partial^2 \phi}{\partial \alpha \partial \beta}
  & = \frac{\partial^2 \phi}{{\partial q}^2} \cdot \frac{\partial q}{\partial \alpha} \cdot \frac{\partial q}{\partial \beta} && + \frac{\partial \phi}{\partial q} \cdot \frac{\partial^2 q}{\partial \alpha \partial \beta}
  && = t\ (k - n q)^2,\\
\frac{\partial^2 \phi}{{\partial \beta}^2}
  & = \frac{\partial^2 \phi}{{\partial q}^2} \left(\frac{\partial q}{\partial \beta}\right)^2 && + \frac{\partial \phi}{\partial q} \cdot \frac{\partial^2 q}{{\partial \beta}^2}
  && = t^2(k - n q)^2.
\end{alignat*}\end{equation*}
$$
-->

<!--
$$
\begin{equation*}\begin{split}
\frac{\partial\ell}{\partial \alpha}
 & = \sum_i \frac{\partial \phi_i}{\partial \alpha}
   = \sum_i \frac{\partial \phi_i}{\partial \hat{p}_ i} \cdot \frac{\partial \hat{p}_ i}{\partial \alpha} 
   = \frac{1}{\beta} \sum_i \frac{k_i - n_i \hat{p}_ i}{\hat{p}_ i (1 - \hat{p}_ i)} \cdot \frac{p_{Ei} - p_{Si}}{t_{Ei} - t_{Si}} \\
\frac{\partial\ell}{\partial \beta}
 & = \sum_i \frac{\partial \phi_i}{\partial \beta}
   = \sum_i \frac{\partial \phi_i}{\partial \hat{p}_ i} \cdot \frac{\partial \hat{p}_ i}{\partial \beta}
   = \frac{1}{\beta} \sum_i \frac{k_i - n_i \hat{p}_ i}{\hat{p}_ i (1 - \hat{p}_ i)} \left(\frac{t_{Ei}\ p_{Ei} - t_{Si}\ p_{Si}}{t_{Ei} - t_{Si}} - \hat{p}_ i\right).  \\
\end{split}\end{equation*}
$$
-->

<!--
__［定義］__
上式の係数 $1/\beta$ を除いた量として $\mu_\alpha,\  \mu_\beta$ を定める。

$$
\begin{equation*}\begin{split}
\mu_\alpha
 & = \beta\ \frac{\partial\ell}{\partial\alpha}
   = \sum_i \frac{\partial \phi_i}{\partial \hat{p}_ i} \cdot \beta\ \frac{\partial \hat{p}_ i}{\partial \alpha}
   = \sum_i \frac{k_i - n_i \hat{p}_ i}{\hat{p}_ i (1 - \hat{p}_ i)} \cdot \frac{p_{Ei} - p_{Si}}{t_{Ei} - t_{Si}},\\
\mu_\beta
 & = \beta\ \frac{\partial\ell}{\partial\beta}
   = \sum_i \frac{\partial \phi_i}{\partial \hat{p}_ i} \cdot \beta\ \frac{\partial \hat{p}_ i}{\partial \beta}
   = \sum_i \frac{k_i - n_i \hat{p}_ i}{\hat{p}_ i (1 - \hat{p}_ i)} \left(\frac{t_{Ei}\ p_{Ei} - t_{Si}\ p_{Si}}{t_{Ei} - t_{Si}} - \hat{p}_i\right).
\end{split}\end{equation*}
$$

__［命題］__
2次元関数 $\mu = (\mu_\alpha, \mu_\beta)$ のヤコビ行列を与えるものとして、このパラメーターによる微分は以下から導かれる。
$$
\begin{equation*}\begin{alignat*}{3}
\frac{\partial \mu_\alpha}{\partial \alpha}
 & = \beta \frac{\partial^2 \ell}{{\partial \alpha}^2}
 && = \beta\ \sum_i \left\{\frac{\partial^2 \phi_i}{{\partial \hat{p}_ i}^2} \left(\frac{\partial \hat{p}_ i}{\partial \alpha}\right)^2 + \frac{\partial \phi_i}{\partial \hat{p}_ i} \cdot \frac{\partial^2 \hat{p}_ i}{{\partial \alpha}^2}\right\},\\
\frac{\partial \mu_\alpha}{\partial \beta}
 = \frac{\partial \mu_\beta}{\partial \alpha}
 & = \beta \frac{\partial^2 \ell}{\partial \alpha \partial \beta}
 && = \beta\ \sum_i \left\{\frac{\partial^2 \phi_i}{{\partial \hat{p}_ i}^2} \cdot \frac{\partial \hat{p}_ i}{\partial \alpha} \cdot \frac{\partial \hat{p}_ i}{\partial \beta} + \frac{\partial \phi_i}{\partial \hat{p}_ i} \cdot \frac{\partial^2 \hat{p}_ i}{\partial \alpha \partial \beta}\right\},\\
\frac{\partial\mu_\beta}{\partial\beta}
 & = \beta \frac{\partial^2 \ell}{{\partial \beta}^2}
 && = \beta\ \sum_i \left\{\frac{\partial^2 \phi_i}{{\partial \hat{p}_ i}^2} \left(\frac{\partial \hat{p}_ i}{\partial \beta}\right)^2 + \frac{\partial \phi_i}{\partial \hat{p}_ i} \cdot \frac{\partial^2 \hat{p}_ i}{{\partial \beta}^2}\right\}.
\end{alignat*}\end{equation*}
$$
-->
