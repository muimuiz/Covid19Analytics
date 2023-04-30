# 複数変異株間のロジットと対数増加率についての覚書き

## 2変異株のロジットの時間推移

単純な場合として、感染力の異なる2つの変異株 0, 1 それぞれの感染者数が時間 $t$ に関して指数関数的に推移している場合を考える。
すなわち、それぞれの感染者数を $x_0(t), x_1(t)$ として、株で決まる定数 $a_i, b_i\ (i = 0, 1)$ を用い、

$$
\begin{equation*}\begin{split}
x_0(t) & = e^{a_0 + b_0 t},\\
x_1(t) & = e^{a_1 + b_1 t},
\end{split}
  \qquad (b_0 \neq b_1)
\end{equation*}
$$

と表す。
2株を合わせた総感染者数 $x(t)$ は、

$$
\begin{equation*}\begin{split}
x(t) & = x_0(t) + x_1(t) = e^{a_0 + b_0 t} + e^{a_1 + b_1 t},
\end{split}\end{equation*}
$$

である。
ここで、定数 $\alpha, \beta$ を、

$$
\begin{equation*}\begin{split}
\alpha & = a_1 - a_0,\\
\beta  & = b_1 - b_0 \quad (\neq 0),
\end{split}\end{equation*}
$$

と置くと、

$$
\begin{equation*}\begin{split}
x_1 & = e^{(a_0 + \alpha) + (b_0 + \beta) t} = x_0\ e^{\alpha + \beta t},
\end{split}\end{equation*}
$$

と表せる。

$x_0$ を基準とする $x_1$ の比（オッズ）の自然対数（ロジット） $\lambda(t)$ を考えると、

$$
\begin{equation*}\begin{split}
\lambda & = \log \frac{x_1}{x_0} = \alpha + \beta t,
\end{split}
  \tag{A}
\end{equation*}
$$

となる。
すなわち、この場合、ロジットは時間に関して線形で推移する。
これは、2株に注目し、それぞれが指数関数的に推移するとした単純な場合において、変異株の感染者数のデータをロジスティック回帰を用いて推定できる根拠を与える。

人の流れのような環境の変化が感染力をそのときどきで変動させるより現実的な場合でも、もしその変動が変異株によらず、0 とならない時間の関数 $m(t)$ を用いて、

$$
\begin{equation*}\begin{split}
x_0(t) & = m(t)\ e^{a_0 + b_0 t},\\
x_1(t) & = m(t)\ e^{a_1 + b_1 t},
\end{split}\end{equation*}
$$

のように表せる場合も、上のロジットの変動における線形の結果 (A) は変わらない。
よって、感染者数そのものの動きに対してロジットの動きは比較的推定・予測しやすいと考えうる。

ただし、実際には、 $m(t)$ のように環境の変動の影響が乗算として記述できない場合もあり、また、感染可能な集団の有限性などから変異株の時間的推移を指数関数として近似することが不適切な場合もあるため、この関係が適用可能かどうかの判断には注意を要する。
より詳細な検討は節を改めて行う。
しかし経験的には、COVID-19 のいくつかの感染の波の初期において、ロジットが線形で推移するとの関係は、実際のデータのよい近似となっており、場合によっては数か月に及ぶよい予測性も持っていることが示されている。

----

## 2株の対数増加率の推移

$x(t)$ の対数微分 $\zeta(t)$ の時間推移を考える。

$$
\begin{equation*}\begin{split}
\zeta(t) = \frac{d}{dt} \log x(t) = \frac{x^\prime}{x}
  & = \frac{b_0 e^{a_0 + b_0 t} + b_1 e^{a_1 + b_1 t}}{e^{a_0 + b_0 t} + e^{a_1 + b_1 t}}\\
  & = \frac{b_0 + (b_0 + \beta) e^{\alpha + \beta t}}{1 + e^{\alpha + \beta t}}\\
  & = b_0 + \beta\ \mathrm{sigmoid}(\alpha + \beta t),
\end{split}
  \tag{B}
\end{equation*}
$$

となる。
ただし、 $\mathrm{sigmoid}$ は標準シグモイド関数（ロジスティック関数）、

$$
\begin{equation*}\begin{split}
\mathrm{sigmoid}\ \lambda & = \frac{e^\lambda}{1 + e^\lambda} = \frac{1}{e^{-\lambda} + 1},
\end{split}\end{equation*}
$$

を表す。

よって、この場合の対数増加率は、株 0 単独の定数値対数増加率 $b_0$ に、大きさ $\beta$ のシグモイド関数が加わることとなる。
すなわち、変異株間の感染力の差である $\beta = b_1 - b_0$ は、変異株の置き換わりの速さを表すとともに、環境が一定の場合の対数増加率の上乗せ分も表している。

----

## 3株以上の場合

互いに感染力の異なる $n\ (> 2)$ 株が存在し、それぞれの感染者数がやはり指数関数的に推移するとする。
すなわち、変異株 $i = 0, \cdots, n - 1$ の感染者数 $x_i(t)$ が、

$$
\begin{equation*}\begin{split}
x_i(t) & = e^{a_i + b_i t}, \qquad (i = 0, \cdots, n - 1)
\end{split}\end{equation*}
$$

と表せるとする。
このうち、任意の2株をとれば、上と同様にその2株に対するロジットは時間に関して直線的に推移する。
よって、実データの2株に注目し、ロジステック回帰を用いることで、そこから2株のパラメーターの差 $a_j - a_i,\ b_j - b_i$ ならば推定することができる。

感染者の総数を、

$$
\begin{equation*}\begin{split}
x(t) & = \sum_{i = 0}^{n - 1} x_i(t),
\end{split}\end{equation*}
$$

で表すとその対数微分 $\zeta(t)$ は、

$$
\begin{equation*}\begin{split}
\zeta(t) = \frac{d}{dt} \log x(t) = \frac{x^\prime}{x}
  & = \frac{\displaystyle\sum_{i = 0}^{n - 1} b_i e^{a_i + b_i t}}{\displaystyle\sum_{i = 0}^{n - 1} e^{a_i + b_i t}},
\end{split}\end{equation*}
$$

となるが、ここで、株 0 を基準にとり、

$$
\begin{equation*}\begin{split}
\alpha_i & = a_i - a_0,\\
\beta_i  & = b_i - b_0,
\end{split}
  \qquad (i = 1, \cdots, n - 1)
\end{equation*}
$$

と置くと、

$$
\begin{equation*}\begin{split}
\zeta(t)
  & = b_0 + \frac{\displaystyle\sum_{i = 1}^{n - 1} \beta_i e^{\alpha_i + \beta_i t}}{\displaystyle 1 + \sum_{i = 1}^{n - 1} e^{\alpha_i + \beta_i t}},
\end{split}
  \tag{C}
\end{equation*}
$$

を得る。
(B) 式のようにシグモイド関数を用いた簡明な表式ではないが、データから $\alpha_i, \beta_i$ が推定できるとき、定数分 $b_0$ を除いて、これから変異株の感染力の差に由来する対数増加率の時間推移を求めることができる。

----

## 感染可能者の有限性および人々の接触機会の変動等、変動要因の影響

前節までの議論は、各変異株が時間に関してそれぞれ指数関数的に推移することを前提としていた。
これは感染の広がりが感染可能な人々 (susceptibles) の数に比べて比較的小さく、かつ、人の接触幾何の変動などに代表される感染性の変動が大きくないときにはよい近似となる。ここでは、そうでない場合、特に2変異株間のロジットの直線的関係がどのように修正されねばならないか定性的に検討する。

### 感染可能者の有限性

感染症のダイナミクスを扱う基礎的な SIR モデルに準じて、感染力が異なる2つの変異株がある場合を記述する微分方程式を考える。
全体の大きさを 1 とし、感染可能者の割合を $y(t)$ （SIR モデルの $S/N$）、2つの株の感染者数の割合をそれぞれ $x_0(t), x_1(t)$ （SIR モデルの $I/N$）とする。
SIR モデルの $\beta N$ にあたる感染力の程度は両株で異なるが、SIR モデルで I から R へ移行する減少の係数は両株で共通だとする。
このとき、この系は、3変数の微分方程式、

$$
\begin{equation*}\begin{split}
y^\prime   & = - (b_0 x_0 + b_1 x_1) y,\\
x_0^\prime & = (b_0 y - \gamma) x_0,\\
x_1^\prime & = (b_1 y - \gamma) x_1,
\end{split}\end{equation*}
$$

として記述される。
ここから $x_0, x_1$ の対数微分は、

$$
\begin{equation*}\begin{split}
(\log x_0)^\prime & = \frac{x_0^\prime}{x_0} = b_0 y - \gamma,\\
(\log x_1)^\prime & = \frac{x_1^\prime}{x_1} = b_1 y - \gamma,
\end{split}\end{equation*}
$$

となる。
よって、株 0 に対する株 1 のロジットの時間変化は、

$$
\begin{equation*}\begin{split}
\left(\log \frac{x_1}{x_0}\right)^\prime
  & = (\log x_1)^\prime - (\log x_0)^\prime = b y(t),
\end{split}\end{equation*}
$$

となる。
ただし、定数 $b = b_1 - b_0$ と置いた。
$y(t)\ (> 0)$ は単調減少するので、上もまた時間に関して単調減少である。
よって、感染可能者の有限性を考える場合、その減少に比例して、ロジットの傾きは直線から外れ鈍っていくと考えうる。

### 感染者数の変動要因

感染者数が変動する要因として次の3つを考える。

1. 検査による隔離
2. 人流など人々の接触機会の変化
3. ワクチンによる能動免疫

最初の検査により陽性者を発見して隔離し、感染を防ぐ場合、もし検査がランダムに行われるのならば、その効果は検査数と感染者数とに比例する。
また、もし一定割合で症状が現れ、受診後の検査の結果、陽性と判明する場合も、感染者数に比例することになる。
よって、検査数に比例する要因1の検査の効果 $s(t)$ は、最も単純には感染者数に比例するものと考えうる。

これに対して、2番目の接触機会の変化と3番目のワクチンの効果は、感染者数とともに感染可能者にも比例すると考えるのが妥当である。
両者の効果を合わせて $r(t)$ と表す。
$r(t)$ は感染機会の量に比例し、効果が大きなほど小さくなる。

また簡単のため $s(t), r(t)$ は2株 0, 1 に対しともに違いはないのだとする。
これにより上述の微分方程式は次のように変更される。

$$
\begin{equation*}\begin{split}
y^\prime   & = - (b_0 x_0 + b_1 x_1) r y,\\
x_0^\prime & = (b_0 r y - \gamma s) x_0,\\
x_1^\prime & = (b_1 r y - \gamma s) x_1,
\end{split}\end{equation*}
$$

ここから、ロジットの時間変化は、

$$
\begin{equation*}\begin{split}
\left(\log \frac{x_1}{x_0}\right)^\prime & = b r(t) y(t),
\end{split}\end{equation*}
$$

となる。
すなわち、要因1の検査による効果 $s(t)$ は打ち消される一方で、要因2と3の感染機会の増減 $r(t)$ は、それに比例してロジットの傾きに影響することとなる。■
