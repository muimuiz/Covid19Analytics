# Covid19Analytics

日本におけるCOVID-19の感染状況を、政府・自治体が発表する感染確認者数などのデータにもとづいて分析し、プロットするプログラム群。
現在は、変異株に関する分析ツールのみ公開している。

## Variants_*.jl

東京都（および一部大阪府）における変異株の割合を、自治体公表のゲノム解析データ *1 *2 から分析し予測する。

言語は Julia を用いており、処理は以下のような複数のファイルと module に分割されている。
なお CSV ファイルはサンプルであり、Github 上での更新の予定はない。

### [Variants.jl](Variants.jl)

変異ゲノム検査の結果をもとに手動で作成した CSV を読み込み、割合やオッズといった統計量を算出して DataFrame とする。
2変異株の比の対数（ロジット）が時間に関する 1 次関数となるというロジット・モデルを元に、ロジスティック回帰手法によってパラメーター空間を探索して最尤値のパラメーターを求める。
探索は Julia の GLM（一般化線形モデル）、R 言語の GLM、および Julia で組んだニュートン＝ラフソン法の3種で行う（これらは理論的にはすべて同じものを求めるが、結果の妥当性を相互に検証するために多重に計算している）。
データが時系列の一定区間ごとのビンの形で集計されているため、時間に関して補正をかけ、繰り返しパラメーター探索を行う。
この点を含むコードに含まれる数学的関係の導出に関しては別ファイルを参照*3。
プログラムの実行のためには、module 内のメイン関数を地域を表すシンボル引数をつけて実行する (Variants.main(:tokyo))。

### [Variants_Plots.jl](Variants_Plots.jl)

上プログラムで作成されたデータに基づき、基準株に対する他の株のロジットの回帰直線プロットを作成する。

### [Variants_Pred.jl](Variants_Pred.jl)

上プログラムで求めた回帰パラメータをもとに、変異株の感染力の差からこれまでと今後の対数増加率を求め、実際の値と比較するプロットを作成する。
感染確認者数から対数増加率を求めた別 CSV データが必要（生成プログラムは現在未公開）。
また、各株の存在比の予測プロットを作成する。

*1 [【令和5年度】東京都新型コロナウイルス感染症モニタリング会議・分析資料](https://www.bousai.metro.tokyo.lg.jp/taisaku/saigai/1023407/index.html); [【令和4年度】東京都新型コロナウイルス感染症モニタリング会議・分析資料](https://www.bousai.metro.tokyo.lg.jp/taisaku/saigai/1021348/index.html) （東京都防災ホームページ）

*2 [新型コロナウイルス感染症患者の発生状況について](https://www.pref.osaka.lg.jp/iryo/osakakansensho/happyo.html); [（過去分）](https://www.pref.osaka.lg.jp/iryo/osakakansensho/happyo_kako.html) (大阪府)

*3 [時間区間で集計されたデータに対する正規分布近似を行わないロジスティック回帰の数値的解法についての覚書き](docs/logitreg.md)
