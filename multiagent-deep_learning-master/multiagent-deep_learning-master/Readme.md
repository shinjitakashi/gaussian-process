MULTI-AGENT DEEP_LEARNING
=========================

書籍『[ゼロから作る Deep Learning](http://www.oreilly.co.jp/books/9784873117584/)』(オライリー・ジャパン発行：著，斎藤康毅)を基に，Multi-agentシステムへの拡張を行ったソースコードです．


## Requirement
ソースコードを実行するには，以下のソフトウェアが必要です．

* Python 3.x
* NumPy
* Matplotlib
* NetworkX 2.x
* retrying
* Progressbar2 (optional)

基となったソースコードに準拠し，Pythonのバージョンは3.x系を利用します．


## Usage
### Training NN
必ずinitialization.pyを実行してからtrain_multiage.pyを実行してください．  また，各パラメータはagent.pyやinitialization.py等を直接編集することにより変更してください．  
```
$ python initialization.py
$ python train_multiage.py
```
データセットはload_mnistメソッドを呼ぶことにより，初回実行時に自動的にダウンロードされます．当該データはdataset/内に保存されます．  

### Format Results
outgraph.pyを実行することにより，通信を行わなかった場合との比較グラフが出力されます．出力にはflag_communicateを変更してtrain_multiage.pyを実行し，データを取得してください．また，エージェントにナンバリングがなされたグラフが必要ならば，別途，labelToAge-Graph.pyを実行してください．
```
$ python outgraph.py
$ python labelToAge-Graph.py
```

### Advanced Settings
* 誤差逆伝搬による勾配計算を数値計算による方法に変更する場合は，Agentクラスのupdateメソッド中で呼び出すメソッドをlayer.numerical_gradientに変更してください．
* NN自体の設定は，Agentクラス内のlayer初期化時の引数によって行います．詳細はcommon/multi_layer_net_extend.pyを参照ください．
* デフォルトではSGDを用いて更新を行っています．他の更新アルゴリズムを用いる場合はcommon/optimizer.pyの対応クラスを必要に応じて書き換え，ないし書き足しを行い，train_multiage.py内でoptimizer変数に代入してください．
* デフォルトでは損失関数に交差エントロピー損失を用いています．2乗和誤差を用いる場合はlayers.pyのSoftmaxWithLoss中のforwardメソッドを書き換えてください．また，それ以外の損失関数を用いる場合は，適宜backwardメソッドも書き直してください．（交差エントロピー誤差と2乗和誤差の逆伝搬は同じだった筈です．違ったらどうかご連絡を．）


## License
[MIT](http://www.opensource.org/licenses/MIT)
