# 目的
- 単一の液状化試験結果から任意の応力経路における応力ひずみ関係を書く

# 今後の方針

- コードの高速化、より実用的に
        - train_dataの読み込みの高速化
            - numpyバイナリに一時的に保存。読み込み時には、可能であれば、ロードを行う
        - vars関数を用いたインスタンスの変数の取得の実装
    - 入力データの分布の確認
        - 特にeaとpが正規分布にはなっていない
            - 入力と出力変数の分布を確認する→2
        - 対数を使って無理やり正規化する?
    - `RNN`のモデルの変更
        - より長いシーケンスに
        - `q_inc`でなく、`ea_inc`を入力にする→1
            - 解が安定しなくなってしまうNo.55や56の結果を参照
        - 損失関数に不等式制約条件を入れる
        - なぜ`RNN`で非線形性が出ないのかを具体的な値を入れて調査する
        ‐ 内挿補完を使って無理やりデータ量を増やす
        - 液状化進行の指標を`CSW`から`累積せん断ひずみ`に変更する
        - 
    - `CNN`の実装
        

# 2024/07/17の解析メモ
- 解が安定しない
- `DataRoader`の使い方があっているかどうか確認。RNNへの入力での0次元目がバッチサイズである必要がある
    - `DataRoader`を使うか、`内包表記`を使うかどちらかにしないといけない
- `RNN`が活性化関数を持っていないため、非線形性が出にくい
    - `RNN`の活性化関数を`ReLU`に変更する

# 2024/07/16の解析メモ

- 合計42パターン`RNN`の結果を出した。
    - 全体としてCNNのような強い非線形性が出にくい...

- 今後の方針
    


# 2024/07/15の勉強メモ

- LSTMとSeq2Seqのコードを完成させる
- 現状のコードで`RNN`の部分は以下の通り
```python
class StressStrainRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StressStrainRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```
- 上記のコードの要点
    - class作成時に、nn.Moduleを継承している
    - `super(StressStrainRNN, self).__init__()`の部分では、親クラスの`__init__`メソッドを呼び出している
        - 親クラスである`nn.Module`のソースコードは[ここ](https://github.com/pytorch/pytorch/blob/1d983bbb289f47f2544d0d1eece47a200d2038c4/torch/nn/modules/module.py#L398)
    - `self.hidden_size`は親クラスには存在しないが、自分で定義している。それ以外も同様
    - `self.fc`は全結合層を表している
    - `forward`メソッドは、入力`x`を受け取り、出力を返すメソッド
        - `h0`は隠れ状態の初期値テンソル
        - `out, _`はRNNの出力と隠れ状態のタプル
        - `out[:, -1, :]`は出力の最後の要素を取得している
- 

# 2024/04/01の勉強メモ
- CNNはひとまず完成→次はRNN，LSTM，Seq2Seqのコードの完成を目指す
- nn.RNNの内容の理解
    - [Pytorchのリンク](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
    - 以下HPからの内容を翻訳して記述
        - RNNは入力シーケンスの各要素に対して再帰的に同じモジュールを適用するモジュールです。各要素には入力と隠れ状態があり、出力は隠れ状態です。
            -「出力は隠れ状態」この意味がよくわからない
                - この意味は、RNNの出力は隠れ状態のみであり、入力の情報は出力には含まれないということ
                - RNNの出力を利用する場合には，全結合層などを用いて出力を変換する必要がある
        - パラメータ：
            - input_size – 入力の特徴量の数
            - hidden_size – 隠れ状態の特徴量の数
            - num_layers – RNNのレイヤーの数．デフォルトは1
            - nonlinearity – 'tanh'か'relu'のいずれか。デフォルトは'tanh'
            - bias – バイアスを使用するかどうか。デフォルトはTrue
            - batch_first – Trueの場合、入力と出力の形状は（バッチサイズ、シーケンス長、特徴量）になります。デフォルトはFalse
            - dropout – ドロップアウト確率。デフォルトは0
                - ドロップアウトは，モデルの過学習を防ぐために使用される手法
                - ドロップアウトは，モデルの一部のユニット(ニューロン)を与えられた確率で無効にすることで，モデルの過学習を防ぐ
                - 一般的な値は入力層で0.1~0.2, 中間層で0.5程度
            - bidirectional – Trueの場合、双方向RNNを使用します。デフォルトはFalse
        - 入力 (batch_first=Trueの場合)：
            - input (batch, seq_len, input_size) – 入力の特徴量のテンソル
                - batch: バッチサイズ，つまりデータの個数，今回の場合であれば，同時に何個のデータを処理するかを示す
                - seq_len: シーケンスの長さ，つまり時系列データの長さ．今回の場合であれば，ある時間ステップからどこまで遡るかを示す
                - input_size: 入力の特徴量の数，今回の場合であれば，5つ(軸差応力，有効平均主応力，軸ひずみ，累積せん断仕事，次の時間ステップまでの軸差応力の増分)
            - h_0 (num_layers * num_directions, batch, hidden_size) - 隠れ状態の初期値テンソル
                - num_layers * num_directions: RNNのレイヤー数と双方向の場合の数の積
                - batch: バッチサイズ，つまりデータの個数
                - hidden_size: 隠れ状態の特徴量の数

        - 出力：
            - output (batch, seq_len, hidden_size * num_directions) – 各要素の出力のテンソル
    - 現在のクラス設計はこちら
        - ```python
            class StressStrainRNN(nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super(StressStrainRNN, self).__init__()
                    self.hidden_size = hidden_size
                    self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)

                def forward(self, x):
                    h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
                    out, _ = self.rnn(x, h0)
                    out = self.fc(out[:, -1, :])
                    return out
          ```
        - forwardメソッドの引数xは入力の特徴量のテンソルであるが，このテンソルの形状は(batch, seq, input_size)である必要がある
            - x.size(0)はxの最初の次元のサイズを取得していて，これはバッチサイズになる．
            - このため，h0の形状は(1, x.size(0), self.hidden_size)となる
            - なぜx.size(1)ではなく，x.size(0)であるのか？
                - 隠れ層のテンソルは一つ前の出力のテンソルを入力として受け取り，その計算方法は以下の通り
                    - $h_t = \tanh(W_{ih}x_t + b_{ih} + W_{hh}h_{(t-1)} + b_{hh})$
                - x.size(0)はバッチサイズを取得しているため，バッチサイズを取得するためにx.size(0)を用いている
                - 一方，x.size(1)はシーケンスの長さを取得しているため，シーケンスの長さを取得するためにx.size(1)を用いている
                - このため，h0の形状は(1, x.size(0), self.hidden_size)となる
