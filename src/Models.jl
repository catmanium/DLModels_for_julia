module Models

include("Layers.jl")
include("Optimizer.jl")
include("DataSet.jl")
include("Func.jl")

export MLP,SimpleRNN,ManyToOneRNN,predict,learn,optimizer,shaping_rnn,save

using Random,JSON

#==============================
共通メンバ変数
・params : 各レイヤに重みをリスト化
・grads : 各レイヤに勾配をリスト化 → Layers.forward()でLayerのメンバに代入したら合わせて変更される
・layers : 各レイヤをリスト化
・loss_layer : 損失関数レイヤ
・optimizer : 最適化アルゴリズム

共通メソッド
・モデル名(;neuron,loss_layer) :コンストラクタ
    ・neuron : 入力～中間～出力　までのニューロン数を指定
    ・loss_layer : 損失関数（フルネーム指定）
        ・重み，バイアス，レイヤ，損失関数レイヤの初期化
        ・生成したレイヤからparams,gradsを取り出しメンバ変数へ保管

・predict(model_params;data) : 推論　→　スコア算出

・learn(model_params;batch_size,max_epoch,data,t_data=[]) 
    ※以下をmax_epoch回ループ,max_ite回ループ
    ・predict()実行　→　損失算出(loss_layerにt_dataが指定してあれば損失を返す)
    ・Model.layersを逆からbackward() → 勾配算出
    ・params更新


===============================#

#========
Model_params
========#
abstract type Model_params end
mutable struct MLP_params <:Model_params
    params
    grads
    layers::Array
    optimizer
    debug::Bool
end
mutable struct SimpleRNN_params <:Model_params
    params
    grads
    layers::Array
    rnn_layer
    optimizer
end
mutable struct ManyToOneRNN_params <: Model_params
    params
    grads
    layers
    optimizer
    padding
end

#==========
コンストラクタ
===========#
function MLP(;activation="Sigmoid",loss="Sigmoid_with_CrossEntropy",neuron,debug=false)
    #neuronには入力，中間1，中間2，...中間n，出力までの各ニューロン数を格納

    if debug
        Random.seed!(123456)
    end

    optimizer = []
    params = []
    grads = []

    #レイヤ初期化
    layers = []
    for i in 1:length(neuron)-1
        layers_tmp = []
        #パラメータ生成
        W = randn(neuron[i],neuron[i+1])
        b = randn(neuron[i+1],1)

        if i == length(neuron)-1
            #活性化関数&損失関数レイヤ
            activation_a = string("Layers.",loss,"()")
        else
            #活性化関数レイヤ
            activation_a = string("Layers.",activation,"()")
        end
        activation_layer = Meta.parse(activation_a)

        #レイヤ生成
        layers_tmp = [
            Layers.Affine(W,b),
            eval(activation_layer),
       ]

       #params,gradsをリストへ
       append!(params,layers_tmp[1].params) #Affineのparamsを取り出し
       append!(grads,layers_tmp[1].grads) #Affineのgradsを取り出し

       append!(layers,layers_tmp)
    end

    return MLP_params(params,grads,layers,optimizer,debug)
end
function SimpleRNN(;data_size,hidden_size,output_size)
    #hidden_size RNNレイヤの隠れ状態の次元数
    D, H, O = data_size, hidden_size, output_size

    #重みの初期化
    rnn_Wx = randn(D,H)/sqrt(D)
    rnn_Wh = randn(H,H)/sqrt(H)
    rnn_b = zeros(1,H)
    affine_W = randn(H,O)/sqrt(H)

    #その他パラメータ
    stateful = true
    padding = nothing

    affine_b = zeros(1,O)

    #勾配の初期化は各レイヤで行う

    #レイヤの初期化
    layers = [
        Layers.TimeRNN(rnn_Wx,rnn_Wh,rnn_b,stateful,padding),
        Layers.TimeAffine(affine_W,affine_b),
        Layers.TimeMse()
    ]

    #params,gradsをリストへ
    params = vcat(layers[1].params,layers[2].params)
    grads = vcat(layers[1].grads,layers[2].grads)
    
    optimizer = []

    return SimpleRNN_params(params,grads,layers,layers[1],[])
end
function ManyToOneRNN(input_size,hidden_size,output_size)
    #p : SimpleRNNとの差異

    #hidden_size RNNレイヤの隠れ状態の次元数
    D, H, O = input_size, hidden_size, output_size

    #その他パラメータ
    stateful = true
    padding = nothing

    #重みの初期化
    # rnn_Wx = randn(D,H)/sqrt(D)
    # rnn_Wh = randn(H,H)/sqrt(H)
    # rnn_b = zeros(1,H)
    rnn_Wx = randn(D,4*H)/sqrt(D)
    rnn_Wh = randn(H,4*H)/sqrt(H)
    rnn_b = zeros(1,4*H)

    affine_W = randn(H,O)/sqrt(H)
    affine_b = zeros(O,1)

    #レイヤの初期化
    layers = [
        Layers.TimeLSTM(rnn_Wx,rnn_Wh,rnn_b,true,padding),
        Layers.Affine(affine_W,affine_b), #p

        Layers.Mean_Squared_Error() #p
    ]    

    #params,gradsをリストへ
    params = vcat(layers[1].params,layers[2].params)
    grads = vcat(layers[1].grads,layers[2].grads)

    optimizer = []
    padding = nothing

    return ManyToOneRNN_params(params,grads,layers,optimizer,padding)
end

function hoge()
    return 0
end

#================
推論
================#
function predict(params::MLP_params,data)
    out = data
    for i in 1:length(params.layers)
        out = Layers.forward(params.layers[i],out)
    end
    
    #推論と学習に応じて，スコアか損失か変わる(学習データの有無で判別)
    return out
end
function predict(params::SimpleRNN_params,data)
    hs = Layers.forward(params.layers[1],data)
    ys = Layers.forward(params.layers[2],hs)
    return ys
end
function predict(this::ManyToOneRNN_params,data)
    hs = Layers.forward(this.layers[1],data)
    #使うのはTの出力のみ
    h = hs[:,end,:]
    y = Layers.forward(this.layers[2],h)

    return y
end

#================
学習
================#
function learn(this::MLP_params;batch_size,max_epoch,data,t_data=[])
    ite = size(data)[1] ÷ batch_size
    epoch_loss = 0
    min_loss = 0
    min_loss_ep = 0
    loss_list = [] #エポック毎の損失の平均を格納
    #1エポック=データを1巡
    for i in 1:max_epoch
        ite_avg_loss_sum = 0 #1イテレーション毎の平均損失の総和
        #データのインデックスシャッフル
        idx = shuffle(1:size(data)[1])
        data = data[idx,:]
        t_data = t_data[idx,:]
        epoch_avg_loss = 0

        for k in 1:ite
            batch_data = data[1+(k-1)*batch_size:k*batch_size,1:end]
            this.layers[end].t = t_data[1+(k-1)*batch_size:k*batch_size,1:end]
            #順伝播
            loss = predict(this,batch_data)
            #逆伝播
            dout=1
            for j in length(this.layers):-1:1
                dout = Layers.backward(this.layers[j],dout)
            end
            ite_avg_loss_sum += sum(loss)/length(loss) #1イテレーションの平均損失を足していく

            #更新
            Optimizer.update(this.optimizer,this)
        end

        epoch_avg_loss = ite_avg_loss_sum/ite #1エポック毎の平均損失

        #10回に一回出力する
        if i%(max_epoch/10) == 0
            println("ep.$i : Loss :　",epoch_avg_loss)
            println("grads:",this.grads[1][1])
        end

        if i==1 || min_loss > epoch_avg_loss
            min_loss = epoch_avg_loss
            min_loss_ep = i
        end

        append!(loss_list,epoch_avg_loss)
    end

    println("min_loss : $min_loss")
    println("min_loss_ep : $min_loss_ep")

    #predict単体で実行するために初期化しておく
    this.layers[end].t = []

    return loss_list
end
function learn(this::SimpleRNN_params;batch_size,max_epoch,window_size,data,t_data=[])
    X = size(data,1) #総データ数
    D = size(data,2) #データ次元数
    T = window_size #RNNレイヤ数
    N = batch_size #バッチ数

    max_ite = (X/N)*(1/T) #イテレーション数
    loss_list = [] #avg_lossのリスト

    for epoch in 1:max_epoch
        total_loss = 0 #損失合計
        avg_loss = 0 #1エポックの平均損失
        for ite in 1:max_ite
            #ミニバッチ作成
            xs = zeros(Float64,(N,T,D))
            ts = zeros(Float64,(N,T,D))
            for n in 1:N
                st = 1+(T*(ite-1))+(X/N)*(n-1)
                ed = T+(T*(ite-1))+(X/N)*(n-1)
                xs[Int(n),:,:] = data[Int(st):Int(ed),:]
                ts[Int(n),:,:] = t_data[Int(st):Int(ed),:]
            end
            #順伝播
            ys = predict(this,xs)
            loss = Layers.forward(this.layers[end],ys,ts)
            total_loss += sum(loss)
            #逆伝播
            dl = ones(Float64,size(loss))
            dys = Layers.backward(this.layers[end],dl)
            dhs = Layers.backward(this.layers[2],dys)
            dxs = Layers.backward(this.layers[1],dhs)
            #更新
            Optimizer.update(this.optimizer,this)
        end

        avg_loss = total_loss/(T*D*N)
        append!(loss_list,avg_loss)
       
        #10回に一回出力する
        if epoch%(max_epoch/10) == 0
            println("ep.$epoch : Loss :　",avg_loss)
            println("grads:",this.grads[1][1])
        end
    end


    return loss_list
end
function learn(this::ManyToOneRNN_params;max_epoch,window_size,data,t_data)
    #先にshapingでデータ加工
    #Tはsize(data,2)を割れる値にする

    D = size(data,3) #データ次元数
    T = window_size #RNNレイヤ数
    N = size(data,1) #バッチ数

    max_ite = size(data,2)÷T #イテレーション数
    loss_list = [] #avg_lossのリスト

    grads_list = []

    for epoch in 1:max_epoch
        ite_total_loss = 0 #損失合計
        avg_loss = 0 #1エポックの平均損失
        st = 0 #data切り取り位置
        ed = 0
        for ite in 1:max_ite
            #ミニバッチ作成
            st = Int(1+(ite-1)*T)
            ed = Int(T*ite)
            xs = data[:,st:ed,:]
            t = t_data[:,ite,:]
            #順伝播
            y = predict(this,xs)
            this.layers[end].t = t #教師データ挿入
            loss = Layers.forward(this.layers[end],y)
            #ite毎の平均損失を加算
            ite_total_loss += sum(loss)/length(loss)
            #逆伝播
            dy = Layers.backward(this.layers[end],0)
            dh = Layers.backward(this.layers[2],dy)
            dxs = Layers.backward(this.layers[1],dh)
            #勾配クリッピング
            # append!(grads_list,this.grads[1][1])
            #更新
            Optimizer.update(this.optimizer,this)
        end

        avg_loss = ite_total_loss/max_ite
        append!(loss_list,avg_loss)
       
        if epoch == 1 || epoch%(max_epoch÷10) == 0
            println("ep.$epoch : Loss :　",avg_loss)
            # println("grads:",this.grads[1][1])
            # println("grads_l:",this.layers[1].grads[2][1])
        end
    end

    #学習の最後に隠れベクトルをリセット->predictする際，ミニバッチサイズが異なる為
    Layers.reset_state(this.layers[1])
    this.layers[end].t = [] #正解データリセット

    return loss_list, grads_list
end

#================
最適化 optimizer
=================#
function optimizer(this::Model_params;name="Adam",learning_rate=0.001,p1=0.95,p2=0.99,e=10^(-12))
    vs = []
    ss = []
    for i in 1:length(this.params)
        v = zeros(Float64,size(this.params[i]))
        s = zeros(Float64,size(this.params[i]))
        append!(vs,[v])
        append!(ss,[s])
    end
    hyper_params = [
        learning_rate,
        p1,
        p2,
        e,
        vs,
        ss
    ]
    #optimizer生成
    optimizer_a = string("Optimizer.$name($hyper_params)")
    this.optimizer = eval(Meta.parse(optimizer_a))
end


#================
データ整形
=================#
function shaping_rnn(model::ManyToOneRNN_params,data,N,padding)
    model.padding = padding
    #re_data (N, X/N, D)
    #ite毎に切り出して使用する
    #Tは2軸目のサイズに応じて後から決める

    X,D = size(data,1),size(data,2)
    re_data = []

    #データの補充
    if size(data,1)%N != 0
        #(n,D)分，paddingで埋める
        n = N - size(data,1)%N
        padding_arr = fill(padding,n,D)
        data = vcat(data,padding_arr) #補充
        X,D = size(data)
    end

    re_data = reshape(data,Int(X/N),N,D)
    re_data = permutedims(re_data,(2,1,3)) #軸の入れ替え

    return re_data
end
function get_rate_of_change(array)
    rate_arr = zeros(length(array)-1)
    for i in 1:length(array)-1
        rate = (array[i+1]-array[i])/array[i]
        rate_arr[i] = rate
    end
    #行列化
    rate_arr = reshape(rate_arr,(length(rate_arr),1))
    return rate_arr
end

#================
サンプルデータ
=================#
function dataset(type::String,n::Int64)
    type_name = string("DataSet.",type,"($n)")
    type_function = Meta.parse(type_name)
    return eval(type_function)
end

#=================
その他
==================#
function save(model::Model_params,file) 
    JSON.json(model)
    f = open("$file","w") #ファイル生成
    JSON.print(f, model)
    close(f)
end

function load(model::Model_params,file)
    if !isfile(file)
        println("no file.")
        return nothing
    end

    f = open("$file","r")
    model_a = JSON.parse(f)

    model.params = model_a["params"]
    model.grads = model_a["grads"]
end


end