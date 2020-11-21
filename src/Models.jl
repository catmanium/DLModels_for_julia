module Models

include("Layers.jl")
include("Optimizer.jl")
include("DataSet.jl")

export MLP,predict,learn,optimizer,dataset

using Random

#========
Model_params
========#
abstract type Model_params end
mutable struct MLP_params <:Model_params
    layers::Array
    optimizer
    debug::Bool
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
    #レイヤ初期化
    layers = []
    for i in 1:length(neuron)-1
        #パラメータ生成
        W = randn(neuron[i],neuron[i+1])
        b = randn(neuron[i+1])

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

       append!(layers,layers_tmp)

    end

    return MLP_params(layers,optimizer,debug)
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


#================
学習
================#
function learn(params::MLP_params;batch_size,epoch,data,t_data=[])
    ite = size(data)[1] ÷ batch_size
    epoch_loss = 0
    min_loss = 0
    min_loss_ep = 0
    loss_list = [] #エポック毎の損失の平均を格納
    #1エポック=データを1巡
    for i in 1:epoch
        ite_avg_loss_sum = 0 #1イテレーション毎の平均損失の総和
        #データのインデックスシャッフル
        idx = shuffle(1:size(data)[1])
        data = data[idx,:]
        t_data = t_data[idx,:]
        epoch_avg_loss = 0

        for k in 1:ite
            batch_data = data[1+(k-1)*batch_size:k*batch_size,1:end]
            params.layers[end].t = t_data[1+(k-1)*batch_size:k*batch_size,1:end]
            loss = predict(params,batch_data)
            #逆伝播
            dout=0
            for j in length(params.layers):-1:1
                dout = Layers.backward(params.layers[j],dout)
            end
            ite_avg_loss_sum += sum(loss)/length(loss) #1イテレーションの平均損失を足していく

            #更新
            Optimizer.update(params.optimizer,params.layers)

        end

        epoch_avg_loss = ite_avg_loss_sum/ite #1エポック毎の平均損失

        #10回に一回出力する
        if i%(epoch/10) == 0
            println("ep.$i : Loss :　",epoch_avg_loss)
        end

        if i==1 || min_loss > epoch_avg_loss
            min_loss = epoch_avg_loss
            min_loss_ep = i
        end

        append!(loss_list,epoch_avg_loss)
    end

    params.layers[end].t = []

    println("min_loss : $min_loss")
    println("min_loss_ep : $min_loss_ep")

    return loss_list
end

#================
最適化 optimizer
=================#
function optimizer(params::Model_params;name="Adam",learning_rate=0.001,p1=0.95,p2=0.99,e=10^(-12))
    hyper_params = [learning_rate,p1,p2,e]
    #optimizer生成
    optimizer_a = string("Optimizer.$name($hyper_params)")
    params.optimizer = eval(Meta.parse(optimizer_a))
end

#================
サンプルデータ
=================#
function dataset(type::String,n::Int64)
    type_name = string("DataSet.",type,"($n)")
    type_function = Meta.parse(type_name)
    return eval(type_function)
end

end