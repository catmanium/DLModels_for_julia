module Optimizer

#=====
params
======#
abstract type Optimizer_params end
mutable struct SGD_params
    learning_rate::Float64
end

#=========
コンストラクタ
==========#
function SGD(learning_rate)
    return SGD_params(learning_rate)    
end

#=========
実行
==========#
function update(params::SGD_params,layers)
    for i in 1:2:length(layers)
        layers[i].W = layers[i].W -params.learning_rate*layers[i].dW
    end
end

end