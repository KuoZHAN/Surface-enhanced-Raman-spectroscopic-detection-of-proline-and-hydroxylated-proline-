%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = readmatrix('Detected1Hyp2NTA3Pro.xlsx');

avg_1=mean(res(1:9140,:),1);
avg_2=mean(res(9141:18922,:),1);
avg_3=mean(res(18923:28900,:),1);
results=[avg_1;avg_2;avg_3];
writematrix(results,'average.xlsx');
disp('average')
%%  CNN ANALYSIS
res = readmatrix('Detected1Hyp2Pro.xlsx');
temp = randperm(19118);

P_train = res(temp(1: 13888), 1: 1189)';
T_train = res(temp(1: 13888), 1190)';
M = size(P_train, 2);

P_test = res(temp(13889: end), 1: 1189)';
T_test = res(temp(13889: end), 1190)';
N = size(P_test, 2);

[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

p_train =  double(reshape(P_train, 1189, 1, 1, M));
p_test  =  double(reshape(P_test , 1189, 1, 1, N));
save('preprocessParams1Hyp2Pro.mat',"ps_input");
layers = [
 imageInputLayer([1189, 1, 1])                                % 输入层
 
  convolution2dLayer([3, 1], 32, 'Padding', 'same','Name','ConvolutionnalNN_1')          % 卷积核大小为 2*1 生成16个卷积
 batchNormalizationLayer                                    % 批归一化层
 reluLayer                                                  % relu 激活层
 
 maxPooling2dLayer([2, 1], 'Stride', [2, 1])                % 最大池化层 大小为 2*1 步长为 [2, 1]

 convolution2dLayer([3, 1], 64, 'Padding', 'same','Name','ConvolutionnalNN_2')          % 卷积核大小为 2*1 生成32个卷积
 batchNormalizationLayer                                    % 批归一化层
 reluLayer                                                  % relu 激活层
 maxPooling2dLayer([2, 1], 'Stride', [2, 1])
 dropoutLayer(0.5) 
 fullyConnectedLayer(64)                                     % 全连接层（类别数） 
 reluLayer
 dropoutLayer(0.5)
 fullyConnectedLayer(2)
 softmaxLayer                                               % 损失函数层
 classificationLayer];                                      % 分类层

options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MaxEpochs', 100, ...                  % 最大训练次数 500
    'InitialLearnRate', 1e-3, ...          % 初始学习率为 0.001
    'L2Regularization', 1e-4, ...          % L2正则化参数
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子 0.1
    'LearnRateDropPeriod', 50, ...        % 经过450次训练后 学习率为 0.001 * 0.1
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

net = trainNetwork(p_train, t_train, layers, options);

t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

analyzeNetwork(layers)

[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
xlim([1, N])
grid

figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

% 保存训练好的模型到当前工作目录
save('trainedCNNModel1Hyp2Pro.mat', 'net');
%%
load('trainedCNNModel1Hyp2P.mat', 'net');

% 获取网络的层
layers = net.Layers;

% 显示每一层的名称和类型
for i = 1:length(layers)
    fprintf('Layer %d: Name = %s, Type = %s\n', i, layers(i).Name, class(layers(i)));
end

% 将网络转换为 layerGraph
lgraph = layerGraph(layers);

% 移除名为 'classoutput' 的输出层
lgraph = removeLayers(lgraph, 'classoutput');

% 检查是否存在 'softmax' 层，若有则移除
softmaxLayerIdx = find(arrayfun(@(l) isa(l, 'nnet.cnn.layer.SoftmaxLayer'), layers));
if ~isempty(softmaxLayerIdx)
    softmaxLayerName = layers(softmaxLayerIdx).Name;
    lgraph = removeLayers(lgraph, softmaxLayerName);
end

% 确保网络是连通的
% analyzeNetwork(lgraph); % 可选

% 将 layerGraph 转换为 dlnetwork
dlnet = dlnetwork(lgraph);

% 导入数据
res = readmatrix('Hypintensity.xlsx');

% 提取前三个样本的特征数据
sample_data = res(1:1, 1:1189)';   % 1189 x 3 矩阵

% 手动指定样本的标签为 1, 2, 3
sample_labels = categorical(1, [1,2,3], {'1','2','3'});

% 对数据进行归一化
[P_train, ps_input] = mapminmax(res(:, 1:1189)', 0, 1);
sample_data = mapminmax('apply', sample_data, ps_input);

% 重塑数据
M_sample = size(sample_data, 2); % 样本数
sample_data = double(reshape(sample_data, [1189, 1, 1, M_sample]));
classes = categories(sample_labels);
num_samples = M_sample;
featureGradients = zeros(1189, num_samples);

% 计算每个样本的特征梯度
for i = 1:num_samples
    inputSample = sample_data(:, :, :, i);
    trueLabel = sample_labels(i);  % 确保标签为 1, 2, 3
    featureGradients(:, i) = computeFeatureGradients(dlnet, inputSample, trueLabel, classes);
end

% 显示特征梯度的尺寸
disp('Size of featureGradients:');
disp(size(featureGradients));  

% 创建特征编号向量
feature_indices =(1:1189)';
T = table(feature_indices, 'VariableNames', {'FeatureIndex'});
% 添加每个样本的特征梯度到表格
data = featureGradients(:, i); 
data(data < 0) = 0;
maxVal = max(data);
normalizedData = data/maxVal;
sampleVarName = ['Sample_', num2str(i)];
T.(sampleVarName) = normalizedData;

% 指定文件名
filename = 'Hypintensityfeatureweightoct20241Hyp2NTA3Pro.xlsx';

% 将表格写入 Excel 文件
writetable(T, filename);

% 显示提示信息
disp(['所有样本的特征梯度已导出到 ', filename]);
% 遍历每个样本，绘制热力图
for i = 1:num_samples
    % 提取第 i 个样本的特征梯度
    data = normalizedData';  
    
    % 绘制热力图（使用 imagesc）
    figure;
    imagesc(feature_indices, 1, data);
    colormap(hot);
    colorbar;
    
    % 矫正 Y 轴方向
    set(gca, 'YDir', 'normal');
    
    % 设置颜色条标签
    cb = colorbar;
    cb.Label.String = '特征梯度';
    
    % 设置轴标签和标题
    xlabel('特征编号');
    ylabel('样本编号');
    title(['样本 ', num2str(i), ' 的特征重要性热力图']);
    
    % 调整 X 轴刻度（可选）
    xticks(1:100:1463);
    xticklabels(arrayfun(@num2str, 1:100:1463, 'UniformOutput', false));
    
    % 调整 Y 轴刻度
    yticks(1);
    yticklabels({num2str(i)});
    
    % 设置 Y 轴范围
    ylim([0.5, 1.5]);
end

% 定义计算梯度的函数
function [loss, gradients] = modelGradients(dlnet, dlX, trueLabel, classes)
    % 前向传播
    dlYPred = forward(dlnet, dlX);

    % 手动添加 softmax 操作
    dlYPred = softmax(dlYPred);

    % 将真实标签转换为 one-hot 编码
    T = onehotencode(trueLabel, 1, 'ClassNames', classes);

    % 将 T 转换为 dlarray
    T = dlarray(T);

    % 调整 T 的尺寸以匹配 dlYPred
    T = reshape(T, size(dlYPred));

    % 计算交叉熵损失
    loss = crossentropy(dlYPred, T);

    % 计算损失相对于输入的梯度
    gradients = dlgradient(loss, dlX);
end

function featureGradients = computeFeatureGradients(dlnet, inputSample, trueLabel, classes)
    % 将输入转换为 dlarray
    dlX = dlarray(inputSample, 'SSC');

    % 启用自动微分，计算输出和梯度
    [loss, gradients] = dlfeval(@modelGradients, dlnet, dlX, trueLabel, classes);

    % 提取梯度并转换为列向量
    featureGradients = extractdata(gradients(:));
end