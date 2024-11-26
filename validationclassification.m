warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

load('trainedCNNModel1Hyp2Pro.mat', 'net');
load('preprocessParams1Hyp2Pro.mat', 'ps_input');

newData = readmatrix('Hyp1Pro2validation.xlsx');  % 替换为您的 Excel 文件名

% 提取特征数据
newFeatures = newData(:, 1:1189)';  % 大小为 [1189, 样本数]

% 如果有真实标签，可以提取（可选）
if size(newData, 2) >= 1190
    trueLabels = newData(:, 1190)';     % 大小为 [1, 样本数]
end

% 使用训练数据的归一化参数对新数据进行归一化
newFeaturesNorm = mapminmax('apply', newFeatures, ps_input);

% 获取新样本的数量
numNewSamples = size(newFeaturesNorm, 2);

% 重塑数据
newFeaturesReshaped = reshape(newFeaturesNorm, [1189, 1, 1, numNewSamples]);

% 使用模型进行分类，获取预测标签和预测概率
[predictedLabels, scores] = classify(net, newFeaturesReshaped);

% 将预测标签转换为数组
predictedLabelsArray = cellstr(predictedLabels);

sampleLabels = {'1', '2'};
% 获取类别列表
classNames = net.Layers(end).Classes;

% 显示预测结果
for i = 1:numNewSamples
    % 获取当前样本的预测分数（概率）
    sampleScores = scores(i, :);

    % 获取最高的预测概率和对应的类别
    [maxScore, idx] = max(sampleScores);
    predictedClass = classNames(idx);

    fprintf('样本 %d 的预测类别是: %s，预测概率: %.2f%%\n', i, char(predictedClass), maxScore * 100);
end

% 创建表格
T = table((1:numNewSamples)', predictedLabelsArray, max(scores, [], 2) * 100, ...
    'VariableNames', {'SampleIndex', 'PredictedLabel', 'PredictionProbability'});

% 保存到 Excel 文件
writetable(T, 'Hyp1Pro2ValidationPredictionResults.xlsx');

fprintf('预测结果已保存到 PredictionResults.xlsx\n');

if exist('trueLabels', 'var')
    % 将真实标签转换为分类变量
    trueLabelsCategorical = categorical(trueLabels, [1,2], {'1','2'})';

    % 计算准确率
    accuracy = sum(predictedLabels == trueLabelsCategorical) / numNewSamples * 100;

    fprintf('预测准确率为: %.2f%%\n', accuracy);

    % 生成并显示混淆矩阵
    figure;
    cm = confusionchart(trueLabelsCategorical, predictedLabels);
    cm.Title = '新数据集的混淆矩阵';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end