# 模型库

`registry.json`预置12个基础候选和24个干扰项，共36项。待补充权重、可信来源、签名、摘要和性能记录的表项不能参加真实实验。

基础模型：D1的SVR、LSBoost；D2的CNN、BiTCN、CBTA；D3的1D-CNN、LSTM、AttentionLSTM；D4的TCN-Bi-LSTM、CNN-Bi-LSTM；Dr的CNN、RNN。原文D3重复写了LSTM且总模型数写12，第三个D3模型暂以AttentionLSTM表示，必须在取得原实现信息后核实；不将此结构称为论文确定架构。

每项包括id、name、kind、task、dataset、architecture、variant、input_dim、output_dim、DataType、artifact、config、dependencies、source、publisher、signature、version、API、environment、validation与baseline_hashes。文件路径必须在模型库根目录内。名称使用场景前缀，防止D2与Dr的CNN冲突。

可使用`train-models`训练重建模型；也可补充原权重并提供符合`imos.models.predict`接口的加载逻辑。第三方权重不能只改文件后缀后直接加载。神经权重使用PyTorch的`weights_only=True`，传统模型使用pickle，仅在可信校验通过后加载。

来源采用Ed25519签名，配置中的`public_keys`是发布者受信公钥，不自动信任注册表自报的公钥。签名覆盖模型ID、发布者、来源、版本、网络配置、依赖及关键制品基准SHA-256。每次调用前重新计算文件摘要。私钥不放入模型库、不上传GitHub。

来源三权重、运行历史窗口、异常权重和无历史记录时的先验由配置指定。环境异常或输入异常不直接扣模型运行分，只有已归因于模型的异常写入惩罚项。
