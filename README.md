# IMOS 多智能体可信计算模型推荐

此仓库为文章内多Agent协同系统的 Python 源码。包括任务解析、SGAP 动作编排、种类匹配、Trust-GUMS 实例调度和计算执行五类智能体，并提供论文全部实验项目的驱动和输出。

## 安装与快速验证

在当前工程根目录运行。核心代码支持 Python 3.9 及以上；实际 MCP 和 LangChain 组合运行请使用 Python 3.10 及以上，论文实验使用 Python 3.10。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[agents]"
python main.py scaffold
python -m unittest discover -s tests -v
python main.py demo --output outputs/my_demo
```

`demo`在独立输出目录生成合成数据、36个小模型与图模型，再运行全部实验。它使用脚本式模拟推理，没有调用五个真实大模型；五个模型名称只用于验证实验矩阵。模拟表现相同或远优于真实模型时均不能用于论文。演示不写入正式`datasets`和`modelbase`，不覆盖已有输出目录。

本次程序验证记录见`docs/verification.md`。真实数据未补齐时，`python main.py audit`会报告缺少的资产，不静默替换为模拟数据。

当前机器另有已安装全部依赖的隔离环境`.venv-mcp`（Python3.12.8、CPU版PyTorch），可用`.\.venv-mcp\Scripts\python.exe main.py ...`直接调用。`requirements-verified.txt`保存该验证环境的依赖版本；它不是论文原环境锁定文件，CPU版PyTorch需使用对应CPU轮子源安装。

## 目录与入口

|位置|作用|
|---|---|
|`main.py`、`imos/cli.py`|所有命令入口|
|`configs/experiment.json`|数据路径、推理参数、训练参数、可信权重、约束与实验次数|
|`configs/rules.json`|任务—动作、动作—模型种类的声明式规则|
|`datasets/`|真实D1、D2-a/b/c、D3、D4、Dr待补充目录|
|`modelbase/registry.json`|36个模型预注册项|
|`imos/networks.py`、`imos/models.py`|候选模型架构、训练、扰动、推理与耗时测量|
|`imos/graph.py`|二部图、GNN、MLP及联合训练|
|`imos/agents.py`|五类智能体与SGAP、Trust-GUMS|
|`imos/pipeline.py`|端到端任务流程、根因反馈与重试|
|`imos/llm.py`|Ollama本地推理、原始响应、参数和耗时记录|
|`imos/transport.py`、`imos/mcp_server.py`|直接调用或真实MCP标准输入输出通信|
|`imos/experiments.py`|表2—表8、消融图、位置交换实验和案例|
|`imos/resources.py`|实际资源快照、可选物理资源压力工具|
|`scripts/extract_paper.py`|提取论文正文、公式文本、表格和图片|
|`tests/test_core.py`|可信边界、数据泄漏、图反向传播及指标测试|
|`docs/reproduction_audit.md`|逐项原文依据、缺失项和实现对应关系|

## 数据与模型补齐

！！！关于数据集与模型！！！
由于数据集和模型是论文实验的核心且GitHub仓库对上传代码内容的容量有严格限制，无法将文中所有数据集与模型集上传至仓库内，故本工程不包含任何数据集和模型，需要自行下载并放置在指定目录，所有数据集与模型来源已在文章中给出，请自行下载并放置在指定目录。
数据契约详见`datasets/README.md`。每个数据目录提供`data.npz`，包含三个明确划分的特征与标签集合。时间序列轴序是样本×时间×通道。不能把同一试件的不同时间窗随机拆入训练和测试，D4也不能跨集合混入同一轴承寿命轨迹。

```powershell
python main.py convert configs/data_mapping.example.json datasets/D1/data.npz
python main.py build-cross
```

`convert`要求先修改映射文件，支持CSV、MAT和NPZ；不猜标签含义。`build-cross`从D2-a构造D2-b/c，尺寸比例1.1与0.9来自论文；`1/比例`时间轴插值、周期延拓和标签保留是重建假设。论文没有给出载荷幅值比例，默认1.0，需依据真实实验修订。

D1：SVR和LSBoost；D2：CNN、BiTCN、CBTA；D3：1D-CNN、LSTM和暂定AttentionLSTM；D4：TCN-Bi-LSTM、CNN-Bi-LSTM；Dr：CNN和RNN。每个基础模型加2个变体，共36项。D3原文重复写LSTM，与“12个基础模型”存在歧义；AttentionLSTM是暂定占位架构，不是已经确定的原模型。

补充原权重时需核对网络结构与调用接口，不能将第三方权重直接改名后加载。若从头训练重建模型，可创建发布者签名密钥：

```powershell
python scripts/make_signing_key.py .private/model_signing.pem
```

管理员确认发布者身份后，将输出的公钥加入`configs/experiment.json`的`public_keys`，键为发布者名称；私钥不上传。随后：

```powershell
python main.py train-models --signing-key .private/model_signing.pem --publisher 实验室 --source 本地实验数据与模型版本记录
python main.py train-selector
python main.py audit
```

`train-models`读取训练集和验证集，训练基础模型和扰动项；从未用测试标签训练。图训练默认读取验证数据的分块交互，也可用`train-selector --interactions path.json`导入真实历史交互。记录需包含`context_id, scene, features[14], eligible[模型ID], labels{模型ID:有用性}`。导入标签应来自历史执行或独立验证数据，不得来自当前待评价测试集。

## Trust-GUMS图结构、训练方式与参数

本工程采用以下配置：

|项目|实现|
|---|---|
|模型节点|14维：验证性能、加载加推理时延、RAM/显存/CPU需求、变体编号、任务类型、7个架构类别特征|
|上下文节点|14维：输入均值/标准差/RMS/峰值、序列长/通道数、CPU/RAM/显存/时延约束、任务类型、尺寸比例、幅值比例、常数|
|边|只连接可行候选与其当前上下文；没有模型—模型或上下文—上下文直接边|
|在线图|经过来源、完整性、环境过滤后的候选节点，加1个当前上下文节点，构成局部星形二部图|
|消息传递|2层同步双向平均邻居聚合；拼接节点自身与邻居表示后Linear+ReLU+Dropout|
|隐层/嵌入|32维|
|有用性MLP|64→32→1，ReLU、Dropout、Sigmoid，输出[0,1]|
|训练标签|独立验证集分块上的accuracy或截断到[0,1]的R²；这是代理有用性，不是原文披露标签|
|联合训练|GNN和MLP一起反向传播，MSE损失、Adam优化器|
|优化参数|learning_rate=0.001，weight_decay=0.0001，dropout=0.1|
|训练长度|最多100轮；验证损失连续15轮不改善则提前停止，保存最佳状态|
|批处理|每批4个局部图，以图级损失均值更新|
|训练/验证|按context_id分组，25%上下文验证，其余训练；不随机拆同一上下文的边|
|随机种子|20261006，实验重复时加重复序号|
|在线行为|只做图前向推理，不在测试时自动训练；选择日志保存供后续显式增量训练|

平均消息传递允许候选通过共享上下文形成间接交互。在线图没有额外历史上下文节点；跨上下文经验体现在预训练参数和历史性能元数据。若要恢复原论文的多上下文历史边，需要补充其真实图规则。默认代理训练标签与历史性能特征共享验证来源，所以图模型的验证MSE不能作为独立泛化证明。

执行顺序严格按正文公式(21)—(28)：

1. 按模型种类召回，再按任务适用场景检索候选。场景索引是重建的适用性实现，避免把D4模型误当作D2候选。
2. 来源分数低于阈值或关键制品SHA-256不一致，立即剔除。
3. 硬件、系统、运行时版本、依赖版本、权限、隐私、时延和可用性硬约束过滤。
4. GNN/MLP预测有用性u，计算`Utility=α1*u+α2*ConstraintFit+α3*HistoryPerf`。
5. `Trust=β*Tsrc+(1-β)*Trun`，`Score=γ*Utility+(1-γ)*Trust`。
6. 按Score降序选择argmax；大模型不能覆盖工具给出的argmax。空集合返回可追踪失败。

重建权重：α=(0.6,0.2,0.2)、β=0.5、γ=0.7。来源权重λ=(0.34,0.33,0.33)、阈值1.0。运行窗口86400秒，五类模型异常权重均为1；无历史时Trun=1是先验假设。ConstraintFit采用时延余量和吞吐达标程度的均值；HistoryPerf为验证集accuracy或截断R²。上述具体值原文均未给定。

算法2伪代码的简化得分遗漏正文可信综合项，本工程以正文公式为准。完整性只作硬过滤，不重复进入Trust加权。

## 候选模型训练参数

候选深度模型的准确结构和训练方式不在原文中。本工程提供可训练参考实现：CNN为两层一维卷积；BiTCN含正反方向膨胀因果残差卷积；CBTA为CNN→BiTCN→时间注意力池化；LSTM/RNN使用时序隐藏状态均值；TCN-Bi-LSTM与CNN-Bi-LSTM包含双向LSTM。均配线性输出层。预处理与模型内编码作为默认步骤，不参加RQ1—RQ3模型推荐，符合论文实验说明；没有额外虚构独立MobileNet权重。

|参数|重建默认值|
|---|---|
|训练轮数/早停|50 / 连续10轮验证损失无改善|
|batch_size|32|
|Adam学习率/weight_decay|0.001 / 0.0001|
|隐层/卷积核/Dropout|32 / 3 / 0.1|
|TCN膨胀率|[1,2,4]|
|分类损失|CrossEntropy；标签映射在训练集确定|
|回归损失|MSE；输入与回归标签标准化参数仅由训练集拟合|
|SVR|RBF核，C=10，epsilon=0.01，gamma采用sklearn默认scale|
|LSBoost|GradientBoostingRegressor的平方损失梯度提升，100棵树，learning_rate=0.05，max_depth=3|
|神经干扰项|基础权重的部分参数张量加高斯扰动，尺度为张量标准差的0.03/0.06|
|传统模型干扰项|降低C/树数量等超参数，实际表现重新评估，不强行保证更差|
|预测区间|验证残差的有限样本修正95%分位数；这是重建补充，原文未说明置信区间算法|

频带、采样频率、降采样比必须按数据真实情况配置；默认不设频带、不降采样，不能称为恢复原论文预处理参数。LSBoost是平方损失提升树重建，不保证与原实现的MATLAB/其他库设置相同。

## 五个本地大模型与推理参数

原文模型版本保持原样：`glm-4.7-flash:q8_0`、`deepseek-r1:32b`、`qwen3:32b`、`llama3.1:8b`、`gemma3:27b`。只有GLM的量化标签明确写出；其余实际量化方式、模型文件摘要、Ollama版本未给定，应在运行前记录，不能自行补称原始量化规格。

通过本地Ollama的`/api/chat`调用。重建统一参数如下，所有模型都可在配置中修改：

|参数|值|
|---|---|
|temperature|0.0|
|top_p / top_k|0.9 / 40|
|num_ctx / num_predict|8192 / 2048|
|repeat_penalty|1.1|
|seed|20261006+重复序号|
|stream / keep_alive|false / 5m|
|请求超时|600秒|
|JSON约束|各结构化智能体启用；纯LLM基线不设系统提示与JSON生成模式|
|think|不额外指定，使用模型服务默认；原文未说明，响应原样保留|

每个智能体的请求、系统提示、原始响应、有效参数与调用耗时均存入`llm_*.json`。温度0不能保证跨硬件完全一致。真实运行默认使用LangChain Runnable编排；`--transport mcp`启用MCP v1 SDK的stdio工具通信。直接调用模式用于本地调试，不能称为实际跨进程MCP实验。

API实现依据：[Ollama Chat接口](https://docs.ollama.com/api/chat)、[MCP Python SDK v1](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x)、[LangChain RunnableLambda](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableLambda)。接口资料用于实现通信，不补充论文训练事实。

## 论文环境与运行耗时

原文报告PC：AMD Ryzen 9 9950X，16核32线程，4.30 GHz；RTX 5090 D，32GB显存；Windows 10；Python 3.10；VS Code。原文将“磁盘64 GB”写为磁盘信息，不能擅自把它认定为内存64GB。

当前`environment`资源数值是重建配置，不是自动确认的原环境。运行前应修改到实际设备可用配额。工程会记录CPU、RAM、GPU资源快照；设备测量与人为配额分开存档。

程序实际输出及含义：

|输出|时间含义|
|---|---|
|`runtime.csv`|每次任务完整墙钟时间、成功与失败、计算模型推理时间|
|`table2/3/5.csv`、`ablation.csv`|全部尝试平均时间与成功样本平均时间、样本数|
|`trials/**/trace.json`|五类智能体各阶段墙钟时间，包括工具与通信开销|
|`llm_timing.json`、`llm_*.json`|LLM墙钟时间和Ollama加载、提示求值、生成耗时及token数量|
|`modelbase/registry.json`|候选模型训练时间、2次预热与10次加载+单样本推理计时|
|`selector.training.json`|图训练总耗时、轮数、训练/验证损失；不冒充原论文训练日志|
|`manifest.json`|实际Python、依赖版本、CUDA/GPU、操作系统和有效实验配置|

Ollama返回的duration纳秒转换为秒。候选模型计时包含文件加载，不能直接当作纯网络前向时延。端到端时间是否包含冷启动取决于`keep_alive`和模型实际缓存状态，应查看load_duration；不跨模型编造统一冷启动状态。

## 运行全部实验

在补齐资产、设置可信发布者、完成图训练及部署上述五个Ollama模型后：

```powershell
python main.py run --output outputs/real_run_001 --transport mcp
```

可使用`--only rq1/rq2/rq3/rq4`分组运行。默认重复次数：RQ1每LLM每场景15次；RQ2每方法每LLM5次；RQ3每版本每LLM5次。RQ4原文未给重复次数，默认每个测试1次。纯代码基线仅在D2-a硬绑定CBTA，不让它复用动态调度器。

|论文内容|本工程输出|
|---|---|
|表1规则映射|`configs/rules.json`、论文参考档案中的表1|
|表2可行性：D1、D2-a、D3、D4×5个LLM|`table2.csv`及`table2_averages.csv`|
|表3跨场景：D2-a/b/c、Dr×5个LLM|`table3.csv`及`table3_averages.csv`|
|表4 D2-a 9个候选性能|`table4.csv`|
|表5协同、规则匹配、纯LLM、纯代码|`table5.csv`|
|注册表首项位置交换|`registry_order_probe.csv`及各次轨迹|
|图3消融SR|`ablation.csv`、`figure3_ablation_SR.png`|
|图4消融MS|`ablation.csv`、`figure4_ablation_MS.png`|
|表6三条冲突指令与一条合规指令|`table6.csv`|
|表7来源失信/制品篡改/可信候选选择|`table7.csv`及独立模型库故障副本|
|表8 Dr幅值×1.01与D1资源不足|`table8.csv`、drift/resource日志|
|完整粗糙度预测案例|`case_study.json`、动作绑定、状态和预测区间|
|中间和最终数值|`trials/**/predictions.npz`、`all_trials.json`、`test_catalog.json`|
|参数、版本与报告|`manifest.json`、`all_tables.json`、`REPORT.md`|

RQ3的文字说只用Deepseek，而原图展示5个LLM；默认覆盖图中的5个，可将`rq3_llms`设置为`deepseek-r1:32b`以复现文字设定。V2按结果描述删除任务模板、保留SGAP冷启动；不是删除整个SGAP。V3移除Trust-GUMS、保留规则。V4同时缺失两算法。

所有次序交换和消融均执行真实决策，不硬编码“规则匹配一定选第一项”或“纯LLM一定失败”。真实实验可能与论文描述不同，应检查逐次日志。

## 可信反馈与资源故障

表7通过实际文件尾部篡改使CBTA-v1完整性失败，在隔离副本操作，原始权重不被修改。CBTA来源失信，CBTA-v2保留真实签名和哈希，不因为性能稍差就跳过可信底线。

Dr实验先向任务输入提供幅值变化事实，不给大模型原始信号或测试标签、不指定处置模型。执行层读取数据并评估结果。这证明的是已知输入变化下的调度稳定性，不能声称智能体自主检测到漂移。`dr_max_error`默认空值，因为原文没有阈值；填入事前任务允许误差后才报告是否满足质量要求。

默认资源测试在第一次调度之后注入“CPU可用配额一半、显存0、RAM 2GB”，执行器检测LSBoost的注册需求不满足，再反馈调度器选择SVR。LSBoost需3GB、SVR需0.1GB是重建需求配置，并非其实际峰值测量。必须区分这项确定性配额测试与论文物理占用程序。

`resource_fault_mode`可切换为`physical`，启动有界CPU/内存/显存占用工具，并记录实际资源快照。独立测试可运行`python -m imos.resources --seconds 15`；退出后释放。GPU为CUDA上下文保留256MiB，不宣称数学意义上占满全部显存。本次没有执行物理资源占用测试。实际压力测试仍同时保留明确的可用配额边界，以使异常触发可审计。

模型原因才影响Trun；资源和数据原因不直接惩罚模型。恢复重試有界，失败也写日志。输入恢复是请求重新读取源数据，不凭空生成替代数据；源未恢复则明确失败。动作重规划会重新运行SGAP，不接受大模型虚构的数值结果。

`monitor.input_bounds`与`monitor.output_bounds`可设置事先确定的正常值域；默认空值表示论文没有提供可采用的物理边界。有限性、来源、完整性和资源满足性始终检查。不能把默认未设置的值域或任务质量要求说成已经验证。

