### config.py

模型使用的列，例如问题一选取的11个特征变量；

模型的配置信息，如神经网络的结构等；

映射表，例如PHY和（nss，mcs）之间的映射关系

### utils.py

使用频率较高的函数，例如误差计算的函数；

画图等

### data_process.py

数据处理的文件，所有训练文件经data_process.py处理后用于后续计算;

### model.py

问题1使用的各种模型；

### rrf.py

RRF（Regularized Random Forest）

### rrf_rag.py

倒数排名融合，用于计算特征重要性排名

### demo.py

问题1的主要代码

### classification.py

问题2的主要代码

### demo3.py

问题3的主要代码
