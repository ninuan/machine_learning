# machine_learning
机器学习大作业
## 第三方库下载
```
pip install -r requirements.txt
```
## 代码结构
```
.
├── README.md
├── data
│   ├── test.csv
│   └── train.csv
├── model
│   ├── __init__.py
│   ├── __pycache__
│   ├── data.py
│   ├── model.py
│   └── utils.py
├── requirements.txt
└── src
    ├── __init__.py
    ├── __pycache__
    ├── main.py
    └── utils.py
```
## 运行
```
python src/main.py
```
## 代码说明
### src
- main.py: 主函数，用于调用model中的函数
- utils.py: 用于数据预处理
- __init__.py: 用于包的导入
- __pycache__: 缓存文件
- data: 数据集
- model: 模型
- requirements.txt: 第三方库
- README.md: 说明文档
- .gitignore: git忽略文件
- .git: git文件
- .idea: pycharm文件

### model
- data.py: 数据预处理
- model.py: 模型
- utils.py: 工具函数
- __init__.py: 用于包的导入
- __pycache__: 缓存文件

## 代码运行结果
```
python src/main.py
```
