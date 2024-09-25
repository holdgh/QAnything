Qanything是一个比较综合完整的开源RAG项目，其完整程度在我的视角是在这几个方面：

    - RAG全流程都具备，从文件上传、处理到在线推理、排序等，关键模块都是有的，而且中文注释，注释和文档也比较完善，所以很适合学习。

    - 包括完整的前后端体系（源码里有看到npm、webpack之类的关键词，相信这块做的也并不含糊）。

    - 部署支持多环境和多种功能，从他多个文档里可以看到，支持linux、windows、Mac等多个环境，而且也有很丰富的教程。

项目目录结构：
|-- Dockerfile  # docker构造文件
|-- FAQ.md  # 常见问题回答
|-- FAQ_zh.md # 常见问题回答
|-- LICENSE  # 证书
|-- README.md # README文档
|-- README_zh.md# README文档
|-- assets  # 一些模型之类的文件会放在这里
|-- close.sh # 关闭脚本
|-- docker-compose-linux.yaml # docker构造所需要的yaml文件，linux专用
|-- docker-compose-windows.yaml # docker构造所需要的yaml文件，windows专用
|-- docs  # 各种使用文档
|-- etc   # 其他文件，此处有一个prompt
|-- qanything_kernel # qanything的核心带代码，基本是python，包括服务、算法等算法
|-- requirements.txt # python依赖
|-- run.sh  # 启动脚本
|-- scripts  # 其他脚本
|-- third_party # 三方库
`-- volumes  # 数据库容器的外部映射地址

核心代码qanything_kernel目录结构：
|-- __init__.py # 空的
|-- configs  # 这里是模型和服务的配置，还有prompt之类的
|-- connector # 中间件的连接工具
|-- core  # milvus、es、mysql的都在里面，大家其实都能参考着用的
|-- dependent_server # 独立服务，这里主要是放大模型服务、ocr服务（图片文字抽取）、重排模型服务的
|-- qanything_server # qanything的核心服务，也是对外的服务，另外还有一些js文件和图片素材，应该是给前端用的。
`-- utils    # 零散的代码工具。这里可以说是宝藏了，里面挺多工具函数自己平时都能用的，类似safe_get等，另外很多文件处理的工具，类似文件加载、切片等，都在里面。