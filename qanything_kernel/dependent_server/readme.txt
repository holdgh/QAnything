./scripts/entrypoint.sh启动脚本中的python文件运行命令如下：
nohup python3 -u qanything_kernel/dependent_server/rerank_server/rerank_server.py > /workspace/QAnything/logs/debug_logs/rerank_server.log 2>&1 &
PID1=$!
nohup python3 -u qanything_kernel/dependent_server/embedding_server/embedding_server.py > /workspace/QAnything/logs/debug_logs/embedding_server.log 2>&1 &
PID2=$!
nohup python3 -u qanything_kernel/dependent_server/pdf_parser_server/pdf_parser_server.py > /workspace/QAnything/logs/debug_logs/pdf_parser_server.log 2>&1 &
PID3=$!
nohup python3 -u qanything_kernel/dependent_server/ocr_server/ocr_server.py > /workspace/QAnything/logs/debug_logs/ocr_server.log 2>&1 &
PID4=$!
nohup python3 -u qanything_kernel/dependent_server/insert_files_serve/insert_files_server.py --port 8110 --workers 1 > /workspace/QAnything/logs/debug_logs/insert_files_server.log 2>&1 &
PID5=$!
nohup python3 -u qanything_kernel/qanything_server/sanic_api.py --host $USER_IP --port 8777 --workers 1 > /workspace/QAnything/logs/debug_logs/main_server.log 2>&1 &
PID6=$!

因此当前目录是程序的入口
rerank模型【1G】下载方式：git clone https://www.modelscope.cn/netease-youdao/bce-reranker-base_v1.git
embedding模型【1G】下载方式：git clone https://www.modelscope.cn/netease-youdao/bce-embedding-base_v1.git
大语言模型【10G】下载方式：git clone https://www.modelscope.cn/netease-youdao/Qwen-7B-QAnything.git
pdf解析模型和ocr模型下载方式：git clone https://www.modelscope.cn/netease-youdao/QAnything-pdf-parser.git
注意事项：下载前，先执行git lfs install