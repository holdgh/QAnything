import os
from dotenv import load_dotenv

load_dotenv()
# 获取环境变量GATEWAY_IP
GATEWAY_IP = os.getenv("GATEWAY_IP", "localhost")
# LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logging.basicConfig(format=LOG_FORMAT)
# 获取项目根目录
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# os.path.dirname(path)取path的上一级路径
# 嵌套使用三次，返回上三级路径
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
UPLOAD_ROOT_PATH = os.path.join(root_path, "QANY_DB", "content")
IMAGES_ROOT_PATH = os.path.join(root_path, "qanything_kernel/qanything_server/dist/qanything/assets", "file_images")
print("UPLOAD_ROOT_PATH:", UPLOAD_ROOT_PATH)
print("IMAGES_ROOT_PATH:", IMAGES_ROOT_PATH)
OCR_MODEL_PATH = os.path.join(root_path, "qanything_kernel", "dependent_server", "ocr_server", "ocr_models")
RERANK_MODEL_PATH = os.path.join(root_path, "qanything_kernel", "dependent_server", "rerank_server", "rerank_models")
EMBED_MODEL_PATH = os.path.join(root_path, "qanything_kernel", "dependent_server", "embed_server", "embed_models")
PDF_MODEL_PATH = os.path.join(root_path, "qanything_kernel/dependent_server/pdf_parser_server/pdf_to_markdown")

# LLM streaming reponse
STREAMING = True

SYSTEM = """
You are a helpful assistant. 
You are always a reliable assistant that can answer questions with the help of external documents.
Today's date is {{today_date}}. The current time is {{current_time}}.
"""

INSTRUCTIONS = """
- All contents between <DOCUMENTS> and </DOCUMENTS> are reference information retrieved from an external knowledge base.
- If you cannot answer based on the given information, you will return the sentence \"抱歉，检索到的参考信息并未提供充足的信息，因此无法回答。\".
- Before answering, confirm the number of key points or pieces of information required, ensuring nothing is overlooked.
- Now, answer the following question based on the above retrieved documents(Let's think step by step):
{{question}}
- Return your answer in Markdown formatting, and in the same language as the question "{{question}}".
"""

PROMPT_TEMPLATE = """
<SYSTEM>
{{system}}
</SYSTEM>

<INSTRUCTIONS>
{{instructions}}
</INSTRUCTIONS>

<DOCUMENTS>
{{context}}
</DOCUMENTS>

<INSTRUCTIONS>
{{instructions}}
</INSTRUCTIONS>
"""

CUSTOM_PROMPT_TEMPLATE = """
<USER_INSTRUCTIONS>
{{custom_prompt}}
</USER_INSTRUCTIONS>

<DOCUMENTS>
{{context}}
</DOCUMENTS>

<INSTRUCTIONS>
- All contents between <DOCUMENTS> and </DOCUMENTS> are reference information retrieved from an external knowledge base.
- Now, answer the following question based on the above retrieved documents(Let's think step by step):
{{question}}
</INSTRUCTIONS>
"""


SIMPLE_PROMPT_TEMPLATE = """
- You are a helpful assistant. You can help me by answering my questions. You can also ask me questions.
- Today's date is {{today}}. The current time is {{now}}.
- User's custom instructions: {{custom_prompt}}
- Before answering, confirm the number of key points or pieces of information required, ensuring nothing is overlooked.
- Now, answer the following question:
{{question}}
Return your answer in Markdown formatting, and in the same language as the question "{{question}}". 
"""

# 缓存知识库数量
CACHED_VS_NUM = 100

# 文本分句长度
SENTENCE_SIZE = 100

# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 30

VECTOR_SEARCH_SCORE_THRESHOLD = 0.3

KB_SUFFIX = '_240625'
# MILVUS_HOST_LOCAL = 'milvus-standalone-local'
# MILVUS_PORT = 19530
MILVUS_HOST_LOCAL = GATEWAY_IP
# MILVUS_PORT = 19540
MILVUS_PORT = 19530
MILVUS_COLLECTION_NAME = 'qanything_collection' + KB_SUFFIX

# ES_URL = 'http://es-container-local:9200/'
ES_URL = f'http://{GATEWAY_IP}:9210/'
ES_USER = None
ES_PASSWORD = None
ES_TOP_K = 30
ES_INDEX_NAME = 'qanything_es_index' + KB_SUFFIX

# MYSQL_HOST_LOCAL = 'mysql-container-local'
# MYSQL_PORT_LOCAL = 3306
MYSQL_HOST_LOCAL = GATEWAY_IP
MYSQL_PORT_LOCAL = 3306
MYSQL_USER_LOCAL = 'root'
MYSQL_PASSWORD_LOCAL = 'cp7R@1K59t3C#Hl4'
MYSQL_DATABASE_LOCAL = 'qanything'

LOCAL_OCR_SERVICE_URL = "localhost:7001"

LOCAL_PDF_PARSER_SERVICE_URL = "localhost:9009"

LOCAL_RERANK_SERVICE_URL = "localhost:8001"
LOCAL_RERANK_MODEL_NAME = 'rerank'
LOCAL_RERANK_MAX_LENGTH = 512
LOCAL_RERANK_BATCH = 1
LOCAL_RERANK_THREADS = 1
LOCAL_RERANK_PATH = os.path.join(root_path, 'qanything_kernel/dependent_server/rerank_server', 'rerank_model_configs_v0.0.1')
# rerank模型路径，在rerank_server启动前，用以创建RerankAsyncBackend对象
LOCAL_RERANK_MODEL_PATH = os.path.join(LOCAL_RERANK_PATH, "rerank.onnx")

LOCAL_EMBED_SERVICE_URL = "localhost:9001"
LOCAL_EMBED_MODEL_NAME = 'embed'
LOCAL_EMBED_MAX_LENGTH = 512
LOCAL_EMBED_BATCH = 1
LOCAL_EMBED_THREADS = 1
# 词嵌入路径
LOCAL_EMBED_PATH = os.path.join(root_path, 'qanything_kernel/dependent_server/embedding_server', 'embedding_model_configs_v0.0.1')
# 词嵌入模型路径，在embedding_server启动前，用以创建一个EmbeddingAsyncBackend对象
LOCAL_EMBED_MODEL_PATH = os.path.join(LOCAL_EMBED_PATH, "embed.onnx")

TOKENIZER_PATH = os.path.join(root_path, 'qanything_kernel/connector/llm/tokenizer_files')

DEFAULT_CHILD_CHUNK_SIZE = 400
DEFAULT_PARENT_CHUNK_SIZE = 800
MAX_CHARS = 1000000  # 单个文件最大字符数，超过此字符数将上传失败，改大可能会导致解析超时

# llm_config = {
#     # 回答的最大token数，一般来说对于国内模型一个中文不到1个token，国外模型一个中文1.5-2个token
#     "max_token": 512,
#     # 附带的上下文数目
#     "history_len": 2,
#     # 总共的token数，如果遇到电脑显存不够的情况可以将此数字改小，如果低于3000仍然无法使用，就更换模型
#     "token_window": 4096,
#     # 如果报错显示top_p值必须在0到1，可以在这里修改
#     "top_p": 1.0
# }

# Bot
BOT_DESC = "一个简单的问答机器人"
BOT_IMAGE = ""
BOT_PROMPT = """
- 你是一个耐心、友好、专业的机器人，能够回答用户的各种问题。
- 根据知识库内的检索结果，以清晰简洁的表达方式回答问题。
- 不要编造答案，如果答案不在经核实的资料中或无法从经核实的资料中得出，请回答“我无法回答您的问题。”（或者您可以修改为：如果给定的检索结果无法回答问题，可以利用你的知识尽可能回答用户的问题。)
"""
BOT_WELCOME = "您好，我是您的专属机器人，请问有什么可以帮您呢？"
