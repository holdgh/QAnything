from qanything_kernel.core.local_file import LocalFile
from qanything_kernel.core.local_doc_qa import LocalDocQA
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.configs.model_config import (BOT_DESC, BOT_IMAGE, BOT_PROMPT, BOT_WELCOME,
                                                   DEFAULT_PARENT_CHUNK_SIZE, MAX_CHARS, VECTOR_SEARCH_TOP_K)
from qanything_kernel.utils.general_utils import *
from langchain.schema import Document
from sanic.response import ResponseStream
from sanic.response import json as sanic_json
from sanic.response import text as sanic_text
from sanic import request, response
import uuid
import json
import asyncio
import urllib.parse
import re
from datetime import datetime
from collections import defaultdict
import os
from tqdm import tqdm
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import base64

__all__ = ["new_knowledge_base", "upload_files", "list_kbs", "list_docs", "delete_knowledge_base", "delete_docs",
           "rename_knowledge_base", "get_total_status", "clean_files_by_status", "upload_weblink", "local_doc_chat",
           "document", "upload_faqs", "get_doc_completed", "get_qa_info", "get_user_id", "get_doc",
           "get_rerank_results", "get_user_status", "health_check", "update_chunks", "get_file_base64",
           "get_random_qa", "get_related_qa", "new_bot", "delete_bot", "update_bot", "get_bot_info"]

INVALID_USER_ID = f"fail, Invalid user_id: . user_id 必须只含有字母，数字和下划线且字母开头"

# 获取环境变量GATEWAY_IP【host.docker.internal，在docker-compose-win.yaml中qanything容器里面的环境变量处有配置】
GATEWAY_IP = os.getenv("GATEWAY_IP", "localhost")
debug_logger.info(f"GATEWAY_IP: {GATEWAY_IP}")

# 异步包装器，用于在后台执行带有参数的同步函数
async def run_in_background(func, *args):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=4) as pool:
        await loop.run_in_executor(pool, func, *args)


# 使用aiohttp异步请求另一个API
async def fetch(session, url, input_json):
    headers = {'Content-Type': 'application/json'}
    async with session.post(url, json=input_json, headers=headers) as response:
        return await response.json()


# 定义一个需要参数的同步函数
def sync_function_with_args(arg1, arg2):
    # 模拟耗时操作
    import time
    time.sleep(5)
    print(f"同步函数执行完毕，参数值：arg1={arg1}, arg2={arg2}")


@get_time_async
async def new_knowledge_base(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("new_knowledge_base %s", user_id)
    kb_name = safe_get(req, 'kb_name')
    debug_logger.info("kb_name: %s", kb_name)
    default_kb_id = 'KB' + uuid.uuid4().hex
    kb_id = safe_get(req, 'kb_id', default_kb_id)
    kb_id = correct_kb_id(kb_id)

    is_quick = safe_get(req, 'quick', False)
    if is_quick:
        kb_id += "_QUICK"

    if kb_id[:2] != 'KB':
        return sanic_json({"code": 2001, "msg": "fail, kb_id must start with 'KB'"})
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not not_exist_kb_ids:
        return sanic_json({"code": 2001, "msg": "fail, knowledge Base {} already exist".format(kb_id)})

    # local_doc_qa.create_milvus_collection(user_id, kb_id, kb_name)
    local_doc_qa.milvus_summary.new_milvus_base(kb_id, user_id, kb_name)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    return sanic_json({"code": 200, "msg": "success create knowledge base {}".format(kb_id),
                       "data": {"kb_id": kb_id, "kb_name": kb_name, "timestamp": timestamp}})


@get_time_async
async def upload_weblink(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("upload_weblink %s", user_id)
    debug_logger.info("user_info %s", user_info)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})

    url = safe_get(req, 'url')
    if url:
        urls = [url]
        # 如果URL以/结尾，先去除这个/
        if url.endswith('/'):
            url = url[:-1]
        titles = [safe_get(req, 'title', url.split('/')[-1]) + '.web']
    else:
        urls = safe_get(req, 'urls')
        titles = safe_get(req, 'titles')
        if len(urls) != len(titles):
            return sanic_json({"code": 2003, "msg": "fail, urls and titles length not equal"})

    for url in urls:
        # url 需要以http开头
        if not url.startswith('http'):
            return sanic_json({"code": 2001, "msg": "fail, url must start with 'http'"})
        # url 长度不能超过2048
        if len(url) > 2048:
            return sanic_json({"code": 2002, "msg": f"fail, url too long, max length is 2048."})

    file_names = []
    for title in titles:
        debug_logger.info('ori name: %s', title)
        file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', title)
        debug_logger.info('cleaned name: %s', file_name)
        file_name = truncate_filename(file_name, max_length=110)
        file_names.append(file_name)

    mode = safe_get(req, 'mode', default='soft')  # soft代表不上传同名文件，strong表示强制上传同名文件
    debug_logger.info("mode: %s", mode)
    chunk_size = safe_get(req, 'chunk_size', default=DEFAULT_PARENT_CHUNK_SIZE)
    debug_logger.info("chunk_size: %s", chunk_size)

    exist_file_names = []
    if mode == 'soft':
        exist_files = local_doc_qa.milvus_summary.check_file_exist_by_name(user_id, kb_id, file_names)
        exist_file_names = [f[1] for f in exist_files]
        for exist_file in exist_files:
            file_id, file_name, file_size, status = exist_file
            debug_logger.info(f"{url}, {status}, existed files, skip upload")
            # await post_data(user_id, -1, file_id, status, msg='existed files, skip upload')
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")

    data = []
    for url, file_name in zip(urls, file_names):
        if file_name in exist_file_names:
            continue
        local_file = LocalFile(user_id, kb_id, url, file_name)
        file_id = local_file.file_id
        file_size = len(local_file.file_content)
        file_location = local_file.file_location
        msg = local_doc_qa.milvus_summary.add_file(file_id, user_id, kb_id, file_name, file_size, file_location,
                                                   chunk_size, timestamp, url)
        debug_logger.info(f"{url}, {file_name}, {file_id}, {msg}")
        data.append({"file_id": file_id, "file_name": file_name, "file_url": url, "status": "gray", "bytes": 0,
                     "timestamp": timestamp})
        # asyncio.create_task(local_doc_qa.insert_files_to_milvus(user_id, kb_id, [local_file]))
    if exist_file_names:
        msg = f'warning，当前的mode是soft，无法上传同名文件{exist_file_names}，如果想强制上传同名文件，请设置mode：strong'
    else:
        msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})


@get_time_async
async def upload_files(req: request):
    # """
    # 功能：上传文件
    # 流程：
    #     1、校验用户合法性
    #     2、校验知识库合法性
    #     3、对于一个知识库，数据库已存在的文件列表和入参中的文件列表总数不得超过10000
    #     4、对于入参中的文件名处理【清洗、拼接】
    #     5、获取知识库已存在的文件名列表，上传模式默认soft，跳过入参中在数据库里已存在的文件
    #     6、上传文件至根目录下\QANY_DB\content\user_id【此处未使用minio】
    #     7、评估文件字符尺寸，若超过100万，则将文件至于失败文件列表中
    #     8、将上述校验处理通过的文件信息连同知识库id、用户id一并插入数据库中
    #     9、构造接口返回值
    # """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    # 校验用户信息合法性
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    # 校验用户不合法时，返回提示信息
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    # 拼接用户id
    user_id = user_id + '__' + user_info
    debug_logger.info("upload_files %s", user_id)
    debug_logger.info("user_info %s", user_info)
    # 获取入参中的知识库id，并进行一定规则处理
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id %s", kb_id)
    mode = safe_get(req, 'mode', default='soft')  # soft代表不上传同名文件，strong表示强制上传同名文件
    debug_logger.info("mode: %s", mode)
    # 获取文件切片尺寸大小，默认800个token
    chunk_size = safe_get(req, 'chunk_size', default=DEFAULT_PARENT_CHUNK_SIZE)
    debug_logger.info("chunk_size: %s", chunk_size)
    # 获取是否使用本地文件参数，默认false
    use_local_file = safe_get(req, 'use_local_file', 'false')
    if use_local_file == 'true':
        # 本地获取，从当前项目的根目录下/data文件夹中获取文件
        files = read_files_with_extensions()
    else:
        # 非本地获取，从请求中获取文件
        files = req.files.getlist('files')
    # 打印文件数量
    debug_logger.info(f"{user_id} upload files number: {len(files)}")
    # 检查知识库id在数据库中是否存在，返回的是无效知识库id列表
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        # 若入参中的知识库id存在无效的，则返回提示信息
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
    # 根据知识库id和用户id查询文件列表
    exist_files = local_doc_qa.milvus_summary.get_files(user_id, kb_id)
    if len(exist_files) + len(files) > 10000:
        # 如果数据库中知识库已有文件数量和入参中文件数量之和大于10000，则超出知识库文件数量限制，返回提示信息
        return sanic_json({"code": 2002,
                           "msg": f"fail, exist files is {len(exist_files)}, upload files is {len(files)}, total files is {len(exist_files) + len(files)}, max length is 10000."})

    data = []
    local_files = []
    file_names = []
    for file in files:
        if isinstance(file, str):
            # 如果file为字符串，则为本地文件绝对路径，通过os.path.basename获取文件名
            file_name = os.path.basename(file)
        else:
            # 如果file为入参中的文件对象，则通过file.name获取文件名
            debug_logger.info('ori name: %s', file.name)
            # 解码文件名
            file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
            debug_logger.info('decode name: %s', file_name)
        # # 使用正则表达式替换以%开头的字符串
        # file_name = re.sub(r'%\w+', '', file_name)
        # 删除掉全角字符
        file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
        debug_logger.info('cleaned name: %s', file_name)
        # max_length = 255 - len(construct_qanything_local_file_nos_key_prefix(file_id)) == 188
        # 对于文件名做处理：对文件名前缀拼接时间戳，并校验文件全名长度不超过110，若超过，则对文件名前缀【未拼接时间戳】进行截取
        file_name = truncate_filename(file_name, max_length=110)
        # 收集文件名到文件名列表
        file_names.append(file_name)

    exist_file_names = []
    if mode == 'soft':
        # 若上传模式为soft，则知识库内存在同名文件时当前文件不再上传
        # 检查文件名列表在数据库中是否存在，输出已存在的文件名
        exist_files = local_doc_qa.milvus_summary.check_file_exist_by_name(user_id, kb_id, file_names)
        exist_file_names = [f[1] for f in exist_files]
        for exist_file in exist_files:
            # 日志打印已存在的文件名
            file_id, file_name, file_size, status = exist_file
            debug_logger.info(f"{file_name}, {status}, existed files, skip upload")
            # await post_data(user_id, -1, file_id, status, msg='existed files, skip upload')

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")

    failed_files = []
    # 将文件列表与文件名列表按照【文件-文件名】一一匹配分组，遍历处理
    for file, file_name in zip(files, file_names):
        # 跳过已存在的文件
        if file_name in exist_file_names:
            continue
        # 创建local_file对象，依赖参数：用户id、知识库id、文件、文件名
        local_file = LocalFile(user_id, kb_id, file, file_name)
        # 评估文件大小【local_file.file_location属性，仅对入参文件有意义，对于本地文件，该字段为'URL'】，返回文件字符数量
        chars = fast_estimate_file_char_count(local_file.file_location)
        debug_logger.info(f"{file_name} char_size: {chars}")
        if chars and chars > MAX_CHARS:
            # 文件超过100万个字符，则给出警告日志，并将文件名追加到失败列表文件列表中，然后对下一个文件进行处理
            debug_logger.warning(f"fail, file {file_name} chars is {chars}, max length is {MAX_CHARS}.")
            # return sanic_json({"code": 2003, "msg": f"fail, file {file_name} chars is too much, max length is {MAX_CHARS}."})
            # 列表的append方法：在列表末尾添加新的对象
            failed_files.append(file_name)
            continue
        # 从local_file对象中获取文件id【uuid】
        file_id = local_file.file_id
        # 获取文件内容尺寸
        file_size = len(local_file.file_content)
        file_location = local_file.file_location
        # 将local_file收集到local_files列表中，这些文件都是校验通过的文件
        local_files.append(local_file)
        # 保存文件信息【文件信息添加到数据库中】
        msg = local_doc_qa.milvus_summary.add_file(file_id, user_id, kb_id, file_name, file_size, file_location,
                                                   chunk_size, timestamp)
        debug_logger.info(f"{file_name}, {file_id}, {msg}")
        # 文件上传结果字段：file_id，file_name，status【gray，什么意思？】，bytes【文件内容长度】，timestamp【时间戳】，estimated_chars【文件字符长度】
        data.append(
            {"file_id": file_id, "file_name": file_name, "status": "gray", "bytes": len(local_file.file_content),
             "timestamp": timestamp, "estimated_chars": chars})

    # asyncio.create_task(local_doc_qa.insert_files_to_milvus(user_id, kb_id, local_files))
    # 构造文件上传结果的提示信息
    if exist_file_names:
        msg = f'warning，当前的mode是soft，无法上传同名文件{exist_file_names}，如果想强制上传同名文件，请设置mode：strong'
    elif failed_files:
        msg = f"warning, {failed_files} chars is too much, max characters length is {MAX_CHARS}, skip upload."
    else:
        msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})


@get_time_async
async def upload_faqs(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("upload_faqs %s", user_id)
    debug_logger.info("user_info %s", user_info)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id %s", kb_id)
    faqs = safe_get(req, 'faqs')
    chunk_size = safe_get(req, 'chunk_size', default=DEFAULT_PARENT_CHUNK_SIZE)
    debug_logger.info("chunk_size: %s", chunk_size)

    # 增加上传文件的功能和上传文件的检查解析
    file_status = {}
    if faqs is None:
        files = req.files.getlist('files')
        faqs = []
        for file in files:
            debug_logger.info('ori name: %s', file.name)
            file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
            debug_logger.info('decode name: %s', file_name)
            # 删除掉全角字符
            file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
            file_name = file_name.replace("/", "_")
            debug_logger.info('cleaned name: %s', file_name)
            file_name = truncate_filename(file_name)
            file_faqs = check_and_transform_excel(file.body)
            if isinstance(file_faqs, str):
                file_status[file_name] = file_faqs
            else:
                faqs.extend(file_faqs)
                file_status[file_name] = "success"

    if len(faqs) > 1000:
        return sanic_json({"code": 2002, "msg": f"fail, faqs too many, max length is 1000."})

    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg})

    data = []
    local_files = []
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    debug_logger.info(f"start insert {len(faqs)} faqs to mysql, user_id: {user_id}, kb_id: {kb_id}")
    for faq in tqdm(faqs):
        ques = faq['question']
        if len(ques) > 512 or len(faq['answer']) > 2048:
            return sanic_json(
                {"code": 2003, "msg": f"fail, faq too long, max length of question is 512, answer is 2048."})
        file_name = f"FAQ_{ques}.faq"
        file_name = file_name.replace("/", "_").replace(":", "_")  # 文件名中的/和：会导致写入时出错
        file_name = simplify_filename(file_name)
        file_size = len(ques) + len(faq['answer'])
        # faq_id = local_doc_qa.milvus_summary.get_faq_by_question(ques, kb_id)
        # if faq_id:
        #     debug_logger.info(f"faq question {ques} already exist, skip")
        #     data.append({
        #         "file_id": faq_id,
        #         "file_name": file_name,
        #         "status": "green",
        #         "length": file_size,
        #         "timestamp": local_doc_qa.milvus_summary.get_file_timestamp(faq_id)
        #     })
        #     continue
        local_file = LocalFile(user_id, kb_id, faq, file_name)
        file_id = local_file.file_id
        file_location = local_file.file_location
        local_files.append(local_file)
        local_doc_qa.milvus_summary.add_faq(file_id, user_id, kb_id, faq['question'], faq['answer'], faq.get('nos_keys', ''))
        local_doc_qa.milvus_summary.add_file(file_id, user_id, kb_id, file_name, file_size, file_location,
                                             chunk_size, timestamp)
        # debug_logger.info(f"{file_name}, {file_id}, {msg}, {faq}")
        data.append(
            {"file_id": file_id, "file_name": file_name, "status": "gray", "length": file_size,
             "timestamp": timestamp})
    debug_logger.info(f"end insert {len(faqs)} faqs to mysql, user_id: {user_id}, kb_id: {kb_id}")

    msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})


@get_time_async
async def list_kbs(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("list_kbs %s", user_id)
    kb_infos = local_doc_qa.milvus_summary.get_knowledge_bases(user_id)
    data = []
    for kb in kb_infos:
        data.append({"kb_id": kb[0], "kb_name": kb[1]})
    debug_logger.info("all kb infos: {}".format(data))
    return sanic_json({"code": 200, "data": data})


@get_time_async
async def list_docs(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("list_docs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id: {}".format(kb_id))
    file_id = safe_get(req, 'file_id')
    page_id = safe_get(req, 'page_id', 1)  # 默认为第一页
    page_limit = safe_get(req, 'page_limit', 10)  # 默认每页显示10条记录
    data = []
    if file_id is None:
        file_infos = local_doc_qa.milvus_summary.get_files(user_id, kb_id)
    else:
        file_infos = local_doc_qa.milvus_summary.get_files(user_id, kb_id, file_id)
    status_count = {}
    # msg_map = {'gray': "已上传到服务器，进入上传等待队列",
    #            'red': "上传出错，请删除后重试或联系工作人员",
    #            'yellow': "已进入上传队列，请耐心等待", 'green': "上传成功"}
    for file_info in file_infos:
        status = file_info[2]
        if status not in status_count:
            status_count[status] = 1
        else:
            status_count[status] += 1
        data.append({"file_id": file_info[0], "file_name": file_info[1], "status": file_info[2], "bytes": file_info[3],
                     "content_length": file_info[4], "timestamp": file_info[5], "file_location": file_info[6],
                     "file_url": file_info[7], "chunks_number": file_info[8], "msg": file_info[9]})
        if file_info[1].endswith('.faq'):
            faq_info = local_doc_qa.milvus_summary.get_faq(file_info[0])
            user_id, kb_id, question, answer, nos_keys = faq_info
            data[-1]['question'] = question
            data[-1]['answer'] = answer

    # data根据timestamp排序，时间越新的越靠前
    data = sorted(data, key=lambda x: int(x['timestamp']), reverse=True)

    # 计算总记录数
    total_count = len(data)
    # 计算总页数
    total_pages = (total_count + page_limit - 1) // page_limit
    if page_id > total_pages and total_count != 0:
        return sanic_json({"code": 2002, "msg": f'输入非法！page_id超过最大值，page_id: {page_id}，最大值：{total_pages}，请检查！'})
    # 计算当前页的起始和结束索引
    start_index = (page_id - 1) * page_limit
    end_index = start_index + page_limit
    # 截取当前页的数据
    current_page_data = data[start_index:end_index]

    # return sanic_json({"code": 200, "msg": "success", "data": {'total': status_count, 'details': data}})
    return sanic_json({
        "code": 200,
        "msg": "success",
        "data": {
            'total_page': total_pages,  # 总页数
            "total": total_count,  # 总文件数
            "status_count": status_count,  # 各状态的文件数【一个字典，key状态标识，value为对应该状态的数量】
            "details": current_page_data,  # 当前页码下的文件目录
            "page_id": page_id,  # 当前页码,
            "page_limit": page_limit  # 每页显示的文件数
        }
    })


@get_time_async
async def delete_knowledge_base(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    # TODO: 确认是否支持批量删除知识库
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("delete_knowledge_base %s", user_id)
    kb_ids = safe_get(req, 'kb_ids')
    kb_ids = [correct_kb_id(kb_id) for kb_id in kb_ids]
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    # milvus = local_doc_qa.match_milvus_kb(user_id, kb_ids)
    for kb_id in kb_ids:
        expr = f"kb_id == \"{kb_id}\""
        asyncio.create_task(run_in_background(local_doc_qa.milvus_kb.delete_expr, expr))
        # local_doc_qa.milvus_kb.delete_expr(expr)
        # milvus.delete_partition(kb_id)
    for kb_id in kb_ids:
        file_infos = local_doc_qa.milvus_summary.get_files(user_id, kb_id)
        file_ids = [file_info[0] for file_info in file_infos]
        file_chunks = [file_info[8] for file_info in file_infos]
        asyncio.create_task(run_in_background(local_doc_qa.es_client.delete_files, file_ids, file_chunks))
        local_doc_qa.milvus_summary.delete_documents(file_ids)
        local_doc_qa.milvus_summary.delete_faqs(file_ids)
        debug_logger.info(f"""delete knowledge base {kb_id} success""")
    local_doc_qa.milvus_summary.delete_knowledge_base(user_id, kb_ids)
    return sanic_json({"code": 200, "msg": "Knowledge Base {} delete success".format(kb_ids)})


@get_time_async
async def rename_knowledge_base(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("rename_knowledge_base %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    new_kb_name = safe_get(req, 'new_kb_name')
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
    local_doc_qa.milvus_summary.rename_knowledge_base(user_id, kb_id, new_kb_name)
    return sanic_json({"code": 200, "msg": "Knowledge Base {} rename success".format(kb_id)})


@get_time_async
async def delete_docs(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("delete_docs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    file_ids = safe_get(req, "file_ids")
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
    valid_file_infos = local_doc_qa.milvus_summary.check_file_exist(user_id, kb_id, file_ids)
    if len(valid_file_infos) == 0:
        return sanic_json({"code": 2004, "msg": "fail, files {} not found".format(file_ids)})
    valid_file_ids = [file_info[0] for file_info in valid_file_infos]
    # milvus_kb = local_doc_qa.match_milvus_kb(user_id, [kb_id])
    # milvus_kb.delete_files(file_ids)
    expr = f"""kb_id == "{kb_id}" and file_id in {valid_file_ids}"""  # 删除数据库中的记录
    asyncio.create_task(run_in_background(local_doc_qa.milvus_kb.delete_expr, expr))
    # local_doc_qa.milvus_kb.delete_expr(expr)
    file_chunks = local_doc_qa.milvus_summary.get_chunk_size(valid_file_ids)
    asyncio.create_task(run_in_background(local_doc_qa.es_client.delete_files, valid_file_ids, file_chunks))

    local_doc_qa.milvus_summary.delete_files(kb_id, valid_file_ids)
    local_doc_qa.milvus_summary.delete_documents(valid_file_ids)
    local_doc_qa.milvus_summary.delete_faqs(valid_file_ids)
    return sanic_json({"code": 200, "msg": "documents {} delete success".format(valid_file_ids)})


@get_time_async
async def get_total_status(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info('get_total_status %s', user_id)
    by_date = safe_get(req, 'by_date', False)
    if not user_id:
        users = local_doc_qa.milvus_summary.get_users()
        users = [user[0] for user in users]
    else:
        users = [user_id]
    res = {}
    for user in users:
        res[user] = {}
        if by_date:
            res[user] = local_doc_qa.milvus_summary.get_total_status_by_date(user)
            continue
        kbs = local_doc_qa.milvus_summary.get_knowledge_bases(user)
        for kb_id, kb_name in kbs:
            gray_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'gray')
            red_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'red')
            yellow_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'yellow')
            green_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'green')
            res[user][kb_name + kb_id] = {'green': len(green_file_infos), 'yellow': len(yellow_file_infos),
                                          'red': len(red_file_infos),
                                          'gray': len(gray_file_infos)}

    return sanic_json({"code": 200, "status": res})


@get_time_async
async def clean_files_by_status(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info('clean_files_by_status %s', user_id)
    status = safe_get(req, 'status', default='gray')
    if status not in ['gray', 'red', 'yellow']:
        return sanic_json({"code": 2003, "msg": "fail, status {} must be in ['gray', 'red', 'yellow']".format(status)})
    kb_ids = safe_get(req, 'kb_ids')
    kb_ids = [correct_kb_id(kb_id) for kb_id in kb_ids]
    if not kb_ids:
        kbs = local_doc_qa.milvus_summary.get_knowledge_bases(user_id)
        kb_ids = [kb[0] for kb in kbs]
    else:
        not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    gray_file_infos = local_doc_qa.milvus_summary.get_file_by_status(kb_ids, status)
    gray_file_ids = [f[0] for f in gray_file_infos]
    gray_file_names = [f[1] for f in gray_file_infos]
    debug_logger.info(f'{status} files number: {len(gray_file_names)}')
    # 删除milvus中的file
    if gray_file_ids:
        # expr = f"file_id in \"{gray_file_ids}\""
        # asyncio.create_task(run_in_background(local_doc_qa.milvus_kb.delete_expr, expr))
        for kb_id in kb_ids:
            local_doc_qa.milvus_summary.delete_files(kb_id, gray_file_ids)
    return sanic_json({"code": 200, "msg": f"delete {status} files success", "data": gray_file_names})


@get_time_async
async def local_doc_chat(req: request):
    """
    问答接口
    流程：
        1、校验用户合法性
        2、校验机器人合法性
        3、获取知识库id和提示词等各种参数
        4、校验知识库合法性
        5、根据知识库id构建问答知识库id，并过滤掉不存在的问答知识库id，将剩余的问答知识库id列表统一收集到知识库id列表中
        6、获取知识库id列表对应的有效文件列表，如果有效文件列表为空，则将知识库id列表置为空，否则更新相应知识库的最新问答时间
        7、判断流式标志，进行答案检索
            - 重构问题，获取源文档列表，对源文档列表去重、重排处理，在源文档列表中检索问题答案，有结果则返回【如果只需要检索文档，则直接返回源文档列表】
            - 在源文档列表中检索不到答案，则构造提示词模板，并对源文档列表进行预处理【满足大模型的token数量限制，图片处理】，如果只检索文档而不需要答案，则返回源文档列表；否则，依据提示词模板、源文档列表和问题构造提示词，调用大模型获取精确答案和新的对话历史
        10、根据流式标志返回问题回答结果
    """
    # time.perf_counter()返回性能计数器的值（以小数秒为单位）作为浮点数，即具有最高可用分辨率的时钟，以测量短持续时间。 它确实包括睡眠期间经过的时间，并且是系统范围的。
    preprocess_start = time.perf_counter()
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    # =============检验用户、机器人、知识库参数的合法性-start===============
    # 获取入参中的用户信息
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    # 校验用户合法性
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    # local_cluster = get_milvus_cluster_by_user_info(user_info)
    # 拼接用户id
    user_id = user_id + '__' + user_info
    # local_doc_qa.milvus_summary.update_user_cluster(user_id, [get_milvus_cluster_by_user_info(user_info)])
    debug_logger.info('local_doc_chat %s', user_id)
    debug_logger.info('user_info %s', user_info)
    # 获取bot_id【机器人id？】
    bot_id = safe_get(req, 'bot_id')
    if bot_id:
        # bot_id非空，从bot信息中获取知识库id列表和提示词
        # 校验bot_id是否存在
        if not local_doc_qa.milvus_summary.check_bot_is_exist(bot_id):
            # bot_id非空且对应bot不存在时，返回提示信息【对应bot不存在】
            return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
        # 依据bot_id获取bot信息
        bot_info = local_doc_qa.milvus_summary.get_bot(None, bot_id)[0]
        # 提取bot信息中的各个字段
        bot_id, bot_name, desc, image, prompt, welcome, model, kb_ids_str, upload_time, user_id = bot_info
        # 获取bot信息中的知识库id列表
        kb_ids = kb_ids_str.split(',')
        if not kb_ids:
            # 如果bot信息中的知识库id列表为空，则返回提示信息【对应bot未绑定知识库】
            return sanic_json({"code": 2003, "msg": "fail, Bot {} unbound knowledge base.".format(bot_id)})
        # 使用bot信息中的提示词
        custom_prompt = prompt
    else:
        # bot_id为空，则从入参中获取知识库id和提示词
        kb_ids = safe_get(req, 'kb_ids')
        custom_prompt = safe_get(req, 'custom_prompt', None)
    if len(kb_ids) > 20:
        # 如果知识库id的个数大于20个，则返回提示信息【知识库id应不超过20个】
        return sanic_json({"code": 2005, "msg": "fail, kb_ids length should less than or equal to 20"})
    # =============检验用户、机器人、知识库参数的合法性-end===============
    # =============获取问题、重排标识、流式回答标识、历史对话、仅检索内容标识、联网搜索标识、大模型路由及超参数-start===============
    # 处理知识库id【主要针对入参中的知识库id？】
    kb_ids = [correct_kb_id(kb_id) for kb_id in kb_ids]
    # 获取入参中的问题
    question = safe_get(req, 'question')
    # 获取入参中的重排标识参数，默认为True
    rerank = safe_get(req, 'rerank', default=True)
    debug_logger.info('rerank %s', rerank)
    # 获取入参中的流式输出标识，默认为False【流式：不用等结果完全出来，边解答边返回。非流式：等待结果完全出来，然后返回】
    streaming = safe_get(req, 'streaming', False)
    # 获取入参中的历史对话列表，默认为空列表
    history = safe_get(req, 'history', [])
    # 获取入参中的仅需检索内容【跳过大模型回答】标识，默认为False
    only_need_search_results = safe_get(req, 'only_need_search_results', False)
    # 获取入参中的联网搜索标识，默认为False
    need_web_search = safe_get(req, 'networking', False)
    # 获取入参中的api_base参数，默认为空字符串
    api_base = safe_get(req, 'api_base', '')
    # 如果api_base中包含0.0.0.0或127.0.0.1或localhost，替换为GATEWAY_IP
    api_base = api_base.replace('0.0.0.0', GATEWAY_IP).replace('127.0.0.1', GATEWAY_IP).replace('localhost', GATEWAY_IP)
    # 获取入参中的api_key，默认为ollama
    api_key = safe_get(req, 'api_key', 'ollama')
    # 获取入参中的api上下文长度，默认4096
    api_context_length = safe_get(req, 'api_context_length', 4096)
    # 获取入参中的推理超参数：top_p，默认0.99；temperature，默认0.5，top_k，默认30
    """
    推理超参数：
    temperature：不小于0的浮点数，取值越高，输出效果越随机，一般介于[0,1]之间。确定性答案，对应值为0
    top_k：大于0的正整数，从k个概率最大的结果中进行采样，值越大，多样性越强，一般设置为20-100之间
    top_p：大于0的浮点数，使所有被考虑的结果的概率之和大于p值，p值越大多样性越强，一般设置为0.7-0.95【top_p比top_k更有效，应优先调节top_p】
    """
    top_p = safe_get(req, 'top_p', 0.99)
    temperature = safe_get(req, 'temperature', 0.5)
    top_k = safe_get(req, 'top_k', VECTOR_SEARCH_TOP_K)

    if top_k > 100:
        # 如果入参中的top_k大于100，则返回提示信息【top_k应不超过100】
        return sanic_json({"code": 2003, "msg": "fail, top_k should less than or equal to 100"})
    # 收集缺失参数列表
    missing_params = []
    if not api_base:
        # 如果入参中的api_base为空字符串或者没传api_base参数，则收集
        missing_params.append('api_base')
    if not api_key:
        # 如果入参中的api_key为空字符串，则收集
        missing_params.append('api_key')
    if not api_context_length:
        # 如果入参中的api_context_length为空，则收集
        missing_params.append('api_context_length')
    if not top_p:
        # 如果入参中的top_p为空，则收集
        missing_params.append('top_p')
    if not top_k:
        # 如果入参中的top_k为空，则收集
        missing_params.append('top_k')
    if top_p == 1.0:
        # 如果入参中的top_p为1，则调整为0.99
        top_p = 0.99
    if not temperature:
        # 如果入参中的temperature为空，则收集
        missing_params.append('temperature')

    if missing_params:
        # 如果缺失参数列表非空，则返回提示信息【告知缺失哪些参数】
        missing_params_str = " and ".join(missing_params) if len(missing_params) > 1 else missing_params[0]
        return sanic_json({"code": 2003, "msg": f"fail, {missing_params_str} is required"})

    if only_need_search_results and streaming:
        # 如果仅需检索结果【不需要经大模型回答】且进行流式输出响应，则返回提示信息【二者不能同时设置为True】
        return sanic_json(
            {"code": 2006, "msg": "fail, only_need_search_results and streaming can't be True at the same time"})
    # 获取模型参数，默认为gpt-3.5-turbo-0613
    model = safe_get(req, 'model', 'gpt-3.5-turbo-0613')
    # 获取入参中LLM输出的最大token数，一般最大设置为上下文长度的1/4，比如4K上下文设置范围应该是[0-1024]
    max_token = safe_get(req, 'max_token')
    # 获取入参中的请求来源标识，默认unknown。可用于区分请求来源，目前共有paas，saas_bot，saas_qa，分别代表api调用，前端bot问答，前端知识库问答
    request_source = safe_get(req, 'source', 'unknown')
    # 获取入参中的混合检索标识，默认False【不开启混合检索】
    hybrid_search = safe_get(req, 'hybrid_search', False)
    # 获取入参中的web内容分块大小，默认值800，该参数在开启联网检索【networking参数为True】后生效，web搜索到的内容文本分块大小
    web_chunk_size = safe_get(req, 'web_chunk_size', DEFAULT_PARENT_CHUNK_SIZE)
    # 打印上述入参
    debug_logger.info("history: %s ", history)
    debug_logger.info("question: %s", question)
    debug_logger.info("kb_ids: %s", kb_ids)
    debug_logger.info("user_id: %s", user_id)
    debug_logger.info("custom_prompt: %s", custom_prompt)
    debug_logger.info("model: %s", model)
    debug_logger.info("max_token: %s", max_token)
    debug_logger.info("request_source: %s", request_source)
    debug_logger.info("only_need_search_results: %s", only_need_search_results)
    debug_logger.info("bot_id: %s", bot_id)
    debug_logger.info("need_web_search: %s", need_web_search)
    debug_logger.info("api_base: %s", api_base)
    debug_logger.info("api_key: %s", api_key)
    debug_logger.info("api_context_length: %s", api_context_length)
    debug_logger.info("top_p: %s", top_p)
    debug_logger.info("top_k: %s", top_k)
    debug_logger.info("temperature: %s", temperature)
    debug_logger.info("hybrid_search: %s", hybrid_search)
    debug_logger.info("web_chunk_size: %s", web_chunk_size)
    # =============获取问题、重排标识、流式回答标识、历史对话、仅检索内容标识、联网搜索标识、大模型路由及超参数-end===============
    # 处理时间字典，用以记录各个阶段的处理时间
    time_record = {}
    # =================收集存在的问答知识库id-start==================
    if kb_ids:
        # 知识库id列表非空时，校验知识库id是否存在，并返回不存在的知识库id列表
        not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            # 如果有不存在的知识库id，则返回提示信息【列出不存在的知识库id】
            return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})
        # 拼接知识库id，组成问答知识库id列表
        faq_kb_ids = [kb + '_FAQ' for kb in kb_ids]
        # 校验问答知识库id是否存在，并返回不存在的问答知识库id列表
        not_exist_faq_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, faq_kb_ids)
        # 过滤出存在的问答知识库id列表
        exist_faq_kb_ids = [kb for kb in faq_kb_ids if kb not in not_exist_faq_kb_ids]
        debug_logger.info("exist_faq_kb_ids: %s", exist_faq_kb_ids)
        # 将存在的问答知识库id列表和知识库id列表拼接起来，赋值给kb_ids
        # 列表间的拼接功能a+=b,相当于a=a+b,比如a=[1,2,3],b=[4,5,6]，那么执行后a=[1,2,3,4,5,6]
        kb_ids += exist_faq_kb_ids
    # =================收集存在的问答知识库id-end==================
    # ==========依据知识库id列表获取有效的文件信息，存在有效文件信息则更新知识库最新问答时间-start============
    file_infos = []
    # 遍历知识库列表收集对应的文件信息
    for kb_id in kb_ids:
        # 将知识库对应的文件信息收集到列表中
        file_infos.extend(local_doc_qa.milvus_summary.get_files(user_id, kb_id))
    # 从知识库文件列表中收集有效文件列表，有效的定义：status字段为green
    valid_files = [fi for fi in file_infos if fi[2] == 'green']
    if len(valid_files) == 0:
        # 如果有效文件列表为空，则清空知识库id列表，采用chat模式？
        debug_logger.info("valid_files is empty, use only chat mode.")
        kb_ids = []
    # 记录预处理消耗时间
    preprocess_end = time.perf_counter()
    time_record['preprocess'] = round(preprocess_end - preprocess_start, 2)
    # 获取格式为'2021-08-01 00:00:00'的时间戳
    qa_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    for kb_id in kb_ids:
        # 更新相应知识库的最新问答时间
        local_doc_qa.milvus_summary.update_knowledge_base_latest_qa_time(kb_id, qa_timestamp)
    # ==========依据知识库id列表获取有效的文件信息，存在有效文件信息则更新知识库最新问答时间-end============
    debug_logger.info("streaming: %s", streaming)
    if streaming:
        # 采取流式回答，边回答边返回
        debug_logger.info("start generate answer")
        # ===========流式回答，定义一个协程函数，利用ResponseStream流式输出-start==========
        async def generate_answer(response):
            """
            异步生成答案
            """
            debug_logger.info("start generate...")
            # 重构问题，获取源文档列表，对源文档列表去重、重排处理，在源文档列表中检索问题答案，有结果则返回【如果只需要检索文档，则直接返回源文档列表】
            # 在源文档列表中检索不到答案，则构造提示词模板，并对源文档列表进行预处理【满足大模型的token数量限制，图片处理】，如果只检索文档而不需要答案，则返回源文档列表；否则，依据提示词模板、源文档列表和问题构造提示词，调用大模型获取精确答案和新的对话历史
            # 这里涉及到了get_knowledge_based_answer利用yield关键字返回的生成器，遍历生成器和遍历列表效果一样，但是占用的存储空间不同
            async for resp, next_history in local_doc_qa.get_knowledge_based_answer(model=model,
                                                                                    max_token=max_token,
                                                                                    kb_ids=kb_ids,
                                                                                    query=question,
                                                                                    retriever=local_doc_qa.retriever,
                                                                                    chat_history=history,
                                                                                    streaming=True,
                                                                                    rerank=rerank,
                                                                                    custom_prompt=custom_prompt,
                                                                                    time_record=time_record,
                                                                                    need_web_search=need_web_search,
                                                                                    hybrid_search=hybrid_search,
                                                                                    web_chunk_size=web_chunk_size,
                                                                                    temperature=temperature,
                                                                                    api_base=api_base,
                                                                                    api_key=api_key,
                                                                                    api_context_length=api_context_length,
                                                                                    top_p=top_p,
                                                                                    top_k=top_k
                                                                                    ):
                # 获取响应中的答案
                chunk_data = resp["result"]
                if not chunk_data:
                    # 如果响应中的答案为空，则继续遍历下一个响应
                    continue
                # 获取答案字符串【原始响应中，answer为起始字符串标识，占用6个，答案内容从第7个开始】
                chunk_str = chunk_data[6:]
                if chunk_str.startswith("[DONE]"):
                    retrieval_documents = format_source_documents(resp["retrieval_documents"])
                    source_documents = format_source_documents(resp["source_documents"])
                    result = next_history[-1][1]
                    # result = resp['result']
                    time_record['chat_completed'] = round(time.perf_counter() - preprocess_start, 2)
                    if time_record.get('llm_completed', 0) > 0:
                       time_record['tokens_per_second'] = round(
                            len(result) / time_record['llm_completed'], 2)
                    formatted_time_record = format_time_record(time_record)
                    chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, "model": model,
                                 "product_source": request_source, 'time_record': formatted_time_record,
                                 'history': history,
                                 'condense_question': resp['condense_question'], 'prompt': resp['prompt'],
                                 'result': result, 'retrieval_documents': retrieval_documents,
                                 'source_documents': source_documents, 'bot_id': bot_id}
                    local_doc_qa.milvus_summary.add_qalog(**chat_data)
                    qa_logger.info("chat_data: %s", chat_data)
                    debug_logger.info("response: %s", chat_data['result'])
                    stream_res = {
                        "code": 200,
                        "msg": "success stream chat",
                        "question": question,
                        "response": result,
                        "model": model,
                        "history": next_history,
                        "condense_question": resp['condense_question'],
                        "source_documents": source_documents,
                        "retrieval_documents": retrieval_documents,
                        "time_record": formatted_time_record,
                        "show_images": resp.get('show_images', [])
                    }
                else:
                    time_record['rollback_length'] = resp.get('rollback_length', 0)
                    if 'first_return' not in time_record:
                        time_record['first_return'] = round(time.perf_counter() - preprocess_start, 2)
                    chunk_js = json.loads(chunk_str)
                    delta_answer = chunk_js["answer"]
                    stream_res = {
                        "code": 200,
                        "msg": "success",
                        "question": "",
                        "response": delta_answer,
                        "history": [],
                        "source_documents": [],
                        "retrieval_documents": [],
                        "time_record": format_time_record(time_record),
                    }
                await response.write(f"data: {json.dumps(stream_res, ensure_ascii=False)}\n\n")
                if chunk_str.startswith("[DONE]"):
                    await response.eof()
                await asyncio.sleep(0.001)

        response_stream = ResponseStream(generate_answer, content_type='text/event-stream')
        # ===========流式回答，定义一个协程函数，利用ResponseStream流式输出-end==========
        return response_stream

    else:
        # ===========非流式回答-start==========
        # 非流式回答，得到完整答案，然后返回
        async for resp, history in local_doc_qa.get_knowledge_based_answer(model=model,
                                                                           max_token=max_token,
                                                                           kb_ids=kb_ids,
                                                                           query=question,
                                                                           retriever=local_doc_qa.retriever,
                                                                           chat_history=history, streaming=False,
                                                                           rerank=rerank,
                                                                           custom_prompt=custom_prompt,
                                                                           time_record=time_record,
                                                                           only_need_search_results=only_need_search_results,
                                                                           need_web_search=need_web_search,
                                                                           hybrid_search=hybrid_search,
                                                                           web_chunk_size=web_chunk_size,
                                                                           temperature=temperature,
                                                                           api_base=api_base,
                                                                           api_key=api_key,
                                                                           api_context_length=api_context_length,
                                                                           top_p=top_p,
                                                                           top_k=top_k
                                                                           ):
            pass
        if only_need_search_results:
            return sanic_json(
                {"code": 200, "question": question, "source_documents": format_source_documents(resp)})
        retrieval_documents = format_source_documents(resp["retrieval_documents"])
        source_documents = format_source_documents(resp["source_documents"])
        formatted_time_record = format_time_record(time_record)
        chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, 'time_record': formatted_time_record,
                     'history': history, "condense_question": resp['condense_question'], "model": model,
                     "product_source": request_source,
                     'retrieval_documents': retrieval_documents, 'prompt': resp['prompt'], 'result': resp['result'],
                     'source_documents': source_documents, 'bot_id': bot_id}
        local_doc_qa.milvus_summary.add_qalog(**chat_data)
        qa_logger.info("chat_data: %s", chat_data)
        debug_logger.info("response: %s", chat_data['result'])
        # ===========非流式回答-end==========
        return sanic_json({"code": 200, "msg": "success no stream chat", "question": question,
                           "response": resp["result"], "model": model,
                           "history": history, "condense_question": resp['condense_question'],
                           "source_documents": source_documents, "retrieval_documents": retrieval_documents,
                           "time_record": formatted_time_record})


@get_time_async
async def document(req: request):
    description = """
# QAnything 介绍
[戳我看视频>>>>>【有道QAnything介绍视频.mp4】](https://docs.popo.netease.com/docs/7e512e48fcb645adadddcf3107c97e7c)

**QAnything** (**Q**uestion and **A**nswer based on **Anything**) 是支持任意格式的本地知识库问答系统。

您的任何格式的本地文件都可以往里扔，即可获得准确、快速、靠谱的问答体验。

**目前已支持格式:**
* PDF
* Word(doc/docx)
* PPT
* TXT
* 图片
* 网页链接
* ...更多格式，敬请期待

# API 调用指南

## API Base URL

https://qanything.youdao.com

## 鉴权
目前使用微信鉴权,步骤如下:
1. 客户端通过扫码微信二维码(首次登录需要关注公众号)
2. 获取token
3. 调用下面所有API都需要通过authorization参数传入这个token

注意：authorization参数使用Bearer auth认证方式

生成微信二维码以及获取token的示例代码下载地址：[微信鉴权示例代码](https://docs.popo.netease.com/docs/66652d1a967e4f779594aef3306f6097)

## API 接口说明
    {
        "api": "/api/local_doc_qa/upload_files"
        "name": "上传文件",
        "description": "上传文件接口，支持多个文件同时上传，需要指定知识库名称",
    },
    {
        "api": "/api/local_doc_qa/upload_weblink"
        "name": "上传网页链接",
        "description": "上传网页链接，自动爬取网页内容，需要指定知识库名称",
    },
    {
        "api": "/api/local_doc_qa/local_doc_chat" 
        "name": "问答接口",
        "description": "知识库问答接口，指定知识库名称，上传用户问题，通过传入history支持多轮对话",
    },
    {
        "api": "/api/local_doc_qa/list_files" 
        "name": "文件列表",
        "description": "列出指定知识库下的所有文件名，需要指定知识库名称",
    },
    {
        "api": "/api/local_doc_qa/delete_files" 
        "name": "删除文件",
        "description": "删除指定知识库下的指定文件，需要指定知识库名称",
    },

"""
    return sanic_text(description)


@get_time_async
async def get_doc_completed(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("get_doc_chunks %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id: {}".format(kb_id))
    file_id = safe_get(req, 'file_id')
    if not file_id:
        return sanic_json({"code": 2005, "msg": "fail, file_id is None"})
    debug_logger.info("file_id: {}".format(file_id))
    page_id = safe_get(req, 'page_id', 1)  # 默认为第一页
    page_limit = safe_get(req, 'page_limit', 10)  # 默认每页显示10条记录

    sorted_json_datas = local_doc_qa.milvus_summary.get_document_by_file_id(file_id)
    # completed_doc = local_doc_qa.get_completed_document(file_id)
    # for json_data in sorted_json_datas:
    #     completed_text += json_data['kwargs']['page_content'] + '\n'
    #     if len(completed_text) > 10000:
    #         return sanic_json({"code": 200, "msg": "failed, completed_text too long, the max length is 10000"})
    chunks = [json_data['kwargs'] for json_data in sorted_json_datas]

    # 计算总记录数
    total_count = len(chunks)
    # 计算总页数
    total_pages = (total_count + page_limit - 1) // page_limit
    if page_id > total_pages and total_count != 0:
        return sanic_json({"code": 2002, "msg": f'输入非法！page_id超过最大值，page_id: {page_id}，最大值：{total_pages}，请检查！'})
    # 计算当前页的起始和结束索引
    start_index = (page_id - 1) * page_limit
    end_index = start_index + page_limit
    # 截取当前页的数据
    current_page_chunks = chunks[start_index:end_index]
    for chunk in current_page_chunks:
        chunk['page_content'] = replace_image_references(chunk['page_content'], file_id)

    # return sanic_json({"code": 200, "msg": "success", "completed_text": completed_doc.page_content,
    #                    "chunks": current_page_chunks, "page_id": page, "total_count": total_count})
    file_location = local_doc_qa.milvus_summary.get_file_location(file_id)
    # 获取file_location的上一级目录
    file_path = os.path.dirname(file_location)
    return sanic_json({"code": 200, "msg": "success", "chunks": current_page_chunks, "file_path": file_path,
                       "page_id": page_id, "page_limit": page_limit, "total_count": total_count})


@get_time_async
async def get_qa_info(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    any_kb_id = safe_get(req, 'any_kb_id')
    user_id = safe_get(req, 'user_id')
    if user_id is None and not any_kb_id:
        return sanic_json({"code": 2005, "msg": "fail, user_id and any_kb_id is None"})
    if any_kb_id:
        any_kb_id = correct_kb_id(any_kb_id)
        debug_logger.info("get_qa_info %s", any_kb_id)
    if user_id:
        user_info = safe_get(req, 'user_info', "1234")
        passed, msg = check_user_id_and_user_info(user_id, user_info)
        if not passed:
            return sanic_json({"code": 2001, "msg": msg})
        user_id = user_id + '__' + user_info
        debug_logger.info("get_qa_info %s", user_id)
    query = safe_get(req, 'query')
    bot_id = safe_get(req, 'bot_id')
    qa_ids = safe_get(req, "qa_ids")
    time_start = safe_get(req, 'time_start')
    time_end = safe_get(req, 'time_end')
    time_range = get_time_range(time_start, time_end)
    if not time_range:
        return {"code": 2002, "msg": f'输入非法！time_start格式错误，time_start: {time_start}，示例：2024-10-05，请检查！'}
    only_need_count = safe_get(req, 'only_need_count', False)
    debug_logger.info(f"only_need_count: {only_need_count}")
    if only_need_count:
        need_info = ["timestamp"]
        qa_infos = local_doc_qa.milvus_summary.get_qalog_by_filter(need_info=need_info, user_id=user_id, time_range=time_range)
        # timestamp = now.strftime("%Y%m%d%H%M")
        # 按照timestamp，按照天数进行统计，比如20240628，20240629，20240630，计算每天的问答数量
        qa_infos = sorted(qa_infos, key=lambda x: x['timestamp'])
        qa_infos = [qa_info['timestamp'] for qa_info in qa_infos]
        qa_infos = [qa_info[:10] for qa_info in qa_infos]
        qa_infos_by_day = dict(Counter(qa_infos))
        return sanic_json({"code": 200, "msg": "success", "qa_infos_by_day": qa_infos_by_day})

    page_id = safe_get(req, 'page_id', 1)
    page_limit = safe_get(req, 'page_limit', 10)
    default_need_info = ["qa_id", "user_id", "bot_id", "kb_ids", "query", "model", "product_source", "time_record",
                         "history", "condense_question", "prompt", "result", "retrieval_documents", "source_documents",
                         "timestamp"]
    need_info = safe_get(req, 'need_info', default_need_info)
    save_to_excel = safe_get(req, 'save_to_excel', False)
    qa_infos = local_doc_qa.milvus_summary.get_qalog_by_filter(need_info=need_info, user_id=user_id, query=query,
                                                               bot_id=bot_id, time_range=time_range,
                                                               any_kb_id=any_kb_id, qa_ids=qa_ids)
    if save_to_excel:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        file_name = f"QAnything_QA_{timestamp}.xlsx"
        file_path = export_qalogs_to_excel(qa_infos, need_info, file_name)
        return await response.file(file_path, filename=file_name,
                                   mime_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   headers={'Content-Disposition': f'attachment; filename="{file_name}"'})

    # 计算总记录数
    total_count = len(qa_infos)
    # 计算总页数
    total_pages = (total_count + page_limit - 1) // page_limit
    if page_id > total_pages and total_count != 0:
        return sanic_json(
            {"code": 2002, "msg": f'输入非法！page_id超过最大值，page_id: {page_id}，最大值：{total_pages}，请检查！'})
    # 计算当前页的起始和结束索引
    start_index = (page_id - 1) * page_limit
    end_index = start_index + page_limit
    # 截取当前页的数据
    current_qa_infos = qa_infos[start_index:end_index]
    msg = f"检测到的Log总数为{total_count}, 本次返回page_id为{page_id}的数据，每页显示{page_limit}条"

    # if len(qa_infos) > 100:
    #     pages = math.ceil(len(qa_infos) // 100)
    #     if page_id is None:
    #         msg = f"检索到的Log数超过100，需要分页返回，总数为{len(qa_infos)}, 请使用page_id参数获取某一页数据，参数范围：[0, {pages - 1}], 本次返回page_id为0的数据"
    #         qa_infos = qa_infos[:100]
    #         page_id = 0
    #     elif page_id >= pages:
    #         return sanic_json(
    #             {"code": 2002, "msg": f'输入非法！page_id超过最大值，page_id: {page_id}，最大值：{pages - 1}，请检查！'})
    #     else:
    #         msg = f"检索到的Log数超过100，需要分页返回，总数为{len(qa_infos)}, page范围：[0, {pages - 1}], 本次返回page_id为{page_id}的数据"
    #         qa_infos = qa_infos[page_id * 100:(page_id + 1) * 100]
    # else:
    #     msg = f"检索到的Log数为{len(qa_infos)}，一次返回所有数据"
    #     page_id = 0
    return sanic_json({"code": 200, "msg": msg, "page_id": page_id, "page_limit": page_limit, "qa_infos": current_qa_infos, "total_count": total_count})


@get_time_async
async def get_random_qa(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    limit = safe_get(req, 'limit', 10)
    time_start = safe_get(req, 'time_start')
    time_end = safe_get(req, 'time_end')
    need_info = safe_get(req, 'need_info')
    time_range = get_time_range(time_start, time_end)
    if not time_range:
        return {"code": 2002, "msg": f'输入非法！time_start格式错误，time_start: {time_start}，示例：2024-10-05，请检查！'}

    debug_logger.info(f"get_random_qa limit: {limit}, time_range: {time_range}")
    qa_infos = local_doc_qa.milvus_summary.get_random_qa_infos(limit=limit, time_range=time_range, need_info=need_info)

    counts = local_doc_qa.milvus_summary.get_statistic(time_range=time_range)
    return sanic_json({"code": 200, "msg": "success", "total_users": counts["total_users"],
                       "total_queries": counts["total_queries"], "qa_infos": qa_infos})


@get_time_async
async def get_related_qa(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    qa_id = safe_get(req, 'qa_id')
    if not qa_id:
        return sanic_json({"code": 2005, "msg": "fail, qa_id is None"})
    need_info = safe_get(req, 'need_info')
    need_more = safe_get(req, 'need_more', False)
    debug_logger.info("get_related_qa %s", qa_id)
    qa_log, recent_logs, older_logs = local_doc_qa.milvus_summary.get_related_qa_infos(qa_id, need_info, need_more)
    # 按kb_ids划分sections
    recent_sections = defaultdict(list)
    for log in recent_logs:
        recent_sections[log['kb_ids']].append(log)
    # 把recent_sections的key改为自增的正整数，且每个log都新增kb_name
    for i, kb_ids in enumerate(list(recent_sections.keys())):
        kb_names = local_doc_qa.milvus_summary.get_knowledge_base_name(json.loads(kb_ids))
        kb_names = [kb_name for user_id, kb_id, kb_name in kb_names]
        kb_names = ','.join(kb_names)
        recent_sections[i] = recent_sections.pop(kb_ids)
        for log in recent_sections[i]:
            log['kb_names'] = kb_names

    older_sections = defaultdict(list)
    for log in older_logs:
        older_sections[log['kb_ids']].append(log)
    # 把older_sections的key改为自增的正整数，且每个log都新增kb_name
    for i, kb_ids in enumerate(list(older_sections.keys())):
        kb_names = local_doc_qa.milvus_summary.get_knowledge_base_name(json.loads(kb_ids))
        kb_names = [kb_name for user_id, kb_id, kb_name in kb_names]
        kb_names = ','.join(kb_names)
        older_sections[i] = older_sections.pop(kb_ids)
        for log in older_sections[i]:
            log['kb_names'] = kb_names

    return sanic_json({"code": 200, "msg": "success", "qa_info": qa_log, "recent_sections": recent_sections,
                       "older_sections": older_sections})


@get_time_async
async def get_user_id(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id: {}".format(kb_id))
    user_id = local_doc_qa.milvus_summary.get_user_by_kb_id(kb_id)
    if not user_id:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(kb_id)})
    else:
        return sanic_json({"code": 200, "msg": "success", "user_id": user_id})


@get_time_async
async def get_doc(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    doc_id = safe_get(req, 'doc_id')
    debug_logger.info("get_doc %s", doc_id)
    if not doc_id:
        return sanic_json({"code": 2005, "msg": "fail, doc_id is None"})
    doc_json_data = local_doc_qa.milvus_summary.get_document_by_doc_id(doc_id)
    return sanic_json({"code": 200, "msg": "success", "doc_text": doc_json_data['kwargs']})


@get_time_async
async def get_rerank_results(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    query = safe_get(req, 'query')
    if not query:
        return sanic_json({"code": 2005, "msg": "fail, query is None"})
    doc_ids = safe_get(req, 'doc_ids')
    doc_strs = safe_get(req, 'doc_strs')
    if not doc_ids and not doc_strs:
        return sanic_json({"code": 2005, "msg": "fail, doc_ids is None and doc_strs is None"})
    if doc_ids:
        rerank_results = await local_doc_qa.get_rerank_results(query, doc_ids=doc_ids)
    else:
        rerank_results = await local_doc_qa.get_rerank_results(query, doc_strs=doc_strs)

    return sanic_json({"code": 200, "msg": "success", "rerank_results": format_source_documents(rerank_results)})


@get_time_async
async def get_user_status(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("get_user_status %s", user_id)
    user_status = local_doc_qa.milvus_summary.get_user_status(user_id)
    if user_status is None:
        return sanic_json({"code": 2003, "msg": "fail, user {} not found".format(user_id)})
    if user_status == 0:
        status = 'green'
    else:
        status = 'red'
    return sanic_json({"code": 200, "msg": "success", "status": status})


@get_time_async
async def health_check(req: request):
    # 实现一个服务健康检查的逻辑，正常就返回200，不正常就返回500
    return sanic_json({"code": 200, "msg": "success"})


@get_time_async
async def get_bot_info(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    bot_id = safe_get(req, 'bot_id')
    if bot_id:
        if not local_doc_qa.milvus_summary.check_bot_is_exist(bot_id):
            return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
    debug_logger.info("get_bot_info %s", user_id)
    bot_infos = local_doc_qa.milvus_summary.get_bot(user_id, bot_id)
    data = []
    for bot_info in bot_infos:
        if bot_info[7] != "":
            kb_ids = bot_info[7].split(',')
            kb_infos = local_doc_qa.milvus_summary.get_knowledge_base_name(kb_ids)
            kb_names = []
            for kb_id in kb_ids:
                for kb_info in kb_infos:
                    if kb_id == kb_info[1]:
                        kb_names.append(kb_info[2])
                        break
        else:
            kb_ids = []
            kb_names = []
        info = {"bot_id": bot_info[0], "user_id": user_id, "bot_name": bot_info[1], "description": bot_info[2],
                "head_image": bot_info[3], "prompt_setting": bot_info[4], "welcome_message": bot_info[5],
                "model": bot_info[6], "kb_ids": kb_ids, "kb_names": kb_names,
                "update_time": bot_info[8].strftime("%Y-%m-%d %H:%M:%S")}
        data.append(info)
    return sanic_json({"code": 200, "msg": "success", "data": data})


@get_time_async
async def new_bot(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    bot_name = safe_get(req, "bot_name")
    desc = safe_get(req, "description", BOT_DESC)
    head_image = safe_get(req, "head_image", BOT_IMAGE)
    prompt_setting = safe_get(req, "prompt_setting", BOT_PROMPT)
    welcome_message = safe_get(req, "welcome_message", BOT_WELCOME)
    model = safe_get(req, "model", 'MiniChat-2-3B')
    kb_ids = safe_get(req, "kb_ids", [])
    kb_ids_str = ",".join(kb_ids)

    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
    debug_logger.info("new_bot %s", user_id)
    bot_id = 'BOT' + uuid.uuid4().hex
    local_doc_qa.milvus_summary.new_qanything_bot(bot_id, user_id, bot_name, desc, head_image, prompt_setting,
                                                  welcome_message, model, kb_ids_str)
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return sanic_json({"code": 200, "msg": "success create qanything bot {}".format(bot_id),
                       "data": {"bot_id": bot_id, "bot_name": bot_name, "create_time": create_time}})


@get_time_async
async def delete_bot(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("delete_bot %s", user_id)
    bot_id = safe_get(req, 'bot_id')
    if not local_doc_qa.milvus_summary.check_bot_is_exist(bot_id):
        return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
    local_doc_qa.milvus_summary.delete_bot(user_id, bot_id)
    return sanic_json({"code": 200, "msg": "Bot {} delete success".format(bot_id)})


@get_time_async
async def update_bot(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("update_bot %s", user_id)
    bot_id = safe_get(req, 'bot_id')
    if not local_doc_qa.milvus_summary.check_bot_is_exist(bot_id):
        return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
    bot_info = local_doc_qa.milvus_summary.get_bot(user_id, bot_id)[0]
    bot_name = safe_get(req, "bot_name", bot_info[1])
    description = safe_get(req, "description", bot_info[2])
    head_image = safe_get(req, "head_image", bot_info[3])
    prompt_setting = safe_get(req, "prompt_setting", bot_info[4])
    welcome_message = safe_get(req, "welcome_message", bot_info[5])
    model = safe_get(req, "model", bot_info[6])
    kb_ids = safe_get(req, "kb_ids")
    if kb_ids is not None:
        not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
            return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
        kb_ids_str = ",".join(kb_ids)
    else:
        kb_ids_str = bot_info[7]
    # 判断哪些项修改了
    if bot_name != bot_info[1]:
        debug_logger.info(f"update bot name from {bot_info[1]} to {bot_name}")
    if description != bot_info[2]:
        debug_logger.info(f"update bot description from {bot_info[2]} to {description}")
    if head_image != bot_info[3]:
        debug_logger.info(f"update bot head_image from {bot_info[3]} to {head_image}")
    if prompt_setting != bot_info[4]:
        debug_logger.info(f"update bot prompt_setting from {bot_info[4]} to {prompt_setting}")
    if welcome_message != bot_info[5]:
        debug_logger.info(f"update bot welcome_message from {bot_info[5]} to {welcome_message}")
    if model != bot_info[6]:
        debug_logger.info(f"update bot model from {bot_info[6]} to {model}")
    if kb_ids_str != bot_info[7]:
        debug_logger.info(f"update bot kb_ids from {bot_info[7]} to {kb_ids_str}")
    #  update_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP 根据这个mysql的格式获取现在的时间
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    debug_logger.info(f"update_time: {update_time}")
    local_doc_qa.milvus_summary.update_bot(user_id, bot_id, bot_name, description, head_image, prompt_setting,
                                           welcome_message, model, kb_ids_str, update_time)
    return sanic_json({"code": 200, "msg": "Bot {} update success".format(bot_id)})


@get_time_async
async def update_chunks(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("update_chunks %s", user_id)
    doc_id = safe_get(req, 'doc_id')
    debug_logger.info(f"doc_id: {doc_id}")
    yellow_files = local_doc_qa.milvus_summary.get_files_by_status("yellow")
    if len(yellow_files) > 0:
        return sanic_json({"code": 2002, "msg": f"fail, currently, there are {len(yellow_files)} files being parsed, please wait for all files to finish parsing before updating the chunk."})
    update_content = safe_get(req, 'update_content')
    debug_logger.info(f"update_content: {update_content}")
    chunk_size = safe_get(req, 'chunk_size', DEFAULT_PARENT_CHUNK_SIZE)
    debug_logger.info(f"chunk_size: {chunk_size}")
    update_content_tokens = num_tokens_embed(update_content)
    if update_content_tokens > chunk_size:
        return sanic_json({"code": 2003, "msg": f"fail, update_content too long, please reduce the length, "
                                                f"your update_content tokens is {update_content_tokens}, "
                                                f"the max tokens is {chunk_size}"})
    doc_json = local_doc_qa.milvus_summary.get_document_by_doc_id(doc_id)
    if not doc_json:
        return sanic_json({"code": 2004, "msg": "fail, DocId {} not found".format(doc_id)})
    doc = Document(page_content=update_content, metadata=doc_json['kwargs']['metadata'])
    doc.metadata['doc_id'] = doc_id
    local_doc_qa.milvus_summary.update_document(doc_id, update_content)
    expr = f'doc_id == "{doc_id}"'
    local_doc_qa.milvus_kb.delete_expr(expr)
    await local_doc_qa.retriever.insert_documents([doc], chunk_size, True)
    return sanic_json({"code": 200, "msg": "success update doc_id {}".format(doc_id)})


@get_time_async
async def get_file_base64(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    file_id = safe_get(req, 'file_id')
    debug_logger.info("get_file_base64 %s", file_id)
    file_location = local_doc_qa.milvus_summary.get_file_location(file_id)
    # file_location = '/home/liujx/Downloads/2021-08-01 00:00:00.pdf'
    if not file_location:
        return sanic_json({"code": 2005, "msg": "fail, file_id is Invalid"})
    with open(file_location, "rb") as f:
        file_base64 = base64.b64encode(f.read()).decode()
    return sanic_json({"code": 200, "msg": "success", "file_base64": file_base64})
