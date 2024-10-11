import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

sys.path.append(root_dir)
print(root_dir)

from sanic import Sanic, response
from qanything_kernel.utils.custom_log import insert_logger
from qanything_kernel.utils.general_utils import get_time_async
from qanything_kernel.core.retriever.general_document import LocalFileForInsert
from qanything_kernel.core.retriever.vectorstore import VectorStoreMilvusClient
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.core.retriever.elasticsearchstore import StoreElasticSearchClient
from qanything_kernel.core.retriever.parent_retriever import ParentRetriever
from qanything_kernel.configs.model_config import MYSQL_HOST_LOCAL, MYSQL_PORT_LOCAL, \
    MYSQL_USER_LOCAL, MYSQL_PASSWORD_LOCAL, MYSQL_DATABASE_LOCAL, MAX_CHARS
from sanic.worker.manager import WorkerManager
import asyncio
import traceback
import time
import random
import aiomysql
import argparse
import json

WorkerManager.THRESHOLD = 6000

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8110, help='port')
parser.add_argument('--workers', type=int, default=4, help='workers')
# 检查是否是local或online，不是则报错
args = parser.parse_args()

INSERT_WORKERS = args.workers
insert_logger.info(f"INSERT_WORKERS: {INSERT_WORKERS}")

# 创建 Sanic 应用
app = Sanic("InsertFileService")

# 数据库配置
db_config = {
    'host': MYSQL_HOST_LOCAL,
    'port': MYSQL_PORT_LOCAL,
    'user': MYSQL_USER_LOCAL,
    'password': MYSQL_PASSWORD_LOCAL,
    'db': MYSQL_DATABASE_LOCAL,
}


@get_time_async
async def process_data(retriever, milvus_kb, mysql_client, file_info, time_record):
    """
    处理文件数据：切片和保存至向量数据库
    流程如下：
    1、设置处理时长限制参数，更新知识库最新插入时间
    2、获取LocalFileForInsert对象
    3、依据文件id更新文件信息中的msg值：处理进度在【1%~5%之间的随机一个数值】之间
    4、文件切片【调用LocalFileForInsert对象的split_file_to_docs方法执行切片--》校验切片所得文档列表的内容长度之和是否超过100万的限制--》超出限制，提示超出并返回--》切片所得文档是否为空，为空，提示为空或链接存在反爬机制或需要登录，并返回--》处理切片异常【区分超时异常和其他异常】】
    5、依据文件id更新msg值：处理进度在【5%~75%之间的随机一个数值】之间
    6、将切片所得文档列表保存至向量数据库【调用retriever对象的insert_documents方法，传参为切片所得文档列表和切片尺寸【来自文件信息】--》处理保存异常【区分超时异常和其他异常】】
    7、返回处理结果：文件状态【green为正常，red为异常】、内容长度、切片数量、文件处理消息
    对文档的切片处理仅有两步：第一步，利用工具类分文件类型进行文档切分【纯文件处理，尚未用到切片尺寸】；第二步【用到了文件信息中的切片尺寸chunk_size】，利用换行符和中文标点符号对前期切分得到的文档列表进行二次切分。TODO 词嵌入在哪里呢？
    """
    # 初始化各变量值
    # 解析时长
    parse_timeout_seconds = 300
    # 插入时长
    insert_timeout_seconds = 300
    # 内容长度
    content_length = -1
    # 文件状态
    status = 'green'
    # 处理起始时间
    process_start = time.perf_counter()
    # 打印开始处理文件的日志。疑问：知识库id从哪里来的？【文件信息中会有知识库id，但是可以直接写成kb_id吗？】
    insert_logger.info(f'Start insert file: {file_info}')
    _, file_id, user_id, file_name, kb_id, file_location, file_size, file_url, chunk_size = file_info
    # 获取格式为'2021-08-01 00:00:00'的时间戳
    insert_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # 更新知识库最新插入时间
    mysql_client.update_knowlegde_base_latest_insert_time(kb_id, insert_timestamp)
    # 获取插入本地文件对象local_file
    local_file = LocalFileForInsert(user_id, kb_id, file_id, file_location, file_name, file_url, chunk_size, mysql_client)
    msg = "success"
    # 初始化文件切片数量为0
    chunks_number = 0
    # 依据文件id更新文件信息中的msg值：处理进度在【1%~5%之间的随机一个数值】之间
    mysql_client.update_file_msg(file_id, f'Processing:{random.randint(1, 5)}%')
    # 这里是把文件做向量化，然后写入Milvus【向量数据库】的逻辑
    # 文件切片起始时间
    start = time.perf_counter()
    try:
        # 执行local_file对象的split_file_to_docs方法进行文件切片，协程等待时长限制300秒【超时触发asyncio.TimeoutError】
        await asyncio.wait_for(
            asyncio.to_thread(local_file.split_file_to_docs),
            timeout=parse_timeout_seconds
        )
        # 文件切片后，得到文档列表【此时还未设置文档id】，遍历文档列表累计所有文档的内容长度，将内容长度之和赋值给content_length
        content_length = sum([len(doc.page_content) for doc in local_file.docs])
        if content_length > MAX_CHARS:
            # 如果总内容长度超过最大限制1000000
            # 设置文件状态为red
            status = 'red'
            # 设置文件处理消息：内容长度太大
            msg = f"{file_name} content_length too large, {content_length} >= MaxLength({MAX_CHARS})"
            # 返回文件处理结果：文件状态、内容长度、切片数量、文件处理消息
            return status, content_length, chunks_number, msg
        elif content_length == 0:
            # 如果总内容长度为0
            # 设置文件状态为red
            status = 'red'
            # 设置文件处理消息：文件内容为空或url存在反爬机制或者要求登录
            msg = f"{file_name} content_length is 0, file content is empty or The URL exists anti-crawling or requires login."
            # 返回文件处理结果：文件状态、内容长度、切片数量、文件处理消息
            return status, content_length, chunks_number, msg
    except asyncio.TimeoutError:
        # 文件切片处理超时异常
        # 将内部标志设置为true。所有等待它变为true的线程都被唤醒。线程一旦标志为true，调用wait()将完全不会阻塞。
        local_file.event.set()
        # 打印文件切片超时日志
        insert_logger.error(f'Timeout: split_file_to_docs took longer than {parse_timeout_seconds} seconds')
        # 设置文件状态为red
        status = 'red'
        # 设置文件处理消息：切片超时
        msg = f"split_file_to_docs timeout: {parse_timeout_seconds}s"
        # 返回文件处理结果：文件状态、内容长度、切片数量、文件处理消息
        return status, content_length, chunks_number, msg
    except Exception as e:
        # 文件处理其他异常
        # 获取文件处理异常信息
        error_info = f'split_file_to_docs error: {traceback.format_exc()}'
        # msg = error_info
        # 打印文件处理异常信息
        insert_logger.error(error_info)
        # 设置文件状态为red
        status = 'red'
        # 设置文件处理消息：文件切片异常
        msg = f"split_file_to_docs error"
        # 返回文件处理结果：文件状态、内容长度、切片数量、文件处理消息
        return status, content_length, chunks_number, msg
    # 文件切片正常且文档内容长度之和未超出最大限制时
    # 文件切片结束时间
    end = time.perf_counter()
    # 记录文件切片处理时长并四舍五入保留2位小数
    time_record['parse_time'] = round(end - start, 2)
    # 打印文件处理时长和所得切片文档个数的日志信息
    insert_logger.info(f'parse time: {end - start} {len(local_file.docs)}')
    # 依据文件id更新msg值：处理进度在【5%~75%之间的随机一个数值】之间
    mysql_client.update_file_msg(file_id, f'Processing:{random.randint(5, 75)}%')
    # 将文件切片得到的文档列表插入向量数据库
    try:
        # 插入向量数据库起始时间
        start = time.perf_counter()
        # 执行retriever对象的insert_documents方法进行文档插入向量数据库，协程等待时长限制300秒【超时触发asyncio.TimeoutError】
        chunks_number, insert_time_record = await asyncio.wait_for(
            retriever.insert_documents(local_file.docs, chunk_size),
            timeout=insert_timeout_seconds)
        # 记录插入时间
        insert_time = time.perf_counter()
        # 更新插入时长
        time_record.update(insert_time_record)
        # 打印文档插入向量数据库时长的日志信息
        insert_logger.info(f'insert time: {insert_time - start}')
        # 依据文件id更新文件的切片数量
        mysql_client.update_chunks_number(local_file.file_id, chunks_number)
    except asyncio.TimeoutError:
        # 执行retriever对象的insert_documents方法超时
        # 打印向量数据库插入操作超时的日志信息
        insert_logger.error(f'Timeout: milvus insert took longer than {insert_timeout_seconds} seconds')
        # 文件id前缀
        expr = f'file_id == \"{local_file.file_id}\"'
        # 清除向量数据中相应文件id的数据
        milvus_kb.delete_expr(expr)
        # 设置文件状态为red
        status = 'red'
        # 插入超时标志为True
        time_record['insert_timeout'] = True
        # 设置文件处理消息：向量数据库插入超时
        msg = f"milvus insert timeout: {insert_timeout_seconds}s"
        # 返回文件处理结果：文件状态、内容长度、切片数量、文件处理消息
        return status, content_length, chunks_number, msg
    except Exception as e:
        # 向量数据库插入时的其他异常
        # 记录异常信息
        error_info = f'milvus insert error: {traceback.format_exc()}'
        # 打印其他异常信息日志
        insert_logger.error(error_info)
        # 设置文件状态为red
        status = 'red'
        # 插入错误标志为True
        time_record['insert_error'] = True
        # 设置文件处理消息：向量数据库插入错误
        msg = f"milvus insert error"
        # 返回文件处理结果：文件状态、内容长度、切片数量、文件处理消息
        return status, content_length, chunks_number, msg
    # 插入向量数据库成功时
    # 依据文件id更新msg值：处理进度在【75%~100%之间的随机一个数值】之间
    mysql_client.update_file_msg(file_id, f'Processing:{random.randint(75, 100)}%')
    # 记录文件上传总时长，四舍五入保留2位小数
    time_record['upload_total_time'] = round(time.perf_counter() - process_start, 2)
    # 依据文件id更新文件上传信息为time_record
    mysql_client.update_file_upload_infos(file_id, time_record)
    # 打印文件保存至向量数据库的日志
    insert_logger.info(f'insert_files_to_milvus: {user_id}, {kb_id}, {file_id}, {file_name}, {status}')
    # 将字典对象time_record字符串化并赋值给msg
    msg = json.dumps(time_record, ensure_ascii=False)
    # 返回文件处理结果：文件状态、内容长度、切片数量、文件处理消息
    return status, content_length, chunks_number, msg


async def check_and_process(pool):
    """
    输入为一个数据库连接池
    检查和处理
    流程如下：
    1、获取worker_id、数据库客户端、向量数据库客户端、es客户端、retriever对象
    2、设置文件处理间隔3秒，也即处理完一个文件后，睡3秒，接着处理下一个文件
    3、文件处理流程：
        1、依据初始状态gray和worker_id条件获取要处理的文件信息【列表，但实际每次只取第一个文件进行处理】
        2、更新文件状态为yellow，表示即将开始处理
        3、进行文件处理【切片、保存至向量数据库】
        4、将文件处理结果【切片数量、切片所得文档总内容尺寸、状态标识、处理信息】更新至数据库
        5、文件处理过程中，若发生向量数据库或数据库连接异常，则打印连接异常日志，并将文件状态置为red，更新至数据库；若发生数据库连接异常，则打印数据库二次连接异常
    """
    # 当SANIC_WORKER_NAME不在系统环境变量中时，赋值为MainProcess，否则赋值为os.environ['SANIC_WORKER_NAME']
    process_type = 'MainProcess' if 'SANIC_WORKER_NAME' not in os.environ else os.environ['SANIC_WORKER_NAME']
    # 获取处理类型以_分隔所得数组的倒数第二个元素取整结果，当其倒数第二个元素不是整数时会报错
    worker_id = int(process_type.split('-')[-2])
    # 打印当前进程的worker_id
    insert_logger.info(f"{os.getpid()} worker_id is {worker_id}")
    # 创建数据库客户端
    mysql_client = KnowledgeBaseManager()
    # 创建向量数据库客户端
    milvus_kb = VectorStoreMilvusClient()
    # 创建es客户端
    es_client = StoreElasticSearchClient()
    # 由上述客户端创建检索对象
    retriever = ParentRetriever(milvus_kb, mysql_client, es_client)
    while True:
        sleep_time = 3
        # worker_id 根据时间变化，每x分钟变一次，获取当前时间的分钟数
        # INSERT_WORKERS默认值为4，在启动脚本entrypoint.sh中的值为1，time.strftime("%M", time.localtime())为当前时间的分钟数
        # 为什么要拿当前时间的分钟数除以INSERT_WORKERS【默认值为4】取整呢？也许仅是一种依据当前时间【分钟数】来动态调整worker_id的规则吧
        minutes = int(int(time.strftime("%M", time.localtime())) / INSERT_WORKERS)
        dynamic_worker_id = (worker_id + minutes) % INSERT_WORKERS
        id = None
        try:
            # with语句：允许用户自定义类来定义运行时上下文，在语句体被执行前进入该上下文，并在语句执行完毕时退出该上下文
            # with语句中的对象对应的类需要实现__enter__【进入运行时执行】和__exit__【退出运行时执行】方法
            async with pool.acquire() as conn:  # 获取连接
                # 游标相当于一次数据库会话
                async with conn.cursor() as cur:  # 创建游标
                    query = f"""
                        SELECT id, timestamp, file_id, file_name FROM File
                        WHERE status = 'gray' AND MOD(id, %s) = %s AND deleted = 0
                        ORDER BY timestamp ASC LIMIT 1;
                    """
                    # 查询逻辑未删除文件状态为gray【注意在上传文件接口，保存文件信息时，该字段的默认值为gray】且满足worker_id筛选规则的文件记录
                    # 此处查询出gray状态的文件，可以推测：上传文件接口，仅是做文件在服务器存储和保存文件信息的逻辑【这样做也许是为了提升接口响应速度】，真正的文件处理在当前操作
                    await cur.execute(query, (INSERT_WORKERS, dynamic_worker_id))
                    # 获取上述查询结果中的第一条【一维元组形式，形如：(id, timestamp, file_id, file_name)】【与之对应的有fetchall方法，返回二维元组，形如：((id, timestamp, file_id, file_name),(id, timestamp, file_id, file_name),...,(id, timestamp, file_id, file_name))】
                    file_to_update = await cur.fetchone()
                    if file_to_update:
                        # 如果文件信息非空
                        # 打印查询的文件信息到日志
                        insert_logger.info(f"{worker_id}, file_to_update: {file_to_update}")
                        # 把files_to_update按照timestamp排序, 获取时间最早的那条记录的id
                        # file_to_update = sorted(files_to_update, key=lambda x: x[1])[0]
                        # 获取文件信息元组中的各字段
                        id, timestamp, file_id, file_name = file_to_update
                        """
                        先更新文件状态，后处理文件数据
                        """
                        # 更新这条文件信息记录的状态
                        await cur.execute("""
                            UPDATE File SET status='yellow'
                            WHERE id=%s;
                        """, (id,))
                        await conn.commit()
                        # 打印文件更新日志【记录文件状态为yellow，标识已经要处理该文件】
                        insert_logger.info(f"UPDATE FILE: {timestamp}, {file_id}, {file_name}, yellow")
                        # 依据id精准查询文件信息
                        await cur.execute(
                            "SELECT id, file_id, user_id, file_name, kb_id, file_location, file_size, file_url, "
                            "chunk_size FROM File WHERE id=%s", (id,))
                        file_info = await cur.fetchone()

                        time_record = {}
                        # 进行文件处理【切片，保存至向量数据库】，获取处理状态、切片后所有文档的内容长度之和、切片数量、处理信息
                        status, content_length, chunks_number, msg = await process_data(retriever, milvus_kb,
                                                                                        mysql_client,
                                                                                        file_info, time_record)
                        # 打印时间记录
                        insert_logger.info('time_record: ' + json.dumps(time_record, ensure_ascii=False))
                        # 更新文件处理后的状态和相关信息
                        await cur.execute(
                            "UPDATE File SET status=%s, content_length=%s, chunks_number=%s, msg=%s WHERE id=%s",
                            (status, content_length, chunks_number, msg, file_info[0]))
                        await conn.commit()
                        # 打印文件更新日志【记录处理结果】
                        insert_logger.info(f"UPDATE FILE: {timestamp}, {file_id}, {file_name}, {status}")
                        sleep_time = 0.1
                    else:
                        # 如果文件信息为空，说明当前没有要处理的文件，直接提交数据库连接
                        await conn.commit()
        except Exception as e:
            # 打印连接异常日志
            insert_logger.error('MySQL或Milvus 连接异常：' + str(e))
            try:
                async with pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        # 打印文件处理错误日志
                        insert_logger.error(f"process_files Error {traceback.format_exc()}")
                        # 如果file的status是yellow，就改为red
                        if id is not None:
                            await cur.execute("UPDATE File SET status='red' WHERE id=%s AND status='yellow'", (id,))
                            await conn.commit()
                            # 查询指定id的文件信息
                            await cur.execute(
                                "SELECT id, file_id, user_id, file_name, kb_id, file_location, file_size FROM File WHERE id=%s",
                                (id,))
                            file_info = await cur.fetchone()
                            # 打印文件状态由黄到红的日志信息
                            insert_logger.info(f"UPDATE FILE: {timestamp}, {file_id}, {file_name}, yellow2red")
                            _, file_id, user_id, file_name, kb_id, file_location, file_size = file_info
                            # await post_data(user_id=user_id, charsize=-1, docid=file_id, status='red', msg="Milvus service exception")
            except Exception as e:
                # 如果此时仍发生异常，则打印MySQL二次连接异常日志
                insert_logger.error('MySQL 二次连接异常：' + str(e))
        finally:
            # 睡3秒，继续处理下一个
            await asyncio.sleep(sleep_time)


@app.listener('after_server_stop')
async def close_db(app, loop):
    # 关闭数据库连接池
    app.ctx.pool.close()
    await app.ctx.pool.wait_closed()


@app.listener('before_server_start')
async def setup_workers(app, loop):
    # 创建数据库连接池
    app.ctx.pool = await aiomysql.create_pool(**db_config, minsize=1, maxsize=16, loop=loop, autocommit=False,
                                              init_command='SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED')  # 更改事务隔离级别
    # 将数据库文档处理异步任务，在insert_files_server启动前，加入到sanic应用任务中
    app.add_task(check_and_process(app.ctx.pool))


# 启动服务
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, workers=INSERT_WORKERS, access_log=False)
