import asyncio
import aiohttp
from typing import List
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.utils.general_utils import get_time_async
from qanything_kernel.configs.model_config import LOCAL_RERANK_SERVICE_URL, LOCAL_RERANK_BATCH
from langchain.schema import Document
import traceback


class YouDaoRerank:
    def __init__(self):
        self.url = f"http://{LOCAL_RERANK_SERVICE_URL}/rerank"

    async def _get_rerank_res(self, query, passages):
        """
        调用本地rerank服务，依据问题和文档内容列表进行rerank，返回rerank结果
        """
        # 构造问题和文档内容列表字典，作为rerank接口入参
        data = {
            'query': query,
            'passages': passages
        }
        headers = {"content-type": "application/json"}
        try:
            async with aiohttp.ClientSession() as session:
                # 调用rerank服务进行rerank，获取响应
                async with session.post(self.url, json=data, headers=headers) as response:
                    if response.status == 200:
                        scores = await response.json()
                        return scores
                    else:
                        debug_logger.error(f'Rerank request failed with status {response.status}')
                        return None
        except Exception as e:
            debug_logger.info(f'rerank query: {query}, rerank passages length: {len(passages)}')
            debug_logger.error(f'rerank error: {traceback.format_exc()}')
            return None

    @get_time_async
    async def arerank_documents(self, query: str, source_documents: List[Document]) -> List[Document]:
        """
        Embed search docs using async calls, maintaining the original order.
        依据重构后的问题，对源文档列表进行rerank处理【调用本地的rerank服务】，并依据rerank结果【得分】从大到小对源文档列表进行排序
        """
        # 获取rerank批处理大小，目前是1
        batch_size = LOCAL_RERANK_BATCH  # 增大客户端批处理大小
        # 初始化源文档列表中每个文档的得分为None
        all_scores = [None for _ in range(len(source_documents))]
        # 获取源文档列表的文档内容列表
        passages = [doc.page_content for doc in source_documents]

        tasks = []
        # range(s, e, step)等差数列，当step为1时，等价于range(s, e)
        # 对文档内容列表进行批处理
        for i in range(0, len(passages), batch_size):
            # a[s:e]为列表切片，提取s,s+1,……,e-1的元素
            # 从i开始【包括i】的batch_size个文档内容做rerank处理【调用本地的rerank服务，以异步方式获取响应结果】，生成异步调用任务
            task = asyncio.create_task(self._get_rerank_res(query, passages[i:i + batch_size]))
            # 将rerank异步任务和批处理批次索引以元组形式收集到任务列表中
            tasks.append((i, task))
        # 遍历任务列表
        for start_index, task in tasks:
            # 获取start_index批次的异步任务【rerank处理】结果
            res = await task
            if res is None:
                # 一旦某一批次的rerank处理结果为None时，直接返回源文档列表
                return source_documents
            # 给对应批次的文档赋值得分，由此可见，rerank是计算文档对于问题的得分情况
            all_scores[start_index:start_index + batch_size] = res
        # 遍历得分列表
        for idx, score in enumerate(all_scores):
            # 为相应文档设置得分元数据score
            source_documents[idx].metadata['score'] = score
        # 按照文档得分从大到小对源文档列表进行排序
        source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)
        # 返回源文档列表
        return source_documents


# 使用示例
# async def main():
#     reranker = YouDaoRerank()
#     query = "Your query here"
#     documents = [Document(page_content="content1"), Document(page_content="content2")]  # 示例文档
#     reranked_docs = await reranker.rerank_documents(query, documents)
#     return reranked_docs
#
#
# # 运行异步主函数
# if __name__ == "__main__":
#     reranked_docs = asyncio.run(main())
