from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, VECTOR_SEARCH_SCORE_THRESHOLD, \
    PROMPT_TEMPLATE, STREAMING, SYSTEM, INSTRUCTIONS, SIMPLE_PROMPT_TEMPLATE, CUSTOM_PROMPT_TEMPLATE, \
    LOCAL_RERANK_MODEL_NAME, LOCAL_EMBED_MAX_LENGTH
from typing import List, Tuple, Union, Dict
import time
from scipy.spatial import cKDTree
from scipy.spatial.distance import cosine
from scipy.stats import gmean
from qanything_kernel.connector.embedding.embedding_for_online_client import YouDaoEmbeddings
from qanything_kernel.connector.rerank.rerank_for_online_client import YouDaoRerank
from qanything_kernel.connector.llm import OpenAILLM
from langchain.schema import Document
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.core.retriever.vectorstore import VectorStoreMilvusClient
from qanything_kernel.core.retriever.elasticsearchstore import StoreElasticSearchClient
from qanything_kernel.core.retriever.parent_retriever import ParentRetriever
from qanything_kernel.utils.general_utils import (get_time, clear_string, get_time_async, num_tokens,
                                                  cosine_similarity, clear_string_is_equal, num_tokens_embed,
                                                  num_tokens_rerank, deduplicate_documents, replace_image_references)
from qanything_kernel.utils.custom_log import debug_logger, qa_logger, rerank_logger
from qanything_kernel.core.chains.condense_q_chain import RewriteQuestionChain
from qanything_kernel.core.tools.web_search_tool import duckduckgo_search
import copy
import requests
import json
import numpy as np
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import traceback
import re


# 问答系统的主体类
class LocalDocQA:
    def __init__(self, port):
        self.port = port
        self.milvus_cache = None
        self.embeddings: YouDaoEmbeddings = None
        self.rerank: YouDaoRerank = None
        self.chunk_conent: bool = True
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
        self.milvus_kb: VectorStoreMilvusClient = None
        self.retriever: ParentRetriever = None
        self.milvus_summary: KnowledgeBaseManager = None
        self.es_client: StoreElasticSearchClient = None
        self.session = self.create_retry_session(retries=3, backoff_factor=1)
        self.doc_splitter = CharacterTextSplitter(
            chunk_size=LOCAL_EMBED_MAX_LENGTH / 2,
            chunk_overlap=0,
            length_function=len
        )

    @staticmethod
    def create_retry_session(retries, backoff_factor):
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def init_cfg(self, args=None):
        self.embeddings = YouDaoEmbeddings()
        self.rerank = YouDaoRerank()
        self.milvus_summary = KnowledgeBaseManager()
        self.milvus_kb = VectorStoreMilvusClient()
        self.es_client = StoreElasticSearchClient()
        self.retriever = ParentRetriever(self.milvus_kb, self.milvus_summary, self.es_client)

    @get_time
    def get_web_search(self, queries, top_k):
        query = queries[0]
        web_content, web_documents = duckduckgo_search(query, top_k)
        source_documents = []
        for idx, doc in enumerate(web_documents):
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', doc.metadata['title'])
            doc.metadata['file_name'] = file_name + '.web'
            doc.metadata['file_url'] = doc.metadata['source']
            doc.metadata['embed_version'] = self.embeddings.embed_version
            doc.metadata['score'] = 1 - (idx / len(web_documents))
            doc.metadata['file_id'] = 'websearch' + str(idx)
            doc.metadata['headers'] = {"新闻标题": file_name}
            source_documents.append(doc)
            if 'description' in doc.metadata:
                desc_doc = Document(page_content=doc.metadata['description'], metadata=doc.metadata)
                source_documents.append(desc_doc)
        return web_content, source_documents

    def web_page_search(self, query, top_k=None):
        # 防止get_web_search调用失败，需要try catch
        try:
            web_content, source_documents = self.get_web_search([query], top_k)
        except Exception as e:
            debug_logger.error(f"web search error: {traceback.format_exc()}")
            return []

        return source_documents

    @get_time_async
    async def get_source_documents(self, query, retriever: ParentRetriever, kb_ids, time_record, hybrid_search, top_k):
        """
        获取源文档列表
        kb_ids：知识库id列表、
        query【检索答案用到的问题，最终问题】、
        time_record【内含：压缩【重构】问题使用的token数量、完整提示词和压缩【重构】问题一共使用的token数量】、
        retriever【ParentRetriever(self.milvus_kb, self.milvus_summary, self.es_client)，在init_cfg方法中有定义】
        hybrid_search：是否联网搜索
        top_k：值越大，结果越呈现多样性
        """
        source_documents = []
        start_time = time.perf_counter()
        query_docs = await retriever.get_retrieved_documents(query, partition_keys=kb_ids, time_record=time_record,
                                                             hybrid_search=hybrid_search, top_k=top_k)
        if len(query_docs) == 0:
            debug_logger.warning("MILVUS SEARCH ERROR, RESTARTING MILVUS CLIENT!")
            retriever.vectorstore_client = VectorStoreMilvusClient()
            debug_logger.warning("MILVUS CLIENT RESTARTED!")
            query_docs = await retriever.get_retrieved_documents(query, partition_keys=kb_ids, time_record=time_record,
                                                                    hybrid_search=hybrid_search, top_k=top_k)
        end_time = time.perf_counter()
        time_record['retriever_search'] = round(end_time - start_time, 2)
        debug_logger.info(f"retriever_search time: {time_record['retriever_search']}s")
        # debug_logger.info(f"query_docs num: {len(query_docs)}, query_docs: {query_docs}")
        for idx, doc in enumerate(query_docs):
            if retriever.mysql_client.is_deleted_file(doc.metadata['file_id']):
                debug_logger.warning(f"file_id: {doc.metadata['file_id']} is deleted")
                continue
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            doc.metadata['embed_version'] = self.embeddings.embed_version
            if 'score' not in doc.metadata:
                doc.metadata['score'] = 1 - (idx / len(query_docs))  # TODO 这个score怎么获取呢
            source_documents.append(doc)
        # if cosine_thresh:
        #     source_documents = [item for item in source_documents if float(item.metadata['score']) > cosine_thresh]

        return source_documents

    def reprocess_source_documents(self, custom_llm: OpenAILLM, query: str,
                                   source_docs: List[Document],
                                   history: List[str],
                                   prompt_template: str) -> Tuple[List[Document], int]:
        """
        处理源文档列表，依据大模型所支持的上下文token数量，减去当前问题、对话历史、提示词、源文档索引消息等的token数量，得出限制的token数量【允许源文档header和内容的token数量最大值】
        在不超过限制的token数量的前提下，将源文档列表中文档加入到新源文档列表中，得出新的源文档列表
        """
        # 组装prompt,根据max_token
        # 获取当前问题的token数量的4倍
        query_token_num = custom_llm.num_tokens_from_messages([query]) * 4
        # 遍历对话历史列表，获取对话历史的token数量
        history_token_num = custom_llm.num_tokens_from_messages([x for sublist in history for x in sublist])
        # 获取提示词的token数量
        template_token_num = custom_llm.num_tokens_from_messages([prompt_template])
        # 依据源文档列表长度，遍历其索引0、1、2、……、len-1，构造消息，获取这些消息的token数量
        reference_field_token_num = custom_llm.num_tokens_from_messages(
            [f"<reference>[{idx + 1}]</reference>" for idx in range(len(source_docs))])
        # 依据大模型的token数量属性，获取【api上下文长度-最大token数量-50-当前问题token数量的4倍-对话历史token数量-提示词token数量-文档索引消息的token数量】限制的token数量
        limited_token_nums = custom_llm.token_window - custom_llm.max_token - custom_llm.offcut_token - query_token_num - history_token_num - template_token_num - reference_field_token_num
        # 日志打印上述token数量数据
        debug_logger.info(f"=============================================")
        debug_logger.info(f"token_window = {custom_llm.token_window}")
        debug_logger.info(f"max_token = {custom_llm.max_token}")
        debug_logger.info(f"offcut_token = {custom_llm.offcut_token}")
        debug_logger.info(f"limited token nums: {limited_token_nums}")
        debug_logger.info(f"template token nums: {template_token_num}")
        debug_logger.info(f"reference_field token nums: {reference_field_token_num}")
        debug_logger.info(f"query token nums: {query_token_num}")
        debug_logger.info(f"history token nums: {history_token_num}")
        debug_logger.info(f"=============================================")

        # if limited_token_nums < 200:
        #     return []
        # 从最后一个往前删除，直到长度合适,这样是最优的，因为超长度的情况比较少见
        # 已知箱子容量，装满这个箱子
        new_source_docs = []
        total_token_num = 0
        # 去重后的文件索引列表
        not_repeated_file_ids = []
        # 遍历源文档列表
        for doc in source_docs:
            headers_token_num = 0
            # 获取文档的文件索引
            file_id = doc.metadata['file_id']
            # 依据文件索引是否在not_repeated_file_ids中对文档列表中的文件进行去重
            if file_id not in not_repeated_file_ids:
                # 如果文件索引不在not_repeated_file_ids，则将其追加
                not_repeated_file_ids.append(file_id)
                if 'headers' in doc.metadata:
                    # 对于文档元数据中有headers字段时，取出headers数据，构造headers变量
                    headers = f"headers={doc.metadata['headers']}"
                    # 获取headers的token数量
                    headers_token_num = custom_llm.num_tokens_from_messages([headers])
            # 对文档内容进行正则匹配过滤，将图片用空字符串替换掉，得到文档有效内容
            doc_valid_content = re.sub(r'!\[figure]\(.*?\)', '', doc.page_content)
            # 获取文档有效内容的token数量
            doc_token_num = custom_llm.num_tokens_from_messages([doc_valid_content])
            # 将headers的token数量累加到文档的token数量上
            doc_token_num += headers_token_num
            if total_token_num + doc_token_num <= limited_token_nums:
                # 如果当前文档的token数量和总token数量的和不超过限制的token数量，则将文档追加到新的源文档列表中，并更新【将当前文档的token数量累加到总token数量上】总token数量
                new_source_docs.append(doc)
                total_token_num += doc_token_num
            else:
                # 如果已超过限制的token数量，则跳出循环
                break
        # 打印新的源文档列表的token数量
        debug_logger.info(f"new_source_docs token nums: {custom_llm.num_tokens_from_docs(new_source_docs)}")
        # 返回新的源文档列表和限制的token数量
        return new_source_docs, limited_token_nums

    def generate_prompt(self, query, source_docs, prompt_template):
        if source_docs:
            context = ''
            not_repeated_file_ids = []
            for doc in source_docs:
                doc_valid_content = re.sub(r'!\[figure]\(.*?\)', '', doc.page_content)  # 生成prompt时去掉图片
                file_id = doc.metadata['file_id']
                if file_id not in not_repeated_file_ids:
                    if len(not_repeated_file_ids) != 0:
                        context += '</reference>\n'
                    not_repeated_file_ids.append(file_id)
                    if 'headers' in doc.metadata:
                        headers = f"headers={doc.metadata['headers']}"
                        context += f"<reference {headers}>[{len(not_repeated_file_ids)}]" + '\n' + doc_valid_content + '\n'
                    else:
                        context += f"<reference>[{len(not_repeated_file_ids)}]" + '\n' + doc_valid_content + '\n'
                else:
                    context += doc_valid_content + '\n'
            context += '</reference>\n'

            # prompt = prompt_template.format(context=context).replace("{{question}}", query)
            prompt = prompt_template.replace("{{context}}", context).replace("{{question}}", query)
        else:
            prompt = prompt_template.replace("{{question}}", query)
        return prompt

    async def get_rerank_results(self, query, doc_ids=None, doc_strs=None):
        docs = []
        if doc_strs:
            docs = [Document(page_content=doc_str) for doc_str in doc_strs]
        else:
            for doc_id in doc_ids:
                doc_json = self.milvus_summary.get_document_by_doc_id(doc_id)
                if doc_json is None:
                    docs.append(None)
                    continue
                user_id, file_id, file_name, kb_id = doc_json['kwargs']['metadata']['user_id'], \
                    doc_json['kwargs']['metadata']['file_id'], doc_json['kwargs']['metadata']['file_name'], \
                    doc_json['kwargs']['metadata']['kb_id']
                doc = Document(page_content=doc_json['kwargs']['page_content'], metadata=doc_json['kwargs']['metadata'])
                doc.metadata['doc_id'] = doc_id
                doc.metadata['retrieval_query'] = query
                doc.metadata['embed_version'] = self.embeddings.embed_version
                if file_name.endswith('.faq'):
                    faq_dict = doc.metadata['faq_dict']
                    page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
                    nos_keys = faq_dict.get('nos_keys')
                    doc.page_content = page_content
                    doc.metadata['nos_keys'] = nos_keys
                docs.append(doc)

        if len(docs) > 1 and num_tokens_rerank(query) <= 300:
            try:
                debug_logger.info(f"use rerank, rerank docs num: {len(docs)}")
                docs = await self.rerank.arerank_documents(query, docs)
                if len(docs) > 1:
                    docs = [doc for doc in docs if float(doc.metadata['score']) >= 0.28]
                return docs
            except Exception as e:
                debug_logger.error(f"query tokens: {num_tokens_rerank(query)}, rerank error: {e}")
                embed1 = await self.embeddings.aembed_query(query)
                for doc in docs:
                    embed2 = await self.embeddings.aembed_query(doc.page_content)
                    doc.metadata['score'] = cosine_similarity(embed1, embed2)
                return docs
        else:
            embed1 = await self.embeddings.aembed_query(query)
            for doc in docs:
                embed2 = await self.embeddings.aembed_query(doc.page_content)
                doc.metadata['score'] = cosine_similarity(embed1, embed2)
            return docs

    async def prepare_source_documents(self, query: str, custom_llm: OpenAILLM, source_documents: List[Document],
                                       chat_history: List[str], prompt_template: str,
                                       need_web_search: bool = False):
        # 删除文档中的图片
        # for doc in source_documents:
        #     doc.page_content = re.sub(r'!\[figure]\(.*?\)', '', doc.page_content)
        # 依据openAI大模型、当前问题、源文档列表、对话历史、提示词，处理源文档列表，得到新的源文档列表和限制的token数量【大模型所支持的上下文剩余token数量】
        retrieval_documents, limited_token_nums = self.reprocess_source_documents(custom_llm=custom_llm, query=query,
                                                                                  source_docs=source_documents,
                                                                                  history=chat_history,
                                                                                  prompt_template=prompt_template)
        debug_logger.info(f"retrieval_documents len: {len(retrieval_documents)}")
        if not need_web_search:
            # 如果没有开启联网搜索
            try:
                # 依据文件id，进行文档聚合【排序，压缩创建新文档，满足限制token数据要求】
                # 为什么源文档列表至多有两个文件id时才会聚合，否则此结果得到的是空列表
                new_docs = self.aggregate_documents(retrieval_documents, limited_token_nums, custom_llm)
                if new_docs:
                    source_documents = new_docs
                else:
                    # 合并所有候选文档，从前往后，所有file_id相同的文档合并，按照doc_id排序
                    merged_documents_file_ids = []
                    for doc in retrieval_documents:
                        if doc.metadata['file_id'] not in merged_documents_file_ids:
                            merged_documents_file_ids.append(doc.metadata['file_id'])
                    source_documents = []
                    for file_id in merged_documents_file_ids:
                        docs = [doc for doc in retrieval_documents if doc.metadata['file_id'] == file_id]
                        docs = sorted(docs, key=lambda x: int(x.metadata['doc_id'].split('_')[-1]))
                        source_documents.extend(docs)

                # source_documents = self.incomplete_table(source_documents, limited_token_nums, custom_llm)
            except Exception as e:
                debug_logger.error(f"aggregate_documents error w/ {e}: {traceback.format_exc()}")
                source_documents = retrieval_documents
        else:
            source_documents = retrieval_documents

        debug_logger.info(f"source_documents len: {len(source_documents)}")
        return source_documents, retrieval_documents

    async def calculate_relevance_optimized(
            self,
            question: str,
            llm_answer: str,
            reference_docs: List[Document],
            top_k: int = 5
    ) -> List[Dict]:
        # 获取问题的scores
        question_scores = [doc.metadata['score'] for doc in reference_docs]
        # 计算问题和LLM回答的embedding
        # question_embedding = await self.embeddings.aembed_query(question)
        llm_answer_embedding = await self.embeddings.aembed_query(llm_answer)

        # 计算所有引用文档分段的embeddings
        all_segments_docs = self.doc_splitter.split_documents(reference_docs)
        all_segments = [doc.page_content for doc in all_segments_docs]
        reference_embeddings = await self.embeddings.aembed_documents(all_segments)

        # 将嵌入向量转换为numpy数组以便使用scipy的cosine函数
        # question_embedding = np.array(question_embedding)
        llm_answer_embedding = np.array(llm_answer_embedding)
        reference_embeddings = np.array(reference_embeddings)

        # 构建KD树
        tree = cKDTree(reference_embeddings)

        # 使用KD树找到最相似的分段
        _, indices = tree.query(llm_answer_embedding.reshape(1, -1), k=top_k)
        if isinstance(indices[0], np.int64):
            indices = [indices]

        def weighted_geometric_mean(scores, weights):
            return gmean([score ** weight for score, weight in zip(scores, weights)])

        # 计算相似度和综合得分
        relevant_docs = []
        for doc_index in indices[0]:
            doc_id = doc_index // len(self.doc_splitter.split_documents([reference_docs[0]]))

            # 使用1 - cosine距离来计算相似度
            # similarity_llm = 1 - cosine(llm_answer_embedding, reference_embeddings[doc_index])
            # similarity_question = 1 - cosine(question_embedding, reference_embeddings[doc_index])
            # 综合得分：结合LLM回答相似度、问题相似度和原始问题得分
            # combined_score = (similarity_llm + similarity_question + question_scores[doc_id]) / 3

            similarity_llm = 1 - cosine(llm_answer_embedding, reference_embeddings[doc_index])
            rerank_score = question_scores[doc_id]

            # 设置rerank分数和LLM回答与文档余弦相似度的权重
            weights = [0.5, 0.5]  # 分别对应similarity_llm和rerank_score
            combined_score = weighted_geometric_mean([similarity_llm, rerank_score], weights)

            relevant_docs.append({
                'document': reference_docs[doc_id],
                'segment': all_segments_docs[doc_index],
                'similarity_llm': float(similarity_llm),
                'question_score': question_scores[doc_id],
                'combined_score': float(combined_score)
            })

        # 按综合得分降序排序
        relevant_docs.sort(key=lambda x: x['combined_score'], reverse=True)

        return relevant_docs

    async def get_knowledge_based_answer(self, model, max_token, kb_ids, query, retriever, custom_prompt, time_record,
                                         temperature, api_base, api_key, api_context_length, top_p, top_k, web_chunk_size,
                                         chat_history=None, streaming: bool = STREAMING, rerank: bool = False,
                                         only_need_search_results: bool = False, need_web_search=False,
                                         hybrid_search=False):
        """
        获取知识库答案
        流程：
            1、获取openai大模型对象【内含指定模型的编码器】
            2、利用历史对话记录列表和当前问题进行重构，生成独立且语义完整的新问题retrieval_query
            3、依据知识库id列表、retrieval_query【检索答案用到的问题，最终问题】、time_record【内含：压缩【重构】问题使用的token数量、完整提示词和压缩【重构】问题一共使用的token数量】、retriever【ParentRetriever(self.milvus_kb, self.milvus_summary, self.es_client)，在init_cfg方法中有定义】以及一些超参数获取源文档列表
            4、判断是否开启联网搜索，若开启，则依据当前问题进行联网搜索，并将搜索到的文档追加到源文档列表中
            5、对源文档列表进行去重、重排rerank、标识及评分过滤处理
            6、遍历源文档列表，依据当前问题对源文档进行逐个匹配，匹配成功，则构造返回结果。如果在源文档列表中没有找到答案，则进行后续处理
            未完待续564行
        """
        # 获取openai大模型对象：内含指定模型的编码器、openai请求客户端
        custom_llm = OpenAILLM(model, max_token, api_base, api_key, api_context_length, top_p, temperature)
        if chat_history is None:
            # 如果对话历史参数为空，则设置其默认值为空列表
            chat_history = []
        # 将问题赋值给retrieval_query【检索查询】
        retrieval_query = query
        # 同时将问题赋值给condense_question【压缩【重构】问题】
        condense_question = query
        if chat_history:
            # 如果对话历史非空
            # 设置一个格式化的对话历史，初始化为空列表
            formatted_chat_history = []
            # 遍历对话历史列表中的对话记录，以“人工问题，ai答案”的顺序格式化收集到列表formatted_chat_history
            for msg in chat_history:
                # 将对话历史中的对话记录按照下述格式收集到格式化对话历史列表中
                formatted_chat_history += [
                    # 记录人类问题
                    HumanMessage(content=msg[0]),
                    # 记录ai答案
                    AIMessage(content=msg[1]),
                ]
            debug_logger.info(f"formatted_chat_history: {formatted_chat_history}")
            # 获取重构问题链对象【用以根据历史对话记录和当前问题，生成一个可以独立理解【不需要依赖聊天记录就可以理解的，语义完整的】的新问题】
            """
            如果当前问题已经语义完整且独立【和历史对话记录没有关系】，则直接返回当前问题
            如果当前问题语义不完整，则需要结合历史对话记录，对当前问题进行修正【语种和意思在构造前后一致】，得到可以独立理解的新问题并返回
            """
            rewrite_q_chain = RewriteQuestionChain(model_name=model, openai_api_base=api_base, openai_api_key=api_key)
            # 利用重构问题链的压缩问题提示词，结合格式化的历史对话记录列表和当前问题，构造完整的提示词列表【系统提示词、格式化的历史对话记录、当前问题】
            full_prompt = rewrite_q_chain.condense_q_prompt.format(
                chat_history=formatted_chat_history,
                question=query
            )
            # 利用openAI大语言模型计算完整提示词列表所使用的token数量，当token数量超过3840时，执行下述循环
            while custom_llm.num_tokens_from_messages([full_prompt]) >= 4096 - 256:
                # 从格式化的历史对话记录列表中去除第一个历史对话【距离当前对话在时间维度上最远，相关性可能最低，因此去除】
                formatted_chat_history = formatted_chat_history[2:]
                # 重新构造完整提示词
                full_prompt = rewrite_q_chain.condense_q_prompt.format(
                    chat_history=formatted_chat_history,
                    question=query
                )
            # 打印历史对话记录的长度变更情况
            debug_logger.info(
                f"Subtract formatted_chat_history: {len(chat_history) * 2} -> {len(formatted_chat_history)}")
            try:
                t1 = time.perf_counter()
                # 异步重构当前问题，并将新问题赋值给condense_question
                condense_question = await rewrite_q_chain.condense_q_chain.ainvoke(
                    {
                        "chat_history": formatted_chat_history,
                        "question": query,
                    },
                )
                t2 = time.perf_counter()
                # 记录压缩【重构】问题消耗时间
                # 时间保留两位小数
                time_record['condense_q_chain'] = round(t2 - t1, 2)
                # 记录重构得到的新问题所使用的token数量
                time_record['rewrite_completion_tokens'] = custom_llm.num_tokens_from_messages([condense_question])
                debug_logger.info(f"condense_q_chain time: {time_record['condense_q_chain']}s")
            except Exception as e:
                # 对于重构问题异常情况，不抛出异常，仅打印重构失败异常日志，且将当前问题【未经过重构处理的原始问题】赋值给condense_question
                debug_logger.error(f"condense_q_chain error: {e}")
                condense_question = query
            # 生成prompt
            # full_prompt = condense_q_prompt.format_messages(
            #     chat_history=formatted_chat_history,
            #     question=query
            # )
            # qa_logger.info(f"condense_q_chain full_prompt: {full_prompt}, condense_question: {condense_question}")
            # 打印经过重构【结合历史对话记录列表和当前问题，利用大模型、提示词进行重构得到独立且语义完整的新问题】的压缩【重构】问题描述
            debug_logger.info(f"condense_question: {condense_question}")
            # 记录完整提示词和压缩【重构】问题一共使用的token数量
            time_record['rewrite_prompt_tokens'] = custom_llm.num_tokens_from_messages([full_prompt, condense_question])
            # 判断两个字符串是否相似：只保留中文，英文和数字
            if clear_string(condense_question) != clear_string(query):
                # 如果压缩【重构】问题和当前问题不相似【说明重构起了作用，也即历史对话记录起了作用】，则将压缩【重构】问题赋值给retrieval_query
                retrieval_query = condense_question

        if kb_ids:
            # 如果知识库id列表非空，则依据知识库id列表、retrieval_query【检索答案用到的问题，最终问题】、time_record【内含：压缩【重构】问题使用的token数量、完整提示词和压缩【重构】问题一共使用的token数量】、retriever【ParentRetriever(self.milvus_kb, self.milvus_summary, self.es_client)，在init_cfg方法中有定义】以及一些超参数获取源文档
            source_documents = await self.get_source_documents(retrieval_query, retriever, kb_ids, time_record,
                                                               hybrid_search, top_k)
        else:
            # 如果知识库id列表为空，则设置源文档为空列表
            source_documents = []

        if need_web_search:
            # 如果开启联网搜索
            t1 = time.perf_counter()
            web_search_results = self.web_page_search(query, top_k=3)
            web_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
                chunk_size=web_chunk_size,
                chunk_overlap=int(web_chunk_size / 4),
                length_function=num_tokens_embed,
            )
            web_search_results = web_splitter.split_documents(web_search_results)
            t2 = time.perf_counter()
            time_record['web_search'] = round(t2 - t1, 2)
            # 将联网搜索结果追加到源文档列表中
            source_documents += web_search_results
        # 对源文档列表进行去重
        source_documents = deduplicate_documents(source_documents)
        if rerank and len(source_documents) > 1 and num_tokens_rerank(query) <= 300:
            # 如果进行重排，且源文档列表非空，且当前问题重排后是使用的token数量不超过300，则进行以下处理
            try:
                t1 = time.perf_counter()
                debug_logger.info(f"use rerank, rerank docs num: {len(source_documents)}")
                # 对源文档列表进行重排处理？
                source_documents = await self.rerank.arerank_documents(condense_question, source_documents)
                t2 = time.perf_counter()
                # 记录源文档列表重排消耗时间
                time_record['rerank'] = round(t2 - t1, 2)
                # 过滤掉低分的文档
                if len(source_documents) > 1:
                    source_documents = [doc for doc in source_documents if float(doc.metadata['score']) >= 0.28]
            except Exception as e:
                # 源文档列表进行重排以及过滤低分的源文档发生异常时，重置重排源文档消耗时间为0
                time_record['rerank'] = 0.0
                # 打印当前问题及知识库重排异常信息
                debug_logger.error(f"query {query}: kb_ids: {kb_ids}, rerank error: {traceback.format_exc()}")

        # rerank之后删除headers，只保留文本内容，用于后续处理
        for doc in source_documents:
            doc.page_content = re.sub(r'^\[headers]\(.*?\)\n', '', doc.page_content)
        # 过滤掉【文件名不以.faq结尾且评分低于0.9的】文档，保留剩余文档列表到high_score_faq_documents
        high_score_faq_documents = [doc for doc in source_documents if
                                    doc.metadata['file_name'].endswith('.faq') and float(doc.metadata['score'] >= 0.9)]
        if high_score_faq_documents:
            # 如果high_score_faq_documents非空，则赋值给源文档列表【换言之，如果high_score_faq_documents为空，则上述过滤像没有发生一样】
            source_documents = high_score_faq_documents
        # FAQ完全匹配处理逻辑
        for doc in source_documents:
            # 对源文档列表中的文档逐个匹配问题
            if doc.metadata['file_name'].endswith('.faq') and clear_string_is_equal(
                    doc.metadata['faq_dict']['question'], query):
                # 如果文档名以.faq结尾且文档中的字典faq_dict中的问题与当前问题一致【说明找到问题的答案了】，则打印匹配到问题的日志，并进行以下处理
                debug_logger.info(f"match faq question: {query}")
                if only_need_search_results:
                    # 如果问答仅需要返回检索结果，则借助yield进行流式迭代返回
                    yield source_documents, None
                    return
                # 获取问题答案
                res = doc.metadata['faq_dict']['answer']
                # 构造新的对话历史列表
                history = chat_history + [[query, res]]
                if streaming:
                    # 如果进行流式回答，则拼接答案和响应
                    res = 'data: ' + json.dumps({'answer': res}, ensure_ascii=False)
                response = {"query": query,
                            "prompt": 'MATCH_FAQ',
                            "result": res,
                            "condense_question": condense_question,
                            "retrieval_documents": source_documents,
                            "source_documents": source_documents}
                # 记录下述时间和token数量信息
                time_record['llm_completed'] = 0.0
                time_record['total_tokens'] = 0
                time_record['prompt_tokens'] = 0
                time_record['completion_tokens'] = 0
                yield response, history
                if streaming:
                    # 流式回答实现
                    response['result'] = "data: [DONE]\n\n"
                    yield response, history
                # 退出函数
                return
        # 在源文档列表中没有找到答案，则进行以下处理
        # es检索+milvus检索结果最多可能是2k
        # 取前30个源文档组成新的源文档列表
        source_documents = source_documents[:top_k]
        # 获取今日日期
        today = time.strftime("%Y-%m-%d", time.localtime())
        # 获取当前时间
        now = time.strftime("%H:%M:%S", time.localtime())

        t1 = time.perf_counter()
        if source_documents:
            # 如果源文档列表非空，则进行下述处理
            if custom_prompt:
                # 如果入参中的提示词非空，则依据custom_prompt替换模板中的相应占位符生成提示词prompt_template
                # escaped_custom_prompt = custom_prompt.replace('{', '{{').replace('}', '}}')
                # prompt_template = CUSTOM_PROMPT_TEMPLATE.format(custom_prompt=escaped_custom_prompt)
                prompt_template = CUSTOM_PROMPT_TEMPLATE.replace("{{custom_prompt}}", custom_prompt)
            else:
                # 如果入参中的提示词为空，则依据当前日期和当前时间替换系统提示词模板中的相应占位符生成提示词
                # system_prompt = SYSTEM.format(today_date=today, current_time=now)
                system_prompt = SYSTEM.replace("{{today_date}}", today).replace("{{current_time}}", now)
                # prompt_template = PROMPT_TEMPLATE.format(system=system_prompt, instructions=INSTRUCTIONS)
                # 将系统提示词内容和说明字符替换到提示词模板中生成提示词prompt_template
                prompt_template = PROMPT_TEMPLATE.replace("{{system}}", system_prompt).replace("{{instructions}}",
                                                                                               INSTRUCTIONS)
        else:
            # 如果源文档列表为空，则进行下述处理
            if custom_prompt:
                # 如果入参中的提示词非空，则依据custom_prompt及当前日期时间替换模板中的相应占位符生成提示词prompt_template
                # escaped_custom_prompt = custom_prompt.replace('{', '{{').replace('}', '}}')
                # prompt_template = SIMPLE_PROMPT_TEMPLATE.format(today=today, now=now, custom_prompt=escaped_custom_prompt)
                prompt_template = SIMPLE_PROMPT_TEMPLATE.replace("{{today}}", today).replace("{{now}}", now).replace(
                    "{{custom_prompt}}", custom_prompt)
            else:
                # 如果入参中的提示词为空，则声明一个simple_custom_prompt，依据simple_custom_prompt、当前日期和当前时间替换系统提示词模板中的相应占位符生成提示词prompt_template
                simple_custom_prompt = """
                - If you cannot answer based on the given information, you will return the sentence \"抱歉，已知的信息不足，因此无法回答。\". 
                """
                # prompt_template = SIMPLE_PROMPT_TEMPLATE.format(today=today, now=now, custom_prompt=simple_custom_prompt)
                prompt_template = SIMPLE_PROMPT_TEMPLATE.replace("{{today}}", today).replace("{{now}}", now).replace(
                    "{{custom_prompt}}", simple_custom_prompt)

        # source_documents_for_show = copy.deepcopy(source_documents)
        # total_images_number = 0
        # for doc in source_documents_for_show:
        #     if 'images' in doc.metadata:
        #         total_images_number += len(doc.metadata['images'])
        #     doc.page_content = replace_image_references(doc.page_content, doc.metadata['file_id'])
        # debug_logger.info(f"total_images_number: {total_images_number}")
        # 依据当前问题、openAI大模型、源文档列表、对话历史、提示词、是否联网检索标识，对源文档列表进行预处理，得到检索文档列表
        source_documents, retrieval_documents = await self.prepare_source_documents(query, custom_llm, source_documents,
                                                                                    chat_history,
                                                                                    prompt_template,
                                                                                    need_web_search)

        total_images_number = 0
        for doc in source_documents:
            if doc.metadata.get('images', []):
                total_images_number += len(doc.metadata['images'])
                doc.page_content = replace_image_references(doc.page_content, doc.metadata['file_id'])
        debug_logger.info(f"total_images_number: {total_images_number}")

        t2 = time.perf_counter()
        time_record['reprocess'] = round(t2 - t1, 2)
        if only_need_search_results:
            yield source_documents, None
            return

        t1 = time.perf_counter()
        has_first_return = False

        acc_resp = ''
        prompt = self.generate_prompt(query=query,
                                      source_docs=source_documents,
                                      prompt_template=prompt_template)
        # debug_logger.info(f"prompt: {prompt}")
        est_prompt_tokens = num_tokens(prompt) + num_tokens(str(chat_history))
        async for answer_result in custom_llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):
            resp = answer_result.llm_output["answer"]
            if 'answer' in resp:
                acc_resp += json.loads(resp[6:])['answer']
            prompt = answer_result.prompt
            history = answer_result.history
            total_tokens = answer_result.total_tokens
            prompt_tokens = answer_result.prompt_tokens
            completion_tokens = answer_result.completion_tokens
            history[-1][0] = query
            response = {"query": query,
                        "prompt": prompt,
                        "result": resp,
                        "condense_question": condense_question,
                        "retrieval_documents": retrieval_documents,
                        "source_documents": source_documents}
            time_record['prompt_tokens'] = prompt_tokens if prompt_tokens != 0 else est_prompt_tokens
            time_record['completion_tokens'] = completion_tokens if completion_tokens != 0 else num_tokens(acc_resp)
            time_record['total_tokens'] = total_tokens if total_tokens != 0 else time_record['prompt_tokens'] + \
                                                                                 time_record['completion_tokens']
            if has_first_return is False:
                first_return_time = time.perf_counter()
                has_first_return = True
                time_record['llm_first_return'] = round(first_return_time - t1, 2)
            if resp[6:].startswith("[DONE]"):
                last_return_time = time.perf_counter()
                time_record['llm_completed'] = round(last_return_time - t1, 2) - time_record['llm_first_return']
                history[-1][1] = acc_resp
                if total_images_number != 0:  # 如果有图片，需要处理回答带图的情况
                    docs_with_images = [doc for doc in source_documents if doc.metadata.get('images', [])]
                    time1 = time.perf_counter()
                    relevant_docs = await self.calculate_relevance_optimized(
                        question=query,
                        llm_answer=acc_resp,
                        reference_docs=docs_with_images,
                        top_k=1
                    )
                    show_images = ["\n### 引用图文如下：\n"]
                    for doc in relevant_docs:
                        print(f"文档: {doc['document']}...")  # 只打印前50个字符
                        print(f"最相关段落: {doc['segment']}...")  # 打印最相关段落的前100个字符
                        print(f"与LLM回答的相似度: {doc['similarity_llm']:.4f}")
                        print(f"原始问题相关性分数: {doc['question_score']:.4f}")
                        print(f"综合得分: {doc['combined_score']:.4f}")
                        print()
                        for image in doc['document'].metadata.get('images', []):
                            image_str = replace_image_references(image, doc['document'].metadata['file_id'])
                            debug_logger.info(f"image_str: {image} -> {image_str}")
                            show_images.append(image_str + '\n')
                    debug_logger.info(f"show_images: {show_images}")
                    time_record['obtain_images'] = round(time.perf_counter() - last_return_time, 2)
                    time2 = time.perf_counter()
                    debug_logger.info(f"obtain_images time: {time2 - time1}s")
                    time_record["obtain_images_time"] = round(time2 - time1, 2)
                    if len(show_images) > 1:
                        response['show_images'] = show_images
            yield response, history

    def get_completed_document(self, file_id, limit=None):
        """
        依据文件id获取经排序【按照文档id从小到大排序】处理的文档json_data字段，分别得到两个完整文档，一个内容含有图片，一个内容不含图片
        """
        sorted_json_datas = self.milvus_summary.get_document_by_file_id(file_id)
        if limit:
            # 如果有文档限制索引，则对sorted_json_datas进行相应索引提取
            sorted_json_datas = sorted_json_datas[limit[0]: limit[1] + 1]

        completed_content_with_figure = ''
        completed_content = ''
        for doc_json in sorted_json_datas:
            # 利用json_data构造文档对象
            doc = Document(page_content=doc_json['kwargs']['page_content'], metadata=doc_json['kwargs']['metadata'])
            # rerank之后删除headers，只保留文本内容，用于后续处理
            # 利用正则过滤headers内容
            doc.page_content = re.sub(r'^\[headers]\(.*?\)\n', '', doc.page_content)
            # if filter_figures:
            #     doc.page_content = re.sub(r'!\[figure]\(.*?\)', '', doc.page_content)  # 删除图片
            if doc_json['kwargs']['metadata']['file_name'].endswith('.faq'):
                # json_data中的文件名以.faq结尾时
                # 获取faq_dict数据
                faq_dict = doc_json['kwargs']['metadata']['faq_dict']
                # 以问答形式构造文档内容
                doc.page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
            # 追加文档内容【结果包含图片】
            completed_content_with_figure += doc.page_content + '\n\n'
            # 删除文档内容中的图片
            completed_content += re.sub(r'!\[figure]\(.*?\)', '', doc.page_content) + '\n\n' # 删除图片
        # 构造含有图片的文件对象
        completed_doc_with_figure = Document(page_content=completed_content_with_figure, metadata=sorted_json_datas[0]['kwargs']['metadata'])
        # 构造不含图片的文件对象
        completed_doc = Document(page_content=completed_content, metadata=sorted_json_datas[0]['kwargs']['metadata'])
        # FIX metadata
        has_table = False
        images = []
        # 遍历处理json_data
        for doc_json in sorted_json_datas:
            if doc_json['kwargs']['metadata'].get('has_table'):
                # 当json_data中出现表格标识时，跳出循环
                has_table = True
                break
            if doc_json['kwargs']['metadata'].get('images'):
                # 当json_data中含有图片列表时，将图片列表追加到images中
                images.extend(doc_json['kwargs']['metadata']['images'])
        # 对于不含图片的文档对象/含有图片的文档对象，针对其元数据，进行以下处理
        # 设置表格标识
        completed_doc.metadata['has_table'] = has_table
        # 设置图片列表
        completed_doc.metadata['images'] = images
        # 设置表格标识
        completed_doc_with_figure.metadata['has_table'] = has_table
        # 设置图片列表
        completed_doc_with_figure.metadata['images'] = images

        # completed_content = ''
        # for doc_json in sorted_json_datas:
        #     doc = Document(page_content=doc_json['kwargs']['page_content'], metadata=doc_json['kwargs']['metadata'])
        #     # rerank之后删除headers，只保留文本内容，用于后续处理
        #     doc.page_content = re.sub(r'^\[headers]\(.*?\)\n', '', doc.page_content)
        #     if filter_figures:
        #         doc.page_content = re.sub(r'!\[figure]\(.*?\)', '', doc.page_content)  # 删除图片
        #     completed_content += doc.page_content + '\n\n'
        # completed_doc = Document(page_content=completed_content, metadata=sorted_json_datas[0]['kwargs']['metadata'])
        return completed_doc, completed_doc_with_figure

    def aggregate_documents(self, source_documents, limited_token_nums, custom_llm):
        """
        对于源文档列表进行聚合处理，得到新的源文档列表
        收集source_documents中的2个文件id的信息：文件id，文档id列表，文档列表
        1、如果source_documents有超过2个文件id，则返回空列表
        2、对第一个文件id，做处理：依据文件id获取文档信息【文档id，json_data】列表，按照文档id从小到大排序，收集所有json_data，生成完整文档【带图片/不带图片】
        3、计算第一个文件id的不带图片完整文档的token数量与第二个文件列表的token数量之和，不超过限制的token数量limited_token_nums时，将第一个文件id的带图片的完整文档收集到结果中；超过限制的token数量时【如果第一个文件id对应的文档列表只有一个文档，则直接返回空列表】，依据第一个文件id对应的文档id列表，取出文档id的上下界，对该文档id上下界范围内的文档生成完整文档，做同样逻辑判断【token数量之和不超过limited_token_nums，则将此时的带图片完整文档收集到结果中；token数量之和超过limited_token_nums时，返回空列表】
        4、对第二个文件id做类似第2、3步的逻辑，有些不同。计算第二个文件id的不带图片完整文档的token数量与第一个文件id的完整文档的token数量之和，不超过限制的token数量limited_token_nums时，将第二个文件id的带图片的完整文档收集到结果中；超过限制的token数量时【如果第二个文件id对应的文档列表只有一个文档，则将其追加到结果中，并返回结果】，依据第二个文件id对应的文档id列表，取出文档id的上下界，对该文档id上下界范围内的文档生成完整文档，做同样逻辑判断【token数量之和不超过limited_token_nums，则将此时的带图片完整文档收集到结果中；token数量之和超过limited_token_nums时，则将第二个文件id对应的文档列表追加到结果中，并返回结果】
        返回结果【一个或两个文件id对应的完整文档列表】
        """
        # 聚合文档，具体逻辑是帮我判断所有候选是否集中在一个或两个文件中，是的话直接返回这一个或两个完整文档，如果tokens不够则截取文档中的完整上下文
        # 用同样的逻辑，收集两个文档
        first_file_dict = {}
        ori_first_docs = []
        second_file_dict = {}
        ori_second_docs = []
        # 遍历新的源文档列表
        for doc in source_documents:
            # 获取文档元数据中的文件id
            file_id = doc.metadata['file_id']
            if not first_file_dict:
                # first_file_dict为空时
                # 将当前文档的文件索引映射到first_file_dict中
                first_file_dict['file_id'] = file_id
                # 将当前文档的文档索引【doc_id以_分隔的最后一段】映射到first_file_dict中
                first_file_dict['doc_ids'] = [int(doc.metadata['doc_id'].split('_')[-1])]
                # 将当前文档追加到ori_first_docs列表中
                ori_first_docs.append(doc)
                # 遍历源文档列表，找到文档元数据中的文件id和当前文档元数据中的文件id一样的文档元数据得分字段，组成一个列表，取该得分列表的最大值，映射到first_file_dict中
                first_file_dict['score'] = max(
                    [doc.metadata['score'] for doc in source_documents if doc.metadata['file_id'] == file_id])
            elif first_file_dict['file_id'] == file_id:
                # 当前文档元数据中的文件id在first_file_dict中时
                # 将当前文档元数据中的文档索引追加到first_file_dict
                first_file_dict['doc_ids'].append(int(doc.metadata['doc_id'].split('_')[-1]))
                # 将当前文档追加到ori_first_docs列表中
                ori_first_docs.append(doc)
            elif not second_file_dict:
                # second_file_dict为空时
                # 将当前文档元数据中的文件id映射到second_file_dict中
                second_file_dict['file_id'] = file_id
                # 将当前文档的文档索引【doc_id以_分隔的最后一段】映射到second_file_dict中
                second_file_dict['doc_ids'] = [int(doc.metadata['doc_id'].split('_')[-1])]
                # 将当前文档追加到ori_second_docs中
                ori_second_docs.append(doc)
                # 遍历源文档列表，找到文档元数据中的文件id和当前文档元数据中的文件id一样的文档元数据得分字段，组成一个列表，取该得分列表的最大值，映射到second_file_dict中
                second_file_dict['score'] = max(
                    [doc.metadata['score'] for doc in source_documents if doc.metadata['file_id'] == file_id])
            elif second_file_dict['file_id'] == file_id:
                # 当前文档元数据中的文件id在second_file_dict中时
                # 将当前文档元数据中的文档索引追加到second_file_dict
                second_file_dict['doc_ids'].append(int(doc.metadata['doc_id'].split('_')[-1]))
                # 将当前文档追加到ori_second_docs列表中
                ori_second_docs.append(doc)
            else:  # 如果有第三个文件，直接返回
                return []
        # 获取第一个/第二个文档列表中文档内容的token数量
        ori_first_docs_tokens = custom_llm.num_tokens_from_docs(ori_first_docs)
        ori_second_docs_tokens = custom_llm.num_tokens_from_docs(ori_second_docs)

        new_docs = []
        # 依据第一个文件id获取经排序【按照文档id从小到大排序】处理的文档json_data字段，分别得到两个完整文档，一个内容含有图片，一个内容不含图片
        first_completed_doc, first_completed_doc_with_figure = self.get_completed_document(first_file_dict['file_id'])
        # 对不含图片的完整文档设置得分
        first_completed_doc.metadata['score'] = first_file_dict['score']
        # 获取不含图片的完整文档的token数量
        first_doc_tokens = custom_llm.num_tokens_from_docs([first_completed_doc])
        if first_doc_tokens + ori_second_docs_tokens > limited_token_nums:
            # 如果第一个文件id的不含图片的完整文档的token数量与第二个文档列表的token数量之和超过限制token数量
            if len(ori_first_docs) == 1:
                # 如果第一个文档列表只有一个文档，则直接返回new_docs【空列表】
                debug_logger.info(f"first_file_docs number is one")
                return new_docs
            # 获取first_file_dict['doc_ids']的最小值和最大值
            doc_limit = [min(first_file_dict['doc_ids']), max(first_file_dict['doc_ids'])]
            # 依据第一个文件id获取经排序【先根据文档索引限制参数doc_limit对文件id查询文档列表结果进行切片，再对切片后的数据按照文档id从小到大排序】处理的文档json_data字段，分别得到两个完整文档，一个内容含有图片，一个内容不含图片
            first_completed_doc_limit, first_completed_doc_limit_with_figure = self.get_completed_document(
                first_file_dict['file_id'], doc_limit)
            # 对不含图片的完整文档设置得分
            first_completed_doc_limit.metadata['score'] = first_file_dict['score']
            # 获取不含图片的完整文档的token数量
            first_doc_tokens = custom_llm.num_tokens_from_docs([first_completed_doc_limit])
            if first_doc_tokens + ori_second_docs_tokens > limited_token_nums:
                # 如果第一个文件id的不含图片的完整文档【过滤了文档id，只取doc_limit范围内的文档列表】的token数量与第二个文档列表的token数量之和超过限制token数量
                debug_logger.info(
                    f"first_limit_doc_tokens {doc_limit}: {first_doc_tokens} + ori_second_docs_tokens: {ori_second_docs_tokens} > limited_token_nums: {limited_token_nums}")
                # 直接返回new_docs【空列表】
                return new_docs
            else:
                # 如果第一个文件id的不含图片的完整文档【过滤了文档id，只取doc_limit范围内的文档列表】的token数量与第二个文档列表的token数量之和不超过限制token数量
                debug_logger.info(
                    f"first_limit_doc_tokens {doc_limit}: {first_doc_tokens} + ori_second_docs_tokens: {ori_second_docs_tokens} <= limited_token_nums: {limited_token_nums}")
                # 将第一个文件id的含图片的完整文档【过滤了文档id，只取doc_limit范围内的文档列表】追加到new_docs中
                new_docs.append(first_completed_doc_limit_with_figure)
        else:
            # 如果第一个文件id的不含图片的完整文档的token数量与第二个文档列表的token数量之和不超过限制token数量
            debug_logger.info(
                f"first_doc_tokens: {first_doc_tokens} + ori_second_docs_tokens: {ori_second_docs_tokens} <= limited_token_nums: {limited_token_nums}")
            # 将第一个文件id的含图片的完整文档追加到new_docs中
            new_docs.append(first_completed_doc_with_figure)
        if second_file_dict:
            # 第二个文件字典非空时
            # 依据第二个文件id获取经排序【按照文档id从小到大排序】处理的文档json_data字段，分别得到两个完整文档，一个内容含有图片，一个内容不含图片
            second_completed_doc, second_completed_doc_with_figure = self.get_completed_document(second_file_dict['file_id'])
            # 设置得分
            second_completed_doc.metadata['score'] = second_file_dict['score']
            # 获取第二个文件id对应的不含图片的完整文档的token数量
            second_doc_tokens = custom_llm.num_tokens_from_docs([second_completed_doc])
            if first_doc_tokens + second_doc_tokens > limited_token_nums:
                # 第一个文件id对应的完整文档的token数量与第二个文档列表的token数量之和超过限制的token数量
                if len(ori_second_docs) == 1:
                    # 第二个文档列表仅有一个文档时，【前期已经判断过：第二个文档列表的token数量与第一个文件id对应的完整文档的token数量之和不超过限制的token数量】将其追加到new_docs中，并返回new_docs
                    debug_logger.info(f"second_file_docs number is one")
                    new_docs.extend(ori_second_docs)
                    return new_docs
                # 获取second_file_dict['doc_ids']的最小值和最大值，用以获取完整文档时过滤文件id对应的文档列表，其实是减少完整文档的token数量
                doc_limit = [min(second_file_dict['doc_ids']), max(second_file_dict['doc_ids'])]
                # 依据第二个文件id获取经排序【先根据文档索引限制参数doc_limit对文件id查询文档列表结果进行切片，再对切片后的数据按照文档id从小到大排序】处理的文档json_data字段，分别得到两个完整文档，一个内容含有图片，一个内容不含图片
                second_completed_doc_limit, second_completed_doc_limit_with_figure = self.get_completed_document(
                    second_file_dict['file_id'], doc_limit)
                # 设置得分
                second_completed_doc_limit.metadata['score'] = second_file_dict['score']
                # 获取不含图片的完整文档的token数量
                second_doc_tokens = custom_llm.num_tokens_from_docs([second_completed_doc_limit])
                if first_doc_tokens + second_doc_tokens > limited_token_nums:
                    # 如果第二个文件id的不含图片的完整文档【过滤了文档id，只取doc_limit范围内的文档列表】的token数量与第二个文档列表的token数量之和超过限制token数量，【前期已经判断过：第二个文档列表的token数量与第一个文件id对应的完整文档的token数量之和不超过限制的token数量】则将第二个文档列表追加到new_docs中，并返回
                    debug_logger.info(
                        f"first_doc_tokens: {first_doc_tokens} + second_limit_doc_tokens {doc_limit}: {second_doc_tokens} > limited_token_nums: {limited_token_nums}")
                    new_docs.extend(ori_second_docs)
                    return new_docs
                else:
                    # 如果第二个文件id的不含图片的完整文档【过滤了文档id，只取doc_limit范围内的文档列表】的token数量与第二个文档列表的token数量之和不超过限制token数量，则将第二个文件id对应的带有图片的完整文档追加到new_docs中
                    debug_logger.info(
                        f"first_doc_tokens: {first_doc_tokens} + second_limit_doc_tokens {doc_limit}: {second_doc_tokens} <= limited_token_nums: {limited_token_nums}")
                    new_docs.append(second_completed_doc_limit_with_figure)
            else:
                # 如果第二个文件id的不含图片的完整文档的token数量与第二个文档列表的token数量之和不超过限制token数量，则将第二个文件id对应的带有图片的完整文档追加到new_docs中
                debug_logger.info(
                    f"first_doc_tokens: {first_doc_tokens} + second_doc_tokens: {second_doc_tokens} <= limited_token_nums: {limited_token_nums}")
                new_docs.append(second_completed_doc_with_figure)
        return new_docs

    def incomplete_table(self, source_documents, limited_token_nums, custom_llm):
        # 若某个doc里包含表格的一部分，则扩展为整个表格
        existing_table_docs = [doc for doc in source_documents if doc.metadata.get('has_table', False)]
        if not existing_table_docs:
            return source_documents
        new_docs = []
        existing_table_ids = []
        verified_table_ids = []
        current_doc_tokens = custom_llm.num_tokens_from_docs(source_documents)
        for doc in source_documents:
            if 'doc_id' not in doc.metadata:
                new_docs.append(doc)
                continue
            if table_doc_id := doc.metadata.get('table_doc_id', None):
                if table_doc_id in existing_table_ids:  # 已经不全了完整表格
                    continue
                if table_doc_id in verified_table_ids:  # 已经确认了完整表格太大放不大
                    new_docs.append(doc)
                    continue
                doc_json = self.milvus_summary.get_document_by_doc_id(table_doc_id)
                if doc_json is None:
                    new_docs.append(doc)
                    continue
                table_doc = Document(page_content=doc_json['kwargs']['page_content'],
                                     metadata=doc_json['kwargs']['metadata'])
                table_doc.metadata['score'] = doc.metadata['score']
                table_doc_tokens = custom_llm.num_tokens_from_docs([table_doc])
                current_table_docs = [doc for doc in source_documents if
                                      doc.metadata.get('table_doc_id', None) == table_doc_id]
                subtract_table_doc_tokens = custom_llm.num_tokens_from_docs(current_table_docs)
                if current_doc_tokens + table_doc_tokens - subtract_table_doc_tokens > limited_token_nums:
                    debug_logger.info(
                        f"Add table_doc_tokens: {table_doc_tokens} > limited_token_nums: {limited_token_nums}")
                    new_docs.append(doc)
                    verified_table_ids.append(table_doc_id)
                    continue
                else:
                    debug_logger.info(f"Incomplete table_doc: {table_doc_id}")
                    new_docs.append(table_doc)
                    existing_table_ids.append(table_doc_id)
                    current_doc_tokens = current_doc_tokens + table_doc_tokens - subtract_table_doc_tokens
        return new_docs
