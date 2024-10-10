from langchain.retrievers import ParentDocumentRetriever
from qanything_kernel.core.retriever.vectorstore import VectorStoreMilvusClient
from qanything_kernel.core.retriever.elasticsearchstore import StoreElasticSearchClient
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.core.retriever.docstrore import MysqlStore
from qanything_kernel.configs.model_config import DEFAULT_CHILD_CHUNK_SIZE, DEFAULT_PARENT_CHUNK_SIZE
from qanything_kernel.utils.custom_log import debug_logger, insert_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qanything_kernel.utils.general_utils import num_tokens_embed, get_time_async
import copy
from typing import List, Optional, Tuple, Dict
from langchain_core.documents import Document
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
)
from langchain_community.vectorstores.milvus import Milvus
from langchain_elasticsearch import ElasticsearchStore
import time
import traceback


class SelfParentRetriever(ParentDocumentRetriever):
    def set_search_kwargs(self, search_type, **kwargs):
        self.search_type = search_type
        self.search_kwargs = kwargs
        debug_logger.info(f"Set search kwargs: {self.search_kwargs}")

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        从向量数据库查询文件
        参数：
            query：问题，用以查询与该问题相关的文档
            run_manager：回调函数
        返回：
            文档列表
        """
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        debug_logger.info(f"Search: query: {query}, {self.search_type} with {self.search_kwargs}")
        # self.vectorstore.col.load()
        scores = []
        if self.search_type == "mmr":
            sub_docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            # 查询类型为：similarity，相似度查询
            # 调用SelfMilvus对象的asimilarity_search_with_score方法【该方法在其爷爷类VectorStore中有定义，具体实现位于其父类Milvus中定义的similarity_search_with_score方法。具体逻辑：调用词向量服务【similarity_search_with_score方法中的self.embedding_func.embed_query(query)，也即YouDaoEmbeddings中的_get_embedding_sync方法】将问题字符转换为词向量，然后依据词向量及其他条件查询向量数据库】进行查询
            res = await self.vectorstore.asimilarity_search_with_score(
                query, **self.search_kwargs
            )
            # 提取查询结果中的得分【这里的得分是怎么来的？】和文档
            scores = [score for _, score in res]
            sub_docs = [doc for doc, _ in res]

        # We do this to maintain the order of the ids that are returned【保持返回结果中的id顺序】
        # 初始化id列表
        ids = []
        # 遍历文档列表
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                # 当前文档的元数据中存在id_key且对应值【id】不在ids中时
                # 将id追加到ids中
                ids.append(d.metadata[self.id_key])
        # 利用MySQL查询文档列表【依据id列表，调用MysqlStore类父类InMemoryStore的amget方法，实际调用MysqlStore类的mget方法，执行self.mysql_client.get_document_by_doc_id(doc_id)】
        # 注意mget方法逻辑：逐个id进行MySQL数据库查询，结果为空，则对应添加None；结果非空，则处理文档元数据后，将查询结果收集到总体结果列表中
        docs = await self.docstore.amget(ids)
        if scores:
            # 向量数据库查询结果中的得分列表非空时
            # 遍历数据库文档查询结果
            for i, doc in enumerate(docs):
                if doc is not None:
                    # 当前文档非空时
                    # 给文档元数据添加得分信息
                    # 【见上述amget方法注释，依据向量数据库查询结果ids去MySQL数据库查询文档信息，查不到的置为None，保持了docs和scores二者的位置一致性，所以这里可以直接按照docs中的索引去scores中取得分赋值给docs中对应索引的文档元数据】
                    doc.metadata['score'] = scores[i]
        # 过滤掉docs中为None的元素
        res = [d for d in docs if d is not None]
        # 获取向量数据库文档查询结果的内容长度列表
        sub_docs_lengths = [len(d.page_content) for d in sub_docs]
        # 获取MySQL数据库文档查询结果的内容长度列表
        res_lengths = [len(d.page_content) for d in res]
        debug_logger.info(
            f"Got child docs: {len(sub_docs)}, {sub_docs_lengths} and Parent docs: {len(res)}, {res_lengths}")
        # 返回MySQL数据库中的非空文档列表
        return res

    async def aadd_documents(
            self,
            documents: List[Document],
            ids: Optional[List[str]] = None,
            add_to_docstore: bool = True,
            parent_chunk_size: Optional[int] = None,
            es_store: Optional[ElasticsearchStore] = None,
            single_parent: bool = False,
    ) -> Tuple[int, Dict]:
        """
        对文档列表进行父切分和子切分，将切分【换行符和中文符号】后的文档列表【经过一些元数据的处理后】存入向量数据库、es和数据库文档存储
        """
        # insert_logger.info(f"Inserting {len(documents)} complete documents, single_parent: {single_parent}")
        # 切分起始时间
        split_start = time.perf_counter()
        # =================父切分-start=================
        if self.parent_splitter is not None and not single_parent:
            # 父切分符非空且非单亲时【对有表格且文档内容token数量超过父切分尺寸的文档进行切分处理【依据换行符和中文标点符号】，然后将所有文档再收集到documents中】
            # documents = self.parent_splitter.split_documents(documents)
            # 初始化已切分文档列表和待切分的文档列表
            split_documents = []
            need_split_docs = []
            # 遍历文档列表
            for doc in documents:
                if doc.metadata['has_table'] or num_tokens_embed(doc.page_content) <= parent_chunk_size:
                    # 当前文档有表格或者当前文档内容的token数量不超过父切分尺寸时
                    if need_split_docs:
                        # 待切分的文档列表非空时【将后面【当前文档没有表格且当前文档内容的token数量超过父切分尺寸时，则将当前文档收集到待切分的文档列表中】这些文档进行切分处理】
                        # 利用分隔符【换行符和中文标点符号】对其进行切分，并收集到已切分的文档列表中
                        split_documents.extend(self.parent_splitter.split_documents(need_split_docs))
                        # 将待切分的文档列表置为空【说明根据换行符和中文标点符号已对待切分的文档列表切分完毕】
                        need_split_docs = []
                    # 当前文档有表格或者当前文档内容的token数量不超过父切分尺寸时，且待切分的文档列表为空，则将当前文档收集到已切分的文档列表中
                    split_documents.append(doc)
                else:
                    # 当前文档没有表格且当前文档内容的token数量超过父切分尺寸时，将当前文档收集到待切分的文档列表中
                    need_split_docs.append(doc)
            if need_split_docs:
                # 待切分的文档列表非空时【将上面【当前文档没有表格且当前文档内容的token数量超过父切分尺寸时，将当前文档收集到待切分的文档列表中】这些文档进行切分处理】
                # 利用分隔符【换行符和中文标点符号】对其进行切分，并收集到已切分文档列表中
                split_documents.extend(self.parent_splitter.split_documents(need_split_docs))
            # 将已切分的文档列表赋值给文档列表
            documents = split_documents
        # 打印父切分处理情况：正在插入**个父切分文档
        insert_logger.info(f"Inserting {len(documents)} parent documents")
        # =================父切分-end==================
        # ===============文档id处理-start==============
        if ids is None:
            # ids为none时，根据insert_documents方法中ids的定义可知【当single_parent为False时，ids为None；当single_parent为True时，ids为文档id列表】，此时single_parent为False【TODO 文档列表来自于多个文件？】
            # 提取第一个文档元数据中的文件id
            file_id = documents[0].metadata['file_id']
            # 按照规则“文件id_文档索引”生成文档id，并收集到doc_ids中
            doc_ids = [file_id + '_' + str(i) for i, _ in enumerate(documents)]
            if not add_to_docstore:
                # add_to_docstore默认值为True，在insert_files_server调用时，此处不会触发异常
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )
        else:
            # ids非空时【此时single_parent为True，没有进行上面的父切分，文档列表的文档数量与ids尺寸应该相等】，TODO 文档列表来自于一个文件
            if len(documents) != len(ids):
                # 文档列表的长度与ids的长度不一致时，触发异常
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            # 将ids直接赋值给doc_ids
            doc_ids = ids
        # ===============文档id处理-end==============
        # =============子切分-start==============
        # 初始化
        # 收集子切分文档信息【多个文档对应一个文档id】，用以保存至向量数据库和es存储
        docs = []
        # 收集文档id和文档信息【一个文档对应一个文档id】，用以保存至MySQL数据库
        full_docs = []
        # 遍历枚举文档列表，获取索引和相应索引的文档
        for i, doc in enumerate(documents):
            # 对应文档索引，获取doc_ids中的对应索引的文档id
            _id = doc_ids[i]
            # 对当前文档，进行子切分【按照换行符和中文标点符号】，得到子切分文档列表sub_docs
            sub_docs = self.child_splitter.split_documents([doc])
            # 当前对象的child_metadata_fields属性并未设置，而child_metadata_fields默认为none
            # 推测意为：子文档元数据属性列表
            if self.child_metadata_fields is not None:
                # 不为none时，遍历子切分得到的文档列表
                for _doc in sub_docs:
                    # 设置当前文档的元数据
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
            # 遍历子切分文档列表，设置文档id元数据和在文档内容中追加headers元数据
            for _doc in sub_docs:
                # 设置子切分文档id元数据【id_key为"doc_id"】
                _doc.metadata[self.id_key] = _id
                # 在子切分文档内容中追加headers元数据
                _doc.page_content = f"[headers]({_doc.metadata['headers']})\n" + _doc.page_content  # 存入page_content，向量检索时会带上headers
            # 收集子切分文档列表结果
            docs.extend(sub_docs)
            # 将当前文档的headers元数据追加到其内容中
            doc.page_content = f"[headers]({doc.metadata['headers']})\n" + doc.page_content  # 存入page_content，等检索后rerank时会带上headers信息
            # 收集文档id和文档元组到列表
            full_docs.append((_id, doc))
        # 打印日志：正在插入多少个子切分文档，并给出第一个子切分文档的元数据和前100行内容
        insert_logger.info(f"Inserting {len(docs)} child documents, metadata: {docs[0].metadata}, page_content: {docs[0].page_content[:100]}...")
        # 记录切分时长
        time_record = {"split_time": round(time.perf_counter() - split_start, 2)}
        # ===============子切分-end===============
        # 执行拷贝，将子切分文档列表拷贝【赋值操作相当于不同的变量指向同一个数据，变量值和地址都是数据的值和地址，通过一个变量对数据进行操作，另一个变量的数据也同时变化。深拷贝相当于将一个变量的数据拷贝一份，放在一个新的地址上，这样两个变量的数据相同，但是地址不同，可以避免相互间的影响】给embed_docs
        embed_docs = copy.deepcopy(docs)
        # 删除一些metadata信息
        for idx, doc in enumerate(embed_docs):
            del doc.metadata['title_lst']
            del doc.metadata['has_table']
            del doc.metadata['images']
            del doc.metadata['file_name']
            del doc.metadata['nos_key']
            del doc.metadata['faq_dict']
            del doc.metadata['page_id']
        # 对embed_docs执行向量数据库存储
        # 【此处为调用SelfParentRetriever对象的vectorstore属性
        # 【VectorStoreMilvusClient对象的local_vectorstore属性，SelfMilvus对象：SelfMilvus(
        #             embedding_function=YouDaoEmbeddings(),
        #             connection_args={"host": self.host, "port": self.port},
        #             collection_name=MILVUS_COLLECTION_NAME,
        #             partition_key_field="kb_id",
        #             # primary_field="doc_id",
        #             auto_id=True,
        #             search_params={"params": {"ef": 64}}
        #         )，见vectorstore.py文件】
        #         的aadd_documents方法】
        # 调用链路：SelfMilvus类继承Milvus类，Milvus类继承VectorStore类，VectorStore类中定义了aadd_documents方法，其aadd_documents方法调用其aadd_texts方法，该方法在SelfMilvus类中有重写实现。因此此处关键逻辑在SelfMilvus类的aadd_texts方法中
        res = await self.vectorstore.aadd_documents(embed_docs, time_record=time_record)
        # 打印日志，在向量数据库中的插入的数量
        insert_logger.info(f'vectorstore insert number: {len(res)}, {res[0]}')
        # es存储操作
        """
        【es_client.es_store，es_client=StoreElasticSearchClient()】
        在类StoreElasticSearchClient的初始化方法中，存在下述定义：
        self.es_store = ElasticsearchStore(
            es_url=ES_URL,
            index_name=ES_INDEX_NAME,
            es_user=ES_USER,
            es_password=ES_PASSWORD,
            strategy=ElasticsearchStore.BM25RetrievalStrategy()
        )
        调用链路：es_store.aadd_documents【实际是VectorStore的aadd_documents方法】--》VectorStore的aadd_texts方法--》VectorStore的add_texts抽象方法--》在ElasticsearchStore中实现了add_texts方法【仅对入参texts做了列表转化】--》VectorStore的add_texts方法
        由此可知，此处调用外部工具，实际调用的是ElasticsearchStore中定义的add_texts方法，设置es存储连接参数，执行es存储
        """
        if es_store is not None:
            # es存储客户端不为none时，也即存在时
            try:
                # es存储开始时间
                es_start = time.perf_counter()
                # docs_ids中的doc_id是file_id + '_' + i
                docs_ids = [doc.metadata['file_id'] + '_' + str(i) for i, doc in enumerate(embed_docs)]
                # 对embed_docs执行es存储【没有调用embedding服务】
                es_res = await es_store.aadd_documents(embed_docs, ids=docs_ids)
                # 记录es存储时长
                time_record['es_insert_time'] = round(time.perf_counter() - es_start, 2)
                # 打印日志：es存储了多少数据
                insert_logger.info(f'es_store insert number: {len(es_res)}, {es_res[0]}')
            except Exception as e:
                # es存储发生异常时，打印日志：es存储异常及异常信息
                insert_logger.error(f"Error in aadd_documents on es_store: {traceback.format_exc()}")

        if add_to_docstore:
            # add_to_docstore默认值为True，在insert_files_server调用时，此处对未进行子切分的文档列表进行存储【在当前方法aadd_documents的调用处，insert_documents方法中，设置了self.retriever的docstore=MysqlStore(self.mysql_client)，所以此处为数据库存储】
            await self.docstore.amset(full_docs)
        # 返回向量数据库存储返回值和时长记录字典
        return len(res), time_record


class ParentRetriever:
    def __init__(self, vectorstore_client: VectorStoreMilvusClient, mysql_client: KnowledgeBaseManager, es_client: StoreElasticSearchClient):
        self.mysql_client = mysql_client
        self.vectorstore_client = vectorstore_client
        # This text splitter is used to create the parent documents
        init_parent_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
            chunk_size=DEFAULT_PARENT_CHUNK_SIZE,
            chunk_overlap=0,
            length_function=num_tokens_embed)
        # # This text splitter is used to create the child documents
        # # It should create documents smaller than the parent
        init_child_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
            chunk_size=DEFAULT_CHILD_CHUNK_SIZE,
            chunk_overlap=int(DEFAULT_CHILD_CHUNK_SIZE / 4),
            length_function=num_tokens_embed)
        # 此处依据ParentRetriever初始化函数中的向量存储client、MySQL数据库client、父子切分符列表生成了一个SelfParentRetriever对象，赋值给当前对象的retriever属性
        self.retriever = SelfParentRetriever(
            vectorstore=vectorstore_client.local_vectorstore,
            docstore=MysqlStore(mysql_client),
            child_splitter=init_child_splitter,
            parent_splitter=init_parent_splitter,
        )
        self.backup_vectorstore: Optional[Milvus] = None
        self.es_store = es_client.es_store
        self.parent_chunk_size = DEFAULT_PARENT_CHUNK_SIZE

    @get_time_async
    async def insert_documents(self, docs, parent_chunk_size, single_parent=False):
        """
        将文档列表插入到向量数据库
        流程如下：
        1、以参数中的切片尺寸【来源于文件信息中的chunk_size】为准
        2、设置父切分符列表和子切分符列表【换行符和中文标点】，设置存储客户端【数据库、向量数据库、es】
        3、对文档列表进行父切分和子切分，将切分【换行符和中文符号】后的文档列表【经过一些元数据的处理后】存入向量数据库、es和数据库文档存储
        """
        # 打印日志：文档列表的长度、切片尺寸
        insert_logger.info(f"Inserting {len(docs)} documents, parent_chunk_size: {parent_chunk_size}, single_parent: {single_parent}")
        if parent_chunk_size != self.parent_chunk_size:
            # 切片尺寸不等，以参数中的切片尺寸为准
            self.parent_chunk_size = parent_chunk_size
            # 对于切分工具，作者写了一个chinese_text_splitter工具类，但是这里没有用到
            # 父切分工具【将换行符和中文标点符号作为分隔符列表，递归切分文档，切分所得文档内容尺寸不超过parent_chunk_size】
            parent_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
                chunk_size=parent_chunk_size,
                # 切分重叠设置为0
                chunk_overlap=0,
                # 长度函数：计算字符串的token数量
                length_function=num_tokens_embed)
            # 设置子切分尺寸，取默认子切分尺寸和当前父切分尺寸一半的最小值
            child_chunk_size = min(DEFAULT_CHILD_CHUNK_SIZE, int(parent_chunk_size / 2))
            # 子切分工具【将换行符和中文标点符号作为分隔符列表，递归切分文档，切分所得文档内容尺寸不超过child_chunk_size】
            child_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
                chunk_size=child_chunk_size,
                # 切分重叠设置为切分尺寸的1/4
                chunk_overlap=int(child_chunk_size / 4),
                # 长度函数：计算字符串的token数量
                length_function=num_tokens_embed)
            self.retriever = SelfParentRetriever(
                # 向量存储设置为向量数据库客户端
                vectorstore=self.vectorstore_client.local_vectorstore,
                # 文档存储设置为数据库客户端
                docstore=MysqlStore(self.mysql_client),
                # 子切分符
                child_splitter=child_splitter,
                # 父切分符
                parent_splitter=parent_splitter
            )
        # insert_logger.info(f'insert documents: {len(docs)}')
        # 当single_parent为False【默认值，表示非单亲，也即文档列表来自于多个文件。TODO 在insert_files_server中，处理文件是逐个处理的，此处为什么默认值为False呢？】时，ids为None；当single_parent为True时，ids为文档id列表
        ids = None if not single_parent else [doc.metadata['doc_id'] for doc in docs]
        # 插入【对文档列表进行父切分和子切分，将切分【换行符和中文符号】后的文档列表【经过一些元数据的处理后】存入向量数据库、es和数据库文档存储】文档列表，入参：文档列表、父切分尺寸、es存储、文档id列表、单亲标识
        return await self.retriever.aadd_documents(docs, parent_chunk_size=parent_chunk_size,
                                                   es_store=self.es_store, ids=ids, single_parent=single_parent)

    async def get_retrieved_documents(self, query: str, partition_keys: List[str], time_record: dict,
                                      hybrid_search: bool, top_k: int):
        """
        依据问题、知识库id、混合检索标识及超参数top_k检索文档列表
        """
        # milvus是向量数据库，用以处理向量相似度搜索，加速非结构化数据检索
        # 设置向量检索起始时间
        milvus_start_time = time.perf_counter()
        # 设置知识库id列表遍历查询条件
        expr = f'kb_id in {partition_keys}'
        # self.retriever.set_search_kwargs("mmr", k=VECTOR_SEARCH_TOP_K, expr=expr)
        # 设置查询条件【注意top_k参数的作用】和查询类型【SelfParentRetriever对象的_age_relevant_documents方法依据查询类型参数进行查询】为相似度查询
        self.retriever.set_search_kwargs("similarity", k=top_k, expr=expr)
        # 依据问题和上述查询条件及查询类型执行查询【调用SelfParentRetriever对象的_age_relevant_documents方法】
        # 逻辑：将问题转化为词向量，进行向量数据库查询得到得分列表和文档id列表，然后利用文档id列表进行MySQL数据库查询，并将向量查询得到的得分追加到MySQL数据库查询结果中，返回MySQL数据库查询结果
        query_docs = await self.retriever.aget_relevant_documents(query)
        # 遍历查询结果
        for doc in query_docs:
            # 为每个文档设置检索来源：milvus
            doc.metadata['retrieval_source'] = 'milvus'
        # 计算向量检索耗时，并收集向量检索耗时数据【四舍五入，保留两位小数】
        milvus_end_time = time.perf_counter()
        time_record['retriever_search_by_milvus'] = round(milvus_end_time - milvus_start_time, 2)
        # 如果没有开启混合检索，则直接返回向量数据库检索结果
        if not hybrid_search:
            return query_docs
        # 当开启混合检索时，执行以下操作
        try:
            # filter = []
            # for partition_key in partition_keys:
            # 设置es检索参数：知识库id列表
            filter = [{"terms": {"metadata.kb_id.keyword": partition_keys}}]
            # 依据问题、超参数top_k和知识库id列表条件，执行es相似度检索【这里没有进行词向量的转化，直接根据query进行查询】
            """
            【es_client.es_store，es_client=StoreElasticSearchClient()】
            在类StoreElasticSearchClient的初始化方法中，存在下述定义：
            self.es_store = ElasticsearchStore(
                es_url=ES_URL,
                index_name=ES_INDEX_NAME,
                es_user=ES_USER,
                es_password=ES_PASSWORD,
                strategy=ElasticsearchStore.BM25RetrievalStrategy()
            )
            调用链路：es_store.asimilarity_search【实际是VectorStore的asimilarity_search方法】--》VectorStore的similarity_search抽象方法--》在ElasticsearchStore中实现了similarity_search方法
            由此可知，此处调用外部工具，直接根据query结合top_k和filter参数进行es检索
            """
            # es检索这里只取了文档，没有取得分
            es_sub_docs = await self.es_store.asimilarity_search(query, k=top_k, filter=filter)
            es_ids = []
            # 获取向量数据库检索【milvus检索】的文档id列表
            milvus_doc_ids = [d.metadata[self.retriever.id_key] for d in query_docs]
            # 遍历es检索结果
            for d in es_sub_docs:
                if self.retriever.id_key in d.metadata and d.metadata[self.retriever.id_key] not in es_ids and d.metadata[self.retriever.id_key] not in milvus_doc_ids:
                    # 如果es检索结果文档元数据中含有文档id且该文档id不在es检索文档id列表和不在milvus检索文档id列表中，则将该文档id收集到es检索文档id列表中
                    es_ids.append(d.metadata[self.retriever.id_key])
            # 依据es检索文档id列表获取es检索文档
            es_docs = await self.retriever.docstore.amget(es_ids)
            # 过滤掉es检索空文档
            es_docs = [d for d in es_docs if d is not None]
            # 遍历es检索文档列表，设置检索来源为：es
            for doc in es_docs:
                doc.metadata['retrieval_source'] = 'es'
            # 计算es检索耗时，并收集es检索耗时数据【四舍五入，保留两位小数】
            time_record['retriever_search_by_es'] = round(time.perf_counter() - milvus_end_time, 2)
            debug_logger.info(f"Got {len(query_docs)} documents from vectorstore and {len(es_sub_docs)} documents from es, total {len(query_docs) + len(es_docs)} merged documents.")
            # 在milvus检索结果中追加es检索文档列表
            query_docs.extend(es_docs)
        except Exception as e:
            debug_logger.error(f"Error in get_retrieved_documents on es_search: {e}")
        # 返回最终检索得到的文档列表
        return query_docs
