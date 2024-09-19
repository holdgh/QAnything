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
            res = await self.vectorstore.asimilarity_search_with_score(
                query, **self.search_kwargs
            )
            scores = [score for _, score in res]
            sub_docs = [doc for doc, _ in res]

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = await self.docstore.amget(ids)
        if scores:
            for i, doc in enumerate(docs):
                if doc is not None:
                    doc.metadata['score'] = scores[i]
        res = [d for d in docs if d is not None]
        sub_docs_lengths = [len(d.page_content) for d in sub_docs]
        res_lengths = [len(d.page_content) for d in res]
        debug_logger.info(
            f"Got child docs: {len(sub_docs)}, {sub_docs_lengths} and Parent docs: {len(res)}, {res_lengths}")
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
        # insert_logger.info(f"Inserting {len(documents)} complete documents, single_parent: {single_parent}")
        split_start = time.perf_counter()
        if self.parent_splitter is not None and not single_parent:
            # documents = self.parent_splitter.split_documents(documents)
            split_documents = []
            need_split_docs = []
            for doc in documents:
                if doc.metadata['has_table'] or num_tokens_embed(doc.page_content) <= parent_chunk_size:
                    if need_split_docs:
                        split_documents.extend(self.parent_splitter.split_documents(need_split_docs))
                        need_split_docs = []
                    split_documents.append(doc)
                else:
                    need_split_docs.append(doc)
            if need_split_docs:
                split_documents.extend(self.parent_splitter.split_documents(need_split_docs))
            documents = split_documents
        insert_logger.info(f"Inserting {len(documents)} parent documents")
        if ids is None:
            file_id = documents[0].metadata['file_id']
            doc_ids = [file_id + '_' + str(i) for i, _ in enumerate(documents)]
            if not add_to_docstore:
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )
        else:
            if len(documents) != len(ids):
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            doc_ids = ids

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            if self.child_metadata_fields is not None:
                for _doc in sub_docs:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
                _doc.page_content = f"[headers]({_doc.metadata['headers']})\n" + _doc.page_content  # 存入page_content，向量检索时会带上headers
            docs.extend(sub_docs)
            doc.page_content = f"[headers]({doc.metadata['headers']})\n" + doc.page_content  # 存入page_content，等检索后rerank时会带上headers信息
            full_docs.append((_id, doc))
        insert_logger.info(f"Inserting {len(docs)} child documents, metadata: {docs[0].metadata}, page_content: {docs[0].page_content[:100]}...")
        time_record = {"split_time": round(time.perf_counter() - split_start, 2)}

        embed_docs = copy.deepcopy(docs)
        # 补充metadata信息
        for idx, doc in enumerate(embed_docs):
            del doc.metadata['title_lst']
            del doc.metadata['has_table']
            del doc.metadata['images']
            del doc.metadata['file_name']
            del doc.metadata['nos_key']
            del doc.metadata['faq_dict']
            del doc.metadata['page_id']

        res = await self.vectorstore.aadd_documents(embed_docs, time_record=time_record)
        insert_logger.info(f'vectorstore insert number: {len(res)}, {res[0]}')
        if es_store is not None:
            try:
                es_start = time.perf_counter()
                # docs的doc_id是file_id + '_' + i
                docs_ids = [doc.metadata['file_id'] + '_' + str(i) for i, doc in enumerate(embed_docs)]
                es_res = await es_store.aadd_documents(embed_docs, ids=docs_ids)
                time_record['es_insert_time'] = round(time.perf_counter() - es_start, 2)
                insert_logger.info(f'es_store insert number: {len(es_res)}, {es_res[0]}')
            except Exception as e:
                insert_logger.error(f"Error in aadd_documents on es_store: {traceback.format_exc()}")

        if add_to_docstore:
            await self.docstore.amset(full_docs)
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
        insert_logger.info(f"Inserting {len(docs)} documents, parent_chunk_size: {parent_chunk_size}, single_parent: {single_parent}")
        if parent_chunk_size != self.parent_chunk_size:
            self.parent_chunk_size = parent_chunk_size
            parent_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
                chunk_size=parent_chunk_size,
                chunk_overlap=0,
                length_function=num_tokens_embed)
            child_chunk_size = min(DEFAULT_CHILD_CHUNK_SIZE, int(parent_chunk_size / 2))
            child_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
                chunk_size=child_chunk_size,
                chunk_overlap=int(child_chunk_size / 4),
                length_function=num_tokens_embed)
            self.retriever = SelfParentRetriever(
                vectorstore=self.vectorstore_client.local_vectorstore,
                docstore=MysqlStore(self.mysql_client),
                child_splitter=child_splitter,
                parent_splitter=parent_splitter
            )
        # insert_logger.info(f'insert documents: {len(docs)}')
        ids = None if not single_parent else [doc.metadata['doc_id'] for doc in docs]
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
        # 设置查询条件【注意top_k参数的作用】和查询类型为相似度查询
        self.retriever.set_search_kwargs("similarity", k=top_k, expr=expr)
        # 依据问题和上述查询条件及查询类型执行查询
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
            # 依据问题、超参数top_k和知识库id列表条件，执行es相似度检索
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
