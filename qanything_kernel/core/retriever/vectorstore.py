from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, List, Any, Iterable, Callable
from qanything_kernel.utils.custom_log import debug_logger, insert_logger
from qanything_kernel.configs.model_config import MILVUS_PORT, MILVUS_COLLECTION_NAME, MILVUS_HOST_LOCAL
from qanything_kernel.connector.embedding.embedding_for_online_client import YouDaoEmbeddings
from qanything_kernel.utils.general_utils import get_time, get_time_async
from langchain_community.vectorstores.milvus import Milvus
from pymilvus.orm.collection import MutationResult
import asyncio
import time


class SelfMilvus(Milvus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_flush_time = 0  # 初始化为0，确保首次调用_should_flush时返回True
        self.inserted_since_last_flush = 0
        self.flush_interval = 600  # 600 seconds
        self.flush_threshold = 10000  # 10,000 entities

    def _should_flush(self) -> bool:
        current_time = time.time()
        time_since_last_flush = current_time - self.last_flush_time
        return (self.inserted_since_last_flush >= self.flush_threshold or
                time_since_last_flush >= self.flush_interval or self.last_flush_time == 0)

    @get_time
    def _milvus_flush(self):
        asyncio.create_task(asyncio.to_thread(self.col.flush))
        self.last_flush_time = time.time()
        self.inserted_since_last_flush = 0
        insert_logger.info(f"Flushed Milvus collection at {self.last_flush_time}")

    def _create_collection(
            self, embeddings: list, metadatas: Optional[list[dict]] = None
    ) -> None:
        """
        功能：收集文档元数据字段名称和值类型，形成表结构【主键、文本内容字段、向量字段、文档元数据各字段】，并将结果赋予col属性
        """
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusException,
        )
        from pymilvus.orm.types import infer_dtype_bydata

        # Determine embedding dim
        dim = len(embeddings[0])
        fields = []
        # 未设置属性_metadata_field
        if self._metadata_field is not None:
            fields.append(FieldSchema(self._metadata_field, DataType.JSON))
        else:
            # Determine metadata schema
            # 走此逻辑
            if metadatas:
                # Create FieldSchema for each entry in metadata.
                # 遍历元数据的键值对，键【字段名】值【获取类型，设置字段类型】对
                for key, value in metadatas[0].items():
                    print(key, value, flush=True)
                    # Infer the corresponding datatype of the metadata
                    # 获取value的数据类型
                    dtype = infer_dtype_bydata(value)
                    # Datatype isn't compatible
                    if dtype == DataType.UNKNOWN or dtype == DataType.NONE:
                        # value为未知类型或None时，抛出异常
                        debug_logger.error(
                            (
                                "Failure to create collection, "
                                "unrecognized dtype for key: %s"
                            ),
                            key,
                        )
                        raise ValueError(f"Unrecognized datatype for {key}.")
                    # Dataype is a string/varchar equivalent
                    elif dtype == DataType.VARCHAR:
                        # value为字符串类型和最大长度限制
                        fields.append(
                            FieldSchema(key, DataType.VARCHAR, max_length=65_535)
                        )
                    else:
                        # 其他类型，直接设置类型即可
                        fields.append(FieldSchema(key, dtype))

        # Create the text field
        fields.append(
            # 设置文本内容字段类型为字符串并限制最大长度【父类Milvus中有声明，self._text_field="text"】
            FieldSchema(self._text_field, DataType.VARCHAR, max_length=65_535)
        )
        # 设置主键
        # Create the primary key field
        if self.auto_id:
            # 自动生成主键
            fields.append(
                # # 设置主键字段类型为数字并设置自动生成标识【父类Milvus中有声明，self._primary_field="pk"】
                FieldSchema(
                    self._primary_field, DataType.INT64, is_primary=True, auto_id=True
                )
            )
        else:
            # 非自动主键，采取字符串类型存储，并限制最大长度
            fields.append(
                FieldSchema(
                    self._primary_field,
                    DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=65_535,
                )
            )
        # Create the vector field, supports binary or float vectors
        # 设置向量字段名称【父类Milvus中有声明，self._vector_field="vector"】和维数
        fields.append(
            FieldSchema(self._vector_field, infer_dtype_bydata(embeddings[0]), dim=dim)
        )

        # Create the schema for the collection
        schema = CollectionSchema(
            fields,
            description=self.collection_description,
            # 在初始化SelfMilvus对象时，设置了partition_key_field="kb_id"
            partition_key_field=self._partition_key_field,
        )

        # Create the collection
        try:
            # 将收集结果赋值给col属性【未设置indexes，因此indexes为None】
            self.col = Collection(
                # 初始化SelfMilvus对象时设置了collection_name=MILVUS_COLLECTION_NAME
                name=self.collection_name,
                schema=schema,
                consistency_level=self.consistency_level,
                using=self.alias,
                num_partitions=64
            )
            # 未设置该属性
            # Set the collection properties if they exist
            if self.collection_properties is not None:
                self.col.set_properties(self.collection_properties)
        except MilvusException as e:
            debug_logger.error(
                "Failed to create collection: %s error: %s", self.collection_name, e
            )
            raise e

    def get_expr_result(self, expr: str, output_fields: List[str]) -> List[int] | None:
        """Get query result with expression

        Args:
            expr: Expression - E.g: "id in [1, 2]", or "title LIKE 'Abc%'"
            output_fields: List of fields to return

        Returns:
            List[int]: List of IDs (Primary Keys)
        """

        from pymilvus import MilvusException

        if self.col is None:
            debug_logger.debug("No existing collection to get pk.")
            return None

        try:
            query_result = self.col.query(
                expr=expr, output_fields=output_fields
            )
        except MilvusException as exc:
            debug_logger.error("Failed to get ids: %s error: %s", self.collection_name, exc)
            raise exc
        return query_result

    # 保存文档内容至向量数据库操作
    # 该方法的逻辑有相当一部分是抄的父类Milvus中的aadd_texts方法，其爷爷类VectorStore(ABC)中定义了抽象方法aadd_texts，被VectorStore(ABC)中定义的aadd_documents方法【提取文档的内容texts和元数据metadatas】调用
    """
    VectorStore(ABC)中定义的aadd_documents方法:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.aadd_texts(texts, metadatas, **kwargs)
    """
    async def aadd_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            timeout: Optional[int] = None,
            batch_size: int = 1000,
            *,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Asynchronously run texts through embeddings and add to the vectorstore."""
        # 从kwargs中获取time_record
        time_record = kwargs.get('time_record', {})

        from pymilvus import Collection, MilvusException

        texts = list(texts)
        # 在初始化SelfMilvus对象时，设置了auto_id=True
        if not self.auto_id:
            assert isinstance(ids, list), "A list of valid ids are required when auto_id is False."
            assert len(set(ids)) == len(texts), "Different lengths of texts and unique ids are provided."
            assert all(len(x.encode()) <= 65_535 for x in ids), "Each id should be a string less than 65535 bytes."

        # Assuming self.embedding_func has an async method embed_documents_async
        # 对文档内容进行embedding操作，转化为词向量
        embedding_start = time.perf_counter()
        try:
            embeddings = await self.embedding_func.aembed_documents(texts)
        except NotImplementedError:
            embeddings = [await self.embedding_func.aembed_query(x) for x in texts]
        time_record['milvus_embedding_time'] = round(time.perf_counter() - embedding_start, 2)

        if len(embeddings) == 0:
            insert_logger.info("Nothing to insert, skipping.")
            return []
        # ===============设置存储结构：表结构、字段名列表、向量字段索引方式、查询参数等-start===============
        # 属性col在其父类Milvus中有定义
        # If the collection hasn't been initialized yet, perform all steps to do so
        if not isinstance(self.col, Collection):
            # 将文档内容换为向量，构造字典参数kwargs
            kwargs = {"embeddings": embeddings, "metadatas": metadatas}
            if self.partition_names:
                kwargs["partition_names"] = self.partition_names
            if self.replica_number:
                kwargs["replica_number"] = self.replica_number
            if self.timeout:
                kwargs["timeout"] = self.timeout
            # 调用父类Milvus中的_init方法
            """
                    if embeddings is not None:
                        self._create_collection(embeddings, metadatas)--》根据文档元数据和向量集合创建表结构【主键、文本内容字段、向量字段、文档元数据各字段】，并将结果赋予col属性
                        self._extract_fields()--》从col中提取字段名列表，赋予fields属性
                        self._create_index()--》设置向量字段索引方式【HNSW、AUTOINDEX】，当前设置的是HNSW类型索引
                        self._create_search_params()--》在创建SelfMilvus对象时设置了search_params={"params": {"ef": 64}}
                        self._load(
                            partition_names=partition_names,
                            replica_number=replica_number,
                            timeout=timeout,
                        )--》该方法的入参都未设置，换言之，入参都为None，【当utility.load_state(self.collection_name, using=self.alias) == LoadState.NotLoad为True时】等价于：
                        conn.load_collection(
                            collection_name=self._name,
                            replica_number=replica_number,
                            timeout=timeout,
                            **kwargs,
                        )
            """
            self._init(**kwargs)

        # ===============设置存储结构：表结构、字段名列表、向量字段索引方式、查询参数等-end===============
        # ===============收集插入数据-start==============
        # Dict to hold all insert columns
        insert_dict: dict[str, list] = {
            self._text_field: texts,
            self._vector_field: embeddings,
        }

        # 在初始化SelfMilvus对象时，设置了auto_id=True
        if not self.auto_id:
            insert_dict[self._primary_field] = ids
        # 未设置属性_metadata_field
        if self._metadata_field is not None:
            for d in metadatas or []:
                insert_dict.setdefault(self._metadata_field, []).append(d)
        else:
            # 走此逻辑
            # Collect the metadata into the insert dict.
            if metadatas is not None:
                for d in metadatas:
                    # TODO 异常：too many values to unpack，太多值了
                    for key, value in d.items():
                        keys = (
                            [x for x in self.fields if x != self._primary_field]
                            if self.auto_id
                            else [x for x in self.fields]
                        )
                        if key in keys:
                            insert_dict.setdefault(key, []).append(value)

        # Total insert count
        vectors: list = insert_dict[self._vector_field]
        total_count = len(vectors)

        # ===============收集插入数据-end==============
        pks: list[str] = []

        insert_start = time.perf_counter()
        # 在设置存储结构时已经对col属性做了处理，此处必成立
        assert isinstance(self.col, Collection)
        # =================分批次【默认1000】存储数据-start==================
        for i in range(0, total_count, batch_size):
            # Grab end index
            # 当前批次的截止索引，i为起始索引
            end = min(i + batch_size, total_count)
            # Convert dict to list of lists batch for insertion
            # 截取各个字段的当前批次数据，并将所有字段的数据整合为一个列表【该列表的每个元素是某个字段的值列表】
            insert_list = [
                insert_dict[x][i:end] for x in self.fields if x in insert_dict
            ]
            # Insert into the collection.
            # 将当前批次的数据插入到数据库
            try:
                res: MutationResult = await asyncio.to_thread(
                    self.col.insert, insert_list, timeout=timeout, **kwargs
                )
                # insert_logger.info(f"insert: {res}, insert keys: {res.primary_keys}")
                insert_logger.info(f"insert: {res}")
                # 收集插入结果中的主键信息
                pks.extend(res.primary_keys)
            except MilvusException as e:
                insert_logger.error(
                    "Failed to insert batch starting at entity: %s/%s", i, total_count
                )
                raise e
            # 收集已经插入的数据数量
            self.inserted_since_last_flush += end - i

        time_record['milvus_insert_time'] = round(time.perf_counter() - insert_start, 2)
        # =================分批次【默认1000】存储数据-end==================
        # 刷新
        asyncio.create_task(asyncio.to_thread(self.col.flush))
        # if self._should_flush():
        #     self._milvus_flush()

        # self.col.flush()
        # 返回插入向量数据库结果中的主键列表【这里定义的是字符串列表，根据代码分析，主键类型为int64。此处可否？】
        return pks


class VectorStoreMilvusClient:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.host = MILVUS_HOST_LOCAL
        self.port = MILVUS_PORT
        self.local_vectorstore: Milvus = SelfMilvus(
            embedding_function=YouDaoEmbeddings(),
            connection_args={"host": self.host, "port": self.port},
            collection_name=MILVUS_COLLECTION_NAME,
            partition_key_field="kb_id",
            # primary_field="doc_id",
            auto_id=True,
            search_params={"params": {"ef": 64}}
        )
        debug_logger.info(
            f'init vectorstore {self.host}, {MILVUS_COLLECTION_NAME}')

    def get_local_chunks(self, expr, timeout=10):
        future = self.executor.submit(
            partial(self.local_vectorstore.get_pks, expr=expr, timeout=timeout))
        return future.result()

    # def delete_chunks(self, chunk_ids):
    #     res = self.vectorstore.delete(expr=f"chunk_id in {chunk_ids}")
    #     debug_logger.info(f'milvus delete chunk number: {len(chunk_ids)} res: {res}')

    @get_time
    def delete_expr(self, expr):
        # 如果expr为空，则不执行删除操作
        if len(self.get_local_chunks(expr)) == 0:
            debug_logger.info(f'expr: {expr} not found in local milvus')
            return
        try:
            res = self.local_vectorstore.delete(expr=expr, timeout=10)
            debug_logger.info(f'local milvus delete expr: {expr} res: {res}')
        except Exception as e:
            debug_logger.error(f'local milvus delete expr: {expr} error: {e}')
