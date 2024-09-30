from qanything_kernel.utils.custom_log import insert_logger
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.configs.model_config import UPLOAD_ROOT_PATH
from qanything_kernel.utils.custom_log import debug_logger
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar
)
import os
import json
from tqdm import tqdm


V = TypeVar("V")


class MysqlStore(InMemoryStore):
    def __init__(self, mysql_client: KnowledgeBaseManager):
        self.mysql_client = mysql_client
        super().__init__()


    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[str, V]]): A sequence of key-value pairs.

        Returns:
            None
        """
        doc_ids = [doc_id for doc_id, _ in key_value_pairs]
        insert_logger.info(f"add documents: {len(doc_ids)}")
        for doc_id, doc in tqdm(key_value_pairs):
            doc_json = doc.to_json()
            if doc_json['kwargs'].get('metadata') is None:
                doc_json['kwargs']['metadata'] = doc.metadata
            self.mysql_client.add_document(doc_id, doc_json)

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        # 初始化结果列表
        docs = []
        # 遍历入参id列表，逐个查询
        for doc_id in keys:
            # 依据当前文档id查询文档信息
            doc_json = self.mysql_client.get_document_by_doc_id(doc_id)
            if doc_json is None:
                # 查询结果为空，对应在结果列表中添加None，继续进行下一次
                docs.append(None)
                continue
            # debug_logger.info(f'doc_id: {doc_id} get doc_json: {doc_json}')
            # 从文档元数据中提取用户id，文件id，文件名称，知识库id
            user_id, file_id, file_name, kb_id = doc_json['kwargs']['metadata']['user_id'], doc_json['kwargs']['metadata']['file_id'], doc_json['kwargs']['metadata']['file_name'], doc_json['kwargs']['metadata']['kb_id'] 
            # 取文档id以_分隔所得列表的最后一个元素，一个数字
            doc_idx = doc_id.split('_')[-1]
            # 构造上传文件时的路径
            upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
            # 构造完整的文档路径
            local_path = os.path.join(upload_path, kb_id, file_id, file_name.rsplit('.', 1)[0] + '_' + doc_idx + '.json')
            # 依据数据查询结果构造文档对象
            doc = Document(page_content=doc_json['kwargs']['page_content'], metadata=doc_json['kwargs']['metadata'])
            # 为上述文档对象元数据添加doc_id信息
            doc.metadata['doc_id'] = doc_id
            if file_name.endswith('.faq'):
                # 文件名以.faq结尾时
                # 取出文档元数据中的faq_dict信息
                faq_dict = doc.metadata['faq_dict']
                # 构造问答内容
                page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
                # 获取faq_dict中的nos_keys信息
                nos_keys = faq_dict.get('nos_keys')
                # 将文档内容置为问答内容
                doc.page_content = page_content
                # 将文档元数据中的nos_keys置为faq_dict中的nos_keys信息
                doc.metadata['nos_keys'] = nos_keys
            # 收集文档结果
            docs.append(doc)
            if not os.path.exists(local_path):
                # 文档不存在时
                #  json字符串写入本地文件
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                # debug_logger.info(f'write local_path: {local_path}')
                with open(local_path, 'w') as f:
                    f.write(json.dumps(doc_json, ensure_ascii=False))
        return docs
