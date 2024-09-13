from typing import Union, Tuple, Dict
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from sanic.request import File
from qanything_kernel.configs.model_config import UPLOAD_ROOT_PATH
import uuid
import os


class LocalFile:
    def __init__(self, user_id, kb_id, file: Union[File, str, Dict], file_name):
        self.user_id = user_id
        self.kb_id = kb_id
        # 随机生成文件id
        self.file_id = uuid.uuid4().hex
        self.file_name = file_name
        # 文件路径【仅本地文件有意义】
        self.file_url = ''
        # 文件处理
        if isinstance(file, Dict):
            # 字典类型文件？
            self.file_location = "FAQ"
            self.file_content = b''
        elif isinstance(file, str):
            # 本地文件，字符串路径
            self.file_location = "URL"
            self.file_content = b''
            self.file_url = file
        else:
            # 文件对象
            # 获取文件内容
            self.file_content = file.body
            # nos_key = construct_nos_key_for_local_file(user_id, kb_id, self.file_id, self.file_name)
            # debug_logger.info(f'file nos_key: {self.file_id}, {self.file_name}, {nos_key}')
            # self.file_location = nos_key
            # upload_res = upload_nos_file_bytes_or_str_retry(nos_key, self.file_content)
            # if 'failed' in upload_res:
            #     debug_logger.error(f'failed init localfile {self.file_name}, {upload_res}')
            # else:
            #     debug_logger.info(f'success init localfile {self.file_name}, {upload_res}')
            # 上传目录：根目录\QANY_DB\content\user_id
            upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
            # 文件目录：根目录\QANY_DB\content\user_id\kb_id\file_id
            file_dir = os.path.join(upload_path, self.kb_id, self.file_id)
            # 创建文件目录，若存在，则跳过【exist_ok：是否在目录存在时触发异常。如果 exist_ok 为 False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；如果 exist_ok 为 True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。】
            os.makedirs(file_dir, exist_ok=True)
            # 文件路径：文件目录拼接文件名
            self.file_location = os.path.join(file_dir, self.file_name)
            #  如果文件不存在，则创建文件f【这里就意味着文件上传到根目录下】，并将文件内容写入文件f中
            if not os.path.exists(self.file_location):
                with open(self.file_location, 'wb') as f:
                    f.write(self.file_content)
