from transformers import AutoTokenizer
from copy import deepcopy
from typing import List
import asyncio
from qanything_kernel.configs.model_config import LOCAL_RERANK_MAX_LENGTH, \
    LOCAL_RERANK_BATCH, LOCAL_RERANK_PATH
from qanything_kernel.utils.general_utils import get_time, get_time_async
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from concurrent.futures import ThreadPoolExecutor
from qanything_kernel.utils.custom_log import rerank_logger
import numpy as np


class RerankAsyncBackend:
    def __init__(self, model_path, use_cpu=True, num_threads=4):
        self.use_cpu = use_cpu
        self.overlap_tokens = 80
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        self.return_tensors = "np"
        # 创建一个ONNX Runtime会话设置，使用GPU执行
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        if use_cpu:
            providers = ['CPUExecutionProvider']
            self.executor = ThreadPoolExecutor(max_workers=num_threads)
            self.batch_size = LOCAL_RERANK_BATCH  # CPU批处理大小
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.executor = ThreadPoolExecutor(max_workers=num_threads)
            self.batch_size = 16  # GPU批处理大小固定为16

        self.session = InferenceSession(model_path, sess_options, providers=providers)
        # 获取rerank分词器
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_RERANK_PATH, use_fast=True)
        self.spe_id = self._tokenizer.sep_token_id

        self.queue = asyncio.Queue()
        asyncio.create_task(self.process_queue())

    @get_time
    def rerank_inference(self, batch):
        """
        rerank处理
        """
        rerank_logger.info(f"rerank shape: {batch['attention_mask'].shape}")
        # 准备输入数据
        inputs = {self.session.get_inputs()[i].name: batch[name]
                  for i, name in enumerate(['input_ids', 'attention_mask', 'token_type_ids'])
                  if name in batch}

        inputs = {k: np.array(v, dtype=np.int64) for k, v in inputs.items()}
        # 执行推理 输出为logits
        result = self.session.run(None, inputs)  # None表示获取所有输出
        # debug_logger.info(f"rerank result: {result}")

        # 应用sigmoid函数
        sigmoid_scores = 1 / (1 + np.exp(-np.array(result[0])))

        return sigmoid_scores.reshape(-1).tolist()

    def merge_inputs(self, chunk1_raw, chunk2):
        """
        对问题分词结果和文档内容分词结果做合并处理
        """
        chunk1 = deepcopy(chunk1_raw)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 1)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1

    def tokenize_preproc(self, query: str, passages: List[str]):
        """
        分词预处理
        对问题和文档内容列表分别进行分词处理，并逐一对问题分词结果和文档内容分词结果进行合并，最后收集合并分词结果列表，以及合并分词结果的索引列表【由于出现一个文档内容分词结果较长的情况【其多个分段对应同一个合并分词结果索引】，因此这里专门设置一个列表来收集合并分词结果索引】
        特别地，对于那些分词长度较长【超过：最大文档内容分词长度：512-问题分词长度-1】的文档内容分词结果，进行截取分段，并逐一对问题分词结果和文档内容分词结果的分段进行合并
        """
        # 对问题进行分词【分token】处理
        query_inputs = self._tokenizer(query, add_special_tokens=False, truncation=False, padding=False)
        # 计算最大文档内容分词长度：512-问题分词长度-1
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 1
        # 断言max_passage_inputs_length大于10
        assert max_passage_inputs_length > 10
        # 设置重叠分词长度：取默认重叠分词长度80与【max_passage_inputs_length乘2后，除以7的向下取整结果】之间的最小值
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length * 2 // 7)

        merge_inputs, merge_inputs_idxs = [], []
        # 遍历文档内容列表枚举对象【文档索引，对应的文档内容】
        for pid, passage in enumerate(passages):
            # 对当前文档内容进行分词
            passage_inputs = self._tokenizer(passage, add_special_tokens=False, truncation=False, padding=False)
            # 获取当前文档内容分词长度
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
                # 当前文档内容分词长度不超过文档内容最大分词长度时
                if not passage_inputs['attention_mask']:
                    # 判断当前文档内容分词结果中的attention_mask字段等价于False时，处理下一个文档内容
                    continue
                # 对问题分词结果和当前文档内容分词结果进行合并处理
                qp_merge_inputs = self.merge_inputs(query_inputs, passage_inputs)
                # 将合并处理结果添加到合并分词列表中
                merge_inputs.append(qp_merge_inputs)
                # 同步收集对应的合并分词索引
                merge_inputs_idxs.append(pid)
            else:
                # 当前文档内容分词长度超过文档内容最大分词长度时，对当前文档内容分词结果进行截取【按照max_passage_inputs_length分段截取】
                # 初始化起始token索引
                start_id = 0
                while start_id < passage_inputs_length:
                    # 当前分段的截止token索引
                    end_id = start_id + max_passage_inputs_length
                    # 当前分段截取结果
                    sub_passage_inputs = {k: v[start_id:end_id] for k, v in passage_inputs.items()}
                    # 更新下一次分段的起始token索引【如果end_id < passage_inputs_length，则start_id=end_id - overlap_tokens【考虑了重叠token长度】；反之，start_id=end_id【分段截取结束了】】
                    start_id = end_id - overlap_tokens if end_id < passage_inputs_length else end_id
                    # 对问题分词结果和当前文档内容分词结果的当前截取分段进行合并处理
                    qp_merge_inputs = self.merge_inputs(query_inputs, sub_passage_inputs)
                    # 将合并处理结果添加到合并分词列表中
                    merge_inputs.append(qp_merge_inputs)
                    # 同步收集对应的合并分词索引【对于截取的多个分段而言，其索引都为pid】
                    merge_inputs_idxs.append(pid)

        return merge_inputs, merge_inputs_idxs

    @get_time_async
    async def get_rerank_async(self, query: str, passages: List[str]):
        """
        使用rerank模型，依据问题query，对passages【文档列表】进行重排处理
        """
        # 对问题和文档内容列表做分词预处理，得到问题分词结果和文档内容分词结果的合并结果列表及合并结果索引列表
        tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)

        futures = []
        mini_batch = 1  # 设置mini_batch为1
        # 遍历逐个处理上述合并分词结果，放入待rerank任务队列中
        for i in range(0, len(tot_batches), mini_batch):
            # 将当前协程结果对象future添加到futures中
            future = asyncio.Future()
            futures.append(future)
            # 将对应合并分词结果连同future放入待rerank任务队列中
            await self.queue.put((tot_batches[i:i + mini_batch], future))
        # 获取rerank结果列表
        results = await asyncio.gather(*futures)
        # 遍历rerank结果列表，转换成单列表形式【由于批次设为1，batch_scores中仅有一个元素】
        tot_scores = [score for batch_scores in results for score in batch_scores]
        # 对应文档内容列表，初始化得分为0
        merge_tot_scores = [0 for _ in range(len(passages))]
        # 将分词结果索引与对应分词结果经rerank处理后的得分一一配对，取得分与0【对于那些分段的文档内容，这里是取几个分段之间的得分最大值】之间的最大值作为相应文档内容的得分
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)
        # 返回对应文档内容的得分列表
        return merge_tot_scores

    async def process_queue(self):
        while True:
            batch_items = []
            futures = []

            try:
                while len(batch_items) < self.batch_size:
                    batch, future = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                    batch_items.extend(batch)
                    futures.append((future, len(batch)))
            except asyncio.TimeoutError:
                pass

            if batch_items:
                loop = asyncio.get_running_loop()
                input_batch = self._tokenizer.pad(
                    batch_items,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors=self.return_tensors
                )
                # 利用线程池对批量任务做rerank处理
                result = await loop.run_in_executor(self.executor, self.rerank_inference, input_batch)

                start = 0
                for future, item_count in futures:
                    end = start + item_count
                    future.set_result(result[start:end])
                    start = end
            else:
                await asyncio.sleep(0.1)

    async def get_rerank(self, query: str, passages: List[str]):
        return await self.get_rerank_async(query, passages)
