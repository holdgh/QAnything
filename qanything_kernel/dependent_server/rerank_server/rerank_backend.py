from transformers import AutoTokenizer
from copy import deepcopy
from typing import List
from qanything_kernel.configs.model_config import LOCAL_RERANK_MAX_LENGTH, \
    LOCAL_RERANK_BATCH, LOCAL_RERANK_PATH, LOCAL_RERANK_THREADS
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.utils.general_utils import get_time
import concurrent.futures
from abc import ABC, abstractmethod


class RerankBackend(ABC):
    def __init__(self, use_cpu: bool = False):
        self.use_cpu = use_cpu
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_RERANK_PATH)
        self.spe_id = self._tokenizer.sep_token_id
        self.overlap_tokens = 80
        self.batch_size = LOCAL_RERANK_BATCH
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        self.return_tensors = None
        self.workers = LOCAL_RERANK_THREADS

    @abstractmethod
    def inference(self, batch) -> List:
        pass
    
    def merge_inputs(self, chunk1_raw, chunk2):
        chunk1 = deepcopy(chunk1_raw)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 1)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1

    def tokenize_preproc(self,
                         query: str,
                         passages: List[str],
                         ):
        """
        逻辑如下：
        先将query进行分词编码【分词，分词索引编码】，计算其分词编码长度
        再基于最大长度减去query的分词编码长度，得出一个段落的最大允许长度max_passage_inputs_length【不得小于10】
        取默认重叠长度和max_passage_inputs_length的最小值作为后续要使用的overlap_tokens
        枚举遍历段落列表，进行下述处理：
            对当前段落进行分词编码【分词，分词索引编码】，计算其分词编码长度，做以下判断处理：
                若其分词编码长度不超过max_passage_inputs_length，直接拼接query分词编码和当前段落的分词编码；
                若其分词编码长度超过max_passage_inputs_length，则对当前段落进行逐段切分【每段长度不超过max_passage_inputs_length，且段与段之间的分词编码重叠长度为overlap_tokens】，逐段拼接query分词编码；
        """
        query_inputs = self._tokenizer.encode_plus(query, truncation=False, padding=False)
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 1
        assert max_passage_inputs_length > 10
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length * 2 // 7)

        # 组[query, passage]对
        merge_inputs = []
        merge_inputs_idxs = []
        for pid, passage in enumerate(passages):
            passage_inputs = self._tokenizer.encode_plus(passage, truncation=False, padding=False,
                                                         add_special_tokens=False)
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
                if passage_inputs['attention_mask'] is None or len(passage_inputs['attention_mask']) == 0:
                    continue
                qp_merge_inputs = self.merge_inputs(query_inputs, passage_inputs)
                merge_inputs.append(qp_merge_inputs)
                # 一个段落对应一个段落索引
                merge_inputs_idxs.append(pid)
            else:
                start_id = 0
                while start_id < passage_inputs_length:
                    end_id = start_id + max_passage_inputs_length
                    sub_passage_inputs = {k: v[start_id:end_id] for k, v in passage_inputs.items()}
                    start_id = end_id - overlap_tokens if end_id < passage_inputs_length else end_id

                    qp_merge_inputs = self.merge_inputs(query_inputs, sub_passage_inputs)
                    merge_inputs.append(qp_merge_inputs)
                    # 多段对应一个父段落索引
                    merge_inputs_idxs.append(pid)

        return merge_inputs, merge_inputs_idxs

    @get_time
    def get_rerank(self, query: str, passages: List[str]):
        tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)

        tot_scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for k in range(0, len(tot_batches), self.batch_size):
                batch = self._tokenizer.pad(
                    tot_batches[k:k + self.batch_size],
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors=self.return_tensors
                )
                future = executor.submit(self.inference, batch)
                futures.append(future)
            # debug_logger.info(f'rerank number: {len(futures)}')
            for future in futures:
                scores = future.result()
                tot_scores.extend(scores)

        merge_tot_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)
        # print("merge_tot_scores:", merge_tot_scores, flush=True)
        return merge_tot_scores
