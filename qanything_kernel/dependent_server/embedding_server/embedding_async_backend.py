import asyncio
import time
import numpy as np
# onnxruntime是一种推理框架
"""
onnxruntime的运行可以分为三个阶段，：
1、session构造：【进行各成员的初始化，得到一个InferenceSession对象】
    1、负责OpKernel管理的KernelRegistryManager对象
    2、持有Session配置信息的SessionOptions对象
    3、负责图分割的GraphTransformerManager
    4、负责log管理的LoggingManager
2、模型加载与初始化：【将onnx模型加载到InferenceSession对象对象中并进行进一步的初始化】
    1、模型加载，提供了8中Load函数，可从url，ModelProto，void* model data，model istream等读取ModelProto。InferenceSession会对ModelProto进行解析然后持有其对应的Model成员
    2、providers注册，在模型加载结束后，InferenceSession会调用RegisterExecutionProviders()函数进行providers注册;【RegisterExecutionProviders函数会完成ExecutionProvider的注册工作。这里解释一下ExecutionProvider，ONNXRuntime用Provider表示不同的运行设备比如CUDAProvider等。目前ONNXRuntimev1.0支持了包括CPU，CUDA，TensorRT，MKL等七种Providers。通过调用sess->RegisterExecutionProvider()函数，InferenceSession通过一个list持有当前运行环境中支持的ExecutionProviders。】
    3、InferenceSession进一步初始化，调用sess->Initialize()【这时InferenceSession会根据自身持有的model和execution providers进行进一步的初始化（在第一阶段Session构造时仅仅持有了空壳子成员变量）。该步骤是InferenceSession初始化的核心，一系列核心操作如内存分配，model partition，kernel注册等都会在这个阶段完成。】
        1、首先，session会根据level注册 graph optimization transformers，并通过GraphTransformerManager成员进行持有。
        2、接下来session会进行OpKernel注册，OpKernel即定义的各个node对应在不同运行设备上的计算逻辑。这个过程会将持有的各个ExecutionProvider上定义的所有node对应的Kernel注册到session中，session通过KernelRegistryManager成员进行持有和管理。
        3、然后session会对Graph进行图变换，包括插入copy节点，cast节点等。
        4、接下来是model partition，也就是根运行设备对graph进行切分，决定每个node运行在哪个provider上。
        5、最后，为每个node创建ExecutePlan，运行计划主要包含了各个op的执行顺序，内存申请管理，内存复用管理等操作。
3、运行
    模型运行，InferenceSession对象每次读入一个batch的数据并进行计算得到模型的最终输出。
"""
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from qanything_kernel.utils.custom_log import embed_logger
from qanything_kernel.configs.model_config import LOCAL_EMBED_MAX_LENGTH, LOCAL_EMBED_PATH, LOCAL_EMBED_BATCH
from qanything_kernel.utils.general_utils import get_time, get_time_async


class EmbeddingAsyncBackend:
    def __init__(self, model_path, use_cpu=True, num_threads=4):
        # cpu使用标识
        self.use_cpu = use_cpu
        self.return_tensors = "np"
        # ========================onnxruntime框架第一阶段：构造session-start=============================
        # 持有Session配置信息的SessionOptions对象
        sess_options = SessionOptions()
        # 负责图分割的GraphTransformerManager
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        if use_cpu:
            # 使用cpu时，设置线程池和cpu批处理大小
            providers = ['CPUExecutionProvider']
            self.executor = ThreadPoolExecutor(max_workers=num_threads)
            self.batch_size = LOCAL_EMBED_BATCH  # CPU批处理大小【当前值为1】
        else:
            # 使用gpu时，设置线程池和gpu批处理大小
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.executor = ThreadPoolExecutor(max_workers=num_threads)
            self.batch_size = 16  # GPU批处理大小固定为16
        # 构造session，这里已经进行了onnxruntime框架第二阶段模型加载等操作
        self.session = InferenceSession(model_path, sess_options=sess_options, providers=providers)
        # ========================onnxruntime框架第一阶段：构造session-end=============================
        # 设置分词器？
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBED_PATH, use_fast=True)  # 请根据实际使用的模型调整
        # 设置任务队列，用以收集待处理的任务【什么时候收集待处理任务呢？】
        self.queue = asyncio.Queue()
        asyncio.create_task(self.process_queue())

    @get_time_async
    async def embed_documents_async(self, texts):
        """
        对文本列表做词向量转换处理
        """
        futures = []
        # 设置mini_batch=1，每次处理1个文本
        mini_batch = 1
        # 遍历文本列表，逐个文本进行词向量转换
        for i in range(0, len(texts), mini_batch):
            future = asyncio.Future()
            futures.append(future)
            # 将当前文本添加到协程队列中，作为待处理的任务
            await self.queue.put((texts[i:i + mini_batch], future))

        results = await asyncio.gather(*futures)
        return [item for sublist in results for item in sublist]

    # @get_time
    @get_time
    def embed_documents(self, texts):
        """
        对文本进行embed处理
        """
        # ========================onnxruntime框架第三阶段：模型运行-start=============================
        # max_length为512
        inputs_onnx = self._tokenizer(texts, padding=True, truncation=True, max_length=LOCAL_EMBED_MAX_LENGTH,
                                      return_tensors=self.return_tensors)
        # inputs_onnx = {k: v for k, v in inputs_onnx.items()}

        inputs_onnx = {k: np.array(v, dtype=np.int64) for k, v in inputs_onnx.items()}
        # start_time = time.time()
        outputs_onnx = self.session.run(output_names=['output'], input_feed=inputs_onnx)
        # debug_logger.info(f"onnx infer time: {time.time() - start_time}")

        # ========================onnxruntime框架第三阶段：模型运行-end=============================
        # 提取词嵌入结果【这里只取第一行，为什么？】
        embedding = outputs_onnx[0][:, 0]
        embed_logger.info(f'embedding shape: {embedding.shape}')
        # 对词嵌入结果做正则化处理？
        # 对每一个行向量计算范数【元素平方和的算术平方根】
        norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
        # 每一行的元素对应除以该行的范数，得到归一化处理后的词向量【每一行的平方和等于1】
        embeddings_normalized = embedding / norm_arr
        # 将每行归一化后【方便计算余弦相似度】的词向量矩阵转换为二维列表，并返回
        return embeddings_normalized.tolist()


    """
    协程函数定义：async def func_name()
    必须将协程函数添加到事件循环【event_loop】中调用运行；单独运行协程函数只会返回一个coroutine对象
    """
    async def process_queue(self):
        while True:
            # 批处理文本列表【待处理】
            batch_texts = []
            # 处理结果列表【已处理】
            futures = []

            try:
                # 只要批处理文本列表长度小于批处理大小，则进行循环体操作【注意batch_texts和futures存在着对应关系】
                while len(batch_texts) < self.batch_size:
                    # 从队列获取待处理的任务【待处理的文本和用来存放处理结果的future】TODO 队列中什么时候加入待处理任务的呢？【见上述embed_documents_async方法，有文本和future的插入】
                    texts, future = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                    # 将待处理的文本追加到批处理文本列表中
                    batch_texts.extend(texts)
                    # 将future和文本长度以元组形式收集到处理结果列表中【len(texts)用以截取处理结果，future用以存放最终处理结果】
                    futures.append((future, len(texts)))
            except asyncio.TimeoutError:
                pass

            # debug_logger.info(f"process_queue embedding texts number: {len(batch_texts)}")
            if batch_texts:
                # 批处理文本列表非空时
                """
                loop=asyncio.get_running_loop() #返回（获取）在当前线程中正在运行的事件循环，如果没有正在运行的事件循环，则会显示错误；它是python3.7中新添加的
 
                loop=asyncio.get_event_loop() #获得一个事件循环，如果当前线程还没有事件循环，则创建一个新的事件循环loop；
 
                loop=asyncio.set_event_loop(loop) #设置一个事件循环为当前线程的事件循环；
 
                loop=asyncio.new_event_loop()  #创建一个新的事件循环
                """
                # 获取当前线程中正在运行的事件循环
                loop = asyncio.get_running_loop()
                # 用上面获取的事件循环：利用线程池self.executor，调用当前对象的embed_documents方法，对待处理的文本列表进行处理【文本转化为词向量】，将处理结果赋值给result
                result = await loop.run_in_executor(self.executor, self.embed_documents, batch_texts)
                # futures与batch_texts存在着一一对应关系【(future, len(texts))--texts】
                start = 0
                for future, text_count in futures:
                    end = start + text_count
                    # 将embed_documents处理结果【相应文本的词向量】收集到future中【start和end用来保持对应关系】【futures与batch_texts存在着一一对应关系【(future, len(texts))--texts】】
                    future.set_result(result[start:end])
                    start = end
            else:
                # 批处理文本列表为空
                await asyncio.sleep(0.1)  # 如果没有文本要处理，短暂休眠
