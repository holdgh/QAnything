"""
pkuseg 是由北京大学语言计算与机器学习研究组研制推出的一套全新的中文分词工具包。pkuseg 具有如下几个特点：
• 多领域分词。相比于其他的中文分词工具包，此工具包同时致力于为不同领域的数据提供个性化的预训练模型。根据待分词文本的领域特点，用户可以自由地选择不同的模型。
• 我们目前支持了新闻领域，网络领域，医药领域，旅游领域，以及混合领域的分词预训练模型。
• 在使用中，如果用户明确待分词的领域，可加载对应的模型进行分词。如果用户无法确定具体领域，你也可以使用 pkuseg 默认的通用模型
• 更高的分词准确率。相比于其他的分词工具包，当使用相同的训练数据和测试数据，pkuseg 可以取得更高的分词准确率。
• 支持用户自训练模型。支持用户使用全新的标注数据进行训练。
• 支持词性标注。
安装出现问题，改用jieba分词
"""
import re

import jieba


def generate_ngrams(text, n):
    """
    生成给定文本的n-grams。

    :param text: 输入的文本字符串
    :param n: n-gram的大小
    :return: 生成的n-grams列表
    """
    # 设置特殊字符，将其替换为空字符
    separators = ["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""]
    pattern = '[' + ''.join(separators) + ']'
    text = re.sub(pattern, '', text)
    # 将文本转换为小写并分割成单词
    # tokens = text.lower().split()
    # jieba精确模式【原文中的每个字相应只出现一次【‘我是中国人’-->我，是，中国人】；对应的是全模式【‘我是中国人’-->我，是，中国，中国人】】分词
    tokens = jieba.lcut(text, cut_all=False)

    # 初始化n-grams列表
    ngrams = []

    # 获取n-gram的起始索引范围
    max_index = len(tokens) - n + 1

    # 遍历文本，生成n-grams
    for i in range(max_index):
        # 提取当前n-gram的单词
        gram = ' '.join(tokens[i:i + n])
        # 将n-gram添加到列表中
        ngrams.append(gram)

    return ngrams


# 示例用法
if __name__ == "__main__":
    text = "这是一个关于n-gram生成器的示例文本，用于演示如何生成n-grams。"
    n = 3

    # 生成n-grams
    ngrams = generate_ngrams(text, n)

    # 打印生成的n-grams
    print(f"生成的{n}-grams如下：")
    for ngram in ngrams:
        print(ngram)
