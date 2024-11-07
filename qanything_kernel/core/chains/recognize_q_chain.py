from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI


class RecognizeQuestionChain:
    def __init__(self, model_name, openai_api_key, openai_api_base):
        self.chat_model = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key,
                                     openai_api_base=openai_api_base,
                                     temperature=0, model_kwargs={"top_p": 0.01, "seed": 1234})
        self.not_answer = '其他'
        self.recognize_q_system_prompt = """
假设你是极其专业的英语和汉语语言专家。你的任务是：给定一个用户输入的问题，请从语义方面，对该问题进行意图分类。

你可以假设这个问题是在用户与聊天机器人对话的背景下。

instructions:
- 意图类别有：历史、政治、体育、金融。
- 根据用户问题，判断该问题的意图类别。如果用户问题的语义涉及上述意图类列，则仅输出相关的意图类别；否则，仅输出：其他。
- 一个问题的语义可能涉及多个意图类别，对于涉及上述意图类别中的多个意图类别的问题，则输出所有相关的意图类别，且意图类别之间用英文逗号隔开。
- 请始终记住，你的任务是对用户问题从语义方面进行意图分类，而不是直接回答问题！

```
Example input:
HumanMessage: `北京明天出门需要带伞吗？`
Example output: `其他`  # 语义涉及天气，不涉及历史、政治、体育、金融，则输出：其他

Example input:
HumanMessage: `NBA历史上最高的人是谁？`
Example output: `体育` # 语义涉及历史、政治、体育、金融中的体育，则输出：体育

Example input:
HumanMessage: `NBA税收占美国体育行业税收的多少？`
Example output: `体育,金融` # 语义涉及历史、政治、体育、金融中的体育和金融，则输出：体育,金融
```

"""
        self.recognize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.recognize_q_system_prompt),
                ("human", "HumanMessage: {question}\n请从语义方面，对该问题进行意图分类。\n所属意图:"),
            ]
        )

        self.recognize_q_chain = self.recognize_q_prompt | self.chat_model | StrOutputParser()
