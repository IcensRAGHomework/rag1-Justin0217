import requests
import base64
from mimetypes import guess_type
from functools import lru_cache
from typing import Optional, List, Dict, Any
from model_configurations import get_model_configuration
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

history = ChatMessageHistory()

class Config:
    def __init__(self):
        self.gpt_chat_version = 'gpt-4o'
        self.gpt_config = get_model_configuration(self.gpt_chat_version)

    @lru_cache()
    def get_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            model=self.gpt_config['model_name'],
            deployment_name=self.gpt_config['deployment_name'],
            openai_api_key=self.gpt_config['api_key'],
            openai_api_version=self.gpt_config['api_version'],
            azure_endpoint=self.gpt_config['api_base'],
            temperature=self.gpt_config['temperature']
        )

class HolidayAPI:
    API_KEY = "Da9B4H14FjzvH2jw5wYjPSjiArTJ7MQc"
    BASE_URL = "https://calendarific.com/api/v2/holidays"

    @staticmethod
    def get_holiday(year: int, month: int) -> str:
        url = f"{HolidayAPI.BASE_URL}?&api_key={HolidayAPI.API_KEY}&country=tw&year={year}&month={month}"
        response = requests.get(url)
        return response.json().get('response')

class GetHolidaySchema(BaseModel):
    year: int = Field(description="specific year")
    month: int = Field(description="specific month")

class AgentManager:
    def __init__(self, config: Config):
        self.config = config
        self.tools = [self._create_holiday_tool()]

    def _create_holiday_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            name="get_holiday",
            description="Fetch holidays for specific year and month",
            func=HolidayAPI.get_holiday,
            args_schema=GetHolidaySchema
        )

    def get_agent(self) -> RunnableWithMessageHistory:
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_functions_agent(self.config.get_llm(), self.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools)
        
        def get_history() -> ChatMessageHistory:
            return history
    
        return RunnableWithMessageHistory(
            agent_executor,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

class OutputFormatter:
    @staticmethod
    def format_holiday_output(llm: BaseChatModel, data: str) -> str:
        
        response_schemas = [
            ResponseSchema(name="date", description="該紀念日的日期", type="YYYY-MM-DD"),
            ResponseSchema(name="name", description="該紀念日的名稱")
        ]

        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_messages([
            ("system", "將我提供的資料整理成指定格式,使用台灣語言,{format_instructions},有幾個答案就回答幾次,將所有答案使用台灣語言放進同個list"),
            ("human", "{data}")
        ])
        prompt = prompt.partial(format_instructions=format_instructions)
        
        return llm.invoke(prompt.format_messages(data=data)).content

    @staticmethod
    def format_result_output(llm: BaseChatModel, data: str, is_list: bool) -> str:
        schema_type = "list" if is_list else "str"
        response_schemas = [
            ResponseSchema(name="Result", description="json內的所有內容", type=schema_type)
        ]
        
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        # First stage formatting
        prompt = ChatPromptTemplate.from_messages([
            ("system", "將提供的json內容輸出成指定json格式,{format_instructions}"),
            ("human", "{question}")
        ])
        prompt = prompt.partial(format_instructions=format_instructions)
        initial_response = llm.invoke(prompt.format_messages(question=data)).content
        
        # Second stage formatting with examples
        examples = [{
            "input": """```json
                    {
                            "Result": [
                                    content
                            ]
                    }
                    ```""",
            "output": """{
                            "Result": [
                                    content
                            ]
                    }"""
        }]
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])
        
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples
        )
        
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "將我提供的文字進行處理"),
            few_shot_prompt,
            ("human", "{input}")
        ])
        
        return llm.invoke(final_prompt.invoke(input=initial_response)).content

    
class ImageProcessor:
    @staticmethod
    def local_image_to_data_url(image_path: str = './baseball.png') -> str:
        mime_type, _ = guess_type(image_path) or ('application/octet-stream', None)
        
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        return f"data:{mime_type};base64,{base64_encoded_data}"

def generate_hw01(question: str) -> str:
    config = Config()
    llm = config.get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "使用台灣語言並回答問題,用格式化的答案呈現,答案的日期包含年月日,除了答案本身以外不要回答其他語句"),
        ("human", "{question}")
    ])
    
    response = llm.invoke(prompt.format_messages(question=question)).content
    response = OutputFormatter.format_holiday_output(llm, response)
    return OutputFormatter.format_result_output(llm, response, True)

def generate_hw02(question: str) -> str:
    config = Config()
    llm = config.get_llm()
    agent_manager = AgentManager(config)
    
    response = agent_manager.get_agent().invoke({"input": question}).get('output')
    response = OutputFormatter.format_holiday_output(llm, response)
    return OutputFormatter.format_result_output(llm, response, True)

def generate_hw03(question2: str, question3: str) -> str:
    generate_hw02(question2)

    config = Config()
    llm = config.get_llm()
    agent_manager = AgentManager(config)
    agent = agent_manager.get_agent()

    response_schemas = [
        ResponseSchema(
            name="add",
            description="該紀念日是否需要加入先前的清單內,若月份相同且該紀念日不被包含在清單內則為true,否則為false",
            type="boolean"
        ),
        ResponseSchema(
            name="reason",
            description="決定該紀念日是否加入清單的理由"
        )
    ]
    
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "使用台灣語言並回答問題,{format_instructions}"),
        ("human", "{question}")
    ])
    prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())
    
    response = agent.invoke({"input": prompt.format_messages(question=question3)}).get('output')
    return OutputFormatter.format_result_output(llm, response, False)

def generate_hw04(question: str) -> str:
    config = Config()
    llm = config.get_llm()
    
    response_schemas = [
        ResponseSchema(
            name="score",
            description="圖片文字表格中顯示的指定隊伍的積分數",
            type="integer"
        )
    ]
    
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "辨識圖片中的文字表格,{format_instructions}"),
        ("user", [{"type": "image_url", "image_url": {"url": ImageProcessor.local_image_to_data_url()}}]),
        ("human", "{question}")
    ])
    
    prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())
    response = llm.invoke(prompt.format_messages(question=question)).content
    return OutputFormatter.format_result_output(llm, response, False)

def demo(question: str) -> Any:
    config = Config()
    llm = config.get_llm()
    message = HumanMessage(content=[{"type": "text", "text": question}])
    return llm.invoke([message])


#print(generate_hw01("2024年台灣10月紀念日有哪些?"))
#print(generate_hw02("台灣2024年10月的紀念日有哪些(請用JSON格式呈現)?"))
#print(generate_hw03('2024年台灣10月紀念日有哪些?', '根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？'))
#print(generate_hw04('請問中華台北的積分是多少'))