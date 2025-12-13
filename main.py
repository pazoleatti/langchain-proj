from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing import List

class Source(BaseModel):
    """Схема для источника используемого агентом"""
    url:str = Field(description="URL источника")

class AgentResponse(BaseModel):
    """Схема для ответа агента"""
    answer:str = Field(description="Ответ агента на запрос и источник")
    sources:List[Source] = Field(default_factory=list, description="Список источников используемых для генерации ответа")

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Поиск 3 вакансий java разработчика на hh.uz в Узбекистане и отображение их деталей")})
    print(result)


if __name__ == "__main__":
    main()
