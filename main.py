from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

@tool
def search(query: str) -> str:
    """
    Инструмент поиска в интернет
    Args:
        query: Запрос для поиска
    Returns:
        Результат поиска
    """
    print(f"Поиск по {query}")
    return "Погода в Токио солнечная"

llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Какая погода в Токио?")})
    print(result)


if __name__ == "__main__":
    main()
