from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

tools = [TavilySearch()]

def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
