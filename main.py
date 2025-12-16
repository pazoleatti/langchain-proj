from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
).partial(format_instructions="")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions
)

agent_executor = AgentExecutor(agent = agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm

def main():
    result = chain.invoke(
        input={
            "input": "Поиск 3 вакансий java разработчика на linkedin в Узбекистане и отображение их деталей"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
