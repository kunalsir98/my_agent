from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY')
os.getenv('TAVILY_API_KEY')
model=ChatGroq(moel='openai/gpt-oss-20b',groq_api_key=groq_api_key)
model

tavily_tool=TavilySearch()
tavily_tool.invoke('what is machine-learning')

def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Example:
        100 * 25
        500 / 10
        (100 + 50) * 2
    """

    try:

        allowed_characters = "0123456789+-*/().% "

        if not all(
            char in allowed_characters
            for char in expression
        ):
            return "Invalid mathematical expression."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as e:

        return f"Calculation error: {str(e)}"

def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """

    return a * b

def percentage_change(
    old_value: float,
    new_value: float
) -> float:
    """
    Calculate percentage change.

    Formula:
        ((new_value - old_value) / old_value) * 100
    """

    if old_value == 0:

        return 0.0

    result = (
        (new_value - old_value)
        / old_value
    ) * 100

    return round(result, 2)

def analyze_text(text: str) -> dict:
    """
    Analyze basic statistics of a text.
    """

    words = text.split()

    sentences = [
        sentence
        for sentence in text.split(".")
        if sentence.strip()
    ]

    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences)
    }

def calculate_average(
    numbers: list[float]
) -> float:
    """
    Calculate the average of a list of numbers.
    """

    if not numbers:

        return 0.0

    return round(
        sum(numbers) / len(numbers),
        2
    )

def find_maximum(
    numbers: list[float]
) -> float:
    """
    Find the maximum value.
    """

    if not numbers:

        return 0.0

    return max(numbers)

def find_minimum(
    numbers: list[float]
) -> float:
    """
    Find the minimum value.
    """

    if not numbers:

        return 0.0

    return min(numbers)

tools=[tavily_tool,calculator,multiply,percentage_change,analyze_text,calculate_average,find_maximum,find_minimum]
tools_list=model.bind_tools(tools)
tools_list

class State(TypedDict):
    messages:Annotated[list,add_messages]

graph_response=StateGraph(State)


def research_agent(state:State):
    response=tools_list.invoke(state['messages'])
    return{'messages':response}

builder=StateGraph(State)

builder.add_node('research_agent',research_agent)
builder.add_node('tools',ToolNode(tools))

builder.add_edge(START,'research_agent')

builder.add_conditional_edges(
    'research_agent',
    tools_condition

)
builder.add_edge('tools','research_agent')

graph=builder.compile()

from langchain.messages import HumanMessage

response=graph.invoke({
    'messages':[
        HumanMessage(content='What is 2+2')
    ]
})