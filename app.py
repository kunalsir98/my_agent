
import ast
import operator
import os
import re
from typing import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load private server credentials
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- CUSTOM UI --------------------

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at top left,
            rgba(79,70,229,.17), transparent 38%),
        #0b1020;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #273449;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg,#312e81,#172554);
    border: 1px solid #4338ca;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    font-size: 34px;
    margin: 0 0 8px 0;
}
.hero p {
    color: #cbd5e1;
    font-size: 16px;
    margin: 0;
}
div[data-testid="stChatMessage"] {
    background: rgba(17,24,39,.78);
    border: 1px solid #263449;
    border-radius: 14px;
    padding: 14px;
}
div.stButton > button {
    border-radius: 10px;
    min-height: 44px;
}
.muted {
    color: #94a3b8;
    font-size: .88rem;
}
</style>
""", unsafe_allow_html=True)


# -------------------- TOOLS --------------------

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(node):
    """Evaluate only supported arithmetic expressions."""

    if isinstance(node, ast.Expression):
        return safe_eval(node.body)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left = safe_eval(node.left)
        right = safe_eval(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Exponent is too large.")

        return BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](
            safe_eval(node.operand)
        )

    raise ValueError("Only basic arithmetic is supported.")


def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression."""

    try:
        tree = ast.parse(expression, mode="eval")
        return str(safe_eval(tree))
    except Exception as exc:
        return f"Calculation error: {exc}"


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""

    if old_value == 0:
        return 0.0

    return round(
        ((new_value - old_value) / old_value) * 100, 2
    )


def analyze_text(text: str) -> dict:
    """Count characters, words and approximate sentences."""

    sentences = [
        part
        for part in re.split(r"[.!?]+", text)
        if part.strip()
    ]

    return {
        "characters": len(text),
        "words": len(text.split()),
        "sentences": len(sentences),
    }


def calculate_average(numbers: list[float]) -> float:
    """Calculate average of a list of numbers."""
    return round(sum(numbers) / len(numbers), 2) if numbers else 0.0


def find_maximum(numbers: list[float]) -> float:
    """Find the largest number in a list."""
    return max(numbers) if numbers else 0.0


def find_minimum(numbers: list[float]) -> float:
    """Find the smallest number in a list."""
    return min(numbers) if numbers else 0.0


# -------------------- LANGGRAPH --------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@st.cache_resource(show_spinner=False)
def create_agent(model_name: str):
    """Create the agent with server-side credentials."""

    if not GROQ_API_KEY or not TAVILY_API_KEY:
        raise RuntimeError("Server credentials are not configured.")

    model = ChatGroq(
        model=model_name,
        groq_api_key=GROQ_API_KEY,
        temperature=0.2,
    )

    search_tool = TavilySearch(
        max_results=5,
        tavily_api_key=TAVILY_API_KEY,
    )

    tools = [
        search_tool,
        calculator,
        multiply,
        percentage_change,
        analyze_text,
        calculate_average,
        find_maximum,
        find_minimum,
    ]

    model_with_tools = model.bind_tools(tools)

    def research_agent(state: AgentState):
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)

    builder.add_node("research_agent", research_agent)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "research_agent")

    builder.add_conditional_edges(
        "research_agent",
        tools_condition,
    )

    builder.add_edge("tools", "research_agent")

    return builder.compile()


# -------------------- HELPERS --------------------

def message_to_text(content) -> str:
    """Convert LangChain message content to displayable text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)

        return "\n".join(part for part in parts if part)

    return str(content)


def extract_urls(messages) -> list[str]:
    """Extract HTTP(S) links found in tool results."""

    sources = []

    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue

        content = message.content

        if not isinstance(content, str):
            content = str(content)

        urls = re.findall(
            r"https?://[^\s\[\]<>'\"]+",
            content,
        )

        for url in urls:
            url = url.rstrip(".,;:)")

            if url not in sources:
                sources.append(url)

    return sources


def reset_chat():
    st.session_state.chat_messages = []
    st.session_state.display_messages = []


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## 🔎 Research Agent")

    st.markdown(
        '<p class="muted">LangGraph · Groq · Tavily</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    model_name = st.selectbox(
        "Select Model",
        [
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
        ],
    )

    if GROQ_API_KEY and TAVILY_API_KEY:
        st.success("AI service is configured")
    else:
        st.warning("AI service is temporarily unavailable")

    st.divider()

    st.markdown("### 🧰 Available Tools")
    st.markdown("""
    - 🌐 Web search
    - 🧮 Calculator
    - ✖️ Multiplication
    - 📈 Percentage change
    - 📝 Text analysis
    - 📊 Average, maximum and minimum
    """)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()

    st.caption(
        "Verify important research claims against their sources."
    )


# -------------------- MAIN HEADER --------------------

st.markdown("""
<div class="hero">
    <h1>🔎 AI Research Assistant</h1>
    <p>
        Research topics, search the web, solve calculations,
        and analyze text with an intelligent LangGraph agent.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.metric("Agent Framework", "LangGraph")
col2.metric("LLM Provider", "Groq")
col3.metric("Web Search", "Tavily")


# -------------------- SESSION STATE --------------------

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []


# -------------------- SUGGESTED PROMPTS --------------------

st.markdown("### 💡 Try asking")

suggestions = [
    "Explain machine learning with examples",
    "Calculate percentage change from 450 to 525",
    "What are the latest developments in Generative AI?",
]

suggestion_cols = st.columns(3)

for index, suggestion in enumerate(suggestions):
    if suggestion_cols[index].button(
        suggestion,
        key=f"suggestion_{index}",
        use_container_width=True,
    ):
        st.session_state.pending_prompt = suggestion


# -------------------- CHAT HISTORY --------------------

for item in st.session_state.display_messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

        if item.get("sources"):
            with st.expander("Links found in tool output"):
                for source in item["sources"]:
                    st.markdown(f"- [{source}]({source})")


# -------------------- CHAT INPUT --------------------

prompt = st.chat_input("Ask your research question...")

if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")


# -------------------- RUN AGENT --------------------

if prompt:

    if not GROQ_API_KEY or not TAVILY_API_KEY:
        st.error(
            "The AI service is temporarily unavailable. "
            "Please try again later."
        )
        st.stop()

    st.session_state.display_messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching and using tools when needed..."):

            try:
                agent = create_agent(model_name)

                # Preserve the complete LangGraph conversation,
                # including assistant tool calls and tool results.
                st.session_state.chat_messages.append(
                    HumanMessage(content=prompt)
                )

                result = agent.invoke({
                    "messages": st.session_state.chat_messages
                })

                st.session_state.chat_messages = result["messages"]

                final_message = next(
                    (
                        message
                        for message in reversed(result["messages"])
                        if isinstance(message, AIMessage)
                        and not getattr(message, "tool_calls", None)
                    ),
                    None,
                )

                if final_message is None:
                    answer = (
                        "I couldn't generate a final answer. "
                        "Please try again."
                    )
                else:
                    answer = message_to_text(
                        final_message.content
                    ).strip()

                if not answer:
                    answer = "I couldn't generate a text response."

                sources = extract_urls(result["messages"])

                st.markdown(answer)

                if sources:
                    with st.expander("Links found in tool output"):
                        for source in sources:
                            st.markdown(f"- [{source}]({source})")

                st.session_state.display_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception:
                # Do not expose raw exception details or credentials.
                st.error(
                    "The assistant couldn't complete this request. "
                    "Please try again later."
                )

                # Remove the failed user turn from graph history.
                if (
                    st.session_state.chat_messages
                    and isinstance(
                        st.session_state.chat_messages[-1],
                        HumanMessage,
                    )
                    and st.session_state.chat_messages[-1].content == prompt
                ):
                    st.session_state.chat_messages.pop()

                if (
                    st.session_state.display_messages
                    and st.session_state.display_messages[-1]["role"] == "user"
                    and st.session_state.display_messages[-1]["content"] == prompt
                ):
                    st.session_state.display_messages.pop()


st.markdown(
    '<p class="muted">AI credentials are configured privately on the server.</p>',
    unsafe_allow_html=True,
)
