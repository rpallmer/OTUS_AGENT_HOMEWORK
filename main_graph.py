from typing import TYPE_CHECKING

from dotenv import load_dotenv

# Импорты LangChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

# Импорты LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# RAG
from agent_rag_qdrant import rag_chain

# Count
from fun_billing_api_cli import run_billing_info_via_ocli


if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig

# --- НАСТРОЙКА ---
load_dotenv()
tool_rag_chain = rag_chain
tool_fun_billing_api=run_billing_info_via_ocli

@tool
def lookup_policy(query: str) -> str:
    """
    Используй этот инструмент для поиска общей информации в базе знаний.
    Применяется, когда запрос клиента не содержит конкретных идентификаторов (например, номера лицевого счёта),
    и требуется найти общую информацию (например, о переезде, оформлении, документах).
    Вход: текст запроса пользователя, сформулированный как поисковый запрос.
    """
    return(rag_chain.invoke({"question": query}))

@tool
def billing_info(type_data: str, personal_account: int)-> str:
    """
    Используй этот инструмент для получения значений показаний, начислений по лицевому счету 
    """
    return tool_fun_billing_api(type_data,personal_account)

tools = [lookup_policy,billing_info]

# 2. Модель
llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

# 3. Память
memory = MemorySaver()


# --- ГРАФ ---


def call_model(state: MessagesState):
    """Узел агента"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=memory)


# --- ИНТЕРФЕЙС ---


def main():
    print("🤖 Ассистент готов к работе! (Введите 'q' для выхода)")

    # 1. Инициализируем хендлер
    try:
        langfuse_handler = CallbackHandler()
        print("✅ Langfuse мониторинг подключен")
    except Exception as e:
        print(f"⚠️ Ошибка подключения Langfuse: {e}")
        langfuse_handler = None

    # 2. Добавляем его в конфиг
    callbacks: list[BaseCallbackHandler] = [langfuse_handler] if langfuse_handler else []

    config: RunnableConfig = {
        "configurable": {"thread_id": "session_1"},
        "callbacks": callbacks,
    }
    
    sys_msg = SystemMessage(
        content=(
            "Ты — ассистент службы поддержки клиентов. Твоя задача — анализировать запросы клиентов и предоставлять точные, полезные и вежливые ответы.\n"
            "\n"
            "**Инструкции:**\n"
            "\n"
            "1.  **Проанализируй** текст запроса пользователя.\n"
            "2.  **Определи**, к какому типу относится запрос, используя следующие критерии:\n"
            "    *   **Тип 1 (Корректный, без внешних данных):** Запрос содержит информацию, связанную с оформлением, переоформлением, расторжением, заключением договоров на коммунальные услуги, открытием новых лицевых счетов, расторжением договоров и необходимыми документами для этих действий, при этом он **не содержит** идентификатор, связанный с конкретным клиентом или объектом обслуживания (например, **номер лицевого счёта**, **номер обращения/заявки**, **конкретный адрес объекта**, **ФИО владельца счёта**, которые требуют обращения к внешней системе). Запрос может быть обобщённым или касаться тем, по которым есть информация в базе знаний. Такой запрос **может быть обработан с использованием локальных инструментов**, таких как **поиск по базе знаний (RAG)**, для предоставления общей информации.\n"
            "    *   **Тип 2 (Корректный, с внешними данными):** Запрос ясен и **содержит** **идентификатор**, связанный с конкретным клиентом или объектом обслуживания, требующий обращения к внешней системе для получения детальной информации. Примеры таких идентификаторов:\n"
            "        *   Явное упоминание \"Лицевой(й) счёт(а)\" с числовым значением (например, \"Лицевой счет объекта: 7709004\").\n"
            "        *   Упоминание \"Обращение\" или \"Заявка\" с числовым номером (например, \"обращение № 11693-3115-КР25\").\n"
            "        *   Указание на конкретный объект недвижимости с адресом, связанным с известным счётом.\n"
            "        *   Упоминание конкретного владельца счёта / клиента (ФИО, статус собственника и т.п.).\n"
            "    *   **Тип 3 (Некорректный):** Запрос неясен, отсутствует осмысленное сообщение или **вопрос не относится к тематике по расчетам с потребителями за тепловую или электрическую энергию и услуги ЖКХ**.\n"
            "\n"
            "3.  **Выполни соответствующее действие в зависимости от типа:**\n"
            "\n"
            "    *   **Если Тип 1 (Нет ключевого идентификатора):**\n"
            "        *   Вызови инструмент `lookup_policy` с текстом запроса пользователя в качестве `query`. Этот инструмент поможет найти общую информацию в базе знаний.\n"
            "        *   **После получения результата** от `lookup_policy`, **проанализируй** его и **сформируй** вежливый и информированный ответ пользователю, основываясь на найденной информации.\n"
            "\n"
            "    *   **Если Тип 2 (Есть ключевой идентификатор):**\n"
            "        *   **Проанализируй** запрос и **извлеки**:\n"
            "            *   **Тип запрашиваемых данных:** Определи, хочет ли клиент информацию о \"Начислениях\" или \"Показаниях\". Обрати внимание на слова вроде \"начисление\", \"долг\", \"перерасчет\", \"квитанция\" (может указывать на \"Начисления\"); \"показание\", \"счетчик\" (может указывать на \"Показания\").\n"
            "            *   **Номер лицевого счёта:** Извлеки **численное значение**, связанное с \"лицевой счёт\", \"счёт\", \"номер счёта\".\n"
            "        *   Вызови инструмент `billing_info`, передав **извлечённые** значения. Этот инструмент реализован как вызов команды `ocli`.\n"
            "        *   **После получения результата** от инструмента `billing_info`, **проанализируй** его и **сформируй** ответ пользователю, предоставив запрошенную информацию, связанную с его счётом.\n"
            "\n"
            "    *   **Если Тип 3 (Некорректный):**\n"
            "        *   **Не вызывай** никакие инструменты.\n"
            "        *   **Сразу сформируй** ответ: \"Извините, я не могу ответить на ваш вопрос. Пожалуйста, уточните или задайте другой вопрос, относящийся к предоставлению коммунальных услуг, тепловой и электрической энергии.\"\n"
            "\n"
            "**Инструменты:**\n"
            "- `lookup_policy(query: string)` - Используй для поиска общей информации в базе знаний. Вход: текст запроса пользователя.\n"
            "- `billing_info(type-data: str, personal-account: int)` - Используй для получения информации о конкретном лицевом счёте из внешней системы через `ocli`. Вход:\n"
            "    - `type-data`: тип запрашиваемых данных, одно из значений: \"Начисления\", \"Показания\".\n"
            "    - `personal-account`: номер лицевого счёта (целое число).\n"
            "- [Другие инструменты, связанные с `ocli`, если есть]\n"
            "\n"
            "**Отвечай на русском языке.**"
        )
    )

    while True:
        try:
            user_input = input("\nВы: ")
            if user_input.lower() in ["q", "exit", "quit"]:
                print("До свидания!")
                break

            print("⏳ Агент думает...", end="", flush=True)

            inputs: MessagesState = {
                "messages": [sys_msg, HumanMessage(content=user_input)],
            }

            for event in app.stream(inputs, config=config):
                if "agent" in event:
                    print(".", end="", flush=True)
                if "tools" in event:
                    print(" [Вызов Инструментов] ", end="", flush=True)

            snapshot = app.get_state(config)
            if snapshot.values["messages"]:
                last_message = snapshot.values["messages"][-1]
                if hasattr(last_message, "content"):
                    print(f"\n\n🤖 Ассистент:\n{last_message.content}")

        except KeyboardInterrupt:
            print("\nВыход...")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
