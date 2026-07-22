import os
import re
from dataclasses import dataclass
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")


def validate_environment() -> None:
    """Check that all required MySQL environment variables exist."""

    required_variables = {
        "MYSQL_USER": MYSQL_USER,
        "MYSQL_PASSWORD": MYSQL_PASSWORD,
        "MYSQL_HOST": MYSQL_HOST,
        "MYSQL_PORT": MYSQL_PORT,
        "MYSQL_DATABASE": MYSQL_DATABASE,
    }

    missing_variables = [
        name
        for name, value in required_variables.items()
        if value is None or value.strip() == ""
    ]

    if missing_variables:
        raise ValueError(
            "Missing required variables in .env: "
            + ", ".join(missing_variables)
        )


validate_environment()


# ============================================================
# 2. Build MySQL connection URL
# ============================================================

encoded_user = quote_plus(MYSQL_USER)
encoded_password = quote_plus(MYSQL_PASSWORD)

database_url = (
    f"mysql+pymysql://{encoded_user}:{encoded_password}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)


# ============================================================
# 3. Connect LangChain to MySQL
# ============================================================

try:
    db = SQLDatabase.from_uri(
        database_url,
        sample_rows_in_table_info=2,
    )

    print("Database connection successful")
    print("Dialect:", db.dialect)
    print("Database:", MYSQL_DATABASE)

except Exception as error:
    print("Could not connect to MySQL.")
    print("Error:", error)
    raise


# ============================================================
# 4. Runtime context
# ============================================================

@dataclass
class RuntimeContext:
    db: SQLDatabase


# ============================================================
# 5. SQL validation
# ============================================================

BLOCKED_SQL_PATTERN = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|REPLACE|"
    r"TRUNCATE|GRANT|REVOKE|CALL|EXECUTE|LOAD|"
    r"LOCK|UNLOCK|SET|USE|RENAME|OPTIMIZE|ANALYZE|"
    r"INTO\s+OUTFILE|INTO\s+DUMPFILE"
    r")\b",
    re.IGNORECASE,
)


def validate_sql_query(query: str) -> str:
    """Allow only one read-only SELECT or WITH query."""

    cleaned_query = query.strip()
    cleaned_query = cleaned_query.rstrip(";").strip()

    if not cleaned_query:
        raise ValueError("The SQL query is empty.")

    if ";" in cleaned_query:
        raise ValueError(
            "Only one SQL statement is allowed per tool call."
        )

    if not re.match(
        r"^(SELECT|WITH)\b",
        cleaned_query,
        re.IGNORECASE,
    ):
        raise ValueError(
            "Only SELECT or WITH...SELECT queries are allowed."
        )

    if BLOCKED_SQL_PATTERN.search(cleaned_query):
        raise ValueError(
            "The query contains a blocked SQL operation."
        )

    return cleaned_query


def add_default_limit(query: str) -> str:
    """Add LIMIT 5 when the query does not already contain LIMIT."""

    if re.search(r"\bLIMIT\s+\d+", query, re.IGNORECASE):
        return query

    return f"{query} LIMIT 5"


# ============================================================
# 6. SQL execution tool
# ============================================================

@tool
def execute_sql(
    query: str,
    runtime: ToolRuntime[RuntimeContext],
) -> str:
    """
    Execute one read-only MySQL SELECT query.

    Only SELECT and WITH...SELECT statements are permitted.
    """

    try:
        safe_query = validate_sql_query(query)
        limited_query = add_default_limit(safe_query)

        print("\nExecuting SQL:")
        print(limited_query)

        result = runtime.context.db.run(limited_query)

        if not result:
            return "The query completed successfully but returned no rows."

        return str(result)

    except Exception as error:
        return f"Error: {error}"


# ============================================================
# 7. Agent system prompt
# ============================================================

available_tables = db.get_usable_table_names()

SYSTEM_PROMPT = f"""
You are a careful MySQL database analyst.

Database name:
{MYSQL_DATABASE}

Available tables:
{", ".join(available_tables)}

Rules:

1. Use the execute_sql tool whenever database data is required.

2. Use MySQL syntax, not SQLite syntax.

3. Send exactly one SQL query in each execute_sql tool call.

4. Only execute SELECT or WITH...SELECT queries.

5. Never execute:
   INSERT
   UPDATE
   DELETE
   ALTER
   DROP
   CREATE
   REPLACE
   TRUNCATE
   GRANT
   REVOKE
   CALL
   LOAD
   SET
   USE

6. Limit normal query results to 5 rows unless the user explicitly
   asks for a different number.

7. Prefer explicit column names instead of SELECT *.

8. Before querying an unfamiliar table, inspect its structure with:

   SELECT
       COLUMN_NAME,
       DATA_TYPE,
       IS_NULLABLE,
       COLUMN_KEY
   FROM INFORMATION_SCHEMA.COLUMNS
   WHERE TABLE_SCHEMA = '{MYSQL_DATABASE}'
     AND TABLE_NAME = 'table_name'
   ORDER BY ORDINAL_POSITION

9. Inspect table columns before creating joins when relationships
   are unknown.

10. If execute_sql returns Error:, correct the query and try again.

11. Do not invent table names or column names.

12. Explain query results clearly.

13. Never reveal database credentials.
"""


# ============================================================
# 8. Initialize local Ollama model
# ============================================================

model = init_chat_model(
    model="qwen3:4b",
    model_provider="ollama",
    temperature=0,
)


# ============================================================
# 9. Create agent
# ============================================================

agent = create_agent(
    model=model,
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=RuntimeContext,
)
# ============================================================
# 10. Ask database
# ============================================================

def ask_database(question: str) -> str:
    """Send a question to the MySQL agent."""

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        context=RuntimeContext(db=db),
    )

    final_message = result["messages"][-1]

    if isinstance(final_message.content, str):
        return final_message.content

    return str(final_message.content)


# ============================================================
# 11. Command-line chat
# ============================================================

def main() -> None:
    print("\nMySQL Ollama agent is ready.")
    print("Model: qwen3:4b")
    print("Type exit or quit to stop.\n")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            if not question:
                continue

            answer = ask_database(question)

            print("\nAgent:")
            print(answer)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        except Exception as error:
            print("\nAgent error:")
            print(error)
            print()


if __name__ == "__main__":
    # Save the agent graph
    try:
        png = agent.get_graph().draw_mermaid_png()

        with open("agent_graph.png", "wb") as file:
            file.write(png)

        print("Saved to agent_graph.png")

    except Exception as error:
        print("Could not generate agent graph:", error)

    question = "Which table has the device_name?"

    input_data = {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
    }

    for step in agent.stream(
        input_data,
        context=RuntimeContext(db=db),
        stream_mode="values",
    ):
        messages = step.get("messages", [])

        if messages:
            messages[-1].pretty_print()