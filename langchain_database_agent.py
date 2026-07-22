import os
import re
from dataclasses import dataclass
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain.agents import create_agent
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
        missing_text = ", ".join(missing_variables)

        raise ValueError(
            f"Missing required variables in .env: {missing_text}"
        )


validate_environment()


# ============================================================
# 2. Build MySQL connection URL
# ============================================================

# Encode credentials so passwords containing @, #, :, /, etc. work.
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
    # print("Tables:", db.get_usable_table_names())

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
# 5. SQL query validation
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
    """
    Validate that the model generated one read-only SQL query.

    Only SELECT and WITH...SELECT statements are allowed.
    """

    cleaned_query = query.strip()

    # Remove one trailing semicolon.
    cleaned_query = cleaned_query.rstrip(";").strip()

    if not cleaned_query:
        raise ValueError("The SQL query is empty.")

    # Prevent multiple statements.
    if ";" in cleaned_query:
        raise ValueError(
            "Only one SQL statement is allowed per tool call."
        )

    # Only SELECT or CTE queries.
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
    """
    Add LIMIT 5 when the model did not provide a LIMIT.

    Information-schema count queries and aggregate queries can still return
    their normal single-row result.
    """

    has_limit = re.search(
        r"\bLIMIT\s+\d+",
        query,
        re.IGNORECASE,
    )

    if has_limit:
        return query

    return f"{query} LIMIT 5"


# ============================================================
# 6. MySQL execution tool
# ============================================================

@tool
def execute_sql(
    query: str,
    runtime: ToolRuntime[RuntimeContext],
) -> str:
    """
    Execute one read-only MySQL SELECT query.

    Use this tool to read data from the MySQL database.
    Only SELECT and WITH...SELECT statements are allowed.
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

SYSTEM_PROMPT = f"""
You are a careful MySQL database analyst.

You are connected to the MySQL database named:
{MYSQL_DATABASE}

The currently available tables are:
{", ".join(db.get_usable_table_names())}

Rules:

1. Use the execute_sql tool whenever the question requires database data.

2. Use MySQL syntax, not SQLite syntax.

3. Send exactly one SQL query in each execute_sql tool call.

4. Only use SELECT queries or WITH...SELECT queries.

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

6. Limit normal query results to 5 rows unless the user explicitly asks
   for a different number.

7. Prefer explicit column names instead of SELECT *.

8. Before querying an unfamiliar table, inspect its structure using:

   SELECT
       COLUMN_NAME,
       DATA_TYPE,
       IS_NULLABLE,
       COLUMN_KEY
   FROM INFORMATION_SCHEMA.COLUMNS
   WHERE TABLE_SCHEMA = '{MYSQL_DATABASE}'
     AND TABLE_NAME = 'table_name'
   ORDER BY ORDINAL_POSITION

9. When joining tables, inspect their columns first if the relationship
   is not known.

10. If execute_sql returns Error:, understand the error, correct the SQL,
    and call the tool again.

11. Do not invent table names or column names.

12. After obtaining the result, explain it clearly in plain language.

13. Do not reveal database passwords or connection credentials.
"""


# ============================================================
# 8. Create agent using local Ollama model
# ============================================================

agent = create_agent(
    model="ollama:qwen3:4b",
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=RuntimeContext,
)


# ============================================================
# 9. Ask the database agent
# ============================================================

def ask_database(question: str) -> str:
    """Send one natural-language question to the database agent."""

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
# 10. Command-line chat
# ============================================================

def main() -> None:
    print("\nMySQL Ollama agent is ready.")
    print("Model: ollama:qwen3:4b")
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
    main()