import json
import os
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from requests.auth import HTTPBasicAuth


load_dotenv()


# ============================================================
# Jira client
# ============================================================

class JiraClient:
    def __init__(
        self,
        jira_url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        project_key: Optional[str] = None,
    ) -> None:
        self.jira_url = (
            jira_url or os.environ["JIRA_URL"]
        ).rstrip("/")

        self.email = email or os.environ["JIRA_EMAIL"]
        self.api_token = api_token or os.environ["JIRA_API_TOKEN"]
        self.project_key = project_key or os.getenv("JIRA_PROJECT_KEY")

        self.auth = HTTPBasicAuth(
            self.email,
            self.api_token,
        )

        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @staticmethod
    def text_to_adf(text: str) -> dict[str, Any]:
        """
        Convert plain text into Atlassian Document Format.
        """

        paragraphs = []

        for line in text.splitlines():
            if line.strip():
                paragraphs.append(
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": line,
                            }
                        ],
                    }
                )
            else:
                paragraphs.append(
                    {
                        "type": "paragraph",
                        "content": [],
                    }
                )

        if not paragraphs:
            paragraphs.append(
                {
                    "type": "paragraph",
                    "content": [],
                }
            )

        return {
            "type": "doc",
            "version": 1,
            "content": paragraphs,
        }

    @classmethod
    def adf_to_text(cls, node: Any) -> str:
        """
        Convert Atlassian Document Format into plain text.
        """

        if node is None:
            return ""

        if isinstance(node, str):
            return node

        if isinstance(node, list):
            return "".join(
                cls.adf_to_text(item)
                for item in node
            )

        if not isinstance(node, dict):
            return str(node)

        node_type = node.get("type")

        if node_type == "text":
            return node.get("text", "")

        if node_type == "hardBreak":
            return "\n"

        content = node.get("content", [])

        text = "".join(
            cls.adf_to_text(item)
            for item in content
        )

        if node_type in {
            "paragraph",
            "heading",
            "blockquote",
            "listItem",
        }:
            return text + "\n"

        return text

    def request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> requests.Response:
        """
        Send a request to Jira and raise a useful error on failure.
        """

        url = f"{self.jira_url}{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                auth=self.auth,
                headers=self.headers,
                timeout=30,
                **kwargs,
            )
        except requests.RequestException as error:
            raise RuntimeError(
                f"Could not connect to Jira: {error}"
            ) from error

        if not response.ok:
            raise RuntimeError(
                f"Jira API request failed\n"
                f"Method: {method}\n"
                f"URL: {url}\n"
                f"Status: {response.status_code}\n"
                f"Response: {response.text}"
            )

        return response

    def create_ticket(
        self,
        summary: str,
        description: str,
        issue_type: str = "Task",
        project_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a Jira ticket.
        """

        selected_project = project_key or self.project_key

        if not selected_project:
            raise ValueError(
                "Project key is missing. "
                "Set JIRA_PROJECT_KEY in the .env file."
            )

        payload = {
            "fields": {
                "project": {
                    "key": selected_project,
                },
                "summary": summary,
                "description": self.text_to_adf(description),
                "issuetype": {
                    "name": issue_type,
                },
            }
        }

        response = self.request(
            method="POST",
            endpoint="/rest/api/3/issue",
            json=payload,
        )

        result = response.json()
        issue_key = result["key"]

        return {
            "id": result["id"],
            "key": issue_key,
            "url": f"{self.jira_url}/browse/{issue_key}",
        }

    def read_ticket(
        self,
        issue_key: str,
    ) -> dict[str, Any]:
        """
        Read a Jira ticket.
        """

        response = self.request(
            method="GET",
            endpoint=f"/rest/api/3/issue/{issue_key}",
            params={
                "fields": (
                    "summary,description,status,issuetype,"
                    "priority,assignee,reporter,labels,"
                    "created,updated"
                )
            },
        )

        issue = response.json()
        fields = issue.get("fields", {})

        status = fields.get("status")
        issue_type = fields.get("issuetype")
        priority = fields.get("priority")
        assignee = fields.get("assignee")
        reporter = fields.get("reporter")

        return {
            "id": issue.get("id"),
            "key": issue.get("key"),
            "url": (
                f"{self.jira_url}/browse/"
                f"{issue.get('key')}"
            ),
            "summary": fields.get("summary"),
            "description": self.adf_to_text(
                fields.get("description")
            ).strip(),
            "status": (
                status.get("name")
                if status
                else None
            ),
            "issue_type": (
                issue_type.get("name")
                if issue_type
                else None
            ),
            "priority": (
                priority.get("name")
                if priority
                else None
            ),
            "assignee": (
                assignee.get("displayName")
                if assignee
                else None
            ),
            "reporter": (
                reporter.get("displayName")
                if reporter
                else None
            ),
            "labels": fields.get("labels", []),
            "created": fields.get("created"),
            "updated": fields.get("updated"),
        }

    def add_comment(
        self,
        issue_key: str,
        comment: str,
    ) -> dict[str, Any]:
        """
        Add a comment to a Jira ticket.
        """

        response = self.request(
            method="POST",
            endpoint=(
                f"/rest/api/3/issue/"
                f"{issue_key}/comment"
            ),
            json={
                "body": self.text_to_adf(comment),
            },
        )

        result = response.json()
        author = result.get("author", {})

        return {
            "comment_id": result.get("id"),
            "issue_key": issue_key,
            "author": author.get("displayName"),
            "comment": self.adf_to_text(
                result.get("body")
            ).strip(),
            "created": result.get("created"),
        }

    def read_comments(
        self,
        issue_key: str,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Read comments from a Jira ticket.
        """

        response = self.request(
            method="GET",
            endpoint=(
                f"/rest/api/3/issue/"
                f"{issue_key}/comment"
            ),
            params={
                "maxResults": max_results,
                "orderBy": "created",
            },
        )

        result = response.json()
        comments = []

        for item in result.get("comments", []):
            author = item.get("author", {})

            comments.append(
                {
                    "id": item.get("id"),
                    "author": author.get("displayName"),
                    "body": self.adf_to_text(
                        item.get("body")
                    ).strip(),
                    "created": item.get("created"),
                    "updated": item.get("updated"),
                }
            )

        return comments


# ============================================================
# Create Jira client
# ============================================================

jira = JiraClient()


# ============================================================
# LangChain Jira tools
# ============================================================

@tool
def create_jira_ticket(
    summary: str,
    description: str,
    issue_type: str = "Task",
) -> str:
    """
    Create a Jira ticket.

    Use issue_type values such as Task, Bug, or Story.
    """

    try:
        result = jira.create_ticket(
            summary=summary,
            description=description,
            issue_type=issue_type,
        )

        return json.dumps(
            {
                "success": True,
                "ticket": result,
            },
            indent=2,
        )

    except Exception as error:
        return json.dumps(
            {
                "success": False,
                "error": str(error),
            },
            indent=2,
        )


@tool
def read_jira_ticket(issue_key: str) -> str:
    """
    Read a Jira ticket using its issue key, such as PROJ-25.
    """

    try:
        result = jira.read_ticket(issue_key)

        return json.dumps(
            {
                "success": True,
                "ticket": result,
            },
            indent=2,
        )

    except Exception as error:
        return json.dumps(
            {
                "success": False,
                "error": str(error),
            },
            indent=2,
        )


@tool
def add_jira_comment(
    issue_key: str,
    comment: str,
) -> str:
    """
    Add a comment to an existing Jira ticket.
    """

    try:
        result = jira.add_comment(
            issue_key=issue_key,
            comment=comment,
        )

        return json.dumps(
            {
                "success": True,
                "result": result,
            },
            indent=2,
        )

    except Exception as error:
        return json.dumps(
            {
                "success": False,
                "error": str(error),
            },
            indent=2,
        )


@tool
def read_jira_comments(issue_key: str) -> str:
    """
    Read all comments from a Jira ticket.
    """

    try:
        result = jira.read_comments(issue_key)

        return json.dumps(
            {
                "success": True,
                "issue_key": issue_key,
                "comments": result,
            },
            indent=2,
        )

    except Exception as error:
        return json.dumps(
            {
                "success": False,
                "error": str(error),
            },
            indent=2,
        )


# ============================================================
# Local Qwen model through Ollama
# ============================================================

llm = init_chat_model(
    "qwen3:4b",
    model_provider="ollama",
    temperature=0.2,
)


# ============================================================
# Create agent
# ============================================================

agent = create_agent(
    model=llm,
    tools=[
        create_jira_ticket,
        read_jira_ticket,
        add_jira_comment,
        read_jira_comments,
    ],
    system_prompt=(
        "You are a Jira assistant. "
        "Use the available Jira tools to create tickets, read tickets, "
        "add comments, and read comments. "
        "An issue key looks like PROJ-25. "
        "If an operation needs an issue key and the user did not provide "
        "one, ask the user for it. "
        "Do not invent issue keys. "
        "Never claim an operation succeeded unless the tool returned "
        "success=true."
    ),
)


# ============================================================
# Run agent
# ============================================================

user_input = (
    "Jira ticket moved today to alpha testing."
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_input,
            }
        ]
    }
)


# Print only the final assistant response
final_message = result["messages"][-1]

if isinstance(final_message.content, str):
    print(final_message.content)
else:
    print(json.dumps(final_message.content, indent=2))