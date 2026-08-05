import json
import time
from urllib.parse import urlparse

from ddgs import DDGS
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

QUERIES = [
    '"coffee industry" news',
    '"organic coffee" news',
    '"specialty coffee" news',
    '"sustainable coffee" news',
    '"coffee farmers" positive news',
]

BLOCKED_DOMAINS = {
    "wikipedia.org",
    "simple.wikipedia.org",
}

IRRELEVANT_TERMS = {
    "coffee run",
    "coffeeroom",
    "murder",
    "celebrity",
    "tom holland",
    "zendaya",
}


def search_coffee_news(
    query: str,
    max_results: int = 10,
    timelimit: str = "w",
) -> list[dict]:
    """Search recent coffee news and return normalized results."""

    try:
        with DDGS(timeout=15) as ddgs:
            raw_results = list(
                ddgs.news(
                    query=query,
                    max_results=max_results,
                    timelimit=timelimit,
                    safesearch="moderate",
                )
            )

        normalized = []

        for item in raw_results:
            url = item.get("url") or item.get("href")
            title = item.get("title", "").strip()
            body = item.get("body", "").strip()

            if not url or not title:
                continue

            normalized.append(
                {
                    "title": title,
                    "body": body,
                    "url": url,
                    "source": item.get("source", ""),
                    "date": item.get("date", ""),
                    "search_query": query,
                }
            )

        return normalized

    except Exception as exc:
        print(f"Search failed for {query!r}: {exc}")
        return []


def is_relevant(result: dict) -> bool:
    """Apply basic relevance and source-quality filtering."""

    title = result.get("title", "")
    body = result.get("body", "")
    url = result.get("url", "")

    combined_text = f"{title} {body}".lower()

    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower().removeprefix("www.")
    except ValueError:
        return False

    if any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in BLOCKED_DOMAINS
    ):
        return False

    if any(term in combined_text for term in IRRELEVANT_TERMS):
        return False

    coffee_terms = {
        "coffee",
        "café",
        "arabica",
        "robusta",
        "roaster",
        "roastery",
        "barista",
    }

    industry_terms = {
        "organic",
        "specialty",
        "sustainable",
        "industry",
        "farmer",
        "producer",
        "production",
        "export",
        "harvest",
        "certification",
        "market",
        "roasting",
    }

    has_coffee_term = any(term in combined_text for term in coffee_terms)
    has_industry_term = any(term in combined_text for term in industry_terms)

    return has_coffee_term and has_industry_term


def collect_news(queries: list[str]) -> list[dict]:
    """Run all searches, flatten results and remove duplicate URLs."""

    unique_results = {}

    for query in queries:
        results = search_coffee_news(query)

        print(f"{query}: {len(results)} raw results")

        for result in results:
            if not is_relevant(result):
                continue

            normalized_url = result["url"].split("#", maxsplit=1)[0].rstrip("/")
            unique_results.setdefault(normalized_url, result)

        # Avoid making several requests in immediate succession.
        time.sleep(1)

    return list(unique_results.values())


def generate_newsletter(search_results: list[dict]) -> str:
    if not search_results:
        return (
            "# This Week in Organic Coffee\n\n"
            "No suitable recent coffee-industry stories were found."
        )

    search_results_json = json.dumps(
        search_results,
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
You are a professional newsletter writer for an organic coffee business.

Use only the supplied search results. Do not invent facts, dates, quotations,
companies, statistics, or URLs.

SEARCH RESULTS:
{search_results_json}

Write a polished weekly newsletter in Markdown.

Requirements:
1. Create an engaging title and a two-sentence introduction.
2. Select only genuinely relevant and positive coffee-industry stories.
3. Prioritize organic coffee, sustainability, farmers, specialty coffee,
   responsible production, innovation, and community impact.
4. Do not include general coffee definitions, product storefronts, celebrity
   stories, crime stories, forums, or unrelated stock pages.
5. Write a separate section for each selected story.
6. Clearly distinguish facts in the search snippets from your interpretation.
7. Place a Markdown source link after every story using that story's exact URL.
8. Never create a URL that is not present in SEARCH RESULTS.
9. If the supplied snippet lacks enough information for a claim, omit the claim.
10. Finish with a short optimistic conclusion.

Use this citation format:

Source: [Publication or article title](exact URL)
"""

    model = init_chat_model(
        "gemini-3.1-flash-lite",
        model_provider="google_genai",
        temperature=0.3,
    )

    response = model.invoke(prompt)

    if isinstance(response.content, str):
        return response.content

    if isinstance(response.content, list):
        text_parts = []

        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])

        return "\n".join(text_parts).strip()

    return str(response.content)


from langchain_ollama import ChatOllama

def generate_newsletter_ollama(search_results):
    prompt = f"""
    You are a professional newsletter writer for an organic coffee business.

    SEARCH RESULTS:
    {search_results}

    Write a professional markdown newsletter.

    Rules:
    - Focus on positive coffee news.
    - Use only information from the search results.
    - Include the source URL after each story.
    """

    model = ChatOllama(
        model="ollama:qwen3:4b",
        temperature=0.3,
    )

    response = model.invoke(prompt)
    return response.content

news_results = collect_news(QUERIES)

print(f"\nRelevant unique stories: {len(news_results)}")

# newsletter = generate_newsletter(news_results)
newsletter=generate_newsletter_ollama(news_results)
with open("coffee_article.md", "w", encoding="utf-8") as file:
    file.write(newsletter)

print(newsletter)