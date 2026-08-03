from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

queries = ["coffee health benefits", "organic coffee benefits studies", "coffee antioxidants wellness"]


def search_for_articles(queries):
    tavily_search = TavilySearch(
        max_results=5,
        topic="news",
        search_depth="advanced",
        time_range="week",
        include_raw_content=False,
        include_answer=False
    )
    all_results = []

    for query in queries:
        results = tavily_search.invoke(query)
        all_results.append(results)
    return all_results


def generate_newsletter(search_results):
    prompt = f"""
    You are a professional newsletter writer for an organic coffee business.
    Below are the search results for this week's coffee news. Each result includes the content and the 
    source url. 

    SEARCH RESULTS:
    {search_results}

    Write the article in markdown format.

    Rules:
    - Focus on the positive news about coffee
    - Provide references using the URLs from the SEARCH RESULTS
    """
    model = init_chat_model("gemini-3.1-flash-lite", model_provider="google_genai", temperature=0.5)
    response = model.invoke(prompt)
    if isinstance(response.content, list):
        return response.content[0]['text']
    return response.content


output = search_for_articles(queries=queries)
article = generate_newsletter(output)
print(article)

with open('article.md', 'w') as file:
    file.write(article)