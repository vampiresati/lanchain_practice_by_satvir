from duckduckgo_search import DDGS
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
query="coffee news last week organic coffee"
queries = [
    "coffee news",
    "organic coffee",
    "coffee industry",
    "specialty coffee",
]



def search_coffee_news(query, max_results=15, timelimit="w"):
    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=max_results,
                    timelimit=timelimit
                )
            )
        return results

    except Exception as e:
        print(f"Search failed: {e}")
        return []


def generate_newsletter(search_results):
    prompt = f"""
    You are a professional newsletter writer for an organic coffee business.
    Below are the search results for this week's coffee news. Each result includes the body and the 
    href. 

    SEARCH RESULTS:
    {search_results}

    Write the article in markdown format.

    Rules:
    - Focus on the positive news about coffee
    - Provide references using the href from the SEARCH RESULTS
    """
    # model = init_chat_model("gemini-3-flash-preview", model_provider="google_genai", temperature=0.5)
    model = init_chat_model("gemini-3.1-flash-lite", model_provider="google_genai")
    response = model.invoke(prompt)
    if isinstance(response.content, list):
        return response.content[0]['text']
    return response.content

# output = search_coffee_news(queries)
# print(output)
output = []
for query in queries:
    results=search_coffee_news(query)
    output.append(results)

article = generate_newsletter(output)


with open('coffe_article.md', 'w') as file:
    file.write(article)

print(article)




from langchain_community.tools import DuckDuckGoSearchResults

# def search_coffee_news(query):
#     search = DuckDuckGoSearchResults()
#     results = search.invoke(query)
#     return results
#
# results=search_coffee_news(query)
# print(results)

# def search_coffee_news(queries, max_results=5, timelimit="w"):
#     all_results = []
#
#     with DDGS() as ddgs:
#         for q in queries:
#             results = list(
#                 ddgs.text(
#                     q,
#                     max_results=max_results,
#                     timelimit=timelimit,
#                 )
#             )
#             all_results.extend(results)
#
#     print(f"Found {len(all_results)} results")
#     return all_results
