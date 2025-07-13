import os
from crewai import LLM, Agent, Process, Task, Crew
from crewai_tools import FirecrawlScrapeWebsiteTool
import sys
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Setup LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.2,
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# Setup tools with proper configuration
def create_scraping_tool():
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable is required")

    return FirecrawlScrapeWebsiteTool(
        api_key=api_key,
    )

# Agent for scraping blog content
blog_scraper = Agent(
    name="Blog Scraper",
    role="Web Content Researcher",
    goal="Extract complete and accurate information from a blog URL",
    backstory="You are an expert web researcher specialized in extracting main content from blogs while filtering out ads and navigation elements.",
    llm=llm,
    tools=[create_scraping_tool()],
    verbose=True,
    allow_delegation=False,
)

# Agent for summarizing blog content
blog_summarizer = Agent(
    name="Blog Summarizer",
    role="Content Analyst",
    goal="Create concise, informative summaries capturing key points from blog content",
    backstory="You are a skilled content analyst with expertise in distilling complex information into clear summaries.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

def scrape_blog_task(url):
    try:
        tool = create_scraping_tool()
        print("[INFO] Trying Firecrawl tool for URL:", url)
        result = tool.run(url=url)
        print("[INFO] Firecrawl success. Creating task with scraped content.")
        return Task(
            description="Content fetched using Firecrawl tool.",
            expected_output=result,
            agent=blog_scraper,
        )
    except Exception as e:
        print(f"[WARNING] Firecrawl tool failed: {e}")
        print("[INFO] Falling back to BeautifulSoup for scraping.")
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)[:4000]  # Truncate to 4K chars
        except Exception as bs_e:
            raise RuntimeError(f"Both Firecrawl and fallback scraping failed: {bs_e}")

        return Task(
            description="Fallback content scraped using BeautifulSoup.",
            expected_output=text,
            agent=blog_scraper,
        )

# Task for summarizing blog content
def summarize_blog_task(scrape_task):
    return Task(
        description=(
            "Create a comprehensive summary of the scraped blog content for generating an AI podcast episode. "
            "Focus on clarity and engagement without referencing the blog or links directly. "
            "Extract key insights, main points, and important details that would be valuable for listeners."
        ),
        expected_output=(
            "Concise summary (500-700 words) with key points, insights, and important details. "
            "Format the summary to be suitable for podcast narration, focusing on clarity and engagement."
        ),
        agent=blog_summarizer,
        context=[scrape_task],
    )

# Define Crew
def create_blog_summary_crew(url):
    if not url.startswith(('http://', 'https://')):
        raise ValueError(f"Invalid URL format: {url}. URL must start with http:// or https://")

    scrape_task = scrape_blog_task(url)
    summarize_task = summarize_blog_task(scrape_task)

    crew = Crew(
        agents=[blog_scraper, blog_summarizer],
        tasks=[scrape_task, summarize_task],
        verbose=True,
        process=Process.sequential,
    )
    return crew

def summarize_blog(url):
    try:
        crew = create_blog_summary_crew(url)
        result = crew.kickoff()
        return result.raw
    except Exception as e:
        return f"[ERROR] Failed to summarize blog: {e}"

# Debugging function to test tool directly
def test_firecrawl_tool(url):
    try:
        tool = create_scraping_tool()
        result = tool.run(url=url)
        print("Tool test successful!")
        print(f"Result: {result[:200]}...")
        return result
    except Exception as e:
        print(f"Tool test failed: {str(e)}")
        return None

def sanitize_filename(url: str) -> str:
    base_name = re.sub(r'https?://', '', url)
    base_name = re.sub(r'\W+', '_', base_name).strip('_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.md"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("🔗 Enter blog URL: ").strip()

    print("\n Testing Firecrawl tool directly")
    test_result = test_firecrawl_tool(url)

    summary = summarize_blog(url)
    markdown_summary = f"# 🎙️ Podcast Summary\n\n**URL**: {url}\n\n{summary}"

    os.makedirs("summaries", exist_ok=True)
    filename = sanitize_filename(url)
    filepath = os.path.join("summaries", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_summary)

    print(f"\n✅ Summary saved to: {filepath}")