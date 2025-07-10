import os
from crewai import LLM, Agent, Process, Task, Crew

from crewai_tools import FirecrawlScapeWebsiteTool

llm=LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.2,
    api_key=os.environ.get("GEMINI_API_KEY"),
)

tools=[FirecrawlScapeWebsiteTool(os.environ.get("FIRECRAWL_API_KEY"))]


###### Define agents

# Agent for scraping blog content
blog_scraper=Agent(
    name="Blog Scraper",
    role="Web Content Researcher",
    goal="Extract complete and accurate information from a blog URLs",
    backstory="You are an expert web researcher specialized in extracting main content from blogs while filtering out ads and navigation elements.",
    llm=llm,
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

###### Define tasks

# Task for scraping blog content
def scrape_blog_task(url):
    return Task(
        description=(
            f"Scrape the blog content from the provided URL: {url} using FirecrawlScapeWebsiteTool."
             "Extract main article text, filtering out navigation/ads. Always use FirecrawlScrapeWebsiteTool."
        )
        expected_output="Full text content of the blog post in markdown format",
        agent=blog_scraper,
    )

# Task for summarizing blog content
def summarize_blog_task(scrape_task):
    return Task(
        description="Create comprehensive summary of scraped blog content for generating AI podcast episode",
        expected_output=(
            "Concise summary (500-700 words) with key points, insights, and important details. "
            "The summary will be used to generate an AI podcast episode using a text to speech model. "
            "Create summary suitable for podcast format, focusing on clarity and engagement."
            "Do not include that this is a blog summary or mention any links or URLs."
        ),
        agent=blog_summarizer,
        context=[scrape_task], # Pass Task object, not string
    )

def create_blog_summary_crew(url):
    scrape_task=scrape_blog_task(url)
    summarize_task=summarize_blog_task(scrape_task)

    crew=Crew(
        agents=[blog_scraper, blog_summarizer],
        tasks=[scrape_task, summarize_task],
        verbose=True,
        process=Process.sequential,
    )
    return crew

def summarize_blog(url):
    crew=create_blog_summary_crew(url)
    result=crew.kickoff()
    return result.raw()

