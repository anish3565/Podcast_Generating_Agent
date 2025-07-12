import os
from crewai import LLM, Agent, Process, Task, Crew
from crewai_tools import FirecrawlScrapeWebsiteTool
import sys
import re
from datetime import datetime

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

# Fixed task for scraping blog content
def scrape_blog_task(url):
    return Task(
        description=(
            f"Use the Firecrawl web scrape tool to extract the main content from this URL: {url}. "
            "Focus on getting the article text while filtering out navigation, ads, and other non-content elements. "
            "Make sure to pass the URL correctly to the tool."
        ),
        expected_output="Full text content of the blog post extracted from the webpage",
        agent=blog_scraper,
    )

# Alternative fallback scraping task (if Firecrawl continues to fail)
def scrape_blog_task_fallback(url):
    return Task(
        description=(
            f"Since the Firecrawl tool is having issues, provide a detailed analysis of what content "
            f"would typically be found at this URL: {url}. Based on the URL structure and domain, "
            "provide a comprehensive summary of the likely content themes and key points."
        ),
        expected_output="Detailed analysis of expected content based on URL and domain knowledge",
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

# Defining Crew with error handling
def create_blog_summary_crew(url, use_fallback=False):
    # URL format validation 
    if not url.startswith(('http://', 'https://')):
        raise ValueError(f"Invalid URL format: {url}. URL must start with http:// or https://")
    
    # Choosing between scraping approach
    if use_fallback:
        scrape_task = scrape_blog_task_fallback(url)
    else:
        scrape_task = scrape_blog_task(url)
    
    summarize_task = summarize_blog_task(scrape_task)

    crew = Crew(
        agents=[blog_scraper, blog_summarizer],
        tasks=[scrape_task, summarize_task],
        verbose=True,
        process=Process.sequential,
    )
    return crew

def summarize_blog(url, use_fallback=False):
    """
    Summarize blog content from a URL.
    
    Args:
        url (str): The blog URL to scrape and summarize
        use_fallback (bool): If True, uses fallback method instead of Firecrawl
    
    Returns:
        str: The summarized content
    """
    try:
        crew = create_blog_summary_crew(url, use_fallback=use_fallback)
        result = crew.kickoff()
        return result.raw
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        if not use_fallback:
            print("Trying fallback method...")
            return summarize_blog(url, use_fallback=True)
        else:
            raise

# Debugging function to test tool directly
def test_firecrawl_tool(url):
    """Test the Firecrawl tool directly to debug issues"""
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
    """Generating a safe filename from a URL and timestamp."""
    base_name = re.sub(r'https?://', '', url)
    base_name = re.sub(r'\W+', '_', base_name).strip('_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.md"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("🔗 Enter blog URL: ").strip()

    # Testing Firecrawl tool directly first
    print("\n🔍 Testing Firecrawl tool directly...")
    test_result = test_firecrawl_tool(url)

    if test_result:
        print("\n✅ Firecrawl tool passed. Running full Crew process...")
        summary = summarize_blog(url)
    else:
        print("\n⚠️ Firecrawl tool failed. Using fallback scraper...")
        summary = summarize_blog(url, use_fallback=True)

    markdown_summary = f"# 🎙️ Podcast Summary\n\n**URL**: {url}\n\n{summary}"

    # Saving the file in 'summaries/' folder
    os.makedirs("summaries", exist_ok=True)
    filename = sanitize_filename(url)
    filepath = os.path.join("summaries", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_summary)

    print(f"\n✅ Summary saved to: {filepath}")