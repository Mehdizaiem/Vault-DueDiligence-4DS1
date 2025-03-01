import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging
import time
import os
from dotenv import load_dotenv
import sys
import hashlib

# Configure paths
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..', '..', '..'))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    logger.warning("newspaper3k library not available. Installing or using fallback extraction.")
    NEWSPAPER_AVAILABLE = False

class CryptoNewsScraper:
    """Scraper for crypto news from reputable sources"""
    
    def __init__(self, cache_dir=None):
        """
        Initialize the crypto news scraper.
        
        Args:
            cache_dir (str, optional): Directory to store cached news articles.
                If None, uses 'Sample_Data/data_ingestion/news_cache'
        """
        self.sources = [
    {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/tag/bitcoin/",
        "article_selector": "article",  # Try simpler selector
        "link_selector": "a[href*='/markets/']",  # Look for market links
        "title_selector": "h5, h4, h3",  # Try multiple heading options
        "base_url": "https://www.coindesk.com"
    },
   {
    "name": "Cointelegraph",
    "url": "https://cointelegraph.com/tags/bitcoin",
    "article_selector": "article",  # Simpler selector
    "link_selector": "a.post-card-inline__title-link",  # Updated class
    "title_selector": "span.post-card-inline__title",  # Updated class
    "base_url": "https://cointelegraph.com"  # Add the base URL
}
]
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Set cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(project_root, "Sample_Data", "data_ingestion", "news_cache")
        else:
            self.cache_dir = cache_dir
            
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"News cache directory: {self.cache_dir}")
        
    def _get_article_content(self, url):
        """
        Extract article content from URL.
        
        Args:
            url (str): URL of the article to extract
            
        Returns:
            dict: Article content including text, date, authors, etc.
        """
        if NEWSPAPER_AVAILABLE:
            try:
                article = Article(url)
                article.download()
                article.parse()
                return {
                    "text": article.text,
                    "publish_date": article.publish_date,
                    "authors": article.authors,
                    "top_image": article.top_image
                }
            except Exception as e:
                logger.error(f"Error extracting article content from {url}: {e}")
        
        # Fallback extraction if newspaper3k fails or is not available
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find main content - this is a simplified approach
            article_text = ""
            for p in soup.find_all("p"):
                article_text += p.get_text() + "\n"
                
            return {
                "text": article_text,
                "publish_date": None,
                "authors": [],
                "top_image": ""
            }
        except Exception as e:
            logger.error(f"Fallback extraction failed for {url}: {e}")
            return {
                "text": "",
                "publish_date": None,
                "authors": [],
                "top_image": ""
            }
    
    def _create_article_id(self, url, title):
        """
        Create a unique ID for an article based on URL and title.
        
        Args:
            url (str): Article URL
            title (str): Article title
            
        Returns:
            str: MD5 hash of URL and title
        """
        content = f"{url}_{title}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def scrape_news(self, limit_per_source=5):
        """
        Scrape news from all configured sources.
        
        Args:
            limit_per_source (int): Maximum number of articles to scrape per source
            
        Returns:
            pd.DataFrame: DataFrame containing scraped articles
        """
        all_articles = []
        
        for source in self.sources:
            logger.info(f"Scraping from {source['name']}...")
            try:
                response = requests.get(source["url"], headers=self.headers, timeout=10)
                if response.status_code != 200:
                    logger.error(f"Failed to fetch from {source['name']}: Status code {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.content, "html.parser")
                articles = soup.select(source["article_selector"])
                
                count = 0
                for article in articles:
                    if count >= limit_per_source:
                        break
                    
                    try:
                        # Extract article link and title
                        link_element = article.select_one(source["link_selector"])
                        if not link_element:
                            continue
                            
                        article_url = link_element.get("href")
                        if not article_url.startswith("http"):
                            article_url = source["base_url"] + article_url
                            
                        title = article.select_one(source["title_selector"]).text.strip()
                        
                        # Create unique ID for caching
                        article_id = self._create_article_id(article_url, title)
                        cache_file = os.path.join(self.cache_dir, f"{article_id}.json")
                        
                        # Check if we've already scraped this article
                        if os.path.exists(cache_file):
                            article_data = pd.read_json(cache_file, typ='series').to_dict()
                            logger.info(f"Retrieved cached article: {title}")
                        else:
                            # Get full article content
                            article_content = self._get_article_content(article_url)
                            
                            # Create article data
                            article_data = {
                                "id": article_id,
                                "source": source["name"],
                                "title": title,
                                "url": article_url,
                                "content": article_content["text"],
                                "date": article_content["publish_date"] or datetime.now(),
                                "authors": article_content["authors"],
                                "image_url": article_content["top_image"],
                                "scraped_at": datetime.now().isoformat()
                            }
                            
                            # Cache the article
                            pd.Series(article_data).to_json(cache_file)
                            logger.info(f"Scraped new article: {title}")
                            
                            # Respect the site by waiting between requests
                            time.sleep(2)
                        
                        all_articles.append(article_data)
                        count += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing article from {source['name']}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error scraping from {source['name']}: {e}")
                continue
                
            # Be respectful of the source and wait between sources
            time.sleep(5)
        
        return pd.DataFrame(all_articles)
    
    def save_to_csv(self, df, filename=None):
        """
        Save scraped news to CSV.
        
        Args:
            df (pd.DataFrame): DataFrame containing news articles
            filename (str, optional): Output filename
                If None, uses 'Sample_Data/data_ingestion/crypto_news.csv'
                
        Returns:
            str: Path to the saved CSV file
        """
        if filename is None:
            output_dir = os.path.join(project_root, "Sample_Data", "data_ingestion", "processed")
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, "crypto_news.csv")
            
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} articles to {filename}")
        return filename
        
    def run(self, limit_per_source=10, output_file=None):
        """
        Run the scraper and save results.
        
        Args:
            limit_per_source (int): Maximum number of articles to scrape per source
            output_file (str, optional): Output file path
            
        Returns:
            pd.DataFrame: DataFrame containing scraped articles
        """
        news_df = self.scrape_news(limit_per_source)
        if not news_df.empty:
            self.save_to_csv(news_df, output_file)
        return news_df

# Example usage
if __name__ == "__main__":
    load_dotenv()  # Load environment variables
    scraper = CryptoNewsScraper()
    news_data = scraper.run(limit_per_source=5)
    print(f"Scraped {len(news_data)} articles")