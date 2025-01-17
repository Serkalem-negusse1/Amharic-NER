import requests
from bs4 import BeautifulSoup
import json

def scrape_telegram_channel(channel_url, output_file="data/telegram_posts.json"):
    """
    Scrapes a Telegram channel and saves posts to a JSON file.
    
    Parameters:
        channel_url (str): The URL of the Telegram channel.
        output_file (str): File to save the scraped data.
    """
    print(f"Scraping data from {channel_url}...")
    
    try:
        response = requests.get(channel_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            data = []
            
            # Extract post content (adjust selectors based on Telegram structure)
            for post in soup.find_all("div", class_="tgme_widget_message_text"):
                text = post.get_text(strip=True)
                data.append({"content": text})
            
            # Save data to JSON
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            print(f"Scraped {len(data)} posts and saved to {output_file}")
        else:
            print(f"Failed to fetch the page. Status code: {response.status_code}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    channel_url = "https://t.me/aradabrand2"
    scrape_telegram_channel(channel_url)
