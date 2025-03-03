import os
import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ✅ Folder to store downloaded PDFs
SAVE_FOLDER = "raw_documents"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ✅ Headers to bypass website restrictions
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# ✅ List of sources (scraping & API-based)
SOURCES = {
    "SEC Enforcement Actions": "https://www.sec.gov/litigation/litreleases.htm",
    "CourtListener API": "https://www.courtlistener.com/api/rest/v3/opinions/?q=crypto"
}

def get_pdf_links(url):
    """Scrape and return a list of PDF links from a given webpage."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            print(f"❌ Failed to fetch {url} (Status {response.status_code})")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        pdf_links = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf"):
                if not href.startswith("http"):
                    href = requests.compat.urljoin(url, href)
                pdf_links.append(href)

        return pdf_links
    except Exception as e:
        print(f"⚠️ Error scraping {url}: {e}")
        return []

def fetch_court_cases():
    """Fetch crypto-related legal cases from CourtListener API."""
    COURTLISTENER_API = "https://www.courtlistener.com/api/rest/v3/opinions/?q=crypto"

    try:
        response = requests.get(COURTLISTENER_API, headers=HEADERS)
        if response.status_code == 200:
            print("✅ Retrieved Court Cases")
            return response.json()  # Returns JSON response
        else:
            print(f"❌ Failed to fetch CourtListener API (Status {response.status_code})")
            return None
    except Exception as e:
        print(f"⚠️ API request error: {e}")
        return None

def download_pdf(url, folder):
    """Download and save a PDF from the given URL."""
    filename = os.path.join(folder, url.split("/")[-1])

    try:
        response = requests.get(url, stream=True, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            with open(filename, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"✅ Downloaded: {filename}")
            return filename
        else:
            print(f"❌ Failed to download: {url} (Status {response.status_code})")
            return None
    except Exception as e:
        print(f"⚠️ Error downloading {url}: {e}")
        return None

if __name__ == "__main__":
    all_pdfs = []

    print("\n🔍 Scraping & Downloading Crypto-Related Reports...\n")

    for source_name, source_url in SOURCES.items():
        print(f"📡 Scraping: {source_name} ({source_url})")

        if "courtlistener.com" in source_url:
            # ✅ Use API instead of scraping
            court_cases = fetch_court_cases()
            if court_cases:
                with open("court_cases.json", "w") as json_file:
                    json.dump(court_cases, json_file, indent=4)
                print("✅ Court cases saved in 'court_cases.json'")
            continue  # Skip PDF scraping for this source

        pdf_links = get_pdf_links(source_url)
        if not pdf_links:
            print(f"⚠️ No PDFs found at {source_name}")
            continue

        print(f"📂 Found {len(pdf_links)} PDFs. Downloading...\n")

        for pdf_url in tqdm(pdf_links[:5]):  # ✅ Limit to first 5 PDFs per site
            saved_file = download_pdf(pdf_url, SAVE_FOLDER)
            if saved_file:
                all_pdfs.append({"Source": source_name, "URL": pdf_url, "Filename": saved_file})

    # ✅ Save metadata to CSV
    if all_pdfs:
        df = pd.DataFrame(all_pdfs)
        df.to_csv("downloaded_reports.csv", index=False)
        print("\n✅ All downloads complete! Metadata saved in 'downloaded_reports.csv'.")
    else:
        print("\n❌ No PDFs were downloaded.")
