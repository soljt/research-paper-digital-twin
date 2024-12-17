import os
import requests
from bs4 import BeautifulSoup
import time

def get_paper_title(ssrn_id):
    """
    Fetch the paper title from SSRN using the abstract ID.
    
    Args:
        ssrn_id (str): The SSRN abstract ID.
    
    Returns:
        str: The paper's title, or None if not found.
    """
    url = f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={ssrn_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            meta_tag = soup.find('meta', attrs={'name': 'citation_title'})
            if meta_tag:
                return meta_tag.get('content')
        else:
            print(f"Failed to fetch URL for SSRN ID {ssrn_id}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error fetching title for SSRN ID {ssrn_id}: {e}")
    return None

def get_paper_title_with_rate_limit(ssrn_id, max_retries=5, initial_wait=10):
    """
    Fetch the paper title from SSRN using the abstract ID, with retry logic for rate limiting.
    
    Args:
        ssrn_id (str): The SSRN abstract ID.
        max_retries (int): Maximum number of retries for rate-limited requests.
        initial_wait (int): Initial wait time in seconds between retries.
    
    Returns:
        str: The paper's title, or None if not found after retries.
    """
    url = f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={ssrn_id}"
    retries = 0
    wait_time = initial_wait
    
    while retries < max_retries:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                meta_tag = soup.find('meta', attrs={'name': 'citation_title'})
                if meta_tag:
                    return meta_tag.get('content')
            elif response.status_code == 429:
                print(f"Rate limit encountered. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2  # Exponential backoff
            else:
                print(f"Failed to fetch URL for SSRN ID {ssrn_id}, status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching title for SSRN ID {ssrn_id}: {e}")
            return None

    print(f"FAIL: Exceeded maximum retries for SSRN ID {ssrn_id}.")
    return None

def rename_files(folder_path):
    """
    Rename SSRN PDF files based on their titles.

    Args:
        folder_path (str): Path to the folder containing SSRN PDF files.
    """
    for filename in os.listdir(folder_path):
        if filename.startswith("ssrn-") and filename.endswith(".pdf"):
            ssrn_id = filename.split("-")[1].split(".")[0]
            title = get_paper_title(ssrn_id)
            if title:
                sanitized_title = "".join(c if c.isalnum() or c in " -_()" else "_" for c in title)
                new_filename = f"{sanitized_title}.pdf"
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed '{filename}' to '{new_filename}'")
            else:
                print(f"Could not fetch title for SSRN ID {ssrn_id}")

# Example usage
if __name__ == "__main__":
    # folder_path = "/home/solthiessen/Desktop/llm-dirk/politicians-digital-twins/ssrn_papers/ssrn_pdfs"
    # rename_files(folder_path)
    print(get_paper_title(4917176))