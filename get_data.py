import asyncio
import json

import aiohttp
from bs4 import BeautifulSoup, Tag


async def fetch_content(session, url, retries=5, delay=5):
    while retries > 0:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    print(f"Bad status code {response.status} for URL: {url}. Retrying...")
                    await asyncio.sleep(delay)
                    retries -= 1
        except aiohttp.ClientError as e:
            print(f"Request error for URL: {url}. Error: {e}. Retrying...")
            await asyncio.sleep(delay)
            retries -= 1
    print(f"Failed to fetch content from {url} after multiple retries.")
    return None


def extract_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    content_div = soup.find('div', id='content')

    if not content_div:
        return None

    result = []

    def process_element(element):
        if isinstance(element, Tag):
            if element.name == 'table':
                table_data = []
                rows = element.find_all('tr')
                for row in rows:
                    row_data = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                    if row_data:
                        table_data.append(row_data)
                if table_data:
                    result.append(table_data)
            elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']:
                header_text = element.get_text(strip=True)
                if header_text:
                    result.append(header_text)
            else:
                for child in element.contents:
                    process_element(child)

    process_element(content_div)

    return result


async def process_link(session, link, semaphore):
    async with semaphore:
        print(f"Processing {link}...")
        html_content = await fetch_content(session, link)
        if html_content:
            extracted_content = extract_content(html_content)
            if extracted_content:
                return {'url': link, 'content': extracted_content}
            else:
                print(f"No content found in <div id='content'> for URL: {link}")
        else:
            print(f"Skipping URL due to failed fetch: {link}")
    return None


async def process_links(links, output_file='data.json', max_concurrent_requests=15):
    data = []
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async with aiohttp.ClientSession() as session:
        tasks = [process_link(session, link, semaphore) for link in links]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is not None:
                data.append(result)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Data extraction completed. Saved to {output_file}.")


with open('links.txt', 'r', encoding='utf-8') as f:
    links = json.load(f)

asyncio.run(process_links(links))
