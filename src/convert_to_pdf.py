import asyncio
from playwright.async_api import async_playwright
import os
import sys

# Workaround for ProactorEventLoop in Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def convert_html_to_pdf(html_path, pdf_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Get absolute file URL
        abs_path = os.path.abspath(html_path)
        file_url = f"file:///{abs_path.replace(chr(92), '/')}"
        
        print(f"Loading {file_url}...")
        await page.goto(file_url, wait_until="networkidle")
        print(f"Printing to {pdf_path}...")
        await page.pdf(path=pdf_path, format="A4", margin={"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"}, print_background=True)
        await browser.close()
        print(f"Finished {pdf_path}")

async def main():
    html_file = "scripts/generation1.html"
    pdf_file = "scripts/generation1.pdf"
    
    if os.path.exists(html_file):
        await convert_html_to_pdf(html_file, pdf_file)
    else:
        print(f"Missing {html_file}")

if __name__ == "__main__":
    asyncio.run(main())
