import asyncio
from playwright.async_api import async_playwright
import os
import sys

# Use default ProactorEventLoopPolicy on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def convert_html_to_pdf(html_path, pdf_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Get absolute file URL
        abs_path = os.path.abspath(html_path)
        file_url = f"file:///{abs_path.replace(chr(92), '/')}"
        
        print(f"Loading {file_url}...")
        await page.goto(file_url, wait_until="networkidle")
        
        # Calculate the full height of the document to make it a one-pager
        dimensions = await page.evaluate('''() => {
            return {
                width: Math.max(document.body.scrollWidth, document.documentElement.scrollWidth, 1200),
                height: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)
            }
        }''')
        
        print(f"Printing to {pdf_path} as a single page...")
        await page.pdf(
            path=pdf_path, 
            width=f"{dimensions['width']}px",
            height=f"{dimensions['height'] + 100}px", # Add a little padding to the bottom
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}
        )
        await browser.close()
        print(f"Finished {pdf_path}")

async def main():
    html_file = "report.html"
    pdf_file = "Report.pdf"
    
    if os.path.exists(html_file):
        await convert_html_to_pdf(html_file, pdf_file)
    else:
        print(f"Missing {html_file}")

if __name__ == "__main__":
    asyncio.run(main())
