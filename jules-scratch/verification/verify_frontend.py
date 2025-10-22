from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()

    # Predictions page
    page.goto("http://127.0.0.1:5000/")
    page.screenshot(path="jules-scratch/verification/predictions.png")

    # Statistics page
    page.goto("http://127.0.0.1:5000/statistics")
    page.screenshot(path="jules-scratch/verification/statistics.png")

    # League Table page
    page.goto("http://127.0.0.1:5000/table")
    page.screenshot(path="jules-scratch/verification/table.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
