#!/usr/bin/env python3
"""
liquidLapse.py

This script loads a specified URL, waits for the heatmap canvas to render,
captures the canvas image, and saves it with a timestamped filename (optionally including
the current Bitcoin price) in a specified output folder.
Configuration settings (such as URL, output folder, headless mode, whether to include
the BTC price in the filename, and the BTC price API URL) are loaded from a YAML config file.
"""

import os
import base64
import re
import yaml
import logging
import requests
import tempfile
import shutil
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_btc_price(btc_price_url):
    """
    Fetch the current Bitcoin price (in USD) from the specified API URL.
    Returns the price as a string (or None on error).
    """
    try:
        response = requests.get(btc_price_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Assume the JSON response is in the CoinGecko format.
        btc_price = data['bitcoin']['usd']
        return str(btc_price)
    except Exception as e:
        logging.error(f"Error fetching BTC price: {e}")
        return None


def setup_driver(headless=True):
    """
    Setup and return a Selenium Chrome driver.
    Uses webdriver-manager to automatically handle ChromeDriver installation.
    """
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    # Create a unique temporary directory for Chrome user data
    temp_dir = tempfile.mkdtemp()
    options.add_argument(f"--user-data-dir={temp_dir}")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver, temp_dir


def dismiss_popups(driver):
    """
    Dismiss any popups (e.g., cookie consent) that may block the page.
    Adjust the XPATH or selector as needed.
    """
    try:
        consent_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Consent")]'))
        )
        consent_btn.click()
        logging.info("Consent popup dismissed.")
    except Exception:
        logging.info("No consent popup detected.")


def capture_canvas_snapshot(driver):
    """
    Locate the heatmap canvas element and extract its image data.
    Uses a flexible CSS selector that matches any canvas whose data-zr-dom-id starts with "zr_".
    Returns the binary image data.
    """
    try:
        canvas = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'canvas[data-zr-dom-id^="zr_"]'))
        )
        data_url = driver.execute_script("return arguments[0].toDataURL('image/png');", canvas)
        image_data = re.sub('^data:image/.+;base64,', '', data_url)
        return base64.b64decode(image_data)
    except Exception as e:
        logging.error(f"Error capturing canvas: {e}")
        return None


def save_snapshot(image_bytes, output_folder, include_price=False, btc_price_url=None):
    """
    Save the binary image data to a PNG file with a timestamped filename.
    If include_price is True, fetch the current BTC price from btc_price_url and append it to the filename.
    """
    if image_bytes is None:
        logging.error("No image data to save.")
        return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    price_str = ""
    if include_price:
        # Use the provided URL or default to CoinGecko if not set.
        if not btc_price_url:
            btc_price_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        btc_price = get_btc_price(btc_price_url)
        if btc_price is not None:
            price_str = f"_BTC-{btc_price}"
        else:
            price_str = "_BTC-NA"
    filename = f"heatmap_{timestamp}{price_str}.png"
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, filename)
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    logging.info(f"Snapshot saved to {out_path}")


def main():
    # Load configuration
    config = load_config()
    url = config.get("url")
    output_folder = config.get("output_folder", "heatmap_snapshots")
    headless = config.get("headless", True)
    include_price = config.get("include_price_in_filename", False)
    btc_price_url = config.get("btc_price_url")  # New key for BTC price API URL

    logging.info("Starting liquidLapse...")
    driver, temp_dir = setup_driver(headless=headless)

    try:
        logging.info(f"Navigating to {url}")
        driver.get(url)
        driver.implicitly_wait(10)
        dismiss_popups(driver)
        image_bytes = capture_canvas_snapshot(driver)
        save_snapshot(image_bytes, output_folder, include_price, btc_price_url)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        driver.quit()
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
