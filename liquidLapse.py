#!/usr/bin/env python3
"""
liquidLapse.py

This script loads a specified URL, waits for the heatmap canvas to render,
captures the canvas image, and saves it with a timestamped filename in a specified output folder.
Configuration settings (such as URL, output folder, and headless mode) are loaded from a YAML config file.
"""

import os
import base64
import re
import yaml
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import tempfile
import shutil

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
        # Wait for any canvas element with data-zr-dom-id attribute starting with "zr_"
        canvas = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'canvas[data-zr-dom-id^="zr_"]'))
        )
        # Use JavaScript to get a base64-encoded PNG from the canvas
        data_url = driver.execute_script("return arguments[0].toDataURL('image/png');", canvas)
        # Remove header "data:image/png;base64," and decode the image data
        image_data = re.sub('^data:image/.+;base64,', '', data_url)
        return base64.b64decode(image_data)
    except Exception as e:
        logging.error(f"Error capturing canvas: {e}")
        return None


def save_snapshot(image_bytes, output_folder):
    """
    Save the binary image data to a PNG file with a timestamped filename.
    """
    if image_bytes is None:
        logging.error("No image data to save.")
        return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"heatmap_{timestamp}.png"
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

    logging.info("Starting liquidLapse...")
    driver, temp_dir = setup_driver(headless=headless)

    try:
        logging.info(f"Navigating to {url}")
        driver.get(url)
        # Wait for the page to load completely
        driver.implicitly_wait(10)
        dismiss_popups(driver)

        image_bytes = capture_canvas_snapshot(driver)
        save_snapshot(image_bytes, output_folder)

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        driver.quit()
        # Clean up the temporary user data directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
