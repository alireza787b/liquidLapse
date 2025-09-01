#!/usr/bin/env python3
"""
liquidLapse.py - Enhanced Professional Version

Robust heatmap capture system with multiple fallback methods
for ECharts-based liquidation heatmaps from CoinGlass.

Features:
- ECharts-specific canvas detection and capture
- Multiple fallback capture methods
- Enhanced error handling and recovery
- Professional logging and monitoring
- Maintains backward compatibility with existing data format
"""

import os
import base64
import re
import yaml
import logging
import requests
import tempfile
import shutil
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class EChartsHeatmapCapture:
    """Professional ECharts heatmap capture handler"""
    
    def __init__(self):
        self.driver = None
        self.temp_dir = None
        
    def setup_driver(self, headless=True):
        """
        Create Chrome driver optimized for ECharts and VPS environments
        """
        options = webdriver.ChromeOptions()
        
        # CRITICAL: Root user and VPS compatibility
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Performance and stability
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")
        
        # Memory management for VPS
        options.add_argument("--memory-pressure-off")
        options.add_argument("--max_old_space_size=4096")
        
        # Window size - important for ECharts rendering
        options.add_argument("--window-size=1920,1080")
        
        # Anti-detection measures
        options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        if headless:
            options.add_argument("--headless=new")
        
        # Create unique temp directory
        self.temp_dir = tempfile.mkdtemp()
        options.add_argument(f"--user-data-dir={self.temp_dir}")
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.set_page_load_timeout(60)
            
            # Disable webdriver detection
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logging.info("Chrome driver initialized successfully")
            return self.driver
            
        except Exception as e:
            logging.error(f"Failed to initialize Chrome driver: {e}")
            if self.temp_dir:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise
    
    def wait_for_echarts_load(self, timeout=30):
        """
        Wait for ECharts to fully load and render
        """
        try:
            # Wait for ECharts container
            echarts_container = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.echarts-for-react'))
            )
            logging.info("ECharts container found")
            
            # Wait for canvas elements
            canvases = WebDriverWait(self.driver, timeout).until(
                lambda d: d.find_elements(By.CSS_SELECTOR, 'canvas[data-zr-dom-id^="zr_"]')
            )
            logging.info(f"Found {len(canvases)} ECharts canvas elements")
            
            # Additional wait for chart rendering
            time.sleep(10)
            
            # Verify canvases have content by checking dimensions
            for i, canvas in enumerate(canvases):
                width = self.driver.execute_script("return arguments[0].width;", canvas)
                height = self.driver.execute_script("return arguments[0].height;", canvas)
                logging.info(f"Canvas {i}: {width}×{height}")
                
                if width == 0 or height == 0:
                    logging.warning(f"Canvas {i} has zero dimensions")
                    
            return canvases
            
        except Exception as e:
            logging.error(f"Failed to wait for ECharts load: {e}")
            return []
    
    def dismiss_popups(self):
        """
        Handle various popups and overlays
        """
        popup_selectors = [
            '//button[contains(text(), "Consent")]',
            '//button[contains(text(), "Accept")]',
            '//button[contains(text(), "OK")]',
            '.modal-close',
            '.close-button',
            '[aria-label="Close"]'
        ]
        
        for selector in popup_selectors:
            try:
                if selector.startswith('//'):
                    element = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                else:
                    element = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                element.click()
                logging.info(f"Dismissed popup with selector: {selector}")
                time.sleep(1)
                return True
            except:
                continue
        
        logging.info("No popups detected or dismissed")
        return False
    
    def capture_echarts_canvas(self):
        """
        Capture ECharts canvas with multiple fallback methods
        """
        canvases = self.wait_for_echarts_load()
        
        if not canvases:
            logging.error("No ECharts canvases found")
            return None
        
        # Method 1: Try to capture the primary canvas (usually the last one for overlays)
        for i, canvas in enumerate(reversed(canvases)):
            canvas_id = canvas.get_attribute("data-zr-dom-id")
            logging.info(f"Attempting capture of canvas: {canvas_id}")
            
            try:
                # Method 1a: Direct toDataURL
                data_url = self.driver.execute_script(
                    "return arguments[0].toDataURL('image/png');", canvas
                )
                
                if data_url and len(data_url) > 1000:  # Ensure substantial data
                    logging.info(f"✓ Canvas {canvas_id} captured via toDataURL ({len(data_url)} chars)")
                    image_data = re.sub('^data:image/.+;base64,', '', data_url)
                    return base64.b64decode(image_data)
                    
            except Exception as e:
                logging.warning(f"toDataURL failed for {canvas_id}: {e}")
            
            try:
                # Method 1b: Validate canvas first
                canvas_data = self.driver.execute_script("""
                    var canvas = arguments[0];
                    if (canvas && canvas.width > 100 && canvas.height > 100) {
                        var ctx = canvas.getContext('2d');
                        if (ctx) {
                            // Check if canvas has actual content
                            var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                            var data = imageData.data;
                            var hasContent = false;
                            
                            // Check for non-transparent pixels
                            for (var i = 3; i < data.length; i += 4) {
                                if (data[i] > 0) {
                                    hasContent = true;
                                    break;
                                }
                            }
                            
                            if (hasContent) {
                                return canvas.toDataURL('image/png');
                            }
                        }
                    }
                    return null;
                """, canvas)
                
                if canvas_data and len(canvas_data) > 1000:
                    logging.info(f"✓ Canvas {canvas_id} captured via validation method")
                    image_data = re.sub('^data:image/.+;base64,', '', canvas_data)
                    return base64.b64decode(image_data)
                    
            except Exception as e:
                logging.warning(f"Validation method failed for {canvas_id}: {e}")
        
        # Method 2: Composite capture (combine all canvases)
        try:
            logging.info("Attempting composite canvas capture...")
            composite_data = self.driver.execute_script("""
                var canvases = document.querySelectorAll('canvas[data-zr-dom-id^="zr_"]');
                if (canvases.length === 0) return null;
                
                var firstCanvas = canvases[0];
                var compositeCanvas = document.createElement('canvas');
                compositeCanvas.width = firstCanvas.width;
                compositeCanvas.height = firstCanvas.height;
                var ctx = compositeCanvas.getContext('2d');
                
                // Draw all canvases onto composite
                for (var i = 0; i < canvases.length; i++) {
                    if (canvases[i].width > 0 && canvases[i].height > 0) {
                        ctx.drawImage(canvases[i], 0, 0);
                    }
                }
                
                return compositeCanvas.toDataURL('image/png');
            """)
            
            if composite_data and len(composite_data) > 1000:
                logging.info("✓ Composite canvas capture successful")
                image_data = re.sub('^data:image/.+;base64,', '', composite_data)
                return base64.b64decode(image_data)
                
        except Exception as e:
            logging.warning(f"Composite capture failed: {e}")
        
        # Method 3: Screenshot fallback of ECharts container
        try:
            logging.info("Attempting ECharts container screenshot...")
            echarts_container = self.driver.find_element(By.CSS_SELECTOR, '.echarts-for-react')
            screenshot = echarts_container.screenshot_as_png
            
            if screenshot and len(screenshot) > 1000:
                logging.info("✓ ECharts container screenshot captured")
                return screenshot
                
        except Exception as e:
            logging.warning(f"Container screenshot failed: {e}")
        
        # Method 4: Full page screenshot as last resort
        try:
            logging.info("Attempting full page screenshot fallback...")
            screenshot = self.driver.get_screenshot_as_png()
            
            if screenshot:
                logging.info("✓ Full page screenshot captured (fallback)")
                return screenshot
                
        except Exception as e:
            logging.error(f"Full page screenshot failed: {e}")
        
        logging.error("All capture methods failed")
        return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
        
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

def get_btc_price(btc_price_url):
    """Fetch current Bitcoin price from API"""
    try:
        response = requests.get(btc_price_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        btc_price = data['bitcoin']['usd']
        logging.info(f"BTC price fetched: ${btc_price:,.2f}")
        return str(btc_price)
    except Exception as e:
        logging.error(f"Error fetching BTC price: {e}")
        return None

def save_heatmap(image_bytes, output_folder, include_price=False, btc_price_url=None):
    """Save heatmap image with timestamped filename"""
    if image_bytes is None:
        logging.error("No image data to save")
        return False
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    price_str = ""
    
    if include_price:
        if not btc_price_url:
            btc_price_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        
        btc_price = get_btc_price(btc_price_url)
        if btc_price is not None:
            price_str = f"_BTC-{btc_price}"
        else:
            price_str = "_BTC-NA"
    
    filename = f"heatmap_{timestamp}{price_str}.png"
    os.makedirs(output_folder, exist_ok=True)
    
    filepath = os.path.join(output_folder, filename)
    
    try:
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        file_size = os.path.getsize(filepath)
        logging.info(f"✓ Heatmap saved: {filepath} ({file_size:,} bytes)")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save heatmap: {e}")
        return False

def main():
    """Main execution function"""
    config = load_config()
    
    url = config.get("url")
    output_folder = config.get("output_folder", "heatmap_snapshots")
    headless = config.get("headless", True)
    include_price = config.get("include_price_in_filename", False)
    btc_price_url = config.get("btc_price_url")
    
    logging.info("Starting liquidLapse Enhanced...")
    
    capture = EChartsHeatmapCapture()
    
    try:
        # Initialize driver
        capture.setup_driver(headless=headless)
        
        # Navigate to page
        logging.info(f"Navigating to {url}")
        capture.driver.get(url)
        
        # Handle popups
        capture.dismiss_popups()
        
        # Capture heatmap
        image_bytes = capture.capture_echarts_canvas()
        
        # Save result
        success = save_heatmap(image_bytes, output_folder, include_price, btc_price_url)
        
        if success:
            logging.info("✓ Heatmap capture completed successfully")
        else:
            logging.error("✗ Heatmap capture failed")
            return False
            
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False
        
    finally:
        capture.cleanup()
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
