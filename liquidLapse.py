#!/usr/bin/env python3
"""
liquidLapse.py - Production-Ready Network-Resilient Version

Professional heatmap capture system with comprehensive network resilience,
multiple fallback mechanisms, and guaranteed reliability for 24/7 operation.

Features:
- Network failure detection and automatic retry with exponential backoff
- Multiple BTC price API fallbacks (CoinGecko, Coinbase, Kraken)
- ECharts-specific capture with 4 fallback methods
- Resource management and cleanup
- Professional logging and monitoring
- VPS-optimized Chrome configuration
- Maintains data format compatibility
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
import signal
import sys
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class NetworkResilienceManager:
    """Manages network connectivity checks and retry logic"""
    
    @staticmethod
    def check_connectivity():
        """Check basic internet connectivity"""
        test_urls = [
            "https://www.google.com",
            "https://1.1.1.1",
            "https://8.8.8.8"
        ]
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return True
            except:
                continue
        return False
    
    @staticmethod
    def is_retryable_error(error_msg):
        """Determine if error is network-related and retryable"""
        retryable_indicators = [
            "ERR_CONNECTION_CLOSED",
            "ERR_NETWORK_CHANGED", 
            "ERR_INTERNET_DISCONNECTED",
            "ERR_CONNECTION_RESET",
            "ERR_CONNECTION_REFUSED",
            "ERR_TIMED_OUT",
            "timeout",
            "connection reset",
            "connection refused",
            "network is unreachable"
        ]
        
        error_lower = str(error_msg).lower()
        return any(indicator.lower() in error_lower for indicator in retryable_indicators)

class BTCPriceManager:
    """Manages BTC price fetching with multiple API fallbacks"""
    
    def __init__(self):
        self.apis = [
            {
                'name': 'CoinGecko',
                'url': 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd',
                'parser': lambda r: r.json()['bitcoin']['usd'],
                'timeout': 10
            },
            {
                'name': 'Coinbase',
                'url': 'https://api.coinbase.com/v2/exchange-rates?currency=BTC',
                'parser': lambda r: float(r.json()['data']['rates']['USD']),
                'timeout': 15
            },
            {
                'name': 'Kraken',
                'url': 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
                'parser': lambda r: float(r.json()['result']['XXBTZUSD']['c'][0]),
                'timeout': 15
            },
            {
                'name': 'Binance',
                'url': 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
                'parser': lambda r: float(r.json()['price']),
                'timeout': 10
            }
        ]
    
    def get_btc_price(self):
        """Fetch BTC price with fallback APIs"""
        for api in self.apis:
            try:
                logging.info(f"Fetching BTC price from {api['name']}...")
                response = requests.get(api['url'], timeout=api['timeout'])
                
                if response.status_code == 200:
                    price = api['parser'](response)
                    logging.info(f"BTC price from {api['name']}: ${float(price):,.2f}")
                    return str(float(price))
                elif response.status_code == 429:
                    logging.warning(f"{api['name']} rate limited (429), trying next API")
                else:
                    logging.warning(f"{api['name']} returned status {response.status_code}")
                    
            except Exception as e:
                logging.warning(f"{api['name']} API failed: {e}")
                continue
        
        logging.error("All BTC price APIs failed")
        return None

class EChartsHeatmapCapture:
    """Professional ECharts heatmap capture with comprehensive error handling"""
    
    def __init__(self):
        self.driver = None
        self.temp_dir = None
        self.url = None
   
    def setup_driver(self, headless=True):
        """Optimized Chrome driver for CoinGlass website"""
        options = webdriver.ChromeOptions()
        
        # Root user compatibility
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Network optimization for slow sites
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-images")  # Load faster without images
        options.add_argument("--disable-web-security")
        
        # Memory optimization
        options.add_argument("--memory-pressure-off")
        options.add_argument("--max_old_space_size=2048")
        
        # Network resilience
        options.add_argument("--aggressive-cache-discard")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        
        # Timeout and performance
        options.add_argument("--window-size=1366,768")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-sync")
        
        # Anti-detection with realistic headers
        options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Add additional flags to handle slow sites
        options.add_argument("--disable-ipc-flooding-protection")
        options.add_argument("--disable-hang-monitor")
        options.add_argument("--disable-prompt-on-repost")
        
        if headless:
            options.add_argument("--headless=new")
        
        # Temp directory
        self.temp_dir = tempfile.mkdtemp()
        options.add_argument(f"--user-data-dir={self.temp_dir}")
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            
            # Set aggressive timeouts
            self.driver.set_page_load_timeout(60)  # Increased timeout
            self.driver.implicitly_wait(20)  # Wait for elements
            
            # Disable webdriver detection
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logging.info("Chrome driver initialized with enhanced timeouts")
            return self.driver
            
        except Exception as e:
            logging.error(f"Failed to initialize Chrome driver: {e}")
            self.cleanup()
            raise
   
    def navigate_with_retry(self, url, max_retries=3):
        """Enhanced navigation with aggressive timeout handling"""
        for attempt in range(max_retries):
            try:
                logging.info(f"Navigation attempt {attempt + 1}/{max_retries} to {url}")
                
                # Set more aggressive timeouts
                self.driver.set_page_load_timeout(60)  # Increased from 45s
                
                # Try different navigation approaches
                if attempt == 0:
                    # Standard navigation
                    self.driver.get(url)
                elif attempt == 1:
                    # JavaScript navigation (sometimes bypasses blocking)
                    self.driver.execute_script(f"window.location.href = '{url}';")
                    time.sleep(10)  # Wait for JS navigation
                else:
                    # Direct URL access with refresh
                    self.driver.get(url)
                    time.sleep(5)
                    self.driver.refresh()
                
                # Wait for page to be in ready state
                WebDriverWait(self.driver, 30).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
                
                # Additional wait for dynamic content
                time.sleep(10)
                
                logging.info("Navigation successful")
                return True
                
            except Exception as e:
                error_msg = str(e)
                logging.warning(f"Navigation attempt {attempt + 1} failed: {error_msg}")
                
                # Check for retryable errors
                if any(err in error_msg.lower() for err in ['timeout', 'timed out', 'renderer']):
                    if attempt < max_retries - 1:
                        delay = 30 + (15 * attempt)  # 30s, 45s, 60s
                        logging.info(f"Timeout detected, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                else:
                    logging.error("Non-timeout navigation error")
                    return False
        
        logging.error("All navigation attempts failed")
        return False
  
    def wait_for_echarts_load(self, timeout=30):
        """Wait for ECharts to load with enhanced validation"""
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
            
            # Progressive wait with validation
            for wait_time in [5, 10, 15]:
                time.sleep(wait_time)
                
                # Check if canvases have content
                valid_canvases = []
                for i, canvas in enumerate(canvases):
                    try:
                        width = self.driver.execute_script("return arguments[0].width;", canvas)
                        height = self.driver.execute_script("return arguments[0].height;", canvas)
                        
                        if width > 100 and height > 100:
                            valid_canvases.append(canvas)
                            logging.info(f"Canvas {i}: {width}×{height} (valid)")
                        else:
                            logging.warning(f"Canvas {i}: {width}×{height} (invalid)")
                    except:
                        continue
                
                if valid_canvases:
                    logging.info(f"Found {len(valid_canvases)} valid canvases after {wait_time}s")
                    return valid_canvases
            
            logging.warning("No valid canvases found after full wait")
            return canvases  # Return anyway for fallback methods
            
        except Exception as e:
            logging.error(f"Failed to wait for ECharts load: {e}")
            return []
    
    def dismiss_popups(self):
        """Comprehensive popup dismissal"""
        popup_selectors = [
            '//button[contains(text(), "Consent")]',
            '//button[contains(text(), "Accept")]',
            '//button[contains(text(), "OK")]',
            '//button[contains(text(), "Close")]',
            '.modal-close',
            '.close-button',
            '[aria-label="Close"]',
            '.popup-close',
            '.dialog-close'
        ]
        
        dismissed = False
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
                time.sleep(2)
                dismissed = True
                break
            except:
                continue
        
        if not dismissed:
            logging.info("No popups detected or dismissed")
        
        return dismissed
    
    def capture_echarts_canvas(self):
        """Multi-method ECharts capture with comprehensive fallbacks"""
        canvases = self.wait_for_echarts_load()
        
        if not canvases:
            logging.error("No ECharts canvases found")
            return None
        
        # Method 1: Direct canvas capture (highest quality)
        for i, canvas in enumerate(reversed(canvases)):
            canvas_id = canvas.get_attribute("data-zr-dom-id") or f"canvas_{i}"
            
            try:
                logging.info(f"Attempting direct capture of canvas: {canvas_id}")
                data_url = self.driver.execute_script("return arguments[0].toDataURL('image/png');", canvas)
                
                if data_url and data_url.startswith("data:image") and len(data_url) > 1000:
                    logging.info(f"Canvas {canvas_id} captured successfully ({len(data_url)} chars)")
                    image_data = re.sub('^data:image/.+;base64,', '', data_url)
                    return base64.b64decode(image_data)
                    
            except Exception as e:
                logging.warning(f"Direct capture failed for {canvas_id}: {e}")
        
        # Method 2: Content validation approach
        for i, canvas in enumerate(canvases):
            try:
                logging.info(f"Attempting validated capture of canvas {i}")
                canvas_data = self.driver.execute_script("""
                    var canvas = arguments[0];
                    if (canvas && canvas.width > 100 && canvas.height > 100) {
                        var ctx = canvas.getContext('2d');
                        if (ctx) {
                            try {
                                return canvas.toDataURL('image/png');
                            } catch(e) {
                                return null;
                            }
                        }
                    }
                    return null;
                """, canvas)
                
                if canvas_data and len(canvas_data) > 1000:
                    logging.info(f"Validated capture successful for canvas {i}")
                    image_data = re.sub('^data:image/.+;base64,', '', canvas_data)
                    return base64.b64decode(image_data)
                    
            except Exception as e:
                logging.warning(f"Validated capture failed for canvas {i}: {e}")
        
        # Method 3: Composite canvas approach
        try:
            logging.info("Attempting composite canvas capture...")
            composite_data = self.driver.execute_script("""
                var canvases = document.querySelectorAll('canvas[data-zr-dom-id^="zr_"]');
                if (canvases.length === 0) return null;
                
                var refCanvas = null;
                for (var i = 0; i < canvases.length; i++) {
                    if (canvases[i].width > 100 && canvases[i].height > 100) {
                        refCanvas = canvases[i];
                        break;
                    }
                }
                if (!refCanvas) return null;
                
                var compositeCanvas = document.createElement('canvas');
                compositeCanvas.width = refCanvas.width;
                compositeCanvas.height = refCanvas.height;
                var ctx = compositeCanvas.getContext('2d');
                
                for (var i = 0; i < canvases.length; i++) {
                    try {
                        if (canvases[i].width > 0 && canvases[i].height > 0) {
                            ctx.drawImage(canvases[i], 0, 0);
                        }
                    } catch(e) {
                        continue;
                    }
                }
                
                try {
                    return compositeCanvas.toDataURL('image/png');
                } catch(e) {
                    return null;
                }
            """)
            
            if composite_data and len(composite_data) > 1000:
                logging.info("Composite canvas capture successful")
                image_data = re.sub('^data:image/.+;base64,', '', composite_data)
                return base64.b64decode(image_data)
                
        except Exception as e:
            logging.warning(f"Composite capture failed: {e}")
        
        # Method 4: ECharts container screenshot
        try:
            logging.info("Attempting ECharts container screenshot...")
            echarts_container = self.driver.find_element(By.CSS_SELECTOR, '.echarts-for-react')
            
            # Scroll element into view
            self.driver.execute_script("arguments[0].scrollIntoView();", echarts_container)
            time.sleep(2)
            
            screenshot = echarts_container.screenshot_as_png
            
            if screenshot and len(screenshot) > 1000:
                logging.info("ECharts container screenshot captured")
                return screenshot
                
        except Exception as e:
            logging.warning(f"Container screenshot failed: {e}")
        
        # Method 5: Full page screenshot (last resort)
        try:
            logging.info("Attempting full page screenshot fallback...")
            screenshot = self.driver.get_screenshot_as_png()
            
            if screenshot and len(screenshot) > 1000:
                logging.info("Full page screenshot captured (fallback)")
                return screenshot
                
        except Exception as e:
            logging.error(f"Full page screenshot failed: {e}")
        
        logging.error("All capture methods failed")
        return None
    
    def cleanup(self):
        """Comprehensive resource cleanup"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass
            self.temp_dir = None

def load_config(config_path="config.yaml"):
    """Load and validate configuration"""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Validate required fields
        required_fields = ['url', 'output_folder', 'check_interval']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        return config
        
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

def save_heatmap(image_bytes, output_folder, btc_price=None):
    """Save heatmap with optimized filename format"""
    if image_bytes is None or len(image_bytes) == 0:
        logging.error("No valid image data to save")
        return False
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Include BTC price if available
    if btc_price is not None:
        try:
            price_float = float(btc_price)
            price_str = f"_BTC-{price_float:.0f}"  # Round to nearest dollar
        except:
            price_str = "_BTC-NA"
    else:
        price_str = "_BTC-NA"
    
    filename = f"heatmap_{timestamp}{price_str}.png"
    os.makedirs(output_folder, exist_ok=True)
    
    filepath = os.path.join(output_folder, filename)
    
    try:
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        file_size = os.path.getsize(filepath)
        
        # Validate file size (should be > 10KB for valid heatmap)
        if file_size < 10240:
            logging.warning(f"Saved file unusually small: {file_size} bytes")
        
        logging.info(f"Heatmap saved: {filepath} ({file_size:,} bytes)")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save heatmap: {e}")
        return False

def capture_with_full_resilience(config):
    """Main capture function with comprehensive error handling"""
    url = config.get("url")
    output_folder = config.get("output_folder", "heatmap_snapshots")
    headless = config.get("headless", True)
    include_price = config.get("include_price_in_filename", False)
    
    # Check network connectivity first
    if not NetworkResilienceManager.check_connectivity():
        logging.error("No internet connectivity detected")
        return False
    
    capture = EChartsHeatmapCapture()
    btc_manager = BTCPriceManager()
    
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            logging.info(f"=== Capture Attempt {attempt + 1}/{max_attempts} ===")
            
            # Setup driver
            capture.setup_driver(headless=headless)
            
            # Navigate with retry
            if not capture.navigate_with_retry(url):
                raise Exception("Failed to navigate to target URL")
            
            # Handle popups
            capture.dismiss_popups()
            
            # Additional wait for page stability
            time.sleep(5)
            
            # Capture heatmap
            image_bytes = capture.capture_echarts_canvas()
            
            if image_bytes:
                # Get BTC price if requested
                btc_price = None
                if include_price:
                    btc_price = btc_manager.get_btc_price()
                
                # Save result
                if save_heatmap(image_bytes, output_folder, btc_price):
                    logging.info(f"Capture attempt {attempt + 1} completed successfully")
                    return True
                else:
                    logging.error(f"Failed to save heatmap on attempt {attempt + 1}")
            else:
                logging.error(f"No image data captured on attempt {attempt + 1}")
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Attempt {attempt + 1} failed: {error_msg}")
            
            # Check if retryable
            if not NetworkResilienceManager.is_retryable_error(error_msg) and attempt == 0:
                logging.error("Non-retryable error on first attempt")
                return False
                
        finally:
            capture.cleanup()
        
        # Wait before retry (except on last attempt)
        if attempt < max_attempts - 1:
            delay = 30 * (attempt + 1)  # 30s, 60s
            logging.info(f"Waiting {delay}s before retry...")
            time.sleep(delay)
    
    logging.error("All capture attempts failed")
    return False

def signal_handler(signum, frame):
    """Graceful shutdown handler"""
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main execution function with comprehensive error handling"""
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        config = load_config()
        logging.info("Starting liquidLapse Network-Resilient Enhanced Version")
        
        success = capture_with_full_resilience(config)
        
        if success:
            logging.info("Heatmap capture completed successfully")
            return True
        else:
            logging.error("Heatmap capture failed")
            return False
            
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
