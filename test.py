import os
import time
import base64
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # optional: run browser headless
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get("https://www.coinglass.com/pro/futures/LiquidationHeatMap")

    # 1) Dismiss popup if it appears
    try:
        consent_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(text(),"Consent")]'))
        )
        consent_btn.click()
        time.sleep(1)
    except:
        pass

    # 2) Wait for any canvas with data-zr-dom-id starting with "zr_"
    #    This might take a while if the site is slow
    canvas = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'canvas[data-zr-dom-id^="zr_"]'))
    )

    # 3) Use JS to convert the canvas to a base64-encoded PNG
    data_url = driver.execute_script("return arguments[0].toDataURL('image/png');", canvas)
    image_data = re.sub('^data:image/.+;base64,', '', data_url)
    image_bytes = base64.b64decode(image_data)

    # 4) Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join("heatmap_snapshots", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"heatmap_{timestamp}.png")
    with open(out_path, 'wb') as f:
        f.write(image_bytes)

    print(f"Saved heatmap to {out_path}")

    driver.quit()

if __name__ == "__main__":
    main()
