Below is a complete README in Markdown that recaps your project details, explains the purpose of the setup script, and provides instructions for installation and usage.

# liquidLapse

**liquidLapse** is a Python automation project that periodically captures snapshots of the CoinGlass liquidation heatmap. Each snapshot is saved with a timestamp in a date-organized folder, allowing you to later playback or review the history of the heatmap.

## Features

- **Automated Snapshot Capture:** Uses Selenium in headless mode to load the CoinGlass heatmap page, capture the heatmap (canvas element) as an image, and save it with an informative timestamp.
- **Configurable Settings:** Reads settings (such as the URL, snapshot interval, output folder, and headless mode) from a `config.yaml` file.
- **Robust Environment Setup:** Includes a setup script (`setup.sh`) that installs required system packages, Python dependencies, and ensures that Google Chrome and ChromeDriver are available.
- **Organized Output:** Snapshots are stored in a folder structure by date, with filenames that include the full date and time.
- **Cross-Platform Support:** The project is designed to work on both Windows and Linux (headless servers) environments.

## Project Structure

```
liquidLapse/
├── config.yaml
├── liquidLapse.py
├── requirements.txt
├── README.md
└── setup.sh
```

## Installation

### On Linux (or any system with bash):

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/liquidLapse.git
   cd liquidLapse
   ```

2. **Run the Setup Script:**

   Make sure the setup script is executable:
   ```bash
   chmod +x setup.sh
   ```

   Then run it:
   ```bash
   ./setup.sh
   ```

   The setup script will:
   - Update your package repositories.
   - Check for and install Python3, pip3, and the `python3-venv` module if they aren’t installed.
   - Check for Google Chrome and install it if missing.
   - Create and activate a Python virtual environment.
   - Upgrade pip and install the required Python dependencies from `requirements.txt`.

3. **Activate the Virtual Environment and Run the Script:**

   ```bash
   source venv/bin/activate
   python liquidLapse.py
   ```

   Press `Ctrl+C` to terminate the script when needed.

### On Windows:

You can use the provided setup instructions in the README and run the equivalent commands in a Git Bash terminal or via PowerShell (with adjustments to the virtual environment activation command).

## Configuration

Edit the `config.yaml` file to set your preferences:

```yaml
# config.yaml
url: "https://www.coinglass.com/pro/futures/LiquidationHeatMap"
check_interval: 300         # Time in seconds between snapshots
output_folder: "heatmap_snapshots"
headless: true              # Set to true to run Chrome in headless mode
```

## Usage

Once setup is complete and the dependencies are installed, run the main script to start capturing snapshots:

1. **Activate your Virtual Environment:**
   ```bash
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

2. **Run the Main Script:**
   ```bash
   python liquidLapse.py
   ```

Snapshots will be saved inside the folder specified in `config.yaml` (organized by date), and each file will be named with a timestamp (e.g., `heatmap_20250328_172525.png`).

## Additional Notes

- **Deployment on a Linux Server:**  
  With headless mode enabled (via the `headless: true` config), the script runs without a GUI, making it perfect for Linux servers.
- **Scheduling:**  
  You can use cron (on Linux) or Windows Task Scheduler to run `liquidLapse.py` at regular intervals if you prefer not to have it running continuously.

## Dependencies

All Python dependencies are listed in `requirements.txt`:
```
selenium
webdriver-manager
PyYAML
```

## License

This project is licensed under the Apache 2.0 License.

