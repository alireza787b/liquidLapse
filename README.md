# liquidLapse

**liquidLapse** is a Python automation project that periodically captures snapshots of the CoinGlass liquidation heatmap. Snapshots are saved with informative date-time stamps for later playback.

## Features

- Loads configuration from a YAML file (`config.yaml`).
- Uses Selenium with headless Chrome (via `webdriver-manager`).
- Captures the heatmap canvas directly by extracting its pixel data.
- Saves snapshots in a directory structure organized by date.
- Runs in an infinite loop at a configurable interval.

## Requirements

- Python 3.6+
- Google Chrome (or Chromium)
- [ChromeDriver](https://sites.google.com/chromium.org/driver/) (automatically managed via `webdriver-manager`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/liquidLapse.git
   cd liquidLapse
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit the `config.yaml` file to adjust:
- `url`: The URL to capture.
- `check_interval`: Time in seconds between snapshots.
- `output_folder`: Base folder where snapshots will be saved.
- `headless`: Set to `true` to run in headless mode.

## Running the Script

To start capturing snapshots, run:
```bash
python liquidLapse.py
```

Press `Ctrl+C` to terminate the script.

## Deployment on Linux (Headless Server)

The script is designed to run in headless mode, making it suitable for deployment on a Linux server without a GUI.

You can use the provided `setup.sh` script (see below) to install dependencies in your server environment.

## License

Apache 2.0 License 
```

---

### 5. **setup.sh**

A sample bash script to help set up your environment on Linux. This script creates a virtual environment and installs dependencies.

```bash
#!/bin/bash
# setup.sh - Setup environment for liquidLapse

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. To run the script, activate the virtual environment with:"
echo "source venv/bin/activate"
echo "then run: python liquidLapse.py"
```

Make the script executable:

```bash
chmod +x setup.sh
```

