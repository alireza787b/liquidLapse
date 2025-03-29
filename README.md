# liquidLapse

**liquidLapse** is a Python automation project that periodically captures snapshots of the CoinGlass liquidation heatmap. Snapshots are saved in a date-organized folder with informative date-time stamps, enabling later playback or review.

## Features

- **Automated Snapshot Capture:**  
  Uses Selenium in headless mode to load the CoinGlass heatmap page, capture the heatmap's canvas element as an image, and save it with a timestamp.
  
- **Configurable Settings:**  
  All key settings (URL, snapshot interval, output folder, and headless mode) are stored in a `config.yaml` file for easy adjustments.

- **Robust Environment Setup:**  
  A `setup.sh` script ensures that all system dependencies (Python, pip, venv, Google Chrome) are installed, creates a virtual environment, and installs the required Python packages.

- **Service Management:**  
  A `service.sh` script allows you to start, stop, restart, and check the status of the liquidLapse service, running the snapshot script in the background and logging its output.

- **Cross-Platform Support:**  
  Designed to run on both Windows and Linux (including headless Linux servers).

## Project Structure

```
liquidLapse/
├── config.yaml
├── liquidLapse.py
├── requirements.txt
├── README.md
├── setup.sh
└── service.sh
```


## Setup & Usage Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/alireza787b/liquidLapse.git
cd liquidLapse
```

### 2. Run the Setup Script

Make the setup script executable:

```bash
chmod +x setup.sh
```

Then run it:

```bash
./setup.sh
```

This script will:
- Update package repositories.
- Install Python3, pip3, python3-venv, and Google Chrome (if needed).
- Create a Python virtual environment.
- Upgrade pip and install the required Python dependencies.

### 3. Manage the liquidLapse Service

Use the service script to start, stop, restart, or check the status of the liquidLapse service:

- **Start the Service:**
  ```bash
  ./service.sh start
  ```
  This will run `liquidLapse.py` in the background and log output to `liquidLapse.log`. The PID is saved in `liquidLapse.pid`.

- **Stop the Service:**
  ```bash
  ./service.sh stop
  ```

- **Restart the Service:**
  ```bash
  ./service.sh restart
  ```

- **Check the Service Status:**
  ```bash
  ./service.sh status
  ```

### 4. Running Directly (Without Service Management)

To run the script directly, activate the virtual environment and execute the Python script:

```bash
source venv/bin/activate   # On Windows use: venv\Scripts\activate
python liquidLapse.py
```

Press `Ctrl+C` to terminate the script.

---

## Configuration

Edit `config.yaml` to set your preferences:

```yaml
# config.yaml
url: "https://www.coinglass.com/pro/futures/LiquidationHeatMap"
check_interval: 300         # Interval (in seconds) between snapshots
output_folder: "heatmap_snapshots"
headless: true              # Run Chrome in headless mode
```

---

## Additional Notes

- **Linux Server Deployment:**  
  With headless mode enabled in `config.yaml`, the script runs without a GUI—ideal for Linux VPS setups.
  
- **Scheduling:**  
  You can set up cron jobs (on Linux) or Windows Task Scheduler to run the service script at regular intervals if desired.

- **Logs:**  
  Runtime logs are saved to `liquidLapse.log`. If the service stops unexpectedly, check the log for troubleshooting details.


## License

This project is licensed under the Apache 2.0 License.

---
