# liquidLapse

**liquidLapse** is a Python automation project that periodically captures snapshots of the CoinGlass liquidation heatmap. Snapshots are saved in a date-organized folder with informative date-time stamps (optionally including the current Bitcoin price), enabling later playback or review. The project now includes advanced sequence processing for AI training pipelines.

## Features

- **Automated Snapshot Capture:**  
  Uses Selenium in headless mode to load the CoinGlass heatmap page, capture the heatmap's canvas element as an image, and save it with a timestamp.

- **BTC Price Integration:**  
  Optionally appends the current Bitcoin price (fetched from a configurable API URL) to the snapshot filename for added context.

- **AI-Ready Sequence Generation:**  
  Processes captured snapshots into chronological sequences suitable for AI model training:
  - Organizes images into fixed-length sequences
  - Handles timestamp gaps with intelligent interpolation
  - Tracks price changes across timeframes
  - Includes future prediction data for supervised learning targets
  - Maintains references to both original and processed image paths

- **Configurable Settings:**  
  All key settings—including the target URL, snapshot interval, output folder, headless mode, BTC price API URL, and whether to include the BTC price in the filename—are stored in a `config.yaml` file for easy adjustments.

- **Robust Environment Setup:**  
  A `setup.sh` script ensures that all system dependencies (Python, pip, venv, Google Chrome) are installed, creates a virtual environment, and installs the required Python packages.

- **Service Management with Detailed Status:**  
  A `service.sh` script lets you start, stop, restart, and check the status of the liquidLapse service. The service runs `liquidLapse.py` in an infinite loop (using the interval defined in `config.yaml`), logs each snapshot execution, and updates a status file with execution details.

- **Cross-Platform Support:**  
  Designed to run on both Windows and Linux (including headless Linux servers).

## Project Structure

```
liquidLapse/
├── config.yaml
├── liquidLapse.py
├── sequence_generator.py     # New script for AI training sequence generation
├── requirements.txt
├── README.md
├── setup.sh
├── service.sh
└── ai_process/               # Directory for AI processing outputs
    └── session_name/         # Named session folders
        ├── dataset_info.json # Dataset metadata
        ├── images/           # Processed images
        └── sequences/        # Generated sequences
            ├── sequences_info.json   # Metadata for all sequences
            └── sequence_n/   # Individual sequence folders
                └── images/   # Sequence images
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
- Upgrade pip and install the required Python dependencies from `requirements.txt`.

### 3. Configure the Application

Edit the `config.yaml` file to set your preferences. For example:

```yaml
# config.yaml
url: "https://www.coinglass.com/pro/futures/LiquidationHeatMap"
check_interval: 300                      # Interval (in seconds) between snapshots
output_folder: "heatmap_snapshots"
headless: true                           # Run Chrome in headless mode
include_price_in_filename: true          # Append BTC price to the filename
btc_price_url: "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
```

### 4. Manage the liquidLapse Service

Use the `service.sh` script to start, stop, restart, or check the status of the liquidLapse service.

- **Start the Service:**
  ```bash
  ./service.sh start
  ```
  This will run `liquidLapse.py` in the background at intervals defined in `config.yaml`, logging output to `liquidLapseService.log` and storing the PID in `liquidLapseService.pid`.

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
  The status command displays whether the service is running and shows details (such as the last snapshot execution time, the configured interval, and any error messages) read from the status file.

### 5. Running Directly (Without Service Management)

To run the snapshot script directly, activate the virtual environment and execute:

```bash
source venv/bin/activate   # On Windows use: venv\Scripts\activate
python liquidLapse.py
```

Press `Ctrl+C` to terminate the script.

### 6. Generating AI Training Sequences

After collecting snapshots, you can process them into training sequences:

```bash
source venv/bin/activate
python sequence_generator.py --session="test1" --length=10
```

Parameters:
- `--session`: Name of the processing session (default: "test1")
- `--length`: Number of images per sequence (default: 10)

The script will:
1. Process the snapshots in the `heatmap_snapshots` directory
2. Generate fixed-length sequences with metadata
3. Include future prediction data for each sequence
4. Save everything in the `ai_process/[session_name]/sequences` directory
5. Create a comprehensive `sequences_info.json` file with all metadata

---

## Additional Notes

- **Linux Server Deployment:**  
  With headless mode enabled in `config.yaml`, the script runs without a GUI—ideal for deployment on Linux VPS or headless servers.

- **Scheduling:**  
  While the service script runs continuously, you can also set up cron jobs (on Linux) or Windows Task Scheduler if you prefer to manage scheduling externally.

- **Logs & Status:**  
  - Runtime logs are saved to `liquidLapseService.log`.  
  - The status file (`liquidLapseService.status`) is updated each snapshot cycle with details such as the last execution time and the snapshot interval.

- **AI Training Pipeline Integration:**  
  The sequence generation feature creates structured data ready for machine learning model training:
  - Each sequence contains a series of chronological snapshots
  - Price change data is included for both the sequence and prediction targets
  - All data is structured in JSON format for easy consumption by ML frameworks
  - Paths to both original and processed images are maintained for flexibility

---

## License

This project is licensed under the Apache 2.0 License.