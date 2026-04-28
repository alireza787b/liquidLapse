# liquidLapse

**liquidLapse** is a Python-based automation project designed to periodically capture snapshots of CoinGlass liquidation heatmaps. The project processes these heatmaps into image sequences for AI model training, offering a seamless pipeline for cryptocurrency market prediction.

### Key Features:

1. **Automated Snapshot Capture**

   * Utilizes **Selenium** in headless mode to periodically capture the CoinGlass heatmap.
   * **BTC Price Integration**: Optionally appends the current Bitcoin price to the snapshot filename for added context.

2. **AI-Ready Sequence Generation**

   * Converts heatmap snapshots into sequences for AI model training, including:

     * Chronologically ordered sequences.
     * Handles timestamp gaps with intelligent interpolation.
     * Tracks price changes and adds future predictions.
     * Stores metadata along with images.

3. **Flexible Configuration**

   * A `config.yaml` file allows easy customization of settings such as target URL, snapshot intervals, BTC price API URL, etc.

4. **Service Management**

   * A `service.sh` script to manage the execution of the snapshot capture service.
   * Supports starting, foreground running, stopping, and checking the status of the service.

5. **AI Model Training & Prediction**

   * Integrates **CNN + LSTM** for sequential prediction based on heatmap snapshots.
   * Trains models and makes real-time predictions.
   * Pushes and pulls models between local and remote servers.

6. **API for Predictions**

   * FastAPI-based server for real-time predictions.
   * Offers flexibility to input the last image and number of frames to predict future market changes.

7. **Cross-Platform Support**

   * Works on both Windows and Linux, including headless Linux servers.

---

## Project Structure

```plaintext
liquidLapse/
├── config.yaml                # Configuration file for snapshot capture and AI settings
├── liquidLapse.py             # Main script for capturing heatmaps and starting the service
├── sequence_generator.py      # AI sequence generator for preparing images for training
├── requirements.txt           # Python dependencies for the project
├── README.md                  # Project documentation
├── setup.sh                   # Setup script for installing dependencies
├── service.sh                 # Service script to manage the liquidLapse process
├── docs/                      # Operational docs for backup / restore workflows
│   └── backup_restore.md
└── ai_process/                # Directory for AI processing outputs
    └── session_name/          # Session folders (named by date/time)
        ├── dataset_info.json  # Metadata for all processed images
        ├── images/            # Processed image files
        └── sequences/         # Generated image sequences
            ├── sequences_info.json   # Metadata for all sequences
            └── sequence_n/   # Individual sequence folders
                └── images/   # Sequence images
```

---

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
chmod +x service.sh
chmod +x backup_heatmaps_to_mega.sh
chmod +x restore_heatmaps_from_mega.sh
```

Then run it:

```bash
./setup.sh
```

This script will:

* Install system dependencies (Python, pip, Google Chrome).
* Set up a virtual environment.
* Install Python dependencies.

### 3. Configure the Application

Edit the `config.yaml` file to customize the settings for snapshot capture:

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

The `service.sh` script can be used to manage the liquidLapse service.

* **Start the Service:**

  ```bash
  ./service.sh start
  ```

  This will run `liquidLapse.py` in the background, logging output to `liquidLapseService.log`.

* **Run the Service in the Foreground:**

  ```bash
  ./service.sh run
  ```

  Use this mode under a process supervisor such as `tmux`, `systemd`, or a container runner. It keeps the long-running loop as the foreground process while still writing `liquidLapseService.pid`, `liquidLapseService.status`, and `service_health.log`.

* **Install as a systemd Service:**

  ```bash
  sudo install -m 0644 deploy/liquidlapse-capture.service /etc/systemd/system/liquidlapse-capture.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now liquidlapse-capture.service
  ```

  This is the preferred production mode on Linux servers because the capture loop survives SSH disconnects and restarts after host reboot.

* **Stop the Service:**

  ```bash
  ./service.sh stop
  ```

* **Restart the Service:**

  ```bash
  ./service.sh restart
  ```

* **Check Service Status:**

  ```bash
  ./service.sh status
  ```

### 5. Running Directly (Without Service Management)

You can also run the snapshot capture script directly:

```bash
source venv/bin/activate   # On Windows use: venv\Scripts\activate
python liquidLapse.py
```

Press `Ctrl+C` to stop the script.

### 6. Generating AI Training Sequences

After capturing snapshots, generate training sequences for AI:

```bash
source venv/bin/activate
python sequence_generator.py
```

This script will:

1. Process snapshots from the `heatmap_snapshots` directory.
2. Generate fixed-length sequences with metadata.
3. Include future prediction data.
4. Save everything in `ai_process/[session_name]/sequences`.

### 7. Training the AI Model

To train the AI model (CNN + LSTM):

```bash
source venv/bin/activate
python train_model.py
```

This script:

1. Loads the sequences generated in step 6.
2. Trains the model using the provided configurations.
3. Saves the best model as `best_model.pt`.

### 8. FastAPI Server for Real-Time Prediction

To run the FastAPI server for real-time predictions:

```bash
source venv/bin/activate
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8100
```

The API provides a `/predict` endpoint where you can get predictions based on the latest model and heatmap images. It supports passing specific model names and image files for prediction or defaults to the latest available model and the most recent images.

---

## Additional Notes

* **AI Training Pipeline Integration:**
  The sequence generation feature creates structured data ready for machine learning model training. Each sequence contains chronological snapshots, price change data, and prediction targets.

* **Deployment to VPS:**
  The system is designed to run in headless mode, making it suitable for deployment on remote VPS servers for both heatmap capturing and prediction tasks.

* **Scheduling:**
  If not using the `service.sh` script, you can also set up cron jobs (Linux) or Windows Task Scheduler for managing snapshot capturing.
  On headless servers, prefer supervising `./service.sh run` directly so the capture loop has a stable parent process.

* **Logs & Status:**

  * Runtime logs are saved to `liquidLapseService.log`.
  * Status information is saved in `liquidLapseService.status`.

* **Heatmap backup / restore:**

  * The live `heatmap_snapshots/` layout stays flat and unchanged.
  * Use `./backup_heatmaps_to_mega.sh` to freeze the current batch, snapshot metadata, generate manifests, and upload to MEGA.
  * Use `./restore_heatmaps_from_mega.sh` to download a batch, verify it, and optionally merge it back into the live flat folder.
  * Existing MEGA backup folders are reused safely, so routine uploads can target the same MEGA root.
  * `sync_heatmaps.sh` remains the legacy rsync pull helper for a different host-to-host workflow.
  * Batch-based backup, MEGA reuse, optional SSH relay, verification, and safe restore steps are documented in [docs/backup_restore.md](docs/backup_restore.md).

---

## License

This project is licensed under the Apache 2.0 License.
