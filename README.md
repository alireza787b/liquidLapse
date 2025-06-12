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

   * A `config.yaml` file allows easy customization of settings such as target URL, snapshot intervals, BTC price API URL, AI prediction settings, and more.

4. **Service Management**

   * A `service.sh` script to manage the execution of the snapshot capture service.
   * Supports starting, stopping, and checking the status of the service.

5. **AI Model Training & Prediction**

   * Integrates **CNN + LSTM** for sequential prediction based on heatmap snapshots.
   * Trains models and makes real-time predictions.
   * Pushes and pulls models between local and remote servers.

6. **Prediction Service (Automated or On-Demand)**

   * **predict_once.py**: Runs a single prediction using the latest model and appends the result to a session-based JSON file.
   * **prediction_service.sh**: Bash script to manage the prediction service (start, stop, status, restart, health).
   * Designed for robust 24/7 operation on VPS or servers, including auto-restart and log monitoring.
   * Uses the same config.yaml for session/model selection and prediction intervals.

7. **API for Predictions**

   * FastAPI-based server for real-time predictions.
   * Offers flexibility to input the last image and number of frames to predict future market changes.

8. **Cross-Platform Support**

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
├── prediction_service.sh      # Service script to manage the prediction service (AI predictions)
└── ai_process/                # Directory for AI processing outputs
    ├── predict_once.py        # Script to run a single prediction and save result
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
chmod +x prediction_service.sh
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

Edit the `config.yaml` file to customize the settings for snapshot capture and prediction:

```yaml
# config.yaml
url: "https://www.coinglass.com/pro/futures/LiquidationHeatMap"
check_interval: 300                      # Interval (in seconds) between snapshots
output_folder: "heatmap_snapshots"
headless: true                           # Run Chrome in headless mode
include_price_in_filename: true          # Append BTC price to the filename
btc_price_url: "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

prediction:
  enabled: true
  prediction_interval: 300               # Interval (in seconds) between predictions
  session: "test1"                       # Model session to use for predictions
  keep_model_in_memory: false            # Whether to cache the model in memory
  prediction_folder: "predictions"       # Where to store prediction results
```

---

## Managing the Prediction Service

The `prediction_service.sh` script manages the AI prediction service.

* **Start the Prediction Service:**

  ```bash
  ./prediction_service.sh start
  ```

* **Stop the Prediction Service:**

  ```bash
  ./prediction_service.sh stop
  ```

* **Restart the Prediction Service:**

  ```bash
  ./prediction_service.sh restart
  ```

* **Check Service Status:**

  ```bash
  ./prediction_service.sh status
  ```

* **Health Check:**

  ```bash
  ./prediction_service.sh health
  ```

The service will run `predict_once.py` at intervals defined in your `config.yaml` and append results to session-based JSON files. Logs are saved to `prediction_service.log`.

### Run a Single Prediction Manually

You can also run a single prediction manually:

```bash
source venv/bin/activate   # On Windows use: venv\Scripts\activate
python ai_process/predict_once.py
```

---

## Example Prediction Output

Each prediction is appended to a compact JSON file for the session:

```json
{
  "prediction_id": "pred_20240612_153000_ab12cd34",
  "timestamp": "2024-06-12T15:30:00.123456Z",
  "model_session": "test1",
  "model_path": "ai_process/test1/train_001/best_model.pt",
  "target_field": "future_future_4h_change_percent",
  "sequence_start": "2024-06-12T14:30:00Z",
  "sequence_end": "2024-06-12T15:25:00Z",
  "btc_price": 67000.12,
  "prediction": 0.0234,
  "processing_time_ms": 120.5,
  "device": "cuda"
}
```

---

## Additional Notes

* **AI Training Pipeline Integration:**
  The sequence generation feature creates structured data ready for machine learning model training. Each sequence contains chronological snapshots, price change data, and prediction targets.

* **Deployment to VPS:**
  The system is designed to run in headless mode, making it suitable for deployment on remote VPS servers for both heatmap capturing and prediction tasks.

* **Scheduling:**
  If not using the `service.sh` or `prediction_service.sh` scripts, you can also set up cron jobs (Linux) or Windows Task Scheduler for managing snapshot capturing and predictions.

* **Logs & Status:**
  * Runtime logs for prediction service are saved to `prediction_service.log`.
  * Status information is available via the `status` and `health` commands in the service script.

---

## Best Practices

- For 24/7 operation and auto-restart after server reboots, use the `prediction_service.sh` script with a process manager (e.g., systemd on Linux).
- Monitor logs (`prediction_service.log`) and use the `status` and `health` commands for troubleshooting.
- Always specify the desired model session in `config.yaml` for reproducible predictions.

---

## License

This project is licensed under the Apache 2.0 License.

