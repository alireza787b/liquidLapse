## Overview

This project demonstrates a novel framework for real-time cognitive monitoring of pilots by predicting the exceedance shape factor—a robust indicator of deviations from ideal flight parameters. The system fuses multimodal data from non-intrusive eye-tracking and control stick inputs using advanced signal processing (Continuous Wavelet Transform) to generate scalograms. These scalograms are then used as inputs to deep Convolutional Neural Networks (CNNs) for regression. The framework has potential applications in aviation, driving, train operations, and air traffic control.

---

## Repository Structure

- **data/**  
  Contains raw and preprocessed data (scalogram images, flight data, etc.).

- **src/**  
  Source code for signal processing, feature extraction, and deep learning model training.

- **models/**  
  Pre-trained CNN models and scripts for network adaptation.

- **docs/**  
  Supplementary materials, documentation, and the paper’s final version.

- **setup.sh**  
  Bash script for setting up the environment and installing dependencies.

- **README.md**  
  This file.

---

## Requirements

- **Python:** 3.7 or higher  
- **MATLAB:** Required for running signal processing scripts  
- **Deep Learning Framework:** TensorFlow or PyTorch (depending on your configuration)  
- **GNU Bash:** For running the setup script  
- **Other Dependencies:** Listed in `requirements.txt`

---

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Run the Setup Script:**

   The `setup.sh` script installs the necessary Python dependencies and configures the environment.
   
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Configure MATLAB:**

   Ensure MATLAB is set up to run the scripts in the `src/` directory. Update any necessary file paths in the MATLAB scripts.

4. **Download Pre-trained Models (if needed):**

   If pre-trained models are not present in the `models/` folder, download them from [Insert Link Here] and place them accordingly.

---

## Running the Project

### 1. Data Preprocessing and Feature Extraction

Run the MATLAB script to preprocess raw data and generate scalograms:
   
```matlab
run('src/preprocess_and_extract.m');
```

### 2. Model Training

Train the deep learning models using the provided configuration:
   
```bash
python src/train_model.py --config configs/training_config.yaml
```

### 3. Model Evaluation

Evaluate the trained model using the test data:
   
```bash
python src/evaluate_model.py --model models/best_model.h5 --data data/test/
```

---

## References

For detailed context and additional documentation, please refer to the supplementary materials in the `docs/` folder.

---

## Contact

For questions or issues, please contact the corresponding author:  
**Fariborz Saghafi**  
Faculty of Aerospace Engineering Department, Sharif University of Technology, Tehran, Iran  
Email: [saghafi@sharif.edu](mailto:saghafi@sharif.edu)

---

## License

This project is licensed under the Apache 2.0 License.