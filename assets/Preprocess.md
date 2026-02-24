# Data Preprocess Tutorial

There is a detailed tutorial for preprocessing the raw data into processed data for `BatteryLife` project. Please follow the steps below carefully to process the raw data.



## Step1: Prepare the preprocess scripts and install the BatteryML package

1. run the code to clone the [BatteryML](https://github.com/microsoft/BatteryML) :

   ```
   git clone https://github.com/microsoft/BatteryML.git
   ```

2. Copy all files under `process_scripts` from `BatteryLife` to `/BatteryML/batteryml/preprocess/` folder from `BatteryML` and run the code below to install the `batteryml`:

   ```
   pip install -r requirements.txt
   pip install .
   ```

   

## Step2: Prepare the raw data file

1. Download raw datasets from **Raw Data Acquisition** of [Data availability](../README.md#data-availability).

2. **Create a folder** named `./datasets/raw/` under the root directory of `BatteryLife` repository and move all raw datasets under it. The final raw data preparation before preprocessing should be like this:

   ```
   /BatteryML-main/datasets/raw/
   |──CALB
   |──CALCE
   |──HNEI
   |──HUST
   |──ISU_ILCC
   |──MATR
   |──MICH
   |──MICH_EXP
   |──NAion
   |──RWTH
   |──SNL
   |──Stanford
   |──Tongji
   |──UL_PUR
   |──XJTU
   |──ZNion
   ```



## Step3: Run the command

1. Run the code:

   ```
   python preprocess_scripts.py
   ```

2. Finally, you can get the processed data under `./datasets/processed/` folder.



