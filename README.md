# ReNF: Rethinking the Principles of Neural Long-Term Time Series Forecasters. 


## Quick Start

1. **Download datasets** from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr)

2. **Update dataset paths** in the `scripts/` directory to match your local setup

3. **Run experiments** using the provided scripts:
   ```bash
   bash ./scripts/traffic.sh      # Traffic dataset
   bash ./scripts/electricity.sh  # Electricity dataset
   bash ./scripts/weather.sh     # Weather dataset
   # ... other datasets available
   ```

## Supported Datasets
- ETTh1/ETTh2, ETTm1/ETTm2
- Electricity, Traffic, Weather, Solar
- Others (M4, PEMS...)
