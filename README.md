# ENTSO-E Data Analysis

This project is a data analysis pipeline for downloading, processing, and visualizing electricity market data from the ENTSO-E (European Network of Transmission System Operators for Electricity) platform.

## Project Structure

```
├── Snakefile           # Snakemake workflow definition
├── config.yaml         # Configuration for the Snakemake pipeline
├── requirements.txt    # Python dependencies
├── environment.yml     # Conda environment definition
├── data/               # Raw and processed data
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── results/            # Output files, such as plots and tables
└── scripts/            # Python scripts for data downloading, processing, and plotting
```

## Setup and Usage
0.  **Get API key:**

    See [here](https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188-How-to-get-security-token). Send an email to transparency@entsoe.eu with “Restful API access” in the subject line. Indicate the email address you entered during registration in the email body. 

1.  **Create the Conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the environment:**

    ```bash
    conda activate cfp-data-analysis
    ```

3.  **Configure the pipeline:**

    Edit the `config.yaml` file to specify the data download period and other parameters.

4.  **Run the Snakemake pipeline:**

    ```bash
    snakemake --cores 1 --scheduler greedy
    ```

    This will execute the workflow, from data downloading to plot generation (depending on configuration).

## Workflow

The `Snakefile` defines the following steps:

1.  **Download data:** The `download_entsoe_data.py` script downloads data from the ENTSO-E API.
2.  **Integrate data:** The `integrate_data.py` script combines the raw download files into bigger files by datatype.
3.  **Process data:** The `processing.py` script processes the raw data and combines it into `processed_data.feather`.
