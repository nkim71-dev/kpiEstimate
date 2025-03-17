# KPI Estimation Tutorial

---
## Prepare Envrionment

Conda environment with python=3.11

* Create a new environment (e.g., "env")

        conda create -n env python=3.11

* Activate conda environment (e.g., "env")

        conda activate env




Install packages

    pip install -r requirements.txt

* If an error occurs

    Install package installer

        conda install pip

    Update pip to the latest version

        pip install --upgrade pip

---
## Processing Data
Preprocess data in the "data" folder

    python src/preprocessData.py

[NOTE] Data parsing will be shortly added.

- Dataset
    - https://github.com/uccmisl/5Gdataset

- Feature
    - The features will be later described in "data/columns.json"

- Normalization
    - The KPI is normalized with standard normalization.
    - All input features are normalized with minmax normalizaiton.


---
## Train Model
Run to train leakage classifier models 

    python src/trainModel.py 

- A folder with trained models is generated under "models" folder 
- The name of the trained models are managed by the time of their creations.

---
## Inference Model
Run to inference the trained leakage classifier models


    python src/inferenceModel.py

- RMSE and MAE are calculated and presented.
- True and predicted KPI are visualized and saved under "figures" folder.





