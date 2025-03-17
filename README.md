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
Run to train estimation models 

    python src/trainModel.py --model-name transformer

- Weights of the trained model are saved in a h5 fil under the "models" folder.
- The name of the trained models are managed by the time of their creations.
- The model name argment can be either dense (default) or transformer.

---
## Inference Model
Run to inference the trained models


    python src/inferenceModel.py --model-name transformer

- RMSE and MAE are calculated and presented.
- True and predicted KPI are visualized and saved under "figures" folder.
- The model name argment can be either dense (default) or transformer.





