# Wine Recommender

Clasification of similars wines project

## Objective
Develop a wine similarity model that can be applied to recommend similar products.

## How the Objective Was Achieved

To build the wine similarity model:

1. The raw wine dataset was cleaned and processed into a structured DataFrame.
2. Relevant features were extracted from the DataFrame for modeling.
3. An autoencoder was trained on these features to learn a latent representation of each wine.
4. The resulting embeddings were used to measure similarity between wines.

## Data
Wine descriptions and metadata in Spanish, collected from [Club de Amantes del Vino](https://cav.cl).

## How to Run

To run the Wine Recommender, follow these steps:

1. **Clone the repository**  

```bash
git clone https://github.com/VCardenasPoza/wine-recommender.git
cd wine-recommender
```

2. **Create and activate a virtual environment**  

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install required packages**  

```bash
pip install -r requirements.txt
```

4. **Run embeddings from zero**  

```bash
python sripts/make_embeddings.py
```

This script generate clean DataFrame from raw, train de autoencoder and generate the embeddings of wines in latent space. Default varietys are set in n = 10, could be modified in the script.

5. **Use the model**  

Open [This notebook]("notebooks/02_encoder_test.ipynb") to explore similar wines.

Open [This notebook]("notebooks/03_visualization.ipynb") to interact with the visualizations.

## Example Similarity 

```python
from src.evualuation.retrieval import top_k
import numpy as np
import pandas as pd

# Load precomputed embeddings
Z = np.load("models/embeddings/wine_embeddings.npy")

# Load clean dataframe

df = pd.read_csv("data/processed/wines_clean.csv")

# Find the 5 most similar wines to wine #42
wine_name =  df["name"].iloc[42]

similar_wines = top_k(wine_id=42,k=5)


print(f"The 5 most similar wines to \033[1m{wine_name}\033[0m are:")

for name in df.iloc[similar_wines]["name"].values:
    print(name)
```
**Expected Output**

```
The 5 most similar wines to Undurraga trama pinot noir 2016 are:
Bodegas tt mane noir pinot noir 2017
Morande gran reserva pinot noir 2016
Santa alba grand reserve pinot noir 2017
Casas del bosque pinot noir 2017
Garces silva boya pinot noir 2016
```
## Visualizations

You can visualize wines in the latent space using UMAP:
![UMAP de vinos](reports/umap_wines.png)

For interactive exploration, download the HTML visualization:

[Interactive UMAP](reports/umap_wines.html)