# Data

## Source
**Dataset of images for model development using transfer-learning and texture features recognition of the
conditions of construction materials with small datasets** (Version 1)
Mengiste, E., Mannem, K. R., Prieto, S. A., & García de Soto, B. — Mendeley Data, 17 May 2023
DOI: [10.17632/zv66h54b8j.1](https://doi.org/10.17632/zv66h54b8j.1) ·
Page: https://data.mendeley.com/datasets/zv66h54b8j/1
License: **CC BY 4.0** — free to use and share with attribution to the authors above.

Photographs of interior wall surfaces taken on a construction site at New York University Abu Dhabi,
labelled by construction stage / material condition.

| Class (folder name) | Images |
|---|---|
| CMU wall | 24 |
| Chiseled concrete | 49 |
| Concrete | 18 |
| First coat plaster | 37 |
| Gypsum | 26 |
| Mesh | 25 |
| Second coat plaster | 29 |
| **Total** | **208** |

Classes are imbalanced (largest/smallest ≈ 2.7), which is why macro-F1 is reported alongside accuracy.

## What is in this folder

| Folder | Tracked in Git? | Contents |
|---|---|---|
| `sample/` | yes | 2 images per class, downscaled to 1024 px on the long side, for a quick look at the data (CC BY 4.0, credited above) |
| `raw/` | **no** (`.gitignore`) | the full dataset — download it yourself as below |

## How to get the full dataset
1. Open the Mendeley page above and click **Download All**.
2. Unzip the archive (if it contains another zip, unzip that too).
3. Move the seven class folders into `data/raw/` so the layout is:

```
data/raw/
├── CMU wall/              CMU wall 0.jpg, CMU wall 1.jpg, ...
├── Chiseled concrete/
├── Concrete/
├── First coat plaster/
├── Gypsum/
├── Mesh/
└── Second coat plaster/
```

**Expected format:** JPEG photographs, one folder per class; the folder name is the class label.
Section 3 of the notebook stops if it does not find exactly 208 images in 7 classes.

On **Google Colab**, place the same `data/raw/` layout inside your Drive project folder
(`MyDrive/wall-material-project/data/raw/`), or change `PROJECT_DIR` in Section 2 of the notebook.
