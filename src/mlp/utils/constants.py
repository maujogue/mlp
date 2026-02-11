FEATURE_COLUMNS = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

LABELS = {
    "B": 0,  # Benign
    "M": 1,  # Malignant
}

DEFAULT_DATASET = "datasets/data.csv"

BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"

LABEL_COLORS = {
    "M": "#D62728",  # Beautiful red
    "B": "#2CA02C",  # Beautiful green
}
