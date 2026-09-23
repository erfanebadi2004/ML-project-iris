from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pipeline_rfc():
    pipeline = Pipeline(steps=[
        ("StandardScaler", StandardScaler()),
        ("PCA", PCA(n_components=3)),
        ("Classifier", RandomForestClassifier(random_state=42)),
    ])
    
    return pipeline