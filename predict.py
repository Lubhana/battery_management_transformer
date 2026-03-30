import numpy as np

def predict(x):
    # TEMPORARY dummy logic (replace with your model)
    score = np.mean(x)

    if score > 30:
        return "⚠️ Anomaly Detected"
    else:
        return "✅ Normal Operation"