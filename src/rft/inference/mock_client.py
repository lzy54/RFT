class MockLLMClient:
    def __init__(self):
        pass

    def generate(self, *args, **kwargs):
        return """
<thinking>
I will generate a CSV with required feature columns and a numeric risk score.
</thinking>

```python
import pandas as pd
from pathlib import Path

def main():
    Path("pred_results").mkdir(exist_ok=True)
    df = pd.DataFrame({
        "x0": [1.0, 2.0],
        "x1": [0.5, 0.3],
        "risk": [0.1, 0.2],
    })
    df.to_csv("pred_results/cox_ph_predictions.csv", index=False)

if __name__ == "__main__":
    main()

```"""

    def close(self):
        """Mock client has no resources to release."""
        return None
