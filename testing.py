import os
import pandas as pd
from IPython.display import display

df = pd.read_csv(os.path.join("sample_data", "time-series.csv"))
display(df)
