import os
import json
import pandas as pd

home = os.path.expanduser("~")
dir  = "/media/sumeyer/SSD_2/ML_DATA/programmatic-dataprovider/data/de/training-datasets/v4/features.out.json/"
filename = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738.json"
filename = "test_sample.json"
datafile = dir + filename

print("loading file : ", datafile),
df = pd.read_json("file://localhost"+datafile)
print("DONE")
print(df.shape)

# Get quick count of rows in a DataFrame
print("len(df.index) = ", len(df.index))

for index, row in df.iterrows():
    print(index, row) # ['some column']
