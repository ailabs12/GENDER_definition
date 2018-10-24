import numpy as np
import scipy.io
from datetime import datetime, timedelta

# print("Load data from mat file...")
data = scipy.io.loadmat("/home/kdg/kdg_projects/Age/imdb_crop/imdb.mat", 
                        squeeze_me=True)
data = data["imdb"]
keys = data.dtype.names

#print("Preprocess data...")
headers = "; ".join(keys)
dt = []
for key in keys:
    dt.append(data[key].flatten()[0].tolist())

for i in range(len(dt[0])):
    if dt[0][i] <= 5000:
        dt[0][i] = None
    else:
        dt[0][i] = (datetime.fromordinal(dt[0][i]) + 
            timedelta(days=dt[0][i]%1) -
            timedelta(days=366))
for i in range(len(dt[5])):
    dt[5][i] = dt[5][i].tolist()

print("Combining features in sets...")
dataset = list(zip(*dt))
print("Done.")

print("Write dataset in csv file...")
i = 1
with open("/home/kdg/kdg_projects/Age/imdb_crop/imdb.csv",
          "w", encoding="utf-8") as f:
    f.write(headers + "\n")
    for row in dataset:
        f.write("; ".join([str(item) for item in row])+"\n")
        print("{0}nt row is written".format(i))
        i += 1
f.close()

""" Testing """
Y = pd.read_csv('/home/kdg/kdg_projects/Age/imdb_crop/imdb.csv', sep=';')  
        
