import pandas as pd

# read the CSV file
df = pd.read_csv('test_dataset.csv')

# shuffle the rows
df = df.sample(frac=1)

# write the shuffled data back to the CSV file
df.to_csv('test_dataset.csv', index=False)

# read the CSV file
df = pd.read_csv('train_dataset.csv')

# shuffle the rows
df = df.sample(frac=1)

# write the shuffled data back to the CSV file
df.to_csv('train_dataset.csv', index=False)