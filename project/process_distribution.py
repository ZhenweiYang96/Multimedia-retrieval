from processing_distribution_function import *

t0 = time.time()
df = pd.read_csv('excel_file\\features.csv')
df[['A3', 'D1', 'D2', 'D3', 'D4']] = df[['A3', 'D1', 'D2', 'D3', 'D4']].apply(convert_to_float)
df = to_bin(df, True)
t1 = time.time()
print(t1 - t0)
df.to_csv('excel_file\\standardized.csv')
