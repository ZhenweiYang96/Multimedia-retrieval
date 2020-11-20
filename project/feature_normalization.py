from matching_function import *

df = pd.read_csv('excel_file\\standardized.csv')
df[['A3', 'D1', 'D2', 'D3', 'D4']] = df[['A3', 'D1', 'D2', 'D3', 'D4']] .apply(adjust_string_to_float)

df['surface_area'] = standardize_single_value(df['surface_area'])
df['sphericity'] = standardize_single_value(df['sphericity'])
df['bounding_box_volume'] = standardize_single_value(df['bounding_box_volume'])
df['diameter'] = standardize_single_value(df['diameter'])
df['eccentricity'] = standardize_single_value(df['eccentricity'])

df['A3'] = to_percentage(df['A3'])
df['D1'] = to_percentage(df['D1'])
df['D2'] = to_percentage(df['D2'])
df['D3'] = to_percentage(df['D3'])
df['D4'] = to_percentage(df['D4'])

df[['A3', 'D1', 'D2', 'D3', 'D4']] = df[['A3', 'D1', 'D2', 'D3', 'D4']] .apply(convert_scientific_notation)
df.to_csv('excel_file\\matching_features.csv')

