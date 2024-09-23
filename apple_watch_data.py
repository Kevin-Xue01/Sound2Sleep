import xml.etree.ElementTree as ET
import pandas as pd
import os

pd.set_option('display.max_columns', None)

file_path = os.path.expanduser('~/Downloads/apple_health_export/export.xml')
tree = ET.parse(file_path) 
root = tree.getroot()
record_list = [x.attrib for x in root.iter('Record')]
data = pd.DataFrame(record_list)

# proper type to dates
for col in ['creationDate', 'startDate', 'endDate']:
    data[col] = pd.to_datetime(data[col])
data = data[data['type'] == 'HKCategoryTypeIdentifierSleepAnalysis']
data = data[data['value'] != 'HKCategoryValueSleepAnalysisInBed']
data['value'] = data['value'].str.replace('HKCategoryValueSleepAnalysis', '', regex=False)
data = data.drop(columns=['type', 'sourceVersion', 'unit', 'creationDate', 'device'])

data.to_csv('apple_watch_sleep_data.csv', index=False)