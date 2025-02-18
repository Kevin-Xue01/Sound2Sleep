import os
import xml.etree.ElementTree as ET

import pandas as pd

pd.set_option('display.max_columns', None)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'data', 'kevin', 'export.xml')
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