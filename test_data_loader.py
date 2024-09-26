import matplotlib.pyplot as plt
import cahDataLoader

data = cahDataLoader.load('output/cah_20210423T132633.json')

hf, hax = plt.subplots()
hax.plot(data['times']['EEG'], data['data']['EEG'])
plt.show()
