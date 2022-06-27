MID.csv file data perpocessing:
______________________

- removed nodes in small connected components ['GUA', 'BLZ', 'GAM', 'SEN', 'SAF', 'LES', 'SWA', 'MZM', 'GNB']
- added Guassian noise of mean=0 and std=1hour to events with same timestamp (Hawkes process doen't handle events occuring at the same time)
- scaled timestamps to be between [0:1000]
- Original dataset duaration = 8380 days