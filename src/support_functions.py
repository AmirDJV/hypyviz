# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:48:49 2020

@author: Amir
"""

def matchPartners(subject1 = None, filesToProcessS2 = None, locationSettings = None):
    
    #Test if all vars were defined
    assert len([t for t in [subject1, filesToProcessS2, locationSettings] if t is None]) == 0, "one of the vars was not defined" 
    
    #Loading subjects
    s1 = subject1
    #Getting the matching file - subject 2
    s2 = filesToProcessS2[filesToProcessS2.index(s1.replace("SUBJECT1", "SUBJECT2"))]
    
#    print(s1, "\n", s2)
    
    #Load files
    epochsS1 = mne.read_epochs(s1, preload=True, verbose = False)
    epochsS2 = mne.read_epochs(s2, preload=True, verbose = False)
        
    #Rename electrodes names
    epochsS1.rename_channels(dict(zip(epochsS1.info["ch_names"], [i + "-0" for i in epochsS1.info["ch_names"]])))
    epochsS2.rename_channels(dict(zip(epochsS2.info["ch_names"], [i + "-1" for i in epochsS2.info["ch_names"]])))
    
    #Combining the subjects to one cap
    combined = combineEpochs(epochsS1 = epochsS1, epochsS2 = epochsS2)
    
    #Adding sensors(channles) locations
    if type(combined) is not str:
        combined.info["chs"] = locationSettings.copy()
    
    return combined


def matchDescription(epochs = None, description = None):
    indexDrop = []
    goodCounter = -1
    for i in range(0, len(epochs.drop_log)):
        #Testing for good data in subject
        if any(epochs.drop_log[i]) == False:
            #Counter of good data, for indexing the epochs file
            goodCounter += 1
            #Testing to see if in the description the data is bad
            if description[i] == ["bad data"]:
                #Adding the index goodCounter to the list for droping
                indexDrop.append(goodCounter)
    #Droping the bad epochs
    epochs.drop(indexDrop, reason ='bad combined')
    epochs.drop_bad()
    return epochs


def combineEpochs(epochsS1 = None, epochsS2 = None):
    #Creating the list of the good/bad epochs in both subjects
    description = []
    for logA, logB in zip(epochsS1.drop_log, epochsS2.drop_log): 
        if (any(logA) == True) | (any(logB) == True):
            description.append(["bad data"])
        else:
            description.append(["good data"])
    
    if len([d for d in description if d == ["good data"]]) < 5: 
        return("Not enoguh good data")
    
    
    #Matching that bad/good epochs for each particiant. It's the intersection of good epochs
    matchDescription(epochs=epochsS1, description=description)
    matchDescription(epochs=epochsS2, description=description)
    
    #Combine matched epochs as one cap, for that I rebuild the two epochs as one epoch structure from scratch
    ##concatenating the data from the caps as one
    
    #Concatenate epochs from subject 1 and subject2
    data = np.concatenate((epochsS1, epochsS2), axis=1)
    ##Creating an info structure
    info = mne.create_info(
            ch_names = list(epochsS1.info["ch_names"] + epochsS2.info["ch_names"]),
            ch_types = np.repeat("eeg", len(list(epochsS1.info["ch_names"] + epochsS2.info["ch_names"]))),
            sfreq = epochsS1.info["sfreq"])
    ##Creating an events structure
    events = np.zeros((data.shape[0], 3), dtype="int32")
    
    #Naming the events by the name of of the original epoch number. 
    #e.g. event == 289 is epoch 289 in the original data
    eventConter = 0
    for i, d in enumerate(description):
        if d == ['good data']:
            events[eventConter][0] = i
            events[eventConter][2] = i 
            eventConter +=1
    ##Creating event ID
    event_id = dict(zip([str(item[2]) for item in events], [item[2] for item in events]))
    ##Time of each epoch
    tmin = -0.5
    ##Building the epoch structure
    combined = mne.EpochsArray(data, info, events, tmin, event_id)
    ##Editing the channels locations
    combined.info["chs"] = epochsS1.info["chs"] + epochsS2.info["chs"]
    
    #test to see that all the good epochs are in the same length
    if len(set(map(len,[epochsS1, epochsS2, combined]))) == 1 and sum([i == ['good data'] for i in description]) == len(epochsS1):
        print("All are the same length")
    else:
        print("ERROR - They are not the same length!") 

    return combined