
import pandas as pd
import pickle
import time
import os

# old
# def process_clinical_data_from_raw(retrieve_from_cache=True):
#     if retrieve_from_cache:
#         # search for cache file
#         for file in os.listdir("cache"):
#             if "KIRC_clinical_processed" in file:
#                 print(f"Clinical cache found: cache/{file}")
#                 # load from cache
#                 with open(f'cache/{file}', 'rb') as f:
#                     return pickle.load(f)

#         print("Clinical cache not found")
#         print("start processing")

#     df = pd.read_csv("data/KIRC.clin.merged.txt", sep="\t").transpose()

#     # transform to first row to column names
#     df.columns = df.iloc[0]
#     df = df[1:]
#     df = df.reset_index(drop=True)

#     # required columns all "days_to_death" and all "days_to_last_followup"
#     allDaysToDeathColumnNames = [i for i in df.columns if "days_to_death" in i]
#     allDaysToLastFollowUpColumnNames = [i for i in df.columns if "days_to_last_followup" in i]

#     daysToDeathList = list()

#     daysToDeathDf = df[allDaysToDeathColumnNames]
#     for index, row in daysToDeathDf.iterrows():
#         maxDaysToDeath = 0
#         for name in allDaysToDeathColumnNames:
#             if type(row[name]) == str:
#                 if maxDaysToDeath < int(row[name]):
#                     maxDaysToDeath = int(row[name])
#         daysToDeathList.append(maxDaysToDeath)

#     daysToLastFollowUpList = list()

#     daysToLastFollowUp = df[allDaysToLastFollowUpColumnNames]
#     for index, row in daysToLastFollowUp.iterrows():
#         maxDaysToLastFollowUp = 0
#         for name in allDaysToLastFollowUpColumnNames:
#             if type(row[name]) == str:
#                 if maxDaysToLastFollowUp < int(row[name]):
#                     maxDaysToLastFollowUp = int(row[name])
#         daysToLastFollowUpList.append(maxDaysToLastFollowUp)
    
#     patientBarcode = list(df["patient.bcr_patient_barcode"].apply(lambda x: x.upper()))
    
#     patientDead = [1 if daysToDeath > 0 else 0 for daysToDeath in daysToDeathList]

#     time_days = list()
#     for index in range(len(patientBarcode)):
#         # check which value is greater
#         if daysToDeathList[index] > daysToLastFollowUpList[index]:
#             time_days.append(daysToDeathList[index])
#         else:
#             time_days.append(daysToLastFollowUpList[index])


#     data = {
#         "patient_barcode": patientBarcode,
#         "time_days": time_days,
#         "patient_dead": patientDead
#     }
#     compiledDf = pd.DataFrame(data=data)

#     # save to cache (include date time)
#     with open(f'cache/KIRC_clinical_processed_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
#         pickle.dump(compiledDf, f)
    
#     print("saved to cache")

#     return compiledDf

def process_clinical_data_from_raw(retrieve_from_cache=True):
    if retrieve_from_cache:
        # search for cache file
        for file in os.listdir("cache"):
            if "KIRC_clinical_processed" in file:
                print(f"Clinical cache found: cache/{file}")
                # load from cache
                with open(f'cache/{file}', 'rb') as f:
                    return pickle.load(f)

        print("Clinical cache not found")
        print("start processing")

    df = pd.read_csv("data/survival_KIRC_survival.txt", sep="\t")
    
    df = df[['_PATIENT', 'OS', 'OS.time']]

    # rename columns
    df.columns = ['patient_barcode', 'dead', 'time(day)']

    # save to cache (include date time)
    with open(f'cache/KIRC_clinical_processed_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    print("saved to cache")

    return df


if __name__ == "__main__":
    df = process_clinical_data_from_raw()
    print(df.head(20))