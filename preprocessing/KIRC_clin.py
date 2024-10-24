import pandas as pd

def main():
    df = pd.read_csv("data/raw/KIRC.clin.merged.txt", sep="\t").transpose()

    # transform to first row to column names
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.reset_index(drop=True)

    # required columns all "days_to_death" and all "days_to_last_followup"
    allDaysToDeathColumnNames = [i for i in df.columns if "days_to_death" in i]
    allDaysToLastFollowUpColumnNames = [i for i in df.columns if "days_to_last_followup" in i]

    daysToDeathList = list()

    daysToDeathDf = df[allDaysToDeathColumnNames]
    for index, row in daysToDeathDf.iterrows():
        maxDaysToDeath = 0
        for name in allDaysToDeathColumnNames:
            if type(row[name]) == str:
                if maxDaysToDeath < int(row[name]):
                    maxDaysToDeath = int(row[name])
        daysToDeathList.append(maxDaysToDeath)

    daysToLastFollowUpList = list()

    daysToLastFollowUp = df[allDaysToLastFollowUpColumnNames]
    for index, row in daysToLastFollowUp.iterrows():
        maxDaysToLastFollowUp = 0
        for name in allDaysToLastFollowUpColumnNames:
            if type(row[name]) == str:
                if maxDaysToLastFollowUp < int(row[name]):
                    maxDaysToLastFollowUp = int(row[name])
        daysToLastFollowUpList.append(maxDaysToLastFollowUp)
    
    patientBarcode = list(df["patient.bcr_patient_barcode"])
    
    patientDead = [1 if daysToDeath > 0 else 0 for daysToDeath in daysToDeathList]

    time_days = list()
    for index in range(len(patientBarcode)):
        # check which value is greater
        if daysToDeathList[index] > daysToLastFollowUpList[index]:
            time_days.append(daysToDeathList[index])
        else:
            time_days.append(daysToLastFollowUpList[index])


    data = {
        "patient_barcode": patientBarcode,
        "time_days": time_days,
        "patient_dead": patientDead
    }
    compiledDf = pd.DataFrame(data=data)
    return compiledDf


if __name__ == "__main__":
    df = main()
    df.to_csv("data/processed/KIRC_clinical.csv", index=False)