import pandas as pd
import math as m

def main():
    df = pd.read_csv("KIRC.clin.merged.txt", sep="\t").transpose()

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

    data = {
        "patient_barcode": patientBarcode,
        "days_to_death": daysToDeathList,
        "days_to_last_followup": daysToLastFollowUpList,
        "patient_dead": patientDead
    }
    compiledDf = pd.DataFrame(data=data)
    return compiledDf


if __name__ == "__main__":
    df = main()
    df.to_excel("output.xlsx")
    df.to_csv("output.txt")
    df.to_html("output.html")