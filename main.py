import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle

# import functions
from preprocessing.process_clinical import process_clinical_data_from_raw
from preprocessing.process_RSEM import process_rsem_data_from_raw
from preprocessing.process_copyno import process_copyno_data_from_raw
from preprocessing.process_miRNA import process_mirna_data_from_raw
from preprocessing.process_RPPA import process_rppa_data_from_raw
from CoxUniScreening import cox_univariate_screening
from CoxUniScreening import extract_top_features

# import models
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, KFold
from pprint import pprint


def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")

    plt.show()

def custom_penalizer(feature_range, penalties):
    # create custom penalizer
    alpha = np.zeros(feature_range[-1])

    # for feature 0-500 have penalty of 0.1
    for i in range(feature_range[0], feature_range[1]):
        alpha[i] = penalties[0]

    # for feature 500-1000 have penalty of 0.5
    for i in range(feature_range[1], feature_range[2]):
        alpha[i] = penalties[1]

    # for feature 1507-1638 have penalty of 0.3
    for i in range(feature_range[2], feature_range[3]):
        alpha[i] = penalties[2]

    # for feature 1638-2000 have penalty of 0.4
    for i in range(feature_range[3], feature_range[4]):
        alpha[i] = penalties[3]

    return alpha


def main():
    clin = process_clinical_data_from_raw()
    rsem = process_rsem_data_from_raw()
    copyno = process_copyno_data_from_raw()
    mirna = process_mirna_data_from_raw()
    rppa = process_rppa_data_from_raw()

    # print out the features size of each dataset
    print(f"Clinical: {clin.shape}")
    print(f"RSEM: {rsem.shape}")
    print(f"CopyNo: {copyno.shape}")
    print(f"miRNA: {mirna.shape}")
    print(f"RPPA: {rppa.shape}")
    # Clinical: (536, 3)
    # RSEM: (533, 20531)
    # CopyNo: (528, 24777)
    # miRNA: (243, 508)
    # RPPA: (454, 132)


    # run cox univariate screening
    rsem_screening_result = cox_univariate_screening(rsem, clin, "KIRC_RSEM")
    copyno_screening_result = cox_univariate_screening(copyno, clin, "KIRC_COPYNO")

    # extract top 500 features
    rsem_top500 = extract_top_features(rsem, rsem_screening_result, top=500)
    copyno_top500 = extract_top_features(copyno, copyno_screening_result, top=500)

    # delete repeated rows
    rsem_top500 = rsem_top500.drop_duplicates(subset=['patient_barcode'], keep="last")
    copyno_top500 = copyno_top500.drop_duplicates(subset=['patient_barcode'], keep="last")
    mirna = mirna.drop_duplicates(subset=['patient_barcode'], keep="last")
    rppa = rppa.drop_duplicates(subset=['patient_barcode'], keep="last")

    # get column indices of each data type
    rsem_cols = rsem_top500.shape[1] - 1
    copyno_cols = copyno_top500.shape[1] - 1
    mirna_cols = mirna.shape[1] - 1
    rppa_cols = rppa.shape[1] - 1

    print(f"rsem_cols: {rsem_cols}")
    print(f"copyno_cols: {copyno_cols}")
    print(f"mirna_cols: {mirna_cols}")
    print(f"rppa_cols: {rppa_cols}")
    # rsem_cols: 500
    # copyno_cols: 500
    # mirna_cols: 507
    # rppa_cols: 131

    # merge all genomic features
    merged = clin.merge(rsem_top500, on='patient_barcode', how='outer')
    merged = merged.merge(copyno_top500, on='patient_barcode', how='outer')
    merged = merged.merge(mirna, on='patient_barcode', how='outer')
    merged = merged.merge(rppa, on='patient_barcode', how='outer')


    print(f"merged: {merged.shape}")
    # merged: (537, 1641)

    # Impute missing values with KNN
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(merged.drop(['patient_barcode', 'dead', 'time(day)'], axis=1))

    # Convert back to DataFrame with column names
    df = pd.DataFrame(imputed_data, columns=merged.drop(['patient_barcode', 'dead', 'time(day)'], axis=1).columns)

    # Add back the patient_barcode column
    df.insert(0, 'patient_barcode', merged['patient_barcode'])

    # Add back the dead column
    df.insert(1, 'dead', merged['dead'])

    # Add back the time(day) column
    df.insert(2, 'time(day)', merged['time(day)'])

    df = df.dropna()

    # extract predictors
    Xt = df.drop(['patient_barcode', 'dead', 'time(day)'], axis=1)

    # extract outcome and convert to numpy array
    y = np.array(list(zip(df['dead'], df['time(day)'])), 
                 dtype=[('dead', bool), ('time(day)', float)])
    
    # fit cox elastic net model
    cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01)

    cox_elastic_net.fit(Xt, y)

    # find out selected features
    coefficients_elastic_net = pd.DataFrame(
    cox_elastic_net.coef_, index=Xt.columns, columns=np.round(cox_elastic_net.alphas_, 5))

    # plot_coefficients(coefficients_elastic_net, n_highlight=5)

    # choosing alphas
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100))
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    coxnet_pipe.fit(Xt, y)

    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv=cv,
        error_score=0.5,
        n_jobs=1,
    ).fit(Xt, y)

    cv_results = pd.DataFrame(gcv.cv_results_)

    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)

    plt.savefig("concordance_index.png")


        
if __name__ == "__main__":
    main()