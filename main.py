import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import logging

logging.basicConfig(filename="myfile.txt",level=logging.DEBUG)
logging.captureWarnings(True)

# import functions
from utils.preprocessing import process_clinical_data_from_raw, process_rsem_data_from_raw, process_copyno_data_from_raw, process_mirna_data_from_raw, process_rppa_data_from_raw
from utils.weight_selection import cox_lasso_group_weight_selection
from utils.coxnet_alpha_selection import cross_valaidate_coxnet, get_top_x_features_name
from utils.cox_univariate_screening import cox_univariate_screening
from utils.cox_univariate_screening import extract_top_features
from utils.graph_plotting_script import c_index_vs_alpha_parameter_tuning_plot, top_features_plot, kaplan_meier_plot, penalty_factors_plot

# import models
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.simplefilter("ignore", UserWarning)

def initialize_data():
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
    rsem_screening_result = cox_univariate_screening(rsem, clin)
    copyno_screening_result = cox_univariate_screening(copyno, clin)

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

    return clin, rsem_top500, copyno_top500, mirna, rppa

def clin_rsem_copyno_mirna_rppa_weighted_alpha(clin, rsem_top500, copyno_top500, mirna, rppa, output_graphs=False):
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
    
    # split data into training and testing
    Xt_train, Xt_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=42)
    
    group_indices = {
        'RSEM': range(0, 500),
        'COPYNO': range(500, 1000),
        'miRNA': range(1000, 1507),
        'RPPA': range(1507, 1638),
    }
    
    # fit adaptive cox lasso model
    weights, group_specific_weights = cox_lasso_group_weight_selection(
        Xt.values, y, group_indices, group_weights=None
    )

    print(pd.unique(weights))

    # cross validate weightedcoxnet model
    gcv = cross_valaidate_coxnet(Xt_train, y_train, weights)

    # test the model
    test_score = gcv.score(Xt_test, y_test)
    print(f"Test score: {test_score}")

    # output graphs
    if output_graphs:
        # check if "output/clin_rsem_copyno_mirna_rppa_weighted_alpha" directory exists
        # if not, create the directory
        if not os.path.exists("output/clin_rsem_copyno_mirna_rppa_weighted_alpha"):
            os.makedirs("output/clin_rsem_copyno_mirna_rppa_weighted_alpha")
            os.makedirs("output/clin_rsem_copyno_mirna_rppa_weighted_alpha/KMCs")

        # plot penalty factors
        penalty_factors_plot(group_indices, weights, show_plot=False, output_path="output/clin_rsem_copyno_mirna_rppa_weighted_alpha/penalty_factors_plot.png")

        # plot c-index vs alpha parameter tuning plot
        c_index_vs_alpha_parameter_tuning_plot(pd.DataFrame(gcv.cv_results_), gcv, show_plot=False, output_path="output/clin_rsem_copyno_mirna_rppa_weighted_alpha/alpha_tuning_plot.png")

        # plot top features
        top_features_plot(gcv, Xt, show_plot=False, output_path="output/clin_rsem_copyno_mirna_rppa_weighted_alpha/top_features_plot.png")
        
        # get top 20 features
        top_features = get_top_x_features_name(gcv, Xt, 20)

        # plot kaplan meier plot
        for feature in top_features:
            kaplan_meier_plot(df, "time(day)", "dead", feature, show_plot=False, output_path=f"output/clin_rsem_copyno_mirna_rppa_weighted_alpha/KMCs/KMC_{feature}.png")

    return gcv, test_score


def clin_rsem_copyno_mirna_rppa_unweighted_alpha(clin, rsem_top500, copyno_top500, mirna, rppa, output_graphs=False):
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
    
    # split data into training and testing
    Xt_train, Xt_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=42)
    
    gcv = cross_valaidate_coxnet(Xt_train, y_train, None)

    # test the model
    test_score = gcv.score(Xt_test, y_test)
    print(f"Test score: {test_score}")

    # output graphs
    if output_graphs:
        # check if "output/unweighted" directory exists
        # if not, create the directory
        if not os.path.exists("output/clin_rsem_copyno_mirna_rppa_unweighted_alpha"):
            os.makedirs("output/clin_rsem_copyno_mirna_rppa_unweighted_alpha")
            os.makedirs("output/clin_rsem_copyno_mirna_rppa_unweighted_alpha/KMCs")

        # plot c-index vs alpha parameter tuning plot
        c_index_vs_alpha_parameter_tuning_plot(pd.DataFrame(gcv.cv_results_), gcv, show_plot=False, output_path="output/clin_rsem_copyno_mirna_rppa_unweighted_alpha/alpha_tuning_plot.png")

        # plot top features
        top_features_plot(gcv, Xt, show_plot=False, output_path="output/clin_rsem_copyno_mirna_rppa_unweighted_alpha/top_features_plot.png")

        # get top 20 features
        top_features = get_top_x_features_name(gcv, Xt, 20)

        # plot kaplan meier plot
        for feature in top_features:
            kaplan_meier_plot(df, "time(day)", "dead", feature, show_plot=False, output_path=f"output/clin_rsem_copyno_mirna_rppa_unweighted_alpha/KMCs/KMC_{feature}.png")

    return gcv, test_score


def clin_rsem(clin, rsem_top500, output_graphs=False):
    # merge all genomic features
    merged = clin.merge(rsem_top500, on='patient_barcode', how='outer')

    print(f"merged: {merged.shape}")

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
    
    # split data into training and testing
    Xt_train, Xt_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=42)
    
    gcv = cross_valaidate_coxnet(Xt_train, y_train, None)

    # test the model
    test_score = gcv.score(Xt_test, y_test)
    print(f"Test score: {test_score}")

    # output graphs
    if output_graphs:
        # check if "output/clin_rsem" directory exists
        # if not, create the directory
        if not os.path.exists("output/clin_rsem"):
            os.makedirs("output/clin_rsem")
            os.makedirs("output/clin_rsem/KMCs")

        # plot c-index vs alpha parameter tuning plot
        c_index_vs_alpha_parameter_tuning_plot(pd.DataFrame(gcv.cv_results_), gcv, show_plot=False, output_path="output/clin_rsem/alpha_tuning_plot.png")

        # plot top features
        top_features_plot(gcv, Xt, show_plot=False, output_path="output/clin_rsem/top_features_plot.png")

        # get top 20 features
        top_features = get_top_x_features_name(gcv, Xt, 20)

        # plot kaplan meier plot
        for feature in top_features:
            kaplan_meier_plot(df, "time(day)", "dead", feature, show_plot=False, output_path=f"output/clin_rsem/KMCs/KMC_{feature}.png")

    return gcv, test_score


def clin_copyno(clin, copyno_top500, output_graphs=False):
    # merge all genomic features
    merged = clin.merge(copyno_top500, on='patient_barcode', how='outer')

    print(f"merged: {merged.shape}")

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
    
    # split data into training and testing
    Xt_train, Xt_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=42)
    
    gcv = cross_valaidate_coxnet(Xt_train, y_train, None)

    # test the model
    test_score = gcv.score(Xt_test, y_test)
    print(f"Test score: {test_score}")

    # output graphs
    if output_graphs:
        # check if "output/clin_copyno" directory exists
        # if not, create the directory
        if not os.path.exists("output/clin_copyno"):
            os.makedirs("output/clin_copyno")
            os.makedirs("output/clin_copyno/KMCs")

        # plot c-index vs alpha parameter tuning plot
        c_index_vs_alpha_parameter_tuning_plot(pd.DataFrame(gcv.cv_results_), gcv, show_plot=False, output_path="output/clin_copyno/alpha_tuning_plot.png")

        # plot top features
        top_features_plot(gcv, Xt, show_plot=False, output_path="output/clin_copyno/top_features_plot.png")

        # get top 20 features
        top_features = get_top_x_features_name(gcv, Xt, 20)

        # plot kaplan meier plot
        for feature in top_features:
            kaplan_meier_plot(df, "time(day)", "dead", feature, show_plot=False, output_path=f"output/clin_copyno/KMCs/KMC_{feature}.png")

    return gcv, test_score


def clin_mirna(clin, mirna, output_graphs=False):
    # merge all genomic features
    merged = clin.merge(mirna, on='patient_barcode', how='outer')

    print(f"merged: {merged.shape}")

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
    
    # split data into training and testing
    Xt_train, Xt_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=42)
    
    gcv = cross_valaidate_coxnet(Xt_train, y_train, None)

    # test the model
    test_score = gcv.score(Xt_test, y_test)
    print(f"Test score: {test_score}")

    # output graphs
    if output_graphs:
        # check if "output/clin_mirna" directory exists
        # if not, create the directory
        if not os.path.exists("output/clin_mirna"):
            os.makedirs("output/clin_mirna")
            os.makedirs("output/clin_mirna/KMCs")

        # plot c-index vs alpha parameter tuning plot
        c_index_vs_alpha_parameter_tuning_plot(pd.DataFrame(gcv.cv_results_), gcv, show_plot=False, output_path="output/clin_mirna/alpha_tuning_plot.png")

        # plot top features
        top_features_plot(gcv, Xt, show_plot=False, output_path="output/clin_mirna/top_features_plot.png")

        # get top 20 features
        top_features = get_top_x_features_name(gcv, Xt, 20)

        # plot kaplan meier plot
        for feature in top_features:
            kaplan_meier_plot(df, "time(day)", "dead", feature, show_plot=False, output_path=f"output/clin_mirna/KMCs/KMC_{feature}.png")

    return gcv, test_score


def clin_rppa(clin, rppa, output_graphs=False):
    # merge all genomic features
    merged = clin.merge(rppa, on='patient_barcode', how='outer')

    print(f"merged: {merged.shape}")

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
    
    # split data into training and testing
    Xt_train, Xt_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=42)
    
    gcv = cross_valaidate_coxnet(Xt_train, y_train, None)

    # test the model
    test_score = gcv.score(Xt_test, y_test)
    print(f"Test score: {test_score}")

    # output graphs
    if output_graphs:
        # check if "output/clin_rppa" directory exists
        # if not, create the directory
        if not os.path.exists("output/clin_rppa"):
            os.makedirs("output/clin_rppa")
            os.makedirs("output/clin_rppa/KMCs")

        # plot c-index vs alpha parameter tuning plot
        c_index_vs_alpha_parameter_tuning_plot(pd.DataFrame(gcv.cv_results_), gcv, show_plot=False, output_path="output/clin_rppa/alpha_tuning_plot.png")

        # plot top features
        top_features_plot(gcv, Xt, show_plot=False, output_path="output/clin_rppa/top_features_plot.png")

        # get top 20 features
        top_features = get_top_x_features_name(gcv, Xt, 20)

        # plot kaplan meier plot
        for feature in top_features:
            kaplan_meier_plot(df, "time(day)", "dead", feature, show_plot=False, output_path=f"output/clin_rppa/KMCs/KMC_{feature}.png")

    return gcv, test_score


def clin_rsem_copyno(clin, rsem_top500, copyno_top500, output_graphs=False):
    # merge all genomic features
    merged = clin.merge(rsem_top500, on='patient_barcode', how='outer')
    merged = merged.merge(copyno_top500, on='patient_barcode', how='outer')

    print(f"merged: {merged.shape}")

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
    
    # split data into training and testing
    Xt_train, Xt_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=42)
    
    gcv = cross_valaidate_coxnet(Xt_train, y_train, None)

    # test the model
    test_score = gcv.score(Xt_test, y_test)
    print(f"Test score: {test_score}")

    # output graphs
    if output_graphs:
        # check if "output/clin_rsem_copyno" directory exists
        # if not, create the directory
        if not os.path.exists("output/clin_rsem_copyno"):
            os.makedirs("output/clin_rsem_copyno")
            os.makedirs("output/clin_rsem_copyno/KMCs")

        # plot c-index vs alpha parameter tuning plot
        c_index_vs_alpha_parameter_tuning_plot(pd.DataFrame(gcv.cv_results_), gcv, show_plot=False, output_path="output/clin_rsem_copyno/alpha_tuning_plot.png")

        # plot top features
        top_features_plot(gcv, Xt, show_plot=False, output_path="output/clin_rsem_copyno/top_features_plot.png")

        # get top 20 features
        top_features = get_top_x_features_name(gcv, Xt, 20)

        # plot kaplan meier plot
        for feature in top_features:
            kaplan_meier_plot(df, "time(day)", "dead", feature, show_plot=False, output_path=f"output/clin_rsem_copyno/KMCs/KMC_{feature}.png")

    return gcv, test_score

    
    
   
        
if __name__ == "__main__":
    clin, rsem_top500, copyno_top500, mirna, rppa = initialize_data()

    weighted_gcv, weighted_score = clin_rsem_copyno_mirna_rppa_weighted_alpha(clin, rsem_top500, copyno_top500, mirna, rppa, output_graphs=True)
    unweighted_gcv, unweighted_score = clin_rsem_copyno_mirna_rppa_unweighted_alpha(clin, rsem_top500, copyno_top500, mirna, rppa, output_graphs=True)
    rsem_gcv, rsem_score = clin_rsem(clin, rsem_top500, output_graphs=True)
    copyno_gcv, copyno_score = clin_copyno(clin, copyno_top500, output_graphs=True)
    mirna_gcv, mirna_score = clin_mirna(clin, mirna, output_graphs=True)
    rppa_gcv, rppa_score = clin_rppa(clin, rppa, output_graphs=True)
    clin_rsem_copyno_gcv, clin_rsem_copyno_score = clin_rsem_copyno(clin, rsem_top500, copyno_top500, output_graphs=True)

    # format the scores and the best parameters into a dictionary
    scores = {
        "weighted_score": weighted_score,
        "unweighted_score": unweighted_score,
        "rsem_score": rsem_score,
        "copyno_score": copyno_score,
        "mirna_score": mirna_score,
        "rppa_score": rppa_score,
        "clin_rsem_copyno_score": clin_rsem_copyno_score,
    }

    alphas = {
        "weighted_alpha": weighted_gcv.best_params_['coxnetsurvivalanalysis__alphas'][0],
        "unweighted_alpha": unweighted_gcv.best_params_['coxnetsurvivalanalysis__alphas'][0],
        "rsem_alpha": rsem_gcv.best_params_['coxnetsurvivalanalysis__alphas'][0],
        "copyno_alpha": copyno_gcv.best_params_['coxnetsurvivalanalysis__alphas'][0],
        "mirna_alpha": mirna_gcv.best_params_['coxnetsurvivalanalysis__alphas'][0],
        "rppa_alpha": rppa_gcv.best_params_['coxnetsurvivalanalysis__alphas'][0],
        "clin_rsem_copyno_alpha": clin_rsem_copyno_gcv.best_params_['coxnetsurvivalanalysis__alphas'][0],
    }

    # save to a file in html
    with open("output/scores.html", "w") as f:
        f.write("<html><body>")
        f.write("<h1>C-Index</h1>")
        f.write("<table>")
        f.write("<tr><th>Model</th><th>Score</th></tr>")
        for key, value in scores.items():
            f.write(f"<tr><td>{key}</td><td>{value}</td></tr>")
        f.write("</table>")
        f.write("</body></html>")

    with open("output/alphas.html", "w") as f:
        f.write("<html><body>")
        f.write("<h1>Alphas</h1>")
        f.write("<table>")
        f.write("<tr><th>Model</th><th>Alpha</th></tr>")
        for key, value in alphas.items():
            f.write(f"<tr><td>{key}</td><td>{value}</td></tr>")
        f.write("</table>")
        f.write("</body></html>")

    print("Done!")