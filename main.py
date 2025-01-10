import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# import functions
from preprocessing import process_clinical_data_from_raw, process_rsem_data_from_raw, process_copyno_data_from_raw, process_mirna_data_from_raw, process_rppa_data_from_raw
from weight_selection import cox_lasso_group_weight_selection
from coxnet_alpha_selection import cross_valaidate_coxnet, get_top_x_features_name
from cox_univariate_screening import cox_univariate_screening
from cox_univariate_screening import extract_top_features
from graph_plotting_script import c_index_vs_alpha_parameter_tuning_plot, top_features_plot, kaplan_meier_plot, penalty_factors_plot

# import models
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.simplefilter("ignore", UserWarning)

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

    cv_results = pd.DataFrame(gcv.cv_results_)

    # plot penalty factors
    penalty_factors_plot(group_indices, weights, show_plot=False, output_path="output/penalty_factors_plot.png")

    # plot c-index vs alpha parameter tuning plot
    c_index_vs_alpha_parameter_tuning_plot(cv_results, gcv, show_plot=False, output_path="output/alpha_tuning_plot.png")

    # plot top features
    top_features_plot(gcv, Xt, show_plot=False, output_path="output/top_features_plot.png")
    
    # get top 20 features
    top_features = get_top_x_features_name(gcv, Xt, 20)

    # plot kaplan meier plot
    for feature in top_features:
        kaplan_meier_plot(df, "time(day)", "dead", feature, show_plot=False, output_path=f"output/KMCs/KMC_{feature}.png")

    # test the model
    test_score = gcv.score(Xt_test, y_test)
    print(f"Test score: {test_score}")

    # Cross validate unweighted coxnet model
    gcv_unweighted = cross_valaidate_coxnet(Xt_train, y_train, None)

    # plot c-index vs alpha parameter tuning plot
    c_index_vs_alpha_parameter_tuning_plot(pd.DataFrame(gcv_unweighted.cv_results_), gcv_unweighted, show_plot=False, output_path="output/alpha_tuning_plot_unweighted.png")

    # plot top features
    top_features_plot(gcv_unweighted, Xt, show_plot=False, output_path="output/top_features_plot_unweighted.png")

    # get top 20 features
    top_features = get_top_x_features_name(gcv_unweighted, Xt, 20)

    # test the model
    test_score = gcv_unweighted.score(Xt_test, y_test)
    print(f"Test score: {test_score}")
        
    
   
        
if __name__ == "__main__":
    main()