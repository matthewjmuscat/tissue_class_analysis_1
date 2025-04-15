import pandas as pd 
import load_files
from pathlib import Path
import os 
import statistical_tests_1_quick_and_dirty
import shape_and_radiomic_features
import misc_funcs
import biopsy_information
import uncertainties_analysis
import production_plots
import pickle
import pathlib # imported for navigating file system
import pyarrow # imported for loading parquet files, although not referenced it is required

def main():
    
    # Main output (files, input) directory
    # This one is 10k 10k containment and dosim, 11 patients, 2 fractions each,  pt 181 MR as well (ran with errors in final figures)
    #main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Apr-01-2025 Time-15,04,17")  # Ensure the directory is a Path object
    # This one is 10k containment and 10 (very low) dosim for speed, 11 patients, 2 fractions each,  pt 181 MR as well (ran with no errors in final figures)
    #main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Apr-02-2025 Time-03,44,41")
    # This one is 10k containment and 10 (very low) dosim for speed, all vitesse patients! 
    #main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Apr-02-2025 Time-19,38,15")
    # This one is 10k containment and 10 (very low) dosim for speed, all vitesse patients, also 2.5^3 for dil, 2.5 only for OARs
    main_output_path = Path("/home/matthew-muscat/Documents/UBC/Research/Data/Output data/MC_sim_out- Date-Apr-03-2025 Time-15,59,46")



    ### Define mimic structs ref dict

    bx_ref = "Bx ref"
    oar_ref = "OAR ref"
    dil_ref = "DIL ref"
    rectum_ref_key = "Rectum ref"
    urethra_ref_key = "Urethra ref"
    # for dataframe builder
    cancer_tissue_label = 'DIL'
    default_exterior_tissue = 'Periprostatic' # For tissue class stuff! Basically dictates what to call tissue that doesnt lie in any defined structure!
    prostate_tissue_label = 'Prostatic'
    rectal_tissue_label = 'Rectal'
    urethral_tissue_label = 'Urethral'


    structs_referenced_dict = { bx_ref: {
                                        'Tissue heirarchy': None, # should always be None
                                        'Tissue class name': None, # Not used for anything as of yet..
                                        }, 
                                oar_ref: {
                                          'Tissue heirarchy': 3,
                                        'Tissue class name': prostate_tissue_label, 
                                          }, 
                                dil_ref: {
                                          'Tissue heirarchy': 0,
                                          'Tissue class name': cancer_tissue_label,

                                          },
                                rectum_ref_key: {
                                          'Tissue heirarchy': 2,
                                          'Tissue class name': rectal_tissue_label,

                                          },
                                urethra_ref_key: {
                                          'Tissue heirarchy': 1,
                                          'Tissue class name': urethral_tissue_label,

                                          } 
                                }




    ### Load master dicts results


    # Use rglob to traverse directories recursively and find folders ending with "pickled data"
    found_folders = [folder for folder in main_output_path.rglob("*") 
                    if folder.is_dir() and folder.name.endswith("pickled data")]

    # Print out the found folders
    if found_folders:
        print("Found the following folder(s) ending with 'pickled data':")
        for folder in found_folders:
            print(folder)
    else:
        print("No folder ending with 'pickled data' was found under the specified directory.")

    # Assuming you want to use the first found folder
    pickled_data_folder = found_folders[0]

    # master_structure_reference_dict
    """
    results_master_structure_reference_dict_path = pickled_data_folder.joinpath("master_structure_reference_dict_results")
    print(f'Loading master_structure_reference_dict from: {results_master_structure_reference_dict_path}')
    with open(results_master_structure_reference_dict_path, "rb") as preprocessed_master_structure_reference_dict_file:
        master_structure_reference_dict = pickle.load(preprocessed_master_structure_reference_dict_file)
    """
    print(f"Done!")

    # master_structure_info_dict
    results_master_structure_info_dict_path = pickled_data_folder.joinpath("master_structure_info_dict_results")
    print(f'Loading master_structure_info_dict from: {results_master_structure_info_dict_path}')
    with open(results_master_structure_info_dict_path, "rb") as preprocessed_master_structure_info_dict_file:
        master_structure_info_dict = pickle.load(preprocessed_master_structure_info_dict_file) 
    print(f"Done!")





    ### Load Dataframes 

    # Set csv directory
    csv_directory = main_output_path.joinpath("Output CSVs")
    cohort_csvs_directory = csv_directory.joinpath("Cohort")






    # Cohort 3d radiomic features all oar and dil structures
    cohort_3d_radiomic_features_all_oar_dil_path = cohort_csvs_directory.joinpath("Cohort: 3D radiomic features all OAR and DIL structures.csv")  # Ensure the directory is a Path object
    cohort_3d_radiomic_features_all_oar_dil_df = load_files.load_csv_as_dataframe(cohort_3d_radiomic_features_all_oar_dil_path)  # Load the CSV file into a DataFrame
    """ NOTE: The columns of the dataframe are:
    cohort_3d_radiomic_features_all_oar_dil_df.columns =
    Index(['Patient ID', 'Structure ID', 'Structure type', 'Structure refnum',
       'Volume', 'Surface area', 'Surface area to volume ratio', 'Sphericity',
       'Compactness 1', 'Compactness 2', 'Spherical disproportion',
       'Maximum 3D diameter', 'PCA major', 'PCA minor', 'PCA least',
       'PCA eigenvector major', 'PCA eigenvector minor',
       'PCA eigenvector least', 'Major axis (equivalent ellipse)',
       'Minor axis (equivalent ellipse)', 'Least axis (equivalent ellipse)',
       'Elongation', 'Flatness', 'L/R dimension at centroid',
       'A/P dimension at centroid', 'S/I dimension at centroid',
       'S/I arclength', 'DIL centroid (X, prostate frame)',
       'DIL centroid (Y, prostate frame)', 'DIL centroid (Z, prostate frame)',
       'DIL centroid distance (prostate frame)', 'DIL prostate sextant (LR)',
       'DIL prostate sextant (AP)', 'DIL prostate sextant (SI)'],
      dtype='object')
    """
    
    
    
    # biopsy basic spatial features
    cohort_biopsy_basic_spatial_features_path = cohort_csvs_directory.joinpath("Cohort: Biopsy basic spatial features dataframe.csv")  # Ensure the directory is a Path object
    cohort_biopsy_basic_spatial_features_df = load_files.load_csv_as_dataframe(cohort_biopsy_basic_spatial_features_path)  # Load the CSV file into a DataFrame
    """ NOTE: The columns of the dataframe are:
    cohort_biopsy_basic_spatial_features_df.columns =
    Index(['Patient ID', 'Bx ID', 'Simulated bool', 'Simulated type',
       'Struct type', 'Bx refnum', 'Bx index', 'Length (mm)', 'Volume (mm3)',
       'Voxel side length (mm)', 'Relative DIL ID', 'Relative DIL index',
       'BX to DIL centroid (X)', 'BX to DIL centroid (Y)',
       'BX to DIL centroid (Z)', 'BX to DIL centroid distance',
       'NN surface-surface distance', 'Relative prostate ID',
       'Relative prostate index', 'Bx position in prostate LR',
       'Bx position in prostate AP', 'Bx position in prostate SI'],
      dtype='object')
      """


    # Cohort: Global sum-to-one mc results
    cohort_global_sum_to_one_tissue_path = cohort_csvs_directory.joinpath("Cohort: global sum-to-one mc results.csv")  # Ensure the directory is a Path object
    cohort_global_sum_to_one_tissue_df = load_files.load_csv_as_dataframe(cohort_global_sum_to_one_tissue_path)
    """ NOTE: The columns of the dataframe are:
    cohort_global_sum_to_one_tissue_df.columns = Index(['Patient ID', 'Bx ID', 'Bx refnum', 'Bx index', 'Tissue class',
       'Simulated bool', 'Simulated type', 'Global Mean BE', 'Global Min BE',
       'Global Max BE', 'Global STD BE', 'Global SEM BE', 'Global Q05 BE',
       'Global Q25 BE', 'Global Q50 BE', 'Global Q75 BE', 'Global Q95 BE',
       'Global CI 95 BE (lower)', 'Global CI 95 BE (upper)'],
      dtype='object')
    """
    # Cohort sum-to-one mc results
    cohort_sum_to_one_mc_results_path = cohort_csvs_directory.joinpath("Cohort: sum-to-one mc results.csv")  # Ensure the directory is a Path object
    cohort_sum_to_one_mc_results_df = load_files.load_csv_as_dataframe(cohort_sum_to_one_mc_results_path)  # Load the CSV file into a DataFrame
    """ NOTE: The columns of the dataframe are:
    cohort_sum_to_one_mc_results_df.columns = Index(['Patient ID', 'Bx ID', 'Bx refnum', 'Bx index', 'Simulated bool',
       'Simulated type', 'Tissue class', 'Original pt index',
       'Total successes', 'Nominal', 'Binomial estimator', 'X (Bx frame)',
       'Y (Bx frame)', 'Z (Bx frame)', 'Binom est STD err', 'CI lower vals',
       'CI upper vals', 'Voxel index', 'Voxel begin (Z)', 'Voxel end (Z)'],
      dtype='object')
    """
    # Cohort tissue class - distances global
    cohort_tissue_class_distances_global_path = cohort_csvs_directory.joinpath("Cohort: Tissue class - distances global results.csv")  # Ensure the directory is a Path object
    # this is a multiindex dataframe
    cohort_tissue_class_distances_global_df = load_files.load_multiindex_csv(cohort_tissue_class_distances_global_path, header_rows=[0, 1])  # Load the CSV file into a DataFrame




    # Load uncertainties csv
    #uncertainties_path = main_output_path.joinpath("uncertainties_file_auto_generated Date-Apr-02-2025 Time-03,45,24.csv")  # Ensure the directory is a Path object

    # Assuming main_output_path is already a Path object
    # Adjust the pattern as needed if the prefix should be "uncertainities"
    csv_files = list(main_output_path.glob("uncertainties*.csv"))
    if csv_files:
        # grab the first one 
        uncertainties_path = csv_files[0]
        uncertainties_df = load_files.load_csv_as_dataframe(uncertainties_path)
    else:
        raise FileNotFoundError("No uncertainties CSV file found in the directory.")

    




    # load all containment and distances results csvs
    mc_sim_results_path = csv_directory.joinpath("MC simulation")  # Ensure the directory is a Path object
    all_paths_containment_and_distances = load_files.find_csv_files(mc_sim_results_path, ['containment and distances (light) results.parquet'])
    # Load and concatenate all containment and distances results csvs
    # Loop through all the paths and load the csv files
    all_containment_and_distances_dfs_list = []
    for path in all_paths_containment_and_distances:
        # Load the csv file into a dataframe
        df = load_files.load_parquet_as_dataframe(path)
        # Append the dataframe to the list
        all_containment_and_distances_dfs_list.append(df)

        del df
    # Concatenate all the dataframes into one dataframe
    all_containment_and_distances_df = pd.concat(all_containment_and_distances_dfs_list, ignore_index=True)
    del all_containment_and_distances_dfs_list
    # Print the shape of the dataframe
    print(f"Shape of all containment and distances dataframe: {all_containment_and_distances_df.shape}")
    # Print the columns of the dataframe
    print(f"Columns of all containment and distances dataframe: {all_containment_and_distances_df.columns}")
    # Print the first 5 rows of the dataframe
    print(f"First 5 rows of all containment and distances dataframe: {all_containment_and_distances_df.head()}")
    # Print the last 5 rows of the dataframe
    print(f"Last 5 rows of all containment and distances dataframe: {all_containment_and_distances_df.tail()}")












    ########### LOADING COMPLETE









    ## Create output directory
    # Output directory 
    output_dir = Path(__file__).parents[0].joinpath("output_data")
    os.makedirs(output_dir, exist_ok=True)


    ### Get unqiue patient IDs
    # Get ALL unique patient IDs from the cohort_3d_radiomic_features_all_oar_dil_df DataFrame
    unique_patient_ids_all = cohort_biopsy_basic_spatial_features_df['Patient ID'].unique().tolist()
    # Print the unique patient IDs
    print("Unique Patient IDs (ALL):")
    print(unique_patient_ids_all)
    # Print the number of unique patient IDs
    print(f"Number of unique patient IDs ALL: {len(unique_patient_ids_all)}")


    # Get unique patient IDs, however the patient IDs actually include the patient ID (F#) where F# indicates the fraction but its actually the same patient, so I want to take only F1, if F1 isnt present for a particular ID then I want to take F2
    # Get the unique patient IDs from the cohort_3d_radiomic_features_all_oar_dil_df DataFrame
    unique_patient_ids_f1_prioritized = misc_funcs.get_unique_patient_ids_fraction_prioritize(cohort_biopsy_basic_spatial_features_df,patient_id_col='Patient ID', priority_fraction='F1')
    # Print the unique patient IDs
    print("Unique Patient IDs (F1) prioritized:")
    print(unique_patient_ids_f1_prioritized)
    # Print the number of unique patient IDs
    print(f"Number of unique patient IDs (F1) prioritized: {len(unique_patient_ids_f1_prioritized)}")

    ### Get unique patient IDs for F1
    # Get the unique patient IDs from the cohort_3d_radiomic_features_all_oar_dil_df DataFrame
    unique_patient_ids_f1 = misc_funcs.get_unique_patient_ids_fraction_specific(cohort_biopsy_basic_spatial_features_df, patient_id_col='Patient ID',fraction='F1')
    # Print the unique patient IDs
    print("Unique Patient IDs (F1) ONLY:")
    print(unique_patient_ids_f1)
    # Print the number of unique patient IDs
    print(f"Number of unique patient IDs F1 ONLY: {len(unique_patient_ids_f1)}")

    ### Get unique patient IDs for F2
    # Get the unique patient IDs from the cohort_3d_radiomic_features_all_oar_dil_df DataFrame
    unique_patient_ids_f2 = misc_funcs.get_unique_patient_ids_fraction_specific(cohort_biopsy_basic_spatial_features_df, patient_id_col='Patient ID',fraction='F2')
    # Print the unique patient IDs
    print("Unique Patient IDs (F2) only:")
    print(unique_patient_ids_f2)
    # Print the number of unique patient IDs
    print(f"Number of unique patient IDs F2 ONLY: {len(unique_patient_ids_f2)}")








    ### Uncertainties analysis (START)
    # Create output directory for uncertainties analysis
    uncertainties_analysis_dir = output_dir.joinpath("uncertainties_analysis")
    os.makedirs(uncertainties_analysis_dir, exist_ok=True)
    # Output filename
    output_filename = 'uncertainties_analysis_statistics_all_patients.csv'
    # Get uncertainties analysis statistics
    uncertainties_analysis_statistics_df = uncertainties_analysis.compute_statistics_by_structure_type(uncertainties_df,
                                                                                           columns=['mu (X)', 'mu (Y)', 'mu (Z)', 'sigma (X)', 'sigma (Y)', 'sigma (Z)', 'Dilations mu (XY)', 'Dilations mu (Z)', 'Dilations sigma (XY)', 'Dilations sigma (Z)', 'Rotations mu (X)', 'Rotations mu (Y)', 'Rotations mu (Z)', 'Rotations sigma (X)', 'Rotations sigma (Y)', 'Rotations sigma (Z)'], 
                                                                                           patient_uids=unique_patient_ids_all)
    # Save the statistics to a CSV file
    uncertainties_analysis_statistics_df.to_csv(uncertainties_analysis_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(uncertainties_analysis_statistics_df)
    ### Uncertainties analysis (END)









    ### Radiomic features analysis (START)
    # Create output directory for radiomic features
    radiomic_features_dir = output_dir.joinpath("radiomic_features")
    os.makedirs(radiomic_features_dir, exist_ok=True)
    # Output filename
    output_filename = 'radiomic_features_statistics_all_patients.csv'
    # Get radiomic statistics
    radiomic_statistics_df = shape_and_radiomic_features.get_radiomic_statistics(cohort_3d_radiomic_features_all_oar_dil_df, 
                                                                                 patient_id= unique_patient_ids_f1_prioritized, 
                                                                                 exclude_columns=['Patient ID', 'Structure ID', 'Structure type', 'Structure refnum','PCA eigenvector major', 'PCA eigenvector minor',	'PCA eigenvector least', 'DIL centroid (X, prostate frame)', 'DIL centroid (Y, prostate frame)', 'DIL centroid (Z, prostate frame)', 'DIL centroid distance (prostate frame)', 'DIL prostate sextant (LR)', 'DIL prostate sextant (AP)', 'DIL prostate sextant (SI)'])

    # Save the statistics to a CSV file
    radiomic_statistics_df.to_csv(radiomic_features_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(radiomic_statistics_df)
    ### Radiomic features analysis (END)




    ### cumulatiove dil volume stats (START)
    mean_cumulative_dil_vol, std_cumulative_dil_vol = shape_and_radiomic_features.cumulative_dil_volume_stats(unique_patient_ids_f1_prioritized, cohort_3d_radiomic_features_all_oar_dil_df)
    # Print the statistics
    print(f"Mean cumulative DIL volume: {mean_cumulative_dil_vol}")
    print(f"Standard deviation of cumulative DIL volume: {std_cumulative_dil_vol}")
    # Save the statistics to a CSV file
    output_filename = 'cumulative_dil_volume_statistics_all_patients.csv'
    # Create output directory for DIL information, use radiomic_features_dir
    os.makedirs(radiomic_features_dir, exist_ok=True)
    # Save the statistics to a CSV file
    cumulative_dil_volume_statistics_df = pd.DataFrame({'Mean cumulative DIL volume (mm^3)': [mean_cumulative_dil_vol], 'Standard deviation of cumulative DIL volume (mm^3)': [std_cumulative_dil_vol]})
    cumulative_dil_volume_statistics_df.to_csv(radiomic_features_dir.joinpath(output_filename), index=False)
    ### cumulatiove dil volume stats (END)










    ### Find DIL double sextant percentages (START)
    # Create output directory for DIL information
    dil_information_dir = output_dir.joinpath("dil_information")
    os.makedirs(dil_information_dir, exist_ok=True)
    # Output filename
    output_filename = 'dil_double_sextant_percentages_all_patients.csv'
    # Get DIL double sextant percentages
    dil_double_sextant_percentages_df = shape_and_radiomic_features.find_dil_double_sextant_percentages(cohort_3d_radiomic_features_all_oar_dil_df, patient_id=unique_patient_ids_f1_prioritized)
    # Save the statistics to a CSV file
    dil_double_sextant_percentages_df.to_csv(dil_information_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(dil_double_sextant_percentages_df)
    ### Find DIL double sextant percentages (END)















    ### Find structure counts (START)
    # Create output directory for DIL information
    radiomic_features_dir = output_dir.joinpath("radiomic_features")
    os.makedirs(radiomic_features_dir, exist_ok=True)
    # Output filename
    output_filename = 'structure_counts_all_patients.csv'
    # Get structure counts
    structure_counts_df, structure_counts_statistics_df = shape_and_radiomic_features.calculate_structure_counts_and_stats(cohort_3d_radiomic_features_all_oar_dil_df, patient_id=unique_patient_ids_f1_prioritized, structure_types=None)
    # Save the statistics to a CSV file
    structure_counts_df.to_csv(radiomic_features_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(structure_counts_df)
    # Save the statistics to a CSV file
    output_filename = 'structure_counts_statistics_all_patients.csv'
    # Save the statistics to a CSV file
    structure_counts_statistics_df.to_csv(radiomic_features_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(structure_counts_statistics_df)
    ### Find structure counts (END)












    ### Biopsy information analysis (START)
    # Create output directory for biopsy information
    biopsy_information_dir = output_dir.joinpath("biopsy_information")
    os.makedirs(biopsy_information_dir, exist_ok=True)
    # Output filename
    output_filename = 'biopsy_information_statistics_all_patients.csv'
    # Get biopsy information statistics
    biopsy_information_statistics_df = biopsy_information.get_filtered_statistics(cohort_biopsy_basic_spatial_features_df, 
                                                                                 columns=['Length (mm)', 
                                                                                          'Volume (mm3)', 
                                                                                          'Voxel side length (mm)',  
                                                                                          'BX to DIL centroid (X)', 
                                                                                          'BX to DIL centroid (Y)',
                                                                                          'BX to DIL centroid (Z)', 
                                                                                          'BX to DIL centroid distance', 
                                                                                          'NN surface-surface distance'], 
                                                                                 patient_id=unique_patient_ids_all,
                                                                                 simulated_type='Real')

    # Save the statistics to a CSV file
    biopsy_information_statistics_df.to_csv(biopsy_information_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(biopsy_information_statistics_df)
    ### Biopsy information analysis (END)

















    ### Find biopsy double sextant percentages (START)
    # Create output directory for biopsy information
    biopsy_information_dir = output_dir.joinpath("biopsy_information")
    os.makedirs(biopsy_information_dir, exist_ok=True)
    # Output filename
    output_filename = 'biopsy_double_sextant_percentages_all_patients.csv'
    # Get biopsy double sextant percentages
    biopsy_double_sextant_percentages_df = biopsy_information.find_biopsy_double_sextant_percentages(cohort_biopsy_basic_spatial_features_df, 
                                                                                                     patient_id=unique_patient_ids_all,
                                                                                                     simulated_type='Real')
    # Save the statistics to a CSV file
    biopsy_double_sextant_percentages_df.to_csv(biopsy_information_dir.joinpath(output_filename), index=True)
    # Print the statistics
    print(biopsy_double_sextant_percentages_df)
    ### Find biopsy double sextant percentages (END)







    ### Find statistics of global tissue scores (START)
    cohort_global_sum_to_one_tissue_scores_statistics = statistical_tests_1_quick_and_dirty.compute_global_tissue_scores_stats_across_all_biopsies(cohort_global_sum_to_one_tissue_df)
    # Print the statistics
    print(cohort_global_sum_to_one_tissue_scores_statistics)
    # Save the statistics to a CSV file
    output_filename = 'global_tissue_scores_statistics_all_patients.csv'
    # use statistical_tests_1_dir
    statistical_tests_1_dir = output_dir.joinpath("statistical_tests_0")
    os.makedirs(statistical_tests_1_dir, exist_ok=True)
    # Save the statistics to a CSV file
    cohort_global_sum_to_one_tissue_scores_statistics.to_csv(statistical_tests_1_dir.joinpath(output_filename), index=True)
    ### Find statistics of global tissue scores (END)


















    ### PLOTS FROM RAW DATA
    # make dirs
    output_fig_directory = output_dir.joinpath("figures")
    os.makedirs(output_fig_directory, exist_ok=True)
    cohort_output_figures_dir = output_fig_directory.joinpath("cohort_output_figures")
    os.makedirs(cohort_output_figures_dir, exist_ok=True)



    # 1. cohort all biopsy voxels histogram by tissue class
    svg_image_scale = 1
    svg_image_width = 1920
    svg_image_height = 1080
    dpi_for_seaborn_plots = 100
    general_plot_name_string = 'cohort - tissue_class_sum-to-one_all_biopsy_voxels_histogram_by_tissue_class'
    bx_sample_pts_vol_element = master_structure_info_dict["Global"]["MC info"]["BX sample pt volume element (mm^3)"]
    production_plots.production_plot_cohort_sum_to_one_all_biopsy_voxels_binom_est_histogram_by_tissue_class(cohort_sum_to_one_mc_results_df,
                                                                                    svg_image_width,
                                                                                    svg_image_height,
                                                                                    dpi_for_seaborn_plots,
                                                                                    general_plot_name_string,
                                                                                    cohort_output_figures_dir,
                                                                                    bx_sample_pts_vol_element,
                                                                                    bin_width=0.05,
                                                                                    bandwidth=0.1)



    # 2. cohort sum to one boxplot global scores
    general_plot_name_string = 'cohort - sum-to-one_boxplot_global_scores'
    production_plots.cohort_global_scores_boxplot_by_bx_type(cohort_global_sum_to_one_tissue_df,
                                 general_plot_name_string,
                                 cohort_output_figures_dir)





    # 3. individual patient histograms of tissue scores by tissue type:
    patient_id_and_bx_index_pairs = [('181 (F1)',0), ('181 (F1)', 1), ('181 (F1)', 2), ('184 (F2)', 0), ('184 (F2)', 1), ('184 (F2)', 2)]
    for patient_id, bx_index in patient_id_and_bx_index_pairs:
        # Create a directory for the patient
        patient_dir = cohort_output_figures_dir.joinpath(patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        # Create the histogram
        # Get bx_id
        bx_id = cohort_global_sum_to_one_tissue_df.loc[(cohort_global_sum_to_one_tissue_df['Patient ID'] == patient_id) & (cohort_global_sum_to_one_tissue_df['Bx index'] == bx_index), 'Bx ID'].values[0]
        production_plots.plot_bx_histograms_by_tissue(cohort_sum_to_one_mc_results_df, 
                                                      patient_id, 
                                                      bx_index, 
                                                      patient_dir, 
                                                      structs_referenced_dict, 
                                                      default_exterior_tissue, 
                                                      fig_name=f"tissue_class_histograms-{bx_index}-{bx_id}.svg", 
                                                      bin_width=0.05, 
                                                      spatial_df=cohort_biopsy_basic_spatial_features_df)


    # 4. individual patient tissue kernel regressions
    patient_id_and_bx_index_pairs = [('181 (F1)',0), ('181 (F1)', 1), ('181 (F1)', 2), ('184 (F2)', 0), ('184 (F2)', 1), ('184 (F2)', 2)]
    for patient_id, bx_index in patient_id_and_bx_index_pairs:
        # Create a directory for the patient
        patient_dir = cohort_output_figures_dir.joinpath(patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        general_plot_name_string = " - tissue_class_sum-to-one_binom_regression_probabilities"
        cohort_sum_to_one_mc_results_df_filtered = cohort_sum_to_one_mc_results_df[(cohort_sum_to_one_mc_results_df['Patient ID'] == patient_id) & (cohort_sum_to_one_mc_results_df['Bx index'] == bx_index)]
        production_plots.production_plot_sum_to_one_tissue_class_binom_regression_matplotlib(cohort_sum_to_one_mc_results_df_filtered,
                                                                        bx_index,
                                                                        patient_id,
                                                                        structs_referenced_dict,
                                                                        default_exterior_tissue,
                                                                        patient_dir,
                                                                        general_plot_name_string)


    # 5. individual patient tissue nominal chart
    patient_id_and_bx_index_pairs = [('181 (F1)',0), ('181 (F1)', 1), ('181 (F1)', 2), ('184 (F2)', 0), ('184 (F2)', 1), ('184 (F2)', 2)]
    for patient_id, bx_index in patient_id_and_bx_index_pairs:
        # Create a directory for the patient
        patient_dir = cohort_output_figures_dir.joinpath(patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        general_plot_name_string = " - tissue_class_sum-to-one_nominal_tissue_class"
        cohort_sum_to_one_mc_results_df_filtered = cohort_sum_to_one_mc_results_df[(cohort_sum_to_one_mc_results_df['Patient ID'] == patient_id) & (cohort_sum_to_one_mc_results_df['Bx index'] == bx_index)]
        
        production_plots.production_plot_sum_to_one_tissue_class_nominal_plotly(cohort_sum_to_one_mc_results_df_filtered,
                                                    patient_id,
                                                    bx_index,
                                                    svg_image_scale,
                                                    svg_image_width,
                                                    svg_image_height,
                                                    general_plot_name_string,
                                                    patient_dir,
                                                    structs_referenced_dict,
                                                    default_exterior_tissue
                                                    )

    ### Find distances statistics (START)

    # Create output directory for distances statistics
    distances_statistics_dir = output_dir.joinpath("distances_statistics")
    os.makedirs(distances_statistics_dir, exist_ok=True)
    # Output filename
    output_filename = 'distances_statistics_all_patients.csv'
    # save dataframe to csv

    cohort_tissue_class_distances_global_df.to_csv(distances_statistics_dir.joinpath(output_filename), index=True)
    
    ### Find distances statistics (END)







    statistical_tests_1_dir = output_dir.joinpath("statistical_tests_0")
    os.makedirs(statistical_tests_1_dir, exist_ok=True)

    output_filename = 'stats_test_1.csv'
    tissue_types = ["DIL", 'Prostatic', 'Periprostatic', 'Urethral', 'Rectal']
    #_ = statistical_tests_1_quick_and_dirty.analyze_data(cohort_global_sum_to_one_tissue_df, tissue_types, statistical_tests_1_dir, output_filename)

    statistical_tests_1_quick_and_dirty.kruskal_wallis_tissue_class_test(cohort_global_sum_to_one_tissue_df)












    # 3. cohort significance testing between global scores (mean)
    paired_wilcoxan_sig_test_tissue_class_df = statistical_tests_1_quick_and_dirty.paired_wilcoxon_signed_rank(cohort_global_sum_to_one_tissue_df, tissue_types, patient_id_col='Patient ID', bx_index_col='Bx index')
    print("Paired Wilcoxon Signed-Rank Test results:")
    print(paired_wilcoxan_sig_test_tissue_class_df)
    
    # Create a heatmap of the p-values
    production_plots.plot_wilcoxon_heatmap(paired_wilcoxan_sig_test_tissue_class_df, tissue_types, cohort_output_figures_dir, title='Wilcoxon Signed-Rank Test p-values', fig_name='wilcoxon_heatmap.svg')



    #  4. cohort effect sizes between global scores (mean)
    # Effect sizes
    effect_sizes = ('cohen', 'hedges', 'cles', 'mean_diff')
    paired_eff_size_test_tissue_class_global_scores_df = statistical_tests_1_quick_and_dirty.paired_effect_size_analysis(cohort_global_sum_to_one_tissue_df, tissue_types, effect_sizes, patient_id_col='Patient ID', bx_index_col='Bx index', value_col='Global Mean BE')

    print("Effect sizes results:")
    print(paired_eff_size_test_tissue_class_global_scores_df)

    # Create a heatmap of the effect sizes
    for effect_size_key in effect_sizes:
        # Create a heatmap for each effect size
        # set the vmin and vmax correctly according to the effect size
        if effect_size_key == 'cohen':
            vmin = -2
            vmax = 2
        elif effect_size_key == 'hedges':
            vmin = -2
            vmax = 2
        elif effect_size_key == 'cles':
            vmin = 0
            vmax = 1
        elif effect_size_key == 'mean_diff':
            vmin = -1
            vmax = 1
        production_plots.plot_effect_size_heatmap(paired_eff_size_test_tissue_class_global_scores_df, 
                                                  tissue_types, 
                                                  effect_size_key, 
                                                  cohort_output_figures_dir, 
                                                  title=f'Effect size heatmap', 
                                                  fig_name=f'{effect_size_key}_heatmap.svg',
                                                  vmin=vmin, 
                                                  vmax=vmax,
                                                  axis_label_size=16, 
                                                  tick_fontsize=14,
                                                  cbar_label_fontsize=16, 
                                                  cbar_tick_fontsize=14)




    print("Done!")

if __name__ == "__main__":
    main()