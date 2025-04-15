import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import copy
import seaborn as sns
import numpy as np
import pandas as pd 
from pathlib import Path
from statsmodels.nonparametric.kernel_regression import KernelReg
import misc_tools
import plotly.express as px
import plotting_funcs

def production_plot_cohort_sum_to_one_all_biopsy_voxels_binom_est_histogram_by_tissue_class(dataframe,
                                       svg_image_width,
                                       svg_image_height,
                                       dpi,
                                       histogram_plot_name_string,
                                       output_dir,
                                       bx_sample_pts_vol_element,
                                       bin_width=0.05,
                                       bandwidth=0.1):
    
    plt.ioff()  # Turn off interactive plotting for batch figure generation
    
    # Deep copy the dataframe to prevent modifications to the original data
    df = copy.deepcopy(dataframe)
    
    # Get the list of unique tissue classes
    tissue_classes = df['Tissue class'].unique()
    
    # Set up the figure and subplots for each tissue class
    fig, axes = plt.subplots(len(tissue_classes), 1, figsize=(svg_image_width / dpi, svg_image_height / dpi), dpi=dpi, sharex=True)
    
    # Increase padding between subplots
    fig.subplots_adjust(hspace=0.8)  # Adjust hspace to increase vertical padding
    
    # Create color mappings for vertical lines
    line_colors = {
        'mean': 'orange',
        'min': 'blue',
        'max': 'purple',
        'q05': 'cyan',
        'q25': 'green',
        'q50': 'red',
        'q75': 'green',
        'q95': 'cyan',
        'max density': 'magenta'
    }
    
    for ax, tissue_class in zip(axes, tissue_classes):
        tissue_data = df[df['Tissue class'] == tissue_class]['Binomial estimator'].dropna()
        
        count = len(tissue_data)
        ax.text(-0.3, 0.85, f'Num voxels: {count}', ha='left', va='top', transform=ax.transAxes, fontsize=14, color='black')
        ax.text(-0.3, 0.7, f'Kernel BW: {bandwidth}', ha='left', va='top', transform=ax.transAxes, fontsize=14, color='black')
        ax.text(-0.3, 0.55, f'Bin width: {bin_width}', ha='left', va='top', transform=ax.transAxes, fontsize=14, color='black')
        ax.text(-0.3, 0.4, f'Bx voxel volume (cmm): {bx_sample_pts_vol_element}', ha='left', va='top', transform=ax.transAxes, fontsize=14, color='black')


        bins = np.arange(0, 1.05, bin_width)  # Create bins from 0 to 1 with steps of 0.05

        # Plot normalized histogram with KDE
        sns.histplot(tissue_data, bins=bins, kde=False, color='skyblue', stat='density', ax=ax)

        # Calculate statistics
        mean_val = tissue_data.mean()
        min_val = tissue_data.min()
        max_val = tissue_data.max()
        quantiles = np.percentile(tissue_data, [5, 25, 50, 75, 95])

        
        try:
            # KDE fit for the binomial estimator values with specified bandwidth
            kde = gaussian_kde(tissue_data, bw_method=bandwidth)
            x_grid = np.linspace(0, 1, 1000)
            y_density = kde(x_grid)
            # Normalize the KDE so the area under the curve equals 1
            y_density /= np.trapz(y_density, x_grid)  # Normalize over the x_grid range

            max_density_value = x_grid[np.argmax(y_density)]

            # Overlay KDE plot
            ax.plot(x_grid, y_density, color='black', linewidth=1.5, label='KDE')

        except np.linalg.LinAlgError as e:
            # If there's a LinAlgError, it likely means all values are identical
            print(f"Cohort sum-to-one histogram plot | Tissue class: {tissue_class} | LinAlgError: {e}")
            constant_value = tissue_data.iloc[0] if len(tissue_data) > 0 else 0
            ax.axvline(constant_value, color='black', linestyle='-', linewidth=1.5, label='All values are identical')
            max_density_value = constant_value  # Set max density to the constant value for further annotations

        except Exception as e:
            # Handle any other unexpected errors and print/log the error message
            print(f"Cohort sum-to-one histogram plot | Tissue class: {tissue_class} | An unexpected error occurred: {e}")
            # Set a fallback for max density value or other defaults
            constant_value = tissue_data.mean() if len(tissue_data) > 0 else 0
            ax.axvline(constant_value, color='red', linestyle='-', linewidth=1.5, label='Fallback line due to error')
            max_density_value = constant_value

        # Add vertical lines for mean, min, max, quantiles, and max density
        line_positions = {
            'Mean': mean_val,
            'Min': min_val,
            'Max': max_val,
            'Q05': quantiles[0],
            'Q25': quantiles[1],
            'Q50': quantiles[2],
            'Q75': quantiles[3],
            'Q95': quantiles[4],
            'Max Density': max_density_value
        }
        
                # Sort line_positions by the x-values (positions of the vertical lines)
        sorted_line_positions = sorted(line_positions.items(), key=lambda item: item[1])

        # Initialize tracking variables to handle overlapping labels
        last_x_val = None
        last_label_y = 1.02  # Initial y position for text labels
        stack_count = 0  # Track count of stacked labels
        offset_x = 0  # Horizontal offset for secondary stacks

        # Iterate over the sorted line positions to add vertical lines and labels
        for label, x_val in sorted_line_positions:
            color = line_colors.get(label.lower(), 'black')
            ax.axvline(x_val, color=color, linestyle='--' if 'Q' in label else '-', label=label)

            # Check for potential overlap and adjust y-position if needed
            if last_x_val is not None and abs(x_val - last_x_val) < 0.1:
                last_label_y += 0.15
                stack_count += 1
            else:
                # Reset position and stack count if no overlap
                last_label_y = 1.02
                stack_count = 0
                offset_x = 0

            # Shift label to the right if stack count exceeds 3
            if stack_count > 2:
                offset_x += 0.03  # Increment horizontal offset
                last_label_y = 1.02  # Reset y-position for the new stack
                stack_count = 0  # Reset stack count for the new column

            # Add text above the plot area with adjusted x and y positions
            ax.text(x_val + offset_x, last_label_y, f'{x_val:.2f}', color=color, ha='center', va='bottom',
                    fontsize=14, transform=ax.get_xaxis_transform())

            # Update last_x_val to current x_val
            last_x_val = x_val


        # Set x-axis limits to [0, 1] and enable grid lines
        ax.set_xlim(0, 1)
        ax.grid(True)
        ax.set_xticks(np.arange(0, 1.1, 0.1))  # Sets vertical grid lines every 0.1
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=16)  # Adjust the number to your desired font size


        # Add title and labels with adjusted title position
        ax.set_title(f'{tissue_class}', fontsize=16, y=1, x = -0.15, ha='left')
        ax.set_ylabel('Density', fontsize=16)
        ax.tick_params(axis='y', labelsize=16)  # Adjust the number to your desired font size

        
    # X-axis label and figure title
    fig.text(0.5, 0.04, 'Multinomial Estimator', ha='center', fontsize=16)
    fig.suptitle('Cohort - Normalized Multinomial Estimator Distribution by Tissue Class For All Biopsy Voxels', fontsize=16)

    # Legend positioned outside the plot area with white background
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.95, 0.5), frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    # Save the figure
    output_path = output_dir.joinpath(f"{histogram_plot_name_string}.svg")
    fig.savefig(output_path, format='svg', dpi=dpi, bbox_inches='tight')

    # Close the figure to free memory
    plt.close(fig)





def cohort_global_scores_boxplot_by_bx_type(cohort_mc_sum_to_one_global_scores_dataframe,
                                 general_plot_name_string,
                                 cohort_output_figures_dir):

    df = cohort_mc_sum_to_one_global_scores_dataframe

    # Melt the DataFrame to bring mean, min, and max into a single column for easier plotting
    df_melted = pd.melt(df, id_vars=['Tissue class', 'Simulated type'], 
                            value_vars=['Global Min BE', 'Global Mean BE', 'Global Max BE', 'Global STD BE'], 
                            var_name='Statistic', value_name='Binomial Estimator')

    # Create a grouped boxplot using seaborn with faceting by 'Simulated type'
    g = sns.catplot(x='Tissue class', y='Binomial Estimator', hue='Statistic', 
                    col='Simulated type', data=df_melted, kind='box', 
                    palette="Set2", height=6, aspect=1.5)

    # Set y-axis limits to be between 0 and 1
    g.set(ylim=(0, 1))

    # Add horizontal grid lines
    g.set_axis_labels("Tissue Class", "Binomial Estimator")
    for ax in g.axes.flat:
        ax.grid(True, which='both', axis='y')  # Add horizontal grid lines to each subplot

    # Customize the plot for better aesthetics
    g.set_titles("Simulated Type: {col_name}")
    g.fig.suptitle('Boxplots of Global Mean, Min, and Max (sum-to-one) Values by Tissue Class', y=1.02, fontsize=12)
    g.set_xticklabels(rotation=45)
    plt.tight_layout()

    # Save the figure
    svg_dose_fig_name = general_plot_name_string + '.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    g.savefig(svg_dose_fig_file_path, format='svg')

    # Close the figure to release memory
    plt.close(g.fig)  # Ensure the figure is closed properly



def plot_wilcoxon_heatmap(results_df, tissue_classes, output_dir, title='Wilcoxon Signed-Rank Test p-values', fig_name='wilcoxon_heatmap.svg'):
    """
    Plots a heatmap of p-values from pairwise Wilcoxon signed-rank tests and saves the figure.

    Args:
        results_df (DataFrame): Output from paired_wilcoxon_signed_rank() containing columns:
                                ['Tissue 1', 'Tissue 2', 'p-value'].
        tissue_classes (list): List of tissue class names tested.
        output_dir (str or Path): Directory to save the heatmap figure.
        title (str): Title for the heatmap plot.
        fig_name (str): Filename for the saved heatmap figure.

    Returns:
        None: Saves and displays a matplotlib heatmap.
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create an empty p-value matrix initialized to ones
    heatmap_matrix = pd.DataFrame(np.ones((len(tissue_classes), len(tissue_classes))),
                                  index=tissue_classes, columns=tissue_classes)

    # Populate matrix with p-values from results_df
    for _, row in results_df.iterrows():
        t1, t2, pval = row['Tissue 1'], row['Tissue 2'], row['p-value']
        heatmap_matrix.loc[t1, t2] = pval
        heatmap_matrix.loc[t2, t1] = pval  # Ensure symmetry

    # Mask the upper triangle and diagonal
    mask = np.triu(np.ones_like(heatmap_matrix, dtype=bool))

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".4f", cmap='coolwarm_r',
                mask=mask, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=0, vmax=0.05)

    plt.title(title, fontsize=14)
    plt.ylabel('')
    plt.xlabel('')
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / fig_name, format='svg')
    print(f"Heatmap saved to {output_dir / fig_name}")


def plot_effect_size_heatmap(results_df, tissue_classes, effect_size_key, output_dir,
                              title=None, fig_name=None, vmin=None, vmax=None,
                              axis_label_size=16, tick_fontsize=14,
                              cbar_label_fontsize=16, cbar_tick_fontsize=14):
    """
    Plots a heatmap of a specified effect size metric from the results dataframe and saves the figure.

    For directional effect sizes (e.g., 'mean_diff', 'cohen_d', 'hedges_g'),
    the value stored is computed as: tissue_1 - tissue_2. Therefore, in a fully
    populated matrix, we would expect an antisymmetric (negative symmetric) matrix.
    For symmetric metrics (e.g., 'cles'), the same value is stored in both cells.

    In order to remove rows/columns that are not useful (i.e. the top row and the last column,
    which often contain masked values or all-zero self comparisons), this function trims the matrix
    using slicing. Then, to hide the remaining values that are not part of the desired lower-triangular
    (directional) display, a strict upper triangle mask (with k=1) is applied.

    The final displayed heatmap shows only the lower portion of the trimmed matrix, free of
    the diagonal and upper-triangular cells.

    Args:
        results_df (DataFrame): Output from paired_effect_size_analysis(), including effect sizes.
        tissue_classes (list): List of tissue class names tested.
        effect_size_key (str): The effect size column to visualize (e.g., 'cohen_d', 'mean_diff').
        output_dir (str or Path): Directory to save the heatmap figure.
        title (str): Title for the heatmap plot.
        fig_name (str): Filename for the saved heatmap figure (optional, defaults to effect_size_key.svg).
        vmin (float): Minimum value for color scaling (defaults to -1).
        vmax (float): Maximum value for color scaling (defaults to 1).
        axis_label_size (int): Font size for axis labels.
        tick_fontsize (int): Font size for tick labels.
        cbar_label_fontsize (int): Font size for the colorbar label.
        cbar_tick_fontsize (int): Font size for the colorbar tick labels.

    Returns:
        None
    """
    # Define colorbar labels.
    cbar_label_dict = {
        'cohen': "Cohen's d",
        'hedges': "Hedges' g",
        'mean_diff': "Mean Difference",
        'cles': "Common Language Effect Size"
    }
    cbar_label = cbar_label_dict.get(effect_size_key, effect_size_key)

    # Prepare the output directory.
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fig_name is None:
        fig_name = f"{effect_size_key}_heatmap.svg"
    if title is None:
        title = f"Effect Size Heatmap ({cbar_label})"
    else:
        title = f"{title} ({cbar_label})"
    if vmin is None:
        vmin = -1
    if vmax is None:
        vmax = 1

    # Initialize the full square matrix.
    matrix = pd.DataFrame(np.nan, index=tissue_classes, columns=tissue_classes)

    # Fill the matrix:
    # - For directional effect sizes, use antisymmetry:
    #       matrix[tissue1, tissue2] = value and matrix[tissue2, tissue1] = -value.
    # - For symmetric metrics (like 'cles'), copy the value to both cells.
    for _, row in results_df.iterrows():
        t1, t2, value = row['Tissue 1'], row['Tissue 2'], row.get(effect_size_key, np.nan)
        if pd.isna(value):
            continue
        if t1 == t2:
            continue  # skip self comparisons
        if effect_size_key in ['mean_diff', 'cohen_d', 'hedges_g']:
            matrix.loc[t1, t2] = value
            matrix.loc[t2, t1] = -value
        else:
            matrix.loc[t1, t2] = value
            matrix.loc[t2, t1] = value

    # (Optional) You could set the diagonal to zero in the full matrix:
    for t in tissue_classes:
        matrix.loc[t, t] = 0

    # Trim the matrix: remove the top row and the last column
    # This removes the first tissue from the rows and the last tissue from the columns.
    trimmed_matrix = matrix.iloc[1:, :-1]

    # Build a mask for the strict upper triangle (k=1 means the diagonal is not masked)
    mask = np.triu(np.ones_like(trimmed_matrix, dtype=bool), k=1)

    # Plot the heatmap using the trimmed matrix and the strict upper triangle mask.
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(trimmed_matrix, annot=True, fmt=".3f", cmap='vlag',
                     mask=mask, linewidths=0.5, vmin=vmin, vmax=vmax,
                     cbar_kws={"shrink": 0.8, "label": cbar_label})
    plt.title(title, fontsize=14)
    ax.set_xlabel("Tissue Class", fontsize=axis_label_size)
    ax.set_ylabel("Tissue Class", fontsize=axis_label_size)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fontsize, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize, rotation=0)

    # Adjust colorbar label and tick label sizes.
    cb = ax.collections[0].colorbar
    cb.set_label(cbar_label, fontsize=cbar_label_fontsize)
    cb.ax.tick_params(labelsize=cbar_tick_fontsize)

    plt.tight_layout()
    save_path = output_dir / fig_name
    plt.savefig(save_path, format='svg')
    plt.close()
    print(f"Heatmap saved to {save_path}")


def plot_bx_histograms_by_tissue(df, patient_id, bx_index, output_dir, structs_referenced_dict, default_exterior_tissue, fig_name="histograms.svg", bin_width=0.05, spatial_df=None):
    """
    Creates overlapping outline histograms of 'Binomial estimator' values per tissue class for a given biopsy.

    Args:
        df (DataFrame): The input dataframe.
        patient_id (str): The patient ID to filter by.
        bx_index (int): The biopsy index to filter by.
        output_dir (str or Path): Directory to save the plot.
        fig_name (str): Filename for the saved plot.
        bin_width (float): Width of histogram bins.
        spatial_df (DataFrame or None): Optional spatial features dataframe to annotate with.

    Returns:
        None
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter dataframe
    filtered_df = df[(df['Patient ID'] == patient_id) & (df['Bx index'] == bx_index)]

    if filtered_df.empty:
        print(f"No data found for Patient ID: {patient_id}, Bx index: {bx_index}")
        return

    # Get unique tissue classes
    #tissue_classes = filtered_df['Tissue class'].unique()

    tissue_classes = misc_tools.tissue_heirarchy_list_tissue_names_creator_func(structs_referenced_dict,
                                       append_default_exterior_tissue = True,
                                       default_exterior_tissue = default_exterior_tissue
                                       )

    # Create figure
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette(n_colors=len(tissue_classes))

    

    # Plot Binomial estimator outline histograms
    for tissue_class, color in zip(tissue_classes, colors):
        vals = filtered_df[filtered_df['Tissue class'] == tissue_class]['Binomial estimator'].dropna()
        total_voxels = len(vals)
        plt.hist(vals, bins=np.arange(0, 1 + bin_width, bin_width), histtype='step', linewidth=2,
                 label=tissue_class, color=color)


    # Optionally annotate with spatial info
    if spatial_df is not None:
        match = spatial_df[(spatial_df['Patient ID'] == patient_id) & (spatial_df['Bx index'] == bx_index)]
        if not match.empty:
            match = match.iloc[0]
            centroid_dist = match['BX to DIL centroid distance']
            #surface_dist = match['NN surface-surface distance']
            length = match['Length (mm)']
            position = f"{match['Bx position in prostate LR']} / {match['Bx position in prostate AP']} / {match['Bx position in prostate SI']}"

            # Add annotation box in plot
            annotation = (f"Bx Length: {length:.1f} mm\n"
                          f"Bx→DIL centroid dist: {centroid_dist:.1f} mm\n"
                          #f"Bx→DIL NN surface dist: {surface_dist:.1f} mm\n"
                          f"Total Voxels: {total_voxels}\n"
                          f"Sector: {position}")
            plt.gca().text(1.02, 0.95, annotation, transform=plt.gca().transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


    # Final plot adjustments
    plt.title("Multinomial Estimator Distribution by Tissue Class", fontsize=16)
    plt.xlabel("Multinomial Estimator", fontsize=16)
    plt.ylabel("Number of Voxels", fontsize=16)

    
    plt.tick_params(axis='both', which='major', labelsize=16)  # Increase the tick label size for both x and y axes


    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / fig_name, format='svg')
    print(f"Histogram saved to {output_dir / fig_name}")




def production_plot_sum_to_one_tissue_class_binom_regression_matplotlib(multi_structure_mc_sum_to_one_pt_wise_results_dataframe,
                                                                        specific_bx_structure_index,
                                                                        patientUID,
                                                                        structs_referenced_dict,
                                                                        default_exterior_tissue,
                                                                        patient_sp_output_figures_dir,
                                                                        general_plot_name_string):


    def stacked_area_plot_with_confidence_intervals(patientUID,
                                                    bx_struct_roi,
                                                    df, 
                                                    stacking_order):
        """
        Create a stacked area plot for binomial estimator values with confidence intervals,
        stacking the areas to sum to 1 at each Z (Bx frame) point. Confidence intervals are 
        shown as black dotted lines, properly shifted to align with stacked lines.

        :param df: pandas DataFrame containing the data
        :param stacking_order: list of tissue class names, ordered by stacking hierarchy
        """
        plt.ioff()
        fig, ax = plt.subplots(figsize=(12, 8))

        # Generate a common x_range for plotting
        x_range = np.linspace(df['Z (Bx frame)'].min(), df['Z (Bx frame)'].max(), 500)

        # Initialize cumulative variables for stacking
        y_cumulative = np.zeros_like(x_range)

        # Set color palette for tissue classes
        #colors = plt.cm.viridis(np.linspace(0, 1, len(stacking_order)))
        colors = sns.color_palette(n_colors=len(stacking_order))

        # Loop through the stacking order
        for i, tissue_class in enumerate(stacking_order):
            tissue_df = df[df['Tissue class'] == tissue_class]

            # Perform kernel regression for binomial estimator
            kr = KernelReg(endog=tissue_df['Binomial estimator'], exog=tissue_df['Z (Bx frame)'], var_type='c', bw=[1])
            y_kr, _ = kr.fit(x_range)

            # Perform kernel regression for CI lower and upper bounds
            kr_lower = KernelReg(endog=tissue_df['CI lower vals'], exog=tissue_df['Z (Bx frame)'], var_type='c', bw=[1])
            kr_upper = KernelReg(endog=tissue_df['CI upper vals'], exog=tissue_df['Z (Bx frame)'], var_type='c', bw=[1])
            ci_lower_kr, _ = kr_lower.fit(x_range)
            ci_upper_kr, _ = kr_upper.fit(x_range)

            # Stack the binomial estimator values (fill between previous and new values)
            ax.fill_between(x_range, y_cumulative, y_cumulative + y_kr, color=colors[i], alpha=0.7, label=tissue_class)

            # Plot the black dotted lines for confidence intervals, shifted by the cumulative values
            ax.plot(x_range, y_cumulative + ci_upper_kr, color='black', linestyle=':', linewidth=1)  # Upper confidence interval
            ax.plot(x_range, y_cumulative + ci_lower_kr, color='black', linestyle=':', linewidth=1)  # Lower confidence interval

            # Update cumulative binomial estimator for stacking
            y_cumulative += y_kr

        # Final plot adjustments
        ax.set_title(f'{patientUID} - {bx_struct_roi} - Stacked Binomial Estimator with Confidence Intervals by Tissue Class',
             fontsize=16,      # Increase the title font size
             #fontname='Arial' # Set the title font family
            )
        ax.set_xlabel("Biopsy Axial Dimension (mm)",
              fontsize=16,    # sets the font size
              #fontname='Arial'
               )   # sets the font family

        ax.set_ylabel("Multinomial Estimator (stacked)",
                    fontsize=16,
                    #fontname='Arial'
                    )


        ax.legend(loc='best', facecolor='white')
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)

        ax.tick_params(axis='both', which='major', labelsize=16)  # Increase the tick label size for both x and y axes

        plt.tight_layout()

        return fig



    tissue_heirarchy_list = misc_tools.tissue_heirarchy_list_tissue_names_creator_func(structs_referenced_dict,
                                       append_default_exterior_tissue = True,
                                       default_exterior_tissue = default_exterior_tissue
                                       )
        
    sp_structure_mc_sum_to_one_pt_wise_results_dataframe = multi_structure_mc_sum_to_one_pt_wise_results_dataframe[multi_structure_mc_sum_to_one_pt_wise_results_dataframe["Bx index"] == specific_bx_structure_index]
    #extract bx_struct_roi
    bx_struct_roi = sp_structure_mc_sum_to_one_pt_wise_results_dataframe["Bx ID"].values[0]

    fig = stacked_area_plot_with_confidence_intervals(patientUID,
                                                bx_struct_roi,
                                                sp_structure_mc_sum_to_one_pt_wise_results_dataframe, 
                                                tissue_heirarchy_list)

    svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

    fig.savefig(svg_dose_fig_file_path, format='svg')

    # clean up for memory
    plt.close(fig)





def production_plot_sum_to_one_tissue_class_nominal_plotly(multi_structure_mc_sum_to_one_pt_wise_results_dataframe,
                                                patientUID,
                                                specific_bx_structure_index,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string,
                                                patient_sp_output_figures_dir,
                                                structs_referenced_dict,
                                                default_exterior_tissue
                                                ):

    def tissue_class_sum_to_one_nominal_plot(df, y_axis_order, patientID, bx_struct_roi):
        df = misc_tools.convert_categorical_columns(df, ['Tissue class', 'Nominal'], [str, int])

        # Generate a list of colors using viridis colormap in Matplotlib
        stacking_order = y_axis_order
        #colors = plt.cm.viridis(np.linspace(0, 1, len(stacking_order)))  # Same method you used in Matplotlib
        colors = sns.color_palette(n_colors=len(stacking_order))

        # Convert the colors to a format Plotly understands (hex strings)
        hex_colors = ['#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
        hex_colors.reverse()
        # Create a color mapping for tissue classes
        color_mapping = dict(zip(stacking_order, hex_colors))
        
        # Hack to adjust the size of the markers
        scale_size = 15
        df["Nominal scaled size"] = df["Nominal"] * scale_size  # Scale size as needed

        # Create the scatter plot and pass the custom color map
        fig = px.scatter(
            df, 
            x='Z (Bx frame)', 
            y='Tissue class', 
            size="Nominal scaled size",  # Size based on Nominal scaled size (0 or 15)
            size_max=scale_size,  # Set size for the points that appear
            color='Tissue class',  # Use tissue class for color assignment
            color_discrete_map=color_mapping,  # Apply the custom color mapping
            title=f'Sum-to-one Nominal tissue class along biopsy major axis (Pt: {patientID}, Bx: {bx_struct_roi})'
        )

        

        # Clear all existing legend entries
        fig.for_each_trace(lambda trace: trace.update(showlegend=False))

        # Add dummy scatter points for the legend with fixed size
        for tissue_class in list(reversed(stacking_order)):
            fig.add_scatter(
                x=[None],  # Dummy invisible point
                y=[None],
                mode='markers',
                marker=dict(size=scale_size, color=color_mapping[tissue_class], symbol='x'),
                name=tissue_class,  # Ensure tissue class appears in legend
                showlegend=True
            )

        # Customize point style
        fig.update_traces(
            marker=dict(
                symbol='x',  # Change to other shapes like 'diamond', 'square', etc.
                #line=dict(width=2, color='DarkSlateGrey'),  # Add border to points
                #size=15,  # Set a base size (adjustable)
                #opacity=1,  # Set point transparency
                #color = 'black'
            )
        )

        # Customize labels and make the plot flatter by tweaking y-axis category settings
        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="Biopsy Axial Dimension (mm)",
                    font=dict(
                        size=30,
                    )
                ),
                tickfont=dict(
                    size=30,
                )
                ),
            yaxis=dict(
                title=dict(
                    text="Nominal Tissue Class",
                    font=dict(
                        size=30,
                        color="black"
                    )
                ),
                tickfont=dict(
                    size=30,
                    color="black"
                ),
                categoryorder='array',  # Set custom order
                categoryarray=y_axis_order,  # Use the provided order for categories
                tickvals=y_axis_order,  # Ensure the ticks follow this order
                tickmode='array',
                ticktext=y_axis_order,
                scaleanchor="x",  # Lock the aspect ratio of x and y
                dtick=1  # Control category spacing
            ),
            height=400,  # Adjust the overall height of the plot to flatten it
            legend_title_text='Tissue class'  # Set legend title
        )

        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = True)
        fig.update_layout(
        paper_bgcolor='white',  # Background of the entire figure
        plot_bgcolor='white'    # Background of the plot area
        )

        return fig 


    # Define the specific order for the y-axis categories
    y_axis_order = misc_tools.tissue_heirarchy_list_tissue_names_creator_func(structs_referenced_dict,
                                       append_default_exterior_tissue = True,
                                       default_exterior_tissue = default_exterior_tissue
                                       )
    y_axis_order.reverse()

    mc_compiled_results_sum_to_one_for_fixed_bx_dataframe = multi_structure_mc_sum_to_one_pt_wise_results_dataframe[multi_structure_mc_sum_to_one_pt_wise_results_dataframe["Bx index"] == specific_bx_structure_index]

    
    # Plotting loop
    bx_struct_roi = mc_compiled_results_sum_to_one_for_fixed_bx_dataframe["Bx ID"].values[0]
    
    fig = tissue_class_sum_to_one_nominal_plot(mc_compiled_results_sum_to_one_for_fixed_bx_dataframe, y_axis_order, patientUID, bx_struct_roi)

    bx_sp_plot_name_string = f"{bx_struct_roi} - " + general_plot_name_string

    svg_dose_fig_name = bx_sp_plot_name_string+'.svg'
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height/3) # added /3 here to make the y axis categories closer together, ie to make the plot flatter so that it can fit beneath the sum-to-one spatial regression plots.

    html_dose_fig_name = bx_sp_plot_name_string+'.html'
    html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 