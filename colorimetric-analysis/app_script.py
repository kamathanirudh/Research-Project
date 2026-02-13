import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from inference_sdk import InferenceHTTPClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import math
from scipy.stats import linregress
from collections import defaultdict
from colormath.color_objects import LabColor
from colormath2.color_diff import delta_e_cie2000
import os


def detect_circles_yolo(image_path, rows, cols, api_key, model_id,reduction_factor, plot_paths=None,concentrations=None):
    """
    Detect circles using Roboflow YOLO model, extract RGB and color info,
    plot graphs, and save results to Excel.

    Args:
        image_path (str): Path to input image.
        rows (int): Number of rows of circles expected.
        cols (int): Number of columns of circles expected.
        api_key (str): Roboflow API key.
        model_id (str): Roboflow model id.

    Returns:
        list: List of tuples containing (center, reduced_radius, points, confidence, number).
    """
    if concentrations is None:
        raise ValueError("concentrations must be provided as a list of values.")
    # --- Binary search for optimal CONFIDENCE_THRESHOLD ---
    low, high = 0.0, 1.0
    best_threshold = None
    target = rows * cols
    epsilon = 0.01
    # Run inference
    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)
    result = client.infer(image_path, model_id=model_id)
    def count_circles(thresh):
        detected = []
        predictions = result['predictions']  # Ensure dict key access, not a slice
        for pred in predictions:
            confidence = pred['confidence']
            class_name = pred['class'].lower()
            if class_name == "cricle" and confidence >= thresh:
                detected.append(pred)
        return detected
    while high - low > epsilon:
        mid = round((low + high) / 2, 4)
        detected = count_circles(mid)
        num_circles = len(detected)
        if num_circles == target:
            best_threshold = mid
            break  # Stop as soon as we get the required number
        elif num_circles > target:
            low = mid
        else:
            high = mid
    CONFIDENCE_THRESHOLD = best_threshold if best_threshold is not None else 0.5
    print(f"Best threshold: {CONFIDENCE_THRESHOLD}")

    # Initialize Roboflow client
    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)

    # Load original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Could not load image.")
        return [], pd.DataFrame()

    # Run inference
    result = client.infer(image_path, model_id=model_id)

    # Extract detected circles with confidence filter
    detected_circles = []
    for pred in result['predictions']:
        confidence = pred['confidence']
        class_name = pred['class'].lower()

        if class_name == "cricle" and confidence >= CONFIDENCE_THRESHOLD:
            points = pred['points']
            polygon_points = np.array([(p['x'], p['y']) for p in points], dtype=np.int32).reshape((-1, 1, 2))

            (x, y), radius = cv2.minEnclosingCircle(polygon_points)
            reduced_radius = max(1, int(radius * reduction_factor))
            center = (int(x), int(y))
            detected_circles.append((center, reduced_radius, points, confidence))

    # 1) Sort purely by y-coordinate (top to bottom)
    sorted_by_y = sorted(detected_circles, key=lambda c: c[0][1])

    # 2) Chop into `rows` chunks of size `cols` (assumes len(detected_circles) == rows*cols)
    row_chunks = [
        sorted_by_y[i*cols : (i+1)*cols]
        for i in range(rows)
    ]

    # 3) Within each chunk (row), sort left-to-right by x-coordinate
    sorted_circles = []
    for chunk in row_chunks:
        sorted_chunk = sorted(chunk, key=lambda c: c[0][0])
        sorted_circles.extend(sorted_chunk)

    # 4) Number them in reading order
    numbered_circles = [
        (c[0], c[1], c[2], c[3], idx + 1)
        for idx, c in enumerate(sorted_circles)
    ]


    # Initialize DataFrame
    df = pd.DataFrame(columns=['Circle Number', 'R', 'G', 'B',
                               'Normalised R', 'Normalised G', 'Normalised B',
                               'X', 'Y', 'Z', 'L', 'a*', 'b*'])

    # RGB to XYZ conversion matrix (sRGB)

    rgb_to_xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    #D65
    
    ''' sRGB D65 to D50 XYZ  0.4360747  0.3850649  0.1430804
                            0.2225045  0.7168786  0.0606169
                            0.0139322  0.0971045  0.7141733
        
        
         sRGB to D65 XYZ     0.4124564  0.3575761  0.1804375
                            0.2126729  0.7151522  0.0721750
                            0.0193339  0.1191920  0.9503041
    
    
    
         D65	95.047	100.000	108.883	~6500K (Daylight)
         D50	96.421	100.000	82.519	~5000K (Warmer)    '''
    
    
    
    
    
    def xyz_to_lab(xyz, white_point=(95.047, 100.0, 108.883)): #D65 White Point
        # Unpack input XYZ and white point values
        X, Y, Z = xyz*100
        Xr, Yr, Zr = white_point

        # Normalize by reference white
        xr = X / Xr
        yr = Y / Yr
        zr = Z / Zr

        # Constants from CIE standard
        epsilon = 216 / 24389  # ≈ 0.008856
        kappa = 24389 / 27     # ≈ 903.3

        # Helper function as per condition
        def f(t):
            if t > epsilon:
                return t ** (1/3)
            else:
                return (kappa * t + 16) / 116

        fx = f(xr)
        fy = f(yr)
        fz = f(zr)

        # Final Lab computation
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return L, a, b



    
        
        
    # Calculate average RGB, normalized RGB, convert to XYZ and LAB
    for center, radius, points, conf, number in numbered_circles:
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, thickness=-1)

        x, y = center
        roi = cv2.bitwise_and(original_image, original_image, mask=mask)
        roi = roi[y-radius:y+radius, x-radius:x+radius]
        mask_roi = mask[y-radius:y+radius, x-radius:x+radius]

                # Assume roi and mask_roi are defined and loaded properly
        circle_pixels = roi[mask_roi == 255]

        # Extract mean BGR channels (OpenCV format)
        avg_b = np.mean(circle_pixels[:, 0])
        avg_g = np.mean(circle_pixels[:, 1])
        avg_r = np.mean(circle_pixels[:, 2])
        # L, a_lab, b_lab= avg_rgb_to_lab(avg_r,avg_g,avg_b)
        # Normalize to [0, 1]
        normalised_r = avg_r / 255.0
        normalised_g = avg_g / 255.0
        normalised_b = avg_b / 255.0
        
        # Create RGB array
        rgb = np.array([normalised_r, normalised_g, normalised_b])
        
        def inverse_srgb_companding(V):
            """Apply inverse sRGB companding to each channel (V in [0,1])"""
            return np.where(V <= 0.04045,
                            V / 12.92,
                            ((V + 0.055) / 1.055) ** 2.4)

        # Apply inverse sRGB companding to get linear RGB
        linear_rgb = inverse_srgb_companding(rgb)

        # Convert linear RGB to XYZ
        xyz = np.dot(rgb_to_xyz_matrix, linear_rgb)
        X, Y, Z = xyz
        
        # def adapt_D65_to_D50(xyz):
        #     M = np.array([[ 0.9555766, -0.0230393,  0.0631636],
        #                 [-0.0282895,  1.0099416,  0.0210077],
        #                 [ 0.0122982, -0.0204830,  1.3299098]])
        #     return np.dot(M, xyz)
        
        # xyz = adapt_D65_to_D50(xyz)

        
        L, a_lab, b_lab = xyz_to_lab(xyz)

        new_row = {'Circle Number': number, 'R': avg_r, 'G': avg_g, 'B': avg_b,
                   'Normalised R': normalised_r, 'Normalised G': normalised_g, 'Normalised B': normalised_b,
                   'X': X, 'Y': Y, 'Z': Z, 'L': L, 'a*': a_lab, 'b*': b_lab}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Define concentration values (assuming uniform increments)
    conc_values = np.array(concentrations)

    # Prepare experiments data dict for plotting
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray']
    if rows > len(colors):
        raise ValueError("Not enough predefined colors for the number of rows")
    
    def rgb_to_hsv(row):
        r, g, b = row['R'] / 255.0, row['G'] / 255.0, row['B'] / 255.0

        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        # Hue calculation
        if delta == 0:
            h = 0
        elif cmax == r:
            h = (60 * ((g - b) / delta)) % 360
        elif cmax == g:
            h = (60 * ((b - r) / delta + 2))
        else:  # cmax == b
            h = (60 * ((r - g) / delta + 4))

        # Saturation calculation
        s = 0 if cmax == 0 else (delta / cmax)

        # Value calculation
        v = cmax

        # Return values in expected format: H in degrees, S and V in percentage
        return pd.Series({'H': h, 'S': s * 100, 'V': v * 100})
    hsv_df = df.apply(rgb_to_hsv, axis=1)
    df = pd.concat([df, hsv_df], axis=1)

    
    experiments = {}
    for i in range(rows):
        start_idx = i * cols
        end_idx = start_idx + cols
        # Filter out column 0 (blank column) - keep only columns 1-4
        experiment_data = df.iloc[start_idx:end_idx]
        # Remove the first row (column 0) from each experiment
        filtered_data = experiment_data.iloc[1:]
        experiments[f"Experiment {i+1} (Circles {start_idx+2}-{end_idx})"] = {
            'color': colors[i],
            'data': filtered_data
        }
        
    # Convert RGB to HSV for each row in df and add columns
    



    def plot_initial(plot_paths=None):
        # Show individual ROI images
        fig, axs = plt.subplots(rows, cols, figsize=(5, 5))
        axs = axs.flatten()  # flatten in case axs is 2D

        for i, (center, radius, points, conf, number) in enumerate(numbered_circles):
            if i >= rows * cols:
                break
            ax = axs[i]
            # Extract ROI and plot on ax
            mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, thickness=-1)
            roi = cv2.bitwise_and(original_image, original_image, mask=mask)
            x, y = center
            roi = roi[y-radius:y+radius, x-radius:x+radius]
            ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title(f"Circle {number}\nConf: {conf:.2f}", fontsize=10)

        plt.tight_layout()
        if plot_paths:
            print(f"Saving plot to {plot_paths[0]}")
            plt.savefig(plot_paths[0])
        plt.close()

        # Draw detections on original image
        image_with_detections = original_image.copy()
        for center, radius, points, conf, number in numbered_circles:
            polygon_points = np.array([(p['x'], p['y']) for p in points], dtype=np.int32)
            cv2.polylines(image_with_detections, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(image_with_detections, center, radius, (255, 0, 0), 2)
            cv2.putText(image_with_detections, f"{conf:.2f}", (center[0], center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(image_with_detections, f"#{number}", (center[0] - 20, center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        if plot_paths:
            print(f"Saving plot to {plot_paths[1]}")
            plt.savefig(plot_paths[1])  # Save the second plot
        plt.close()


        def plot_measurements_grid(experiments, conc_values):
            ylabels = [
                'X Value', 'Y Value', 'Z Value',
                'R', 'G', 'B',
                'Hue', 'Saturation', 'Value',
                'L', 'a*', 'b*'
            ]
            columns = [
                'X', 'Y', 'Z',
                'R', 'G', 'B',
                'H', 'S', 'V',
                'L', 'a*', 'b*'
            ]

            fig, axs = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows x 3 columns grid
            axs = axs.flatten()

            for ax, col, ylabel in zip(axs, columns, ylabels):
                for label, exp in experiments.items():
                    ax.plot(conc_values, exp['data'][col], marker='o', linestyle='-', linewidth=2, color=exp['color'], label=label)
                ax.set_xlabel('Concentration')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{ylabel} vs Concentration')
                ax.legend(fontsize=8)
                ax.grid(True)

            plt.tight_layout()
            if plot_paths:
                print(f"Saving plot to {plot_paths[2]}")
                plt.savefig(plot_paths[2]) # Save the third plot
            plt.close()


        plot_measurements_grid(experiments, conc_values)
    plot_initial(plot_paths)

    return numbered_circles, df


def delta_e_calc(df, rows, cols):
    total = rows * cols
    if len(df) < total:
        raise ValueError("Not enough rows in DataFrame for given grid dimensions.")
    
    # Assign column index: repeats [0, 1, ..., cols-1] for each row
    df['column index'] = [i % cols for i in range(total)]
    
    # Assign experiment: increases every 'cols' rows
    df['experiment'] = [i // cols + 1 for i in range(total)]

    # Copy dataframe
    df_filtered = df.copy()
    delta_e_values = []

    for exp in range(1, rows + 1):
        exp_df = df_filtered[df_filtered['experiment'] == exp]
        ref_row = exp_df[exp_df['column index'] == 0]

        if ref_row.empty:
            raise ValueError(f"Reference (column index 0) not found for experiment {exp}")

        ref_L, ref_A, ref_B = ref_row.iloc[0][['L', 'a*', 'b*']]
        ref_lab = LabColor(lab_l=ref_L, lab_a=ref_A, lab_b=ref_B)

        for _, row in exp_df.iterrows():
            sample_lab = LabColor(lab_l=row['L'], lab_a=row['a*'], lab_b=row['b*'])
            delta_e = delta_e_cie2000(ref_lab, sample_lab)
            # Ensure scalar output
            if hasattr(delta_e, 'item'):
                delta_e = delta_e.item()
            delta_e_values.append(delta_e)

    df_filtered['Delta E'] = delta_e_values
    df_filtered.drop(['experiment', 'column index'], axis=1, inplace=True)
    return df_filtered


def gray_scale_conv(df):
    df['Grayscale'] = (
        0.2989 * df['R'] +
        0.5870 * df['G'] +
        0.1140 * df['B']
    )
    return df


def filter_df(df,cols):
    # Drop unnecessary columns
    df2 = df.drop(columns=['Normalised R', 'Normalised G', 'Normalised B'], errors='ignore')

    # Compute Column Index and filter out blank column
    df2["Column Index"] = (df2["Circle Number"] - 1) % cols
    df2 = df2[df2["Column Index"] != 0]

    # Final filtered dataframe
    filtered_df = df2.drop(columns=["Column Index"], errors='ignore')

    return df2,filtered_df


# def plot_wo_blank_value(filtered_df, rows, cols, concentrations):

#     # Parameters to plot (excluding Circle Number)
#     parameters = [col for col in filtered_df.columns if col != "Circle Number"]

#     # ---------- 1. Line Plot Grid ----------
#     max_cols = 3
#     num_params = len(parameters)
#     num_rows = -(-num_params // max_cols)  # Ceiling division

#     fig1, axes1 = plt.subplots(num_rows, max_cols, figsize=(15, 5 * num_rows))
#     axes1 = axes1.flatten()

#     for i, param in enumerate(parameters):
#         ax = axes1[i]

#         for exp in range(rows):
#             start = exp * (cols - 1)
#             end = (exp + 1) * (cols - 1)
#             exp_data = filtered_df.iloc[start:end]

#             ax.plot(concentrations, exp_data[param], marker='o', linestyle='-', label=f"Experiment {exp+1}")

#         ax.set_title(f"{param} vs. Concentration")
#         ax.set_xlabel("Concentration")
#         ax.set_ylabel(param)
#         ax.legend()
#         ax.grid(True)

#     # Hide unused axes
#     for j in range(i + 1, len(axes1)):
#         fig1.delaxes(axes1[j])
#     plt.tight_layout()
#     plt.show()

#     # # ---------- 2. Scatter Plot Grid (Flipped Axes) ----------
#     # fig2, axes2 = plt.subplots(num_rows, max_cols, figsize=(15, 5 * num_rows))
#     # axes2 = axes2.flatten()

#     # for i, param in enumerate(parameters):
#     #     ax = axes2[i]

#     #     for exp in range(rows):
#     #         start = exp * (cols - 1)
#     #         end = (exp + 1) * (cols - 1)
#     #         exp_data = filtered_df.iloc[start:end]

#     #         ax.scatter(exp_data[param], concentrations, marker='o', label=f"Experiment {exp+1}")

#     #     ax.set_title(f"Concentration vs. {param}")
#     #     ax.set_xlabel(param)
#     #     ax.set_ylabel("Concentration")
#     #     ax.legend()
#     #     ax.grid(True)

#     # for j in range(i + 1, len(axes2)):
#     #     fig2.delaxes(axes2[j])
#     # plt.tight_layout()
#     # plt.show()


def add_experiment_column(df2, total_experiments, cols_per_experiment):
    # Make a copy of df2 to avoid modifying original
    df3 = df2.copy().reset_index(drop=True)

    # Total number of data points per experiment
    rows_per_experiment = cols_per_experiment

    # Sanity check: make sure the expected number of rows is correct
    expected_rows = total_experiments * rows_per_experiment
    actual_rows = len(df3)

    if actual_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows (from {total_experiments} experiments with {rows_per_experiment} points), "
                         f"but got {actual_rows} rows in df2.")

    # Assign experiment numbers
    experiment_numbers = []
    for i in range(total_experiments):
        experiment_numbers.extend([i + 1] * rows_per_experiment)

    df3["Experiment"] = experiment_numbers

    return df3


def parameter_linear_regression_evaluation(df3,plot_path=None, experiment_type="Albumin"):
    # Map Column Index values 1,2,3,4 to 2,4,6,8
    mapping = {1: 2, 2: 4, 3: 6, 4: 8}
    df3 = df3.copy()
    if set(df3['Column Index'].unique()).issubset({1, 2, 3, 4}):
        df3['Column Index'] = df3['Column Index'].map(mapping)

    features = ['R', 'G', 'B', 'X', 'Y', 'Z', 'L', 'a*', 'b*', 'H', 'S', 'V', 'Grayscale', 'Delta E']

    metrics = {
        'Feature': [],
        'R²': [],
        'RMSE': [],
        'MAE': [],
        'AIC': [],
        'BIC': []
    }

    X = df3[['Column Index']].values  # X is always Column Index
    n = len(df3)

    

    # Prepare for plotting
    num_features = len(features)
    ncols = 3
    nrows = (num_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f'{experiment_type} Linear Regression model', fontsize=20, y=0.99)
    experiment_numbers = sorted(df3['Experiment'].unique())
    # Use specific dark colors for up to 3 experiments
    exp_colors = ['navy', 'darkgreen', 'red']

    for idx, feature in enumerate(features):
        y = df3[feature].values  # y is the color feature
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        # Residuals
        residuals = y - y_pred

        # Metrics
        r2 = model.score(X, y)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        # Log-likelihood for AIC/BIC
        sse = np.sum(residuals ** 2)
        k = X.shape[1] + 1  # parameters (slope + intercept)
        aic = n * np.log(sse / n) + 2 * k
        bic = n * np.log(sse / n) + k * np.log(n)

        # Store results
        metrics['Feature'].append(feature)
        metrics['R²'].append(r2)
        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
        metrics['AIC'].append(aic)
        metrics['BIC'].append(bic)

        # Plotting
    
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        for i, exp_num in enumerate(experiment_numbers):
            exp_data = df3[df3['Experiment'] == exp_num]
            color = exp_colors[i % len(exp_colors)]
            ax.plot(
                exp_data['Column Index'],
                exp_data[feature],
                marker='o', linestyle='-',
                color=color, label=f'Row {exp_num}', alpha=0.8
            )
        # Regression line (dotted)
        x_range = np.linspace(df3['Column Index'].min(), df3['Column Index'].max(), 100).reshape(-1, 1)
        y_reg = model.predict(x_range)
        ax.plot(x_range, y_reg, 'k--', label=f'Regression (R²={r2:.2f})', linewidth=2)
        ax.set_title(feature)
        ax.set_xlabel('Concentration (g/dL)', fontsize=15)
        ax.set_ylabel(feature, fontsize=15)
        ax.grid(True)
        ax.legend(fontsize=12)

# Hide unused subplots and show plot

    for j in range(num_features, nrows * ncols):
        row, col = divmod(j, ncols)
        if row < nrows and col < ncols and axes[row][col] in fig.axes:
            fig.delaxes(axes[row][col])
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if plot_path:
        print("Saving regression plots to", plot_path)
        plt.savefig(plot_path)
    plt.close()
    # Convert to DataFrame and sort by R² (descending) or AIC/BIC (ascending)
    result_df = pd.DataFrame(metrics)
    result_df = result_df.sort_values(by='R²', ascending=False).reset_index(drop=True)

    return result_df, df3

def compute_lod_table(df_with_blank, df_reg, cols):
    """
    Returns a DataFrame:
    Color Space | Blank Row 1 | Blank Row 2 | ... | Std Dev | LOD
    """

    features = [
        'R', 'G', 'B',
        'X', 'Y', 'Z',
        'L', 'a*', 'b*',
        'H', 'S', 'V',
        'Grayscale', 'Delta E'
    ]

    rows = []

    # Prepare blank dataframe
    df_blank = df_with_blank.copy()
    df_blank["Column Index"] = (df_blank["Circle Number"] - 1) % cols
    df_blank["Experiment"] = (df_blank["Circle Number"] - 1) // cols + 1

    for feature in features:
        if feature not in df_blank.columns:
            continue

        # --- Blank values per row ---
        blank_vals = (
            df_blank[df_blank["Column Index"] == 0]
            .sort_values("Experiment")[feature]
            .values
        )

        # Need at least 2 blanks
        if len(blank_vals) < 2:
            continue

        sigma_blank = np.std(blank_vals, ddof=1)

        # --- Regression slope ---
        X = df_reg[['Column Index']].values
        y = df_reg[feature].values
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]

        if slope == 0:
            continue

        lod = (3 * sigma_blank) / abs(slope)

        row = {"Color Space": feature}
        for i, val in enumerate(blank_vals, start=1):
            row[f"Blank Row {i}"] = val

        row["Std Dev (σ_blank)"] = sigma_blank
        row["LOD (g/dL)"] = lod

        rows.append(row)

    return pd.DataFrame(rows)


# if __name__ == "__main__":
#     print("Current working directory:", os.getcwd())
#     api_key = "y6u7lWMqSmjyVGQh4KJJ"
#     model_id = "my-first-project-xz1fx/5"
#     image_path = "test3.jpg"
#     rows = 4
#     cols = 5
#     concentrations = [2, 4, 6, 8]  # For columns 1-4 (excluding blank column 0)
#     reduction_factor= 0.8
#     def on_click():
#         numbered_circles, df = detect_circles_yolo(image_path, rows, cols, api_key, model_id,reduction_factor) #2 plots
#         df=delta_e_calc(df, rows,cols)
#         df=gray_scale_conv(df)
#         df2, filtered_df = filter_df(df, cols)
#         df3 = add_experiment_column(df2, total_experiments=rows, cols_per_experiment=cols-1)
#         print("all circles without blank")
#         print(df3)
#         r2_df, df3 = parameter_linear_regression_evaluation(df3)#1 plot
#         print("ranking linear regression")
#         print(r2_df)
#     on_click()