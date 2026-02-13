from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import logging
import uuid
import pandas as pd

# Import your workflow functions here
from app_script import (
    detect_circles_yolo, delta_e_calc, gray_scale_conv,
    filter_df, add_experiment_column, parameter_linear_regression_evaluation, compute_lod_table

)

app = FastAPI()

# Allow CORS for development (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
API_KEY = "y6u7lWMqSmjyVGQh4KJJ"
MODEL_ID = "my-first-project-xz1fx/5"
REDUCTION_FACTOR = 0.8

logging.basicConfig(level=logging.INFO)

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    experiment_type: str = Form(...),
    rows: int = Form(...),
    cols: int = Form(...),
    concentrations: str = Form(...),
):
    try:
        logging.info("Received /analyze request")
        import uuid, json, os, shutil

        image_id = str(uuid.uuid4())
        image_path = f"uploads/{image_id}_{image.filename}"
        os.makedirs("uploads", exist_ok=True)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        logging.info(f"Saved image to {image_path}")

        concentrations_list = json.loads(concentrations)
        logging.info(f"Parsed concentrations: {concentrations_list}")
        
        logging.info(f"Final concentrations list: {concentrations_list}")

        results_dir = f"results/{image_id}"
        os.makedirs(results_dir, exist_ok=True)
        plot_paths = [os.path.join(results_dir, f"plot{i}.png") for i in range(1, 5)]
        logging.info(f"Plot paths: {plot_paths}")

        numbered_circles, df = detect_circles_yolo(
            image_path, rows, cols, API_KEY, MODEL_ID, REDUCTION_FACTOR, plot_paths=plot_paths[:3], concentrations=concentrations_list
        )
        logging.info("detect_circles_yolo completed")

        df = delta_e_calc(df, rows, cols)
        logging.info("delta_e_calc completed")
        df = gray_scale_conv(df)
        logging.info("gray_scale_conv completed")

        df_with_blank,df_no_blank = filter_df(df,cols)
        logging.info("Initial filter_df completed")
        df3 = add_experiment_column(df_no_blank, total_experiments=rows, cols_per_experiment=cols-1)
        logging.info("add_experiment_column completed")


        r2_df, df3_reg = parameter_linear_regression_evaluation(df3, plot_path=plot_paths[3], experiment_type=experiment_type)
        logging.info("parameter_linear_regression_evaluation completed")

        lod_df = compute_lod_table(df, df3_reg, cols)

        lod_path = os.path.join(results_dir, "lod_table.csv")
        lod_df.to_csv(lod_path, index=False)

        logging.info(f"LOD table saved to {lod_path}")


        df3_path = os.path.join(results_dir, "all_color_data.csv")
        r2_path = os.path.join(results_dir, "linear_regression_ranking.csv")
        df3.to_csv(df3_path, index=False)
        r2_df.to_csv(r2_path, index=False)
        logging.info(f"Saved datasets: {df3_path}, {r2_path}")

        files_to_check = plot_paths + [df3_path, r2_path, lod_path]
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                logging.error(f"Expected file not found: {file_path}")

        existing_plots = [p for p in plot_paths if os.path.exists(p)]
        datasets = {
            "all_color_data": df3_path if os.path.exists(df3_path) else None,
            "linear_regression_ranking": r2_path if os.path.exists(r2_path) else None,
            "lod_table": lod_path if os.path.exists(lod_path) else None
        }

        return {
            "plots": existing_plots,
            "datasets": datasets,
            "lod": lod_df.to_dict(orient="records")
        }
    except Exception as e:
        import traceback
        logging.error(f"Exception in /analyze: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=400,
            content={"error": f"{str(e)}\n{traceback.format_exc()}"}
        )
    


    
@app.get("/download/{file_path:path}")
def download_file(file_path: str):
    if not file_path.startswith("results/"):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    return FileResponse(file_path)