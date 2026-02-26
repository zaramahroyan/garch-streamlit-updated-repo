# import streamlit as st
# import pandas as pd
# import numpy as np
# from arch import arch_model
# from scipy.stats import t as student_t
# import io

# st.set_page_config(page_title="Risk Dept GARCH Analyzer", layout="wide")

# st.title("📈 GARCH(1,1)-t Volatility & VaR Analyzer")
# st.write("Targeting Seed Variance: 751st-850th values from the bottom.")

# # 1. FILE UPLOADER
# uploaded_file = st.file_uploader("Choose the returns.xlsx file", type="xlsx")

# if uploaded_file:
#     try:
#         st.info("Reading and cleaning data... Please wait.")

#         raw = pd.read_excel(uploaded_file, sheet_name="Prices", header=0)
#         raw.rename(columns={raw.columns[0]: "Date"}, inplace=True)
#         raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
#         raw.set_index("Date", inplace=True)

#         df_prices = raw.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")
#         df_prices = df_prices.dropna(axis=1, how="all")

#         all_returns = pd.DataFrame(index=df_prices.index)
#         all_stdevs = pd.DataFrame(index=df_prices.index)
#         all_var_99 = pd.DataFrame(index=df_prices.index)
#         model_params = []

#         progress_bar = st.progress(0)
#         assets = df_prices.columns

#         for i, asset in enumerate(assets):
#             series = df_prices[asset].dropna()

#             # NEW REQUIREMENT: At least 850 returns 
#             # (Buffer 750 + Window 100)
#             ret = 100 * np.log(series / series.shift(1)).dropna()
#             T = len(ret)

#             if T < 851:
#                 # If an asset has less than 851 returns, it cannot support this window
#                 progress_bar.progress((i + 1) / len(assets))
#                 continue

#             try:
#                 # --- STEP A: Parameter Estimation ---
#                 model = arch_model(ret, vol="GARCH", p=1, q=1, dist="t")
#                 res = model.fit(disp="off")

#                 om, al, be, nu = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"], res.params["nu"]

#                 # --- STEP B: Recursive GARCH with 750-day Offset Seed ---
#                 sigma2 = np.full(T, np.nan)

#                 # NEW LOGIC:
#                 # We want 100 values for variance, ending 750 values from the bottom.
#                 # Seed end = T - 750
#                 # Seed start = T - 850
#                 seed_start = T - 850
#                 seed_end = T - 750
                
#                 # Calculate variance for the window [T-850 : T-750]
#                 seed_variance = np.var(ret.values[seed_start:seed_end])
                
#                 # Place the seed value at the end of the window (index seed_end - 1)
#                 sigma2[seed_end - 1] = seed_variance

#                 r_sq = ret.values ** 2

#                 # Recursive GARCH fills the remaining 750 days to the present
#                 for t in range(seed_end, T):
#                     sigma2[t] = om + al * r_sq[t - 1] + be * sigma2[t - 1]

#                 stdev = np.sqrt(sigma2)
#                 t_quantile = student_t.ppf(0.01, nu)

#                 all_returns[asset] = ret
#                 all_stdevs[asset] = pd.Series(stdev, index=ret.index)
#                 all_var_99[asset] = pd.Series(t_quantile * stdev, index=ret.index)

#                 model_params.append({
#                     "Asset": asset, "Omega": om, "Alpha": al, "Beta": be,
#                     "Persistence": al + be, "Nu (DF)": nu,
#                     "Seed_Used_At_Index": seed_end
#                 })

#             except Exception:
#                 pass

#             progress_bar.progress((i + 1) / len(assets))

#         # --------------------------------------------------
#         # 4. EXCEL EXPORT
#         # --------------------------------------------------
#         output = io.BytesIO()
#         df_prices.index = pd.to_datetime(df_prices.index)
#         all_returns.index = pd.to_datetime(all_returns.index)
#         all_stdevs.index = pd.to_datetime(all_stdevs.index)
#         all_var_99.index = pd.to_datetime(all_var_99.index)

#         with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="dd-mmm-yy") as writer:
#             workbook = writer.book
#             date_format = workbook.add_format({"num_format": "dd-mmm-yy"})
            
#             def save_fmt(df, name):
#                 df.to_excel(writer, sheet_name=name)
#                 ws = writer.sheets[name]
#                 ws.set_column("A:A", 18, date_format)

#             save_fmt(df_prices, "Original_Prices")
#             save_fmt(all_returns, "Returns_Scaled")
#             save_fmt(all_stdevs, "GARCH_Stdev")
#             save_fmt(all_var_99, "GARCH_VaR")
#             pd.DataFrame(model_params).to_excel(writer, sheet_name="Model_Parameters", index=False)

#         st.success("✅ Analysis Complete!")
#         st.download_button(
#             label="Download GARCH Analysis Report",
#             data=output.getvalue(),
#             file_name="Risk_Dept_750_Buffer_GARCH.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         )

#     except Exception as e:
#         st.error(f"Error: {e}")

# ====================================================================================

import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t as student_t
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Risk Dept GARCH Analyzer", layout="wide")

# --- CUSTOM CSS FOR THE DARK RED OMBRE THEME ---
# --- CUSTOM CSS FOR THE DARK RED OMBRE THEME ---
st.markdown("""
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #0a0a0a 50%, #4a0505 100%);
        color: #e0e0e0;
    }

    /* Glowing White Headings */
    h1, h2, h3, h4{
        color: #ffffff !important;
        text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.4);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Customizing the File Uploader Box */
    .stFileUploader section {
        background-color: rgba(255, 255, 0, 0.05) !important;
        border: 1px solid #4a0a0a !important;
        border-radius: 10px;
    }

    /* Target the "Drag and drop", "Limit", and "XLSX" text specifically */
    .stFileUploader [data-testid='stFileUploadDropzone'] div div span {
        color: #ffffff !important;
    }
    
    .stFileUploader [data-testid='stFileUploadDropzone'] div div small {
        color: #fffccc !important;
    }

    /* General text colors for labels and markdown */
    .stMarkdown, p, label, [data-testid="stWidgetLabel"] {
        color: #d3d3d3 !important;
    }

    /* Success Message Styling */
    .stAlert {
        background-color: rgba(139, 0, 0, 0.2) !important;
        color: #ffcccc !important;
        border: 1px solid #8b0000 !important;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #4a0a0a !important;
        color: white !important;
        border-radius: 5px;
        border: 1px solid #8b0000;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #8b0000 !important;
        box-shadow: 0px 0px 15px #8b0000;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 GARCH(1,1)-t Volatility & VaR Analyzer")
st.write("Targeting Seed Variance: 751st-850th values from the bottom.")

# 1. FILE UPLOADER
uploaded_file = st.file_uploader("Choose the returns.xlsx file", type="xlsx")

if uploaded_file:
    try:
        st.info("Reading and cleaning data... Please wait.")

        raw = pd.read_excel(uploaded_file, sheet_name="Prices", header=0)
        raw.rename(columns={raw.columns[0]: "Date"}, inplace=True)
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw.set_index("Date", inplace=True)

        df_prices = raw.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")
        df_prices = df_prices.dropna(axis=1, how="all")

        all_returns = pd.DataFrame(index=df_prices.index)
        all_stdevs = pd.DataFrame(index=df_prices.index)
        all_var_99 = pd.DataFrame(index=df_prices.index)
        model_params = []

        progress_bar = st.progress(0)
        assets = df_prices.columns

        for i, asset in enumerate(assets):
            series = df_prices[asset].dropna()

            # NEW REQUIREMENT: At least 850 returns 
            ret = 100 * np.log(series / series.shift(1)).dropna()
            T = len(ret)

            if T < 851:
                progress_bar.progress((i + 1) / len(assets))
                continue

            try:
                # --- STEP A: Parameter Estimation ---
                model = arch_model(ret, vol="GARCH", p=1, q=1, dist="t")
                res = model.fit(disp="off")
                om, al, be, nu = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"], res.params["nu"]

                # --- STEP B: Recursive GARCH with 750-day Offset Seed ---
                sigma2 = np.full(T, np.nan)
                seed_start = T - 850
                seed_end = T - 750
                
                seed_variance = np.var(ret.values[seed_start:seed_end])
                sigma2[seed_end - 1] = seed_variance
                r_sq = ret.values ** 2

                for t in range(seed_end, T):
                    sigma2[t] = om + al * r_sq[t - 1] + be * sigma2[t - 1]

                stdev = np.sqrt(sigma2)
                t_quantile = student_t.ppf(0.01, nu)

                all_returns[asset] = ret
                all_stdevs[asset] = pd.Series(stdev, index=ret.index)
                all_var_99[asset] = pd.Series(t_quantile * stdev, index=ret.index)

                model_params.append({
                    "Asset": asset, "Omega": om, "Alpha": al, "Beta": be,
                    "Persistence": al + be, "Nu (DF)": nu,
                    "Seed_Used_At_Index": seed_end
                })
            except Exception:
                pass

            progress_bar.progress((i + 1) / len(assets))

        # --- EXCEL EXPORT ---
        output = io.BytesIO()
        df_prices.index = pd.to_datetime(df_prices.index)
        all_returns.index = pd.to_datetime(all_returns.index)
        all_stdevs.index = pd.to_datetime(all_stdevs.index)
        all_var_99.index = pd.to_datetime(all_var_99.index)

        with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="dd-mmm-yy") as writer:
            workbook = writer.book
            date_format = workbook.add_format({"num_format": "dd-mmm-yy"})
            
            def save_fmt(df, name):
                df.to_excel(writer, sheet_name=name)
                ws = writer.sheets[name]
                ws.set_column("A:A", 18, date_format)

            save_fmt(df_prices, "Original_Prices")
            save_fmt(all_returns, "Returns_Scaled")
            save_fmt(all_stdevs, "GARCH_Stdev")
            save_fmt(all_var_99, "GARCH_VaR")
            pd.DataFrame(model_params).to_excel(writer, sheet_name="Model_Parameters", index=False)

        st.success("✅ Analysis Complete!")
        st.download_button(
            label="Download GARCH Analysis Report",
            data=output.getvalue(),
            file_name="Risk_Dept_750_Buffer_GARCH.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Error: {e}")

