import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t as student_t
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Risk Dept GARCH Analyzer", layout="wide")

st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(180deg, #000000 0%, #0a0a0a 50%, #4a0505 100%);
}

/* Default text white */
body, .stApp {
    color: white;
}

/* Headings */
h1, h2, h3, h4 {
    color: white !important;
}

/* Dropdown text BLACK */
div[data-baseweb="select"] * {
    color: black !important;
}

/* Widget labels (including radio / select labels) */
[data-testid="stWidgetLabel"] > div > p {
    color: #ffffff !important;
}

/* File uploader area: make filename and helper text white */
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] div {
    color: #ffffff !important;
}

/* Browse button text BLACK */
.stFileUploader button {
    color: black !important;
}

/* Download button text BLACK */
.stDownloadButton > button {
    color: black !important;
}

/* Buttons styling */
.stButton>button,
.stDownloadButton>button {
    background-color: #8b0000 !important;
    border: 1px solid #ff4d4d;
    border-radius: 5px;
    color: #ffffff !important;
    font-weight: 600;
}

.stButton>button:hover,
.stDownloadButton>button:hover {
    background-color: #b30000 !important;
    box-shadow: 0px 0px 8px #ff4d4d;
}

/* Alerts */
.stAlert {
    background-color: rgba(139, 0, 0, 0.4) !important;
    color: white !important;
    border: 1px solid #ff4d4d !important;
}

/* Make radio / option labels white for visibility */
div[role="radiogroup"] * {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# st.markdown("""
#     <style>
#     /* Background Gradient */
#     .stApp {
#         background: linear-gradient(180deg, #000000 0%, #0a0a0a 50%, #4a0505 100%);
#     }

#     /* FORCE ALL TEXT TO WHITE */
#     html, body, [class*="css"], .stApp, .stMarkdown, 
#     p, span, div, label, small, strong, em, 
#     h1, h2, h3, h4, h5, h6,
#     [data-testid="stWidgetLabel"],
#     [data-testid="stFileUploadDropzone"] * {
#         color: #ffffff !important;
#     }

#     /* File Uploader Box */
#     .stFileUploader section {
#         background-color: rgba(255, 255, 255, 0.05) !important;
#         border: 1px solid #8b0000 !important;
#         border-radius: 10px;
#     }

#     /* Success / Info / Alert Boxes */
#     .stAlert {
#         background-color: rgba(139, 0, 0, 0.2) !important;
#         color: #ffffff !important;
#         border: 1px solid #8b0000 !important;
#     }

#     /* Buttons */
#     .stButton>button {
#         background-color: #4a0a0a !important;
#         color: white !important;
#         border-radius: 5px;
#         border: 1px solid #8b0000;
#         transition: 0.3s;
#     }

#     .stButton>button:hover {
#         background-color: #8b0000 !important;
#         box-shadow: 0px 0px 15px #8b0000;
#     }
#     </style>
# """, unsafe_allow_html=True)

st.title("📈 GARCH(1,1)-t Volatility & VaR Analyzer")
st.write("Targeting Seed Variance: 751st-850th values from the bottom.")

distribution_choice = st.radio(
    "Select GARCH Distribution",
    ["Student-t", "Normal"]
)

# 1. FILE UPLOADER
uploaded_file = st.file_uploader("Choose the returns.xlsx file", type="xlsx")

if uploaded_file:
    try:
        st.info("Reading and cleaning data... Please wait.")

        # raw = pd.read_excel(uploaded_file, sheet_name="Prices", header=0)
        # raw.rename(columns={raw.columns[0]: "Date"}, inplace=True)
        # raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        # raw.set_index("Date", inplace=True)

        # df_prices = raw.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")
        # df_prices = df_prices.dropna(axis=1, how="all")

        raw = pd.read_excel(uploaded_file, sheet_name="Prices", header=0)
        raw.rename(columns={raw.columns[0]: "Date"}, inplace=True)
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw.set_index("Date", inplace=True)

        # Replace only cells that are exactly 0 with "-"
        raw = raw.applymap(lambda x: "-" if x == 0 else x)

        df_prices = raw.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")
        df_prices = df_prices.dropna(axis=1, how="all")

        all_returns = pd.DataFrame(index=df_prices.index)
        all_stdevs = pd.DataFrame(index=df_prices.index)
        all_var_99 = pd.DataFrame(index=df_prices.index)
        model_params = []

        progress_bar = st.progress(0)
        assets = df_prices.columns

        all_rsquare = pd.DataFrame(index=df_prices.index)
        all_sigma2 = pd.DataFrame(index=df_prices.index)
        all_loglik = []

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
                dist_type = "t" if distribution_choice == "Student-t" else "normal"

                model = arch_model(ret, vol="GARCH", p=1, q=1, dist=dist_type)
                res = model.fit(disp="off")

                om = res.params["omega"]
                al = res.params["alpha[1]"]
                be = res.params["beta[1]"]

                if distribution_choice == "Student-t":
                    nu = res.params["nu"]
                    from scipy.stats import t
                    quantile = t.ppf(0.01, nu)
                else:
                    nu = None
                    from scipy.stats import norm
                    quantile = norm.ppf(0.01)

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

                all_rsquare[asset] = pd.Series(r_sq, index=ret.index)
                all_sigma2[asset] = pd.Series(sigma2, index=ret.index)
                all_loglik.append({
                    "Asset": asset,
                    "LogLikelihood": res.loglikelihood
                })


                # t_quantile = student_t.ppf(0.01, nu)
                if distribution_choice == "Student-t":
                    nu = res.params["nu"]
                    quantile = student_t.ppf(0.01, nu)
                else:
                    from scipy.stats import norm
                    quantile = norm.ppf(0.01)

                all_returns[asset] = ret
                all_stdevs[asset] = pd.Series(stdev, index=ret.index)
                # all_var_99[asset] = pd.Series(t_quantile * stdev, index=ret.index)
                all_var_99[asset] = pd.Series(quantile * stdev, index=ret.index)

                model_params.append({
                    "Asset": asset,
                    "Omega": om,
                    "Alpha": al,
                    "Beta": be,
                    "Persistence": al + be,
                    "Nu (DF)": nu if nu is not None else "Normal",
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
            save_fmt(all_rsquare, "R_Squared")
            save_fmt(all_sigma2, "GARCH_Variance")
            pd.DataFrame(all_loglik).to_excel(writer, sheet_name="GARCH_LogLikelihood", index=False)
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

