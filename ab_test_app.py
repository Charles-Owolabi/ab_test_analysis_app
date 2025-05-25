import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import urllib.request
from scipy.stats import chi2_contingency, norm
import io # io is imported but not used in this specific version. Will keep for now.


# Set up the page config
st.set_page_config(
    page_title="A/B Test Analysis",
    layout="wide",
    page_icon="📊"
)

# Download a sample corporate image (you can replace this with your own image)
@st.cache_data
def load_image():
    # Using a placeholder image URL as the original might be subject to change/deletion
    # Consider hosting your own image or using a more permanent source
    try:
        url = "https://images.unsplash.com/photo-1551434678-e076c223a692?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80"
        # It's better to save to a temporary path or handle image bytes directly
        # For simplicity, using "corporate.jpg" but be mindful of write permissions in deployment
        image_filename = "corporate.jpg"
        urllib.request.urlretrieve(url, image_filename)
        return Image.open(image_filename)
    except Exception as e:
        st.warning(f"Could not load image from URL: {e}. Using a placeholder.")
        # Create a simple placeholder image if download fails
        img = Image.new('RGB', (600, 400), color = ('#A9A9A9'))
        # You could add text to this placeholder if desired
        return img

corporate_img = load_image()

# --- Sidebar Content (Moved outside of file upload check) ---
st.sidebar.header("🔍 More Info")
st.sidebar.markdown("---")
st.sidebar.markdown("### Project By")
st.sidebar.markdown("**Name:** Charles Owolabi")
st.sidebar.markdown("**Email:** [ctowolabi@gmail.com](mailto:ctowolabi@gmail.com)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Powered By")
# Using a more permanent image URL for "Powered By" is advisable if rb.gy is a shortener
# For now, assuming it's a placeholder. Let's use a direct known image for robustness if possible
# If using rb.gy/hcs089 specifically, ensure it's a direct image link.
# Using a generic placeholder for demonstration if that link is problematic.
try:
    powered_by_img = load_image_from_url("https://www.streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png") # Example
    st.sidebar.image(powered_by_img, width=150)
except:
    st.sidebar.image("https://rb.gy/hcs089", width=100) # Fallback to original if above fails for any reason

st.sidebar.markdown("---")


# Create two columns for the main content area
col1, col2 = st.columns([2, 1])

# First column with all your content
with col1:
    st.title("📊 A/B Test Analyzer")
    st.markdown("""
    This interactive app helps analyze an A/B test conducted on a marketing campaign comparing two groups: an **Ad group** and a **PSA group**.
    Use the insights to make data-driven decisions and estimate the potential impact on revenue.
    """)

    # Expander 1: Executive Summary
    with st.expander("📌 **Executive Summary**", expanded=False):
        st.markdown("""
        In a fast-paced world, companies continuously seek to maintain profitability and gain market share. Marketing departments, in particular, focus on running successful campaigns in a complex and competitive market.

        To achieve this, **A/B testing systems** are often employed. This randomized experimentation method involves presenting different versions of a variable (e.g., a web page, banner, or ad) to separate audience segments simultaneously. This approach helps determine which version yields the most significant impact on key business metrics.

        This analysis focuses on an **A/B test** where:
        - The majority of participants were exposed to ads (**the experimental group**).
        - A smaller portion saw a **Public Service Announcement (PSA)** or nothing (**the control group**).
        """)

    # Expander 2: About This Project
    with st.expander("📝 **About This Project**", expanded=False):
        st.markdown("""
        This report evaluates the effectiveness of a marketing campaign using **A/B testing data**.
        We compare the **Ad Group (experimental)** and the **PSA Group (control)** to answer:

        - Did the marketing ads improve conversion rates?
        - Is the observed difference statistically significant?
        - What is the estimated revenue uplift from the campaign?
        """)

    # Expander 3: Objectives
    with st.expander("🎯 **Primary Objectives**", expanded=False): # Changed emoji for variety
            st.markdown("""
        The primary objectives of this case study are to:

        - Perform hypothesis testing on the e-commerce dataset to compare the experimental and control groups.
        - Analyze whether the ads were successful in improving key metrics.
        - Estimate the potential revenue generated or uplift attributed to the ads.
        - Assess the statistical significance of the differences between the groups using A/B testing techniques.
            """)


    # Expander 4: About Dataset
    with st.expander("ℹ️ **About the Dataset**", expanded=False): # Changed emoji
        st.markdown("""
        The dataset includes the following columns:

        * **user_id**: Unique identifier for users.
        * **test_group**: A/B test assignment — "ad" (treatment) or "psa" (control).
        * **converted**: Whether the user converted (True/False after processing).
        * **total_ads**: Number of ads shown to the user.
        * **most_ads_day**: Day of the week with the highest ad exposure for the user.
        * **most_ads_hour**: Hour of the day with the highest ad exposure for the user.
        """)

    # Expander 5: Key Metrics
    with st.expander("🔑 **Key Metrics**", expanded=False):
        st.markdown("""
        - **Conversion Rate**: Percentage of users who converted in each group.
        - **Uplift %**: Percentage improvement in conversion rate (treatment vs. control).
        - **Statistical Significance (p-value)**: Probability that the observed difference is due to chance.
        - **Estimated Revenue Impact**: Projected revenue gain or loss from the campaign.
        """)

# Second column with the corporate image and file uploader
with col2:
    #st.image(corporate_img, caption="Marketing Analytics Environment", use_column_width=True)
    try:
        st.image(corporate_img, caption="Marketing Analytics Environment", use_container_width=True)
    except TypeError:
        st.image(corporate_img, caption="Marketing Analytics Environment", width=None)

    st.markdown("""
    <style>
    .stImage > img { /* Target the img tag directly within .stImage for better specificity */
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # Upload data
    uploaded_file = st.file_uploader("**Upload your CSV file for A/B Test Analysis:**", type="csv", help="Ensure your CSV has 'test_group' and 'converted' columns.")


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded and initially processed successfully!")

        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # --- Process 'test_group' column ---
        if 'test_group' in df.columns:
            df['test_group'] = df['test_group'].astype(str).str.strip().str.lower()
            # Ensure only 'ad' and 'psa' are present, or notify user
            valid_groups = ['ad', 'psa'] # These are the hardcoded expected group names
            
            # Check unique values in the 'test_group' column
            unique_test_groups = df['test_group'].unique()
            if not all(item in valid_groups for item in unique_test_groups) and \
               not all(item in unique_test_groups for item in valid_groups): # Check if there's a mismatch
                 st.warning(f"Found values in 'test_group': {unique_test_groups}. Expected 'ad' and 'psa' for some calculations. Analysis will proceed, but ensure these groups are present for full functionality or interpret uplift/significance sections carefully.")
        else:
            st.error("Critical column 'test_group' is missing. Please upload a valid CSV file. Use the expanders above to learn more about this tool, the A/B testing methodology, and the expected data format.")
            st.stop()

        # --- Process 'converted' column ---
        # if 'converted' in df.columns:
        #     true_like_values = ['true', '1', 'yes', 't', '□', 'converted'] # Added 'converted' as a potential true value
        #     df['converted'] = df['converted'].astype(str).str.strip().str.lower().isin(true_like_values)
        # else:
        #     st.error("Critical column 'converted' is missing. Please upload a valid CSV.")
        #     st.stop()




        # Data Processing (replace your current conversion code with this)
        if 'converted' in df.columns:
            # Convert to boolean (keep this for all calculations)
            true_values = ['true', '1', 'yes', 't', '□', 'converted']
            df['converted'] = df['converted'].astype(str).str.strip().str.lower().isin(true_values)
            
            # Create a string version ONLY for display (optional)
            df['converted_display'] = df['converted'].astype(str)
        else:
            st.error("Critical column 'converted' is missing. Please upload a valid CSV.")
            st.stop()



        st.header("🔬 Data Exploration & Pre-Analysis")

        # with st.expander("📄 **Data Preview**", expanded=False):
        #     st.dataframe(df.head())
        #     st.markdown(f"**Shape of the dataset:** {df.shape[0]} rows, {df.shape[1]} columns")


        with st.expander("📄 **Data Preview**", expanded=False):
            # Show the display version without affecting calculations
            preview_df = df.head().copy()
            preview_df['converted'] = preview_df['converted'].astype(str)  # Temporarily convert to string for display
            st.dataframe(preview_df)
            st.markdown(f"**Shape of the dataset:** {df.shape[0]} rows, {df.shape[1]} columns")




        with st.expander("⚠️ **Missing Values Check**", expanded=False):
            missing = df.isnull().sum()
            if missing.sum() == 0:
                st.success("🎉 No missing values found!")
            else:
                st.warning(f"Found {missing.sum()} missing values:")
                st.dataframe(missing[missing > 0].reset_index().rename(columns={'index': 'Column', 0: 'Missing Count'}))

        with st.expander("📊 **Group Distribution**", expanded=False):
            dist_col1, dist_col2 = st.columns(2)

            with dist_col1:
                st.subheader("Test Group Counts")
                group_counts = df['test_group'].value_counts()
                st.dataframe(group_counts.reset_index().rename(columns={'index': 'Group', 'count': 'Count'})) # Updated column name
                if not group_counts.empty:
                    fig_group, ax_group = plt.subplots(figsize=(6,4))
                    sns.barplot(x=group_counts.index, y=group_counts.values, ax=ax_group, palette=['#4A90E2', '#FF6B6B'] if len(group_counts) > 1 else ['#4A90E2'])
                    ax_group.set_title("Distribution of Test Groups")
                    ax_group.set_ylabel("Number of Users")
                    st.pyplot(fig_group)
                else:
                    st.info("No data to display for test group distribution.")


            with dist_col2:
                st.subheader("Conversion Status Counts")
                conversion_counts = df['converted'].value_counts() # 'converted' is now boolean
                st.dataframe(conversion_counts.reset_index().rename(columns={'index': 'Converted', 'count': 'Count'})) # Updated column name
                if not conversion_counts.empty:
                    fig_conv, ax_conv = plt.subplots(figsize=(6,4))
                    sns.barplot(x=conversion_counts.index, y=conversion_counts.values, ax=ax_conv, palette=['#FFC107', '#4CAF50'] if len(conversion_counts) > 1 else ['#FFC107'])
                    ax_conv.set_title("Overall Conversion Status")
                    ax_conv.set_ylabel("Number of Users")
                    ax_conv.set_xticklabels(['Not Converted (False)', 'Converted (True)'] if len(conversion_counts.index) == 2 else [str(conversion_counts.index[0])])
                    
                    # ax_conv.set_xticks(range(len(conversion_counts.index)))
                    # ax_conv.set_xticklabels(['Not Converted (False)', 'Converted (True)'] if len(conversion_counts.index) == 2 else [str(conversion_counts.index[0])])
                                       
                    
                    st.pyplot(fig_conv)
                else:
                    st.info("No data to display for conversion status distribution.")


        with st.expander("📈 **Statistical Summary of Numerical Features**", expanded=False):
            numerical_df = df.select_dtypes(include=np.number)
            if not numerical_df.empty:
                st.write(numerical_df.describe())
            else:
                st.info("No numerical columns found in the dataset.")
        
        st.header("📈 Conversion Rate Analysis")

        # Calculate conversion rates and CIs
        if 'test_group' in df.columns and 'converted' in df.columns:
            # In Conversion Rate Analysis (ensure this uses the BOOLEAN column)
            conversion_stats = df.groupby('test_group')['converted'].agg(
                converted_count='sum',  # This works because 'converted' is boolean
                total_count='count'
            ).reset_index()


            

            if not conversion_stats.empty and 'total_count' in conversion_stats.columns and conversion_stats['total_count'].sum() > 0:
                conversion_stats['rate'] = conversion_stats['converted_count'] / conversion_stats['total_count']

                # CI calculation
                z = norm.ppf(0.975) # 95% confidence interval
                # Ensure total_count is not zero to avoid division by zero error
                conversion_stats['se'] = np.sqrt(
                    conversion_stats['rate'] * (1 - conversion_stats['rate']) / np.maximum(conversion_stats['total_count'], 1) # Use np.maximum to avoid 0 in denominator
                )
                conversion_stats['ci_lower'] = conversion_stats['rate'] - z * conversion_stats['se']
                conversion_stats['ci_upper'] = conversion_stats['rate'] + z * conversion_stats['se']
                
                # Set 'test_group' as index again for easier lookup if needed
                conversion_stats_display = conversion_stats.set_index('test_group')

                with st.expander("🔢 **Overall Conversion Statistics by Group**", expanded=False): # Expanded by default
                    st.dataframe(conversion_stats_display.style.format({
                        'converted_count': '{:,.0f}',
                        'total_count': '{:,.0f}',
                        'rate': '{:.2%}',
                        'se': '{:.4f}',
                        'ci_lower': '{:.2%}',
                        'ci_upper': '{:.2%}'
                    }).background_gradient(subset=['rate', 'ci_lower', 'ci_upper'], cmap='viridis_r'))
            else:
                st.warning("Not enough data to calculate conversion statistics after grouping.")
                conversion_stats_display = pd.DataFrame() # Empty dataframe

        else:
            st.warning("'test_group' or 'converted' column not found for conversion rate analysis.")
            conversion_stats_display = pd.DataFrame() # Empty dataframe


        if 'most_ads_day' in df.columns and 'test_group' in df.columns and 'converted' in df.columns:
            with st.expander("📅 **Conversion Rates by Day of Week**", expanded=False):
                # Ensure 'most_ads_day' is in a consistent format if necessary (e.g., title case)
                df['most_ads_day'] = df['most_ads_day'].astype(str).str.strip().str.title()
                
                conversion_by_day = df.groupby(['test_group', 'most_ads_day'])['converted'].mean().unstack()
                days_ordered = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Reindex to ensure all days are present and in order, fill missing with NaN or 0
                conversion_by_day = conversion_by_day.reindex(columns=days_ordered).fillna(0)

                if not conversion_by_day.empty:
                    fig_day, ax_day = plt.subplots(figsize=(12, 6)) # Increased figure size
                    conversion_by_day.plot(kind='bar', ax=ax_day, width=0.8)

                    ax_day.set_title('Conversion Rates by Test Group and Day of Week', pad=20, fontsize=16)
                    ax_day.set_ylabel('Conversion Rate', fontsize=12)
                    ax_day.set_xlabel('Test Group', labelpad=10, fontsize=12)
                    ax_day.legend(title='Day of Week', bbox_to_anchor=(1.02, 1), loc='upper left')
                    ax_day.grid(axis='y', linestyle='--', alpha=0.7)
                    ax_day.set_axisbelow(True)
                    ax_day.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
                    plt.xticks(rotation=0, ha='center')
                    plt.tight_layout()
                    st.pyplot(fig_day)
                else:
                    st.info("Not enough data to plot conversion rates by day of the week.")
        elif 'test_group' in df.columns and 'converted' in df.columns: # only if most_ads_day is missing
            st.info("Column 'most_ads_day' not found. Skipping 'Conversion Rates by Day of Week' analysis.")


        # --- CORRECTED THIS SECTION ---
        st.header("🚀 Uplift & Significance") # This should not be used with 'with'
        # --- END OF CORRECTION ---

        with st.expander("🔍 **Uplift & Significance Details**", expanded=False): # Changed title slightly for clarity, expanded by default
            p_value_chi2 = None # Initialize for later checks
            chi2_stat = None
            dof_chi2 = None

            uplift_col, pval_col = st.columns(2)

            with uplift_col:
                st.subheader("Effectiveness Uplift") # Added subheader
                if 'ad' in conversion_stats_display.index and 'psa' in conversion_stats_display.index:
                    control_rate = conversion_stats_display.loc['psa', 'rate']
                    treatment_rate = conversion_stats_display.loc['ad', 'rate']
                    
                    if control_rate > 0: # Avoid division by zero
                        uplift = (treatment_rate - control_rate) / control_rate
                        st.metric("Relative Uplift (ad vs. psa)", value=f"{uplift:.2%}",
                                  help="Percentage change in conversion rate of Ad group compared to PSA group.")
                    else:
                        st.warning("Control group (PSA) has zero conversion rate, relative uplift cannot be meaningfully calculated as a percentage.")
                    
                    abs_diff = treatment_rate - control_rate
                    st.metric("Absolute Difference", value=f"{abs_diff:.2%}",
                              help="Absolute difference in conversion rates (Ad rate - PSA rate).")
                else:
                    st.warning("Could not calculate uplift. Required groups ('ad' and 'psa') not found in the processed conversion statistics.")

            with pval_col:
                st.subheader("Statistical Significance")
                if 'test_group' in df.columns and 'converted' in df.columns:
                    contingency = pd.crosstab(df['test_group'], df['converted'])
                    # Ensure contingency table is 2x2 for chi2_contingency to work as expected for A/B
                    if contingency.shape == (2,2): # Only proceed if we have two groups and two outcomes
                        try:
                            chi2_stat, p_value_chi2, dof_chi2, expected = chi2_contingency(contingency)
                            p_color = "inverse" if p_value_chi2 < 0.05 else "normal"
                            st.metric("P-value (Chi-Square Test)",
                                      f"{p_value_chi2:.4f}",
                                      delta="Result is Significant" if p_value_chi2 < 0.05 else "Result is Not Significant",
                                      delta_color=p_color,
                                      help="P-value from Chi-Square test of independence. Lower p-values (typically < 0.05) suggest a significant difference between groups.")
                        except ValueError as ve:
                            st.error(f"Could not perform Chi-Square test. This often happens if one group has no conversions or no non-conversions, or if a group is missing. Details: {ve}")
                            p_value_chi2 = None # Set p to None if test fails
                    else:
                        st.warning(f"Chi-Square test requires a 2x2 contingency table (2 groups, 2 outcomes). Current table shape: {contingency.shape}. Ensure 'ad' and 'psa' groups both exist and have data.")
                        p_value_chi2 = None
                        contingency = pd.DataFrame() # Ensure contingency is empty if not 2x2 for display later
                else:
                    st.warning("Required columns for Chi-Square test not available.")
                    contingency = pd.DataFrame()
                    p_value_chi2 = None


            st.subheader("🔍 **Detailed Chi-Square Test Results & Interpretation**") # Moved out of general expander to be more specific
            if 'test_group' in df.columns and 'converted' in df.columns and not contingency.empty:
                st.markdown("The **Chi-Square Test of Independence** is used to determine if there's a statistically significant association between the test group (Ad/PSA) and the conversion outcome.")

                st.markdown("#### Contingency Table:")
                # Ensure contingency table has both True and False columns, even if one is all zeros for a group
                if True not in contingency.columns and False in contingency.columns : contingency[True] = 0
                if False not in contingency.columns and True in contingency.columns : contingency[False] = 0
                contingency = contingency.reindex(columns=[False, True], fill_value=0) # Standardize column order

                st.dataframe(
                    contingency.style.format("{:,}")
                    .set_caption("Observed counts of conversions by test group")
                    .background_gradient(cmap='Blues', subset=pd.IndexSlice[:, [True, False]])
                )
            
                if p_value_chi2 is not None and chi2_stat is not None: # Check if p-value was calculated
                    st.markdown("#### Test Statistics:")
                    stat_col1, stat_col2 = st.columns(2)
                    stat_col1.metric("Chi2 Statistic (χ²)", f"{chi2_stat:.4f}")
                    stat_col2.metric("Degrees of Freedom", dof_chi2)

                    st.markdown("### Interpretation:")
                    if p_value_chi2 < 0.05:
                        st.success(f"""
                        ✅ **Statistically significant difference found (p-value = {p_value_chi2:.4f}).**
                        We can reject the null hypothesis that the test groups and conversion rates are independent.
                        The observed difference in conversion rates between the Ad and PSA groups is unlikely to be due to random chance.
                        """)
                    else:
                        st.warning(f"""
                        ⚠️ **No statistically significant difference found (p-value = {p_value_chi2:.4f}).**
                        We cannot reject the null hypothesis. The observed difference in conversion rates
                        could be due to random variation or chance.
                        """)
                elif not contingency.empty: # If contingency table was made, but test failed for other reasons
                    st.error("Chi-Square test could not be performed or results are incomplete. Interpretation is not available.")
                # else: # Contingency table itself was empty
                    # st.info("No contingency table to interpret.")


            st.markdown("#### Relevant Group Conversion Rates (from above):")
            if not conversion_stats_display.empty:
                st.dataframe(conversion_stats_display.style.format({
                    'converted_count': '{:,}',
                    'total_count': '{:,}',
                    'rate': '{:.2%}',
                    'se': '{:.4f}',
                    'ci_lower': '{:.2%}',
                    'ci_upper': '{:.2%}'
                }))
            else:
                st.info("Conversion statistics not available for display here.")


        st.header("💰 Advanced Business Metrics Projection")
        with st.expander("📈 **Campaign Financial Impact Analysis**", expanded=False):
            st.markdown("Estimate the potential financial impact of the campaign. Adjust the inputs below based on your business context.")

            col_roi1, col_roi2, col_roi3 = st.columns(3)
            default_additional_conversions = 0
            
            # Recalculate default_additional_conversions based on current conversion_stats_display
            if 'ad' in conversion_stats_display.index and 'psa' in conversion_stats_display.index:
                ad_stats_fin = conversion_stats_display.loc['ad']
                psa_stats_fin = conversion_stats_display.loc['psa']
                if psa_stats_fin['total_count'] > 0 and ad_stats_fin['total_count'] > 0 : # Ensure counts are positive
                    estimated_conversions_ad_group_at_psa_rate = psa_stats_fin['rate'] * ad_stats_fin['total_count']
                    default_additional_conversions = int(ad_stats_fin['converted_count'] - estimated_conversions_ad_group_at_psa_rate)
            else:
                 st.info("Default additional conversions cannot be pre-calculated as 'ad' or 'psa' group is missing from stats.")


            with col_roi1:
                ad_spend = st.number_input(
                    "Total Ad Spend ($)",
                    min_value=0.0, value=5000.0, step=100.0, key="ad_spend",
                    help="Enter the total cost of running the ad campaign."
                )
            with col_roi2:
                conversion_value = st.number_input(
                    "Average Value per Conversion ($)",
                    min_value=0.01, value=50.0, step=5.0, key="conversion_value",
                    help="What is the average revenue or profit generated by a single conversion?"
                )
            with col_roi3:
                n_additional_conversions = st.number_input(
                    "Net Additional Conversions from Ads",
                    min_value=0, value=max(0, default_additional_conversions), step=1, key="n_additional_conversions",
                    help=f"How many *extra* conversions are attributed to the ad campaign compared to the baseline (PSA)? Estimated based on data: {default_additional_conversions}"
                )

            if n_additional_conversions > 0:
                revenue_gain = conversion_value * n_additional_conversions
                net_profit = revenue_gain - ad_spend

                st.markdown("#### Projected Financial Outcomes:")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Gross Revenue Gain from Ads", f"${revenue_gain:,.2f}")

                if ad_spend > 0:
                    roi = net_profit / ad_spend
                    delta_color_roi = "normal" if roi >= 0 else "inverse"
                    metric_col2.metric("Return on Investment (ROI)",
                                    f"{roi:.1%}",
                                    delta=f"${net_profit:,.2f} Net Profit",
                                    delta_color=delta_color_roi)
                else: # No ad spend, so ROI is infinite if profit > 0, or N/A
                    metric_col2.metric("Return on Investment (ROI)", "N/A (No Ad Spend)", delta=f"${net_profit:,.2f} Net Profit")
                
                metric_col3.metric("Net Profit from Ads", f"${net_profit:,.2f}")
            else:
                st.info("Enter positive 'Net Additional Conversions from Ads' to calculate financial impact, or ensure 'default_additional_conversions' is positive based on data.")
        
        st.markdown("---")
        st.balloons() # A little celebration for reaching the end of analysis!
        st.success("Analysis Complete! Review the sections above for detailed insights.")


    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a valid CSV file.")
    except KeyError as e:
        st.error(f"A data processing error occurred due to a missing column: {e}. Please check your CSV file structure and ensure it matches the expected format, especially for 'test_group' and 'converted' columns after cleaning.")
        st.exception(e)
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        st.exception(e) # Shows the full traceback for easier debugging if needed
        st.warning("Please ensure your CSV file is correctly formatted and contains the expected columns (e.g., 'test_group', 'converted'). Refer to 'About the Dataset' for details.")

else:
    st.info("👋 Welcome! Please upload a CSV file to begin your A/B test analysis.")
    st.markdown("Use the expanders above to learn more about this tool, the A/B testing methodology, and the expected data format.")
