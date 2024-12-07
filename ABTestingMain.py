import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict
import io

#Set Page COnfig
st.set_page_config(page_title="A/B Testing WebApp", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Add custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.title("📊 A/B Test Analyzer")
    st.markdown("""
        Upload your A/B testing data and get instant statistical analysis and insights.
        The tool supports CSV files with conversion data for your control (A) and variant (B) groups.
    """)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        confidence_level = st.slider("Confidence Level", min_value=0.80, max_value=0.99, value=0.95, step=0.01, help="The confidence level for the statistical test")
        st.markdown("----")
        st.markdown("#### About")
        st.markdown("""
            This web application is a simple A/B Test Analyzer tool that allows you to upload your A/B testing data and get instant statistical analysis and insights.
            The tool supports CSV files with conversion data for your control (A) and variant (B) groups.
        """)
        st.markdown("----")

    # File upload section
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], help="Upload a CSV file with conversion data for your A/B test")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File successfully uploaded! Preview of your data:")
            st.dataframe(df.head())
            group_column, target_column = build_columnselection(df)
            print(group_column, target_column)  # Debugging
            if group_column and target_column:
                validation_passed, df = display_validation_results(df, group_column, target_column)
                if validation_passed:
                    tab1, tab2 = st.tabs(["Visualizations", "Statistical Analysis"])
                    
                    with tab1:
                        # Display visualizations
                        display_visualizations(df, group_column, target_column)
                    
                    with tab2:
                        # Statistical Analysis Configuration
                        st.subheader("Configure Analysis Parameters")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            alpha = st.slider(
                                "Significance Level (α)",
                                min_value=0.01,
                                max_value=0.10,
                                value=0.05,
                                step=0.01,
                                help="Probability of Type I error"
                            )
                        
                        with col2:
                            effect_size = st.slider(
                                "Minimum Detectable Effect",
                                min_value=0.01,
                                max_value=0.20,
                                value=0.05,
                                step=0.01,
                                help="Smallest meaningful difference to detect"
                            )
                        
                        with col3:
                            power = st.slider(
                                "Statistical Power",
                                min_value=0.70,
                                max_value=0.95,
                                value=0.80,
                                step=0.05,
                                help="Probability of detecting true effect"
                            )
                        
                        # Display statistical results
                        display_statistical_results(
                            df=df,
                            group_column=group_column,
                            target_column=target_column
                        )
                        
                        # Store results in session state for potential export
                        st.session_state['analysis_complete'] = True
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
       # Placeholder for future features
    if st.checkbox("Show planned features"):
        st.info("""
            Coming soon:
            - Multiple statistical tests
            - Power analysis
            - Sample size calculator
            - Downloadable reports
            - Advanced visualizations
        """)

def build_columnselection(df)-> Tuple[str, str]:
    st.subheader("Column Selection")
    col1, col2 = st.columns(2)
    with col1:
        group_column = st.selectbox("**Select the one Column that identifies the control/variant groups**", df.columns.tolist(), help="This column should contain two unique values, one identifying the control and the other identifying the variant group", placeholder="Select a column")
        if group_column:
            groups = df[group_column].unique()
            st.info(f"Groups found: {groups}")
            if len(groups) != 2:
                st.error("Please select a column with exactly two unique values")
    with col2:
        target_column = st.selectbox("Select the column with the target metric", df.columns.tolist(), help="This column should contain the target metric you want to compare between the control and variant groups")
        if target_column:
            st.info(f"Target metric selected: {target_column}")
    return group_column, target_column


"""
Data Valdaition Functions
"""
#Function to validate the data structure
def validate_data_structure(df: pd.DataFrame, group_column: str, target_column: str) -> Dict[str,bool]:
    """
    Validate the structure of the uploaded data
    Returns a dictionary of validation Results
    """
    validation_results = {
        "valid_groups": False,
        "valid_target": False,
        "no_nulls": False,
        "message:": None
    }

    # Check if the group column has exactly two unique values
    unique_groups = df[group_column].unique()
    if len(unique_groups) != 2:
        validation_results["message"] = "The group column must have exactly two unique values"
        print(validation_results["message"]) # Debugging
        return validation_results
    validation_results["valid_groups"] = True

    # Check if the target column is numeric
    if df[target_column].dtype not in [np.float64, np.int64]:
        validation_results["message"] = "The target column must be numeric"
        print(validation_results["message"])# Debugging
        return validation_results
    validation_results["valid_target"] = True

    # Check for null values
    if df[group_column].isnull().sum() > 0 or df[target_column].isnull().sum() > 0:
        validation_results["message"] = "The group and target columns must not contain null values"
        print(validation_results["message"])# Debugging
        return validation_results
    validation_results["no_nulls"] = True

    validation_results["message"] = "Data validation successful"
    print("debug_validate_data_structure->",validation_results["message"])# Debugging
    return validation_results

#Function to check the group balance
def check_group_balance(df: pd.DataFrame, group_column: str) -> Tuple[bool, Optional[str], pd.DataFrame]:
    """
    Check if the control and variant groups are balanced
    Returns a tuple with a boolean and an optional message
    """
    group_counts = df[group_column].value_counts()
    total_sample = len(df)
    print("Debug_check_group_balance->Group Counts:",group_counts) # Debugging

    #Calculate Proportions
    proportions = group_counts/total_sample

    # Create a summary dataframe
    summary_df = pd.DataFrame({
        "Group": group_counts.index,
        "Count": group_counts.values,
        "Proportion": proportions.values
    })

    #Calculate SPlit Ration
    split_ratio = group_counts[0]/group_counts[1]

    # Check if the split ratio is within 0.9 and 1.1
    if split_ratio < 0.9 or split_ratio > 1.1:
        return False, "The control and variant groups are not balanced"

    return True, None, summary_df

#Function to balance the groups
def balance_groups(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """
    Balance the control and variant groups by downsampling the larger group
    Returns a new DataFrame with balanced groups
    """
    group_counts = df[group_column].value_counts()
    min_group = group_counts.idxmin()
    min_group_count = group_counts.min()
    max_group = group_counts.idxmax()

    # Downsample the larger group
    max_group_data = df[df[group_column] == max_group].sample(min_group_count, random_state=42)
    min_group_data = df[df[group_column] == min_group]

    # Combine the downsampled data
    balanced_df = pd.concat([max_group_data, min_group_data])

    return balanced_df

#Function to display the validation results
def display_validation_results(df: pd.DataFrame, group_column: str, target_column: str) -> Tuple[bool,pd.DataFrame]:
    """
    Display the validation results
    Returns a tuple with a boolean indicating if the data is valid and a processed DataFrame (if balancing was required)
    """
    #Call the validation function and check if all values are true
    validation_results = validate_data_structure(df, group_column, target_column)
    print("debug_display_validation_results->",validation_results) # Debugging
    if (validation_results["message"]!="Data validation successful"):
        print("debug_display_validation_results->",validation_results["message"])
        st.error(validation_results["message"])
        return False, df
    
    #Check the group balance
    group_balance, balance_message, summary_df = check_group_balance(df, group_column)
    print(group_balance, balance_message) # Debugging
    #Display the validation results
    st.subheader("Data Validation Results - Group Balance")
    st.dataframe(summary_df)

    if not group_balance:
        st.warning(balance_message)
        st.info("Balancing the groups...")
        df = balance_groups(df, group_column)
        st.success("Groups balanced successfully!")
        return True, df
    else:
        st.success("Data validation successful!- groups are balanced")
        return True, df

"""
Data Visualization Functions

"""

#Violin chart for distribution comparision
def create_distribution_plot(df: pd.DataFrame, group_column: str, target_column: str) -> go.Figure:
    """
    Create distribution comparision between control and variant groups using box plots
    Function returns a plotly figure object
    """
    print("Debug_create_distribution_plot->\n",df.head(3)) # Debugging
    fig = go.Figure()
    for group in df[group_column].unique():
        data = df[df[group_column] == group][target_column]
        fig.add_trace(go.Violin(x=[group]*len(data),y=data, name=group, box_visible=True, meanline_visible=True ))

    fig.update_layout(title=f"Distribution of {target_column} by Group", xaxis_title=group_column, yaxis_title=target_column, showlegend=False, height=600)
    return fig


#Cumulative plot for the target metric
def create_cumulative_plot(df: pd.DataFrame, group_column: str, target_column: str, date_column: Optional[str] = None) -> go.Figure:
    """
    Create a cumulative sum plot of the target metric by group
    Function returns a plotly figure object
    """
    fig = go.Figure()
    if date_column and date_column in df.columns:
        try:
            # Try to convert to timedelta for MM:SS.MS format
            try:
                df[date_column] = pd.to_timedelta(df[date_column])
            except ValueError:
                def convert_special_time(time_str):
                    # Split into minutes and seconds
                    minutes, seconds = time_str.split(':')
                    # Convert to timedelta
                    return pd.Timedelta(minutes=float(minutes), seconds=float(seconds))
            
                # Convert the time strings to timedelta
                df['date_column'] = df[date_column].apply(convert_special_time)

            if df[date_column].isnull().any():
                print("Debug_create_cumulative_plot->Conversion to timedelta failed") # Debugging
                raise ValueError("Conversion to timedelta failed")
            df_sorted = df.sort_values([group_column,date_column])
            
            # Create plot with actual times
            for group in df_sorted[group_column].unique():
                group_data = df_sorted[df_sorted[group_column] == group]
                cumulative_mean = group_data[target_column].expanding().mean()
                
                fig.add_trace(go.Scatter(
                    x=group_data[date_column],
                    y=cumulative_mean,
                    name=group,
                    mode='lines',
                    hovertemplate=(
                        "Time: %{x}<br>" +
                        f"{target_column}: %{{y:.4f}}<br>" +
                        "<extra></extra>"
                    )
                ))
            
            fig.update_layout(
                xaxis_title="Time",
                xaxis_tickformat='%H:%M:%S.%f'
            )
        
        except ValueError:
            # If conversion to timedelta fails, try to convert to datetime
            try:
                df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                df[date_column] = pd.to_datetime(df[date_column])  # Fallback to default parsing
            df_sorted = df.sort_values(date_column)
            
            # Create plot with actual dates
            for group in df_sorted[group_column].unique():
                group_data = df_sorted[df_sorted[group_column] == group]
                cumulative_mean = group_data[target_column].expanding().mean()
                
                fig.add_trace(go.Scatter(
                    x=group_data[date_column],
                    y=cumulative_mean,
                    name=group,
                    mode='lines',
                    hovertemplate=(
                        "Date: %{x}<br>" +
                        f"{target_column}: %{{y:.4f}}<br>" +
                        "<extra></extra>"
                    )
                ))
            
            fig.update_layout(
                xaxis_title="Date",
                xaxis_tickformat='%Y-%m-%d'
            )
    
    else:
        # If no date column, use sample order with a warning
        st.warning("""
            No date column selected. The cumulative plot is using sample order, 
            which may not reflect actual temporal patterns. Consider adding a date 
            column for more accurate temporal analysis.
        """)
        
        df_sorted = df.sort_index()
        for group in df_sorted[group_column].unique():
            group_data = df_sorted[df_sorted[group_column] == group]
            cumulative_mean = group_data[target_column].expanding().mean()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(group_data) + 1)),
                y=cumulative_mean,
                name=group,
                mode='lines',
                hovertemplate=(
                    "Sample: %{x}<br>" +
                    f"{target_column}: %{{y:.4f}}<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            xaxis_title="Sample Size"
        )
    
    fig.update_layout(
        title=f"Cumulative Mean of {target_column} by Group",
        yaxis_title=target_column,
        height=600
    )
    
    return fig

#Function to create the summary statistics plot
def create_summary_stats_plot(df: pd.DataFrame, group_column: str, target_column: str) -> go.Figure:
    """
    Create summary statistics visualization
    Returns a Plotly figure with mean, CI, and sample sizes
    """
    summary_stats = df.groupby(group_column).agg({
        target_column: ['mean', 'std', 'count']
    })[target_column]
    
    # Calculate confidence intervals
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    summary_stats['ci_lower'] = summary_stats['mean'] - z_score * (summary_stats['std'] / np.sqrt(summary_stats['count']))
    summary_stats['ci_upper'] = summary_stats['mean'] + z_score * (summary_stats['std'] / np.sqrt(summary_stats['count']))
    
    fig = go.Figure()
    
    # Add bars for means
    fig.add_trace(go.Bar(
        x=summary_stats.index,
        y=summary_stats['mean'],
        name='Mean',
        error_y=dict(
            type='data',
            symmetric=False,
            array=summary_stats['ci_upper'] - summary_stats['mean'],
            arrayminus=summary_stats['mean'] - summary_stats['ci_lower']
        )
    ))
    
    # Add sample sizes as text
    for group in summary_stats.index:
        fig.add_annotation(
            x=group,
            y=summary_stats.loc[group, 'mean'],
            text=f"n={int(summary_stats.loc[group, 'count'])}",
            showarrow=False,
            yshift=10
        )
    
    fig.update_layout(
        title=f"Summary Statistics for {target_column}",
        xaxis_title=group_column,
        yaxis_title=f"Mean {target_column}",
        showlegend=False,
        height=500
    )
    
    return fig

#Function for visualizations
def display_visualizations(df: pd.DataFrame, group_column: str, target_column: str):
    """
    Display all visualizations with proper layout and explanations
    """
    st.subheader("Data Visualizations")

    # Add date column selection
    date_column = None
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if date_columns:
        date_column = st.selectbox(
            "Select date/time column for temporal analysis (optional)",
            options=[None] + date_columns,
            help="Select the column containing date/time information for temporal analysis"
        )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Distribution", "Cumulative Metrics", "Summary Statistics"])
    
    with tab1:
        st.plotly_chart(
            create_distribution_plot(df, group_column, target_column),
            use_container_width=True
        )
        st.markdown("""
            **Distribution Plot Interpretation:**
            - Compare the spread and central tendency of both groups
            - Check for outliers and unusual patterns
            - Assess if the distributions look similar in shape
        """)
    
    with tab2:
        st.plotly_chart(
            create_cumulative_plot(df, group_column, target_column, date_column),
            use_container_width=True
        )
        st.markdown("""
            **Cumulative Plot Interpretation:**
            - Shows how the metric stabilizes over time
            - Helps identify if the test ran long enough
            - Reveals any temporal patterns or trends
            - Important for detecting seasonality or time-based effects
            - Can show if the test was stopped at an appropriate time
        """)
    
    with tab3:
        st.plotly_chart(
            create_summary_stats_plot(df, group_column, target_column),
            use_container_width=True
        )
        st.markdown("""
            **Summary Statistics Interpretation:**
            - Compare mean values between groups
            - Error bars show 95% confidence intervals
            - Sample sizes (n) shown for each group
        """)

"""
Statistical Analysis Functions
"""
def check_sample_size_requirements(df: pd.DataFrame, 
                                 group_column: str, 
                                 target_column: str,
                                 effect_size: float = 0.05,
                                 power: float = 0.8,
                                 alpha: float = 0.05) -> dict:
    """
    Check if sample size is adequate for detecting the specified effect size
    """
    from scipy.stats import norm
    
    # Calculate required sample size
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # Get current sample stats
    group_sizes = df[group_column].value_counts()
    pooled_std = df[target_column].std()
    
    # Calculate required sample size per group
    required_n = 2 * ((z_alpha + z_beta) ** 2) * (pooled_std ** 2) / (effect_size ** 2)
    required_n = int(np.ceil(required_n))
    
    return {
        'required_samples_per_group': required_n,
        'current_samples': dict(group_sizes),
        'has_adequate_samples': all(size >= required_n for size in group_sizes),
        'effect_size': effect_size,
        'power': power,
        'alpha': alpha
    }

def check_normality(df: pd.DataFrame, group_column: str, target_column: str) -> dict:
    """
    Check normality assumption using Shapiro-Wilk test
    """
    from scipy.stats import shapiro
    
    normality_results = {}
    
    for group in df[group_column].unique():
        group_data = df[df[group_column] == group][target_column]
        statistic, p_value = shapiro(group_data)
        
        normality_results[group] = {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    return normality_results

def check_variance_homogeneity(df: pd.DataFrame, group_column: str, target_column: str) -> dict:
    """
    Check homogeneity of variances using Levene's test
    """
    from scipy.stats import levene
    
    groups = [group for name, group in df.groupby(group_column)[target_column]]
    statistic, p_value = levene(*groups)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'has_equal_variance': p_value > 0.05
    }

def perform_ab_test(df: pd.DataFrame, 
                   group_column: str, 
                   target_column: str,
                   alpha: float = 0.05) -> dict:
    """
    Perform appropriate statistical test based on data characteristics
    """
    # Check assumptions
    normality_results = check_normality(df, group_column, target_column)
    variance_results = check_variance_homogeneity(df, group_column, target_column)
    
    # Get groups
    groups = df[group_column].unique()
    group1_data = df[df[group_column] == groups[0]][target_column]
    group2_data = df[df[group_column] == groups[1]][target_column]
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((group1_data.var() + group2_data.var()) / 2)
    effect_size = abs(group1_data.mean() - group2_data.mean()) / pooled_std
    
    # Choose appropriate test based on assumptions
    all_normal = all(result['is_normal'] for result in normality_results.values())
    equal_variance = variance_results['has_equal_variance']
    
    if all_normal and equal_variance:
        # Use Student's t-test
        from scipy.stats import ttest_ind
        statistic, p_value = ttest_ind(group1_data, group2_data)
        test_name = "Student's t-test"
    elif all_normal and not equal_variance:
        # Use Welch's t-test
        from scipy.stats import ttest_ind
        statistic, p_value = ttest_ind(group1_data, group2_data, equal_var=False)
        test_name = "Welch's t-test"
    else:
        # Use Mann-Whitney U test
        from scipy.stats import mannwhitneyu
        statistic, p_value = mannwhitneyu(group1_data, group2_data)
        test_name = "Mann-Whitney U test"
    
    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': effect_size,
        'effect_size_interpretation': interpret_cohens_d(effect_size),
        'group_means': {
            groups[0]: group1_data.mean(),
            groups[1]: group2_data.mean()
        },
        'group_sizes': {
            groups[0]: len(group1_data),
            groups[1]: len(group2_data)
        },
        'assumptions': {
            'normality': normality_results,
            'variance_homogeneity': variance_results
        }
    }

def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size
    """
    d = abs(d)
    if d < 0.2:
        return "Negligible effect"
    elif d < 0.5:
        return "Small effect"
    elif d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"

def display_statistical_results(df: pd.DataFrame, group_column: str, target_column: str):
    """
    Display statistical analysis results in Streamlit
    """
    st.subheader("Statistical Analysis")
    
    # Check sample size requirements
    st.write("### Sample Size Analysis")
    size_results = check_sample_size_requirements(df, group_column, target_column)
    
    if size_results['has_adequate_samples']:
        st.success("✅ Sample size is adequate for the analysis")
    else:
        st.warning("⚠️ Sample size might be insufficient")
    
    st.write("Required samples per group:", size_results['required_samples_per_group'])
    st.write("Current sample sizes:", size_results['current_samples'])
    
    # Perform and display A/B test results
    st.write("### A/B Test Results")
    results = perform_ab_test(df, group_column, target_column)
    
    # Display test results
    st.write(f"Test performed: {results['test_name']}")
    st.write(f"P-value: {results['p_value']:.4f}")
    
    if results['significant']:
        st.success("✅ Statistically significant difference detected!")
    else:
        st.info("ℹ️ No statistically significant difference detected")
    
    st.write(f"Effect size: {results['effect_size']:.4f} ({results['effect_size_interpretation']})")
    
    # Display group means
    st.write("### Group Comparison")
    means_df = pd.DataFrame({
        'Group': results['group_means'].keys(),
        'Mean': results['group_means'].values(),
        'Sample Size': results['group_sizes'].values()
    })
    st.dataframe(means_df)
    
    # Display assumption checks
    with st.expander("View Assumption Checks"):
        st.write("#### Normality Test Results")
        for group, result in results['assumptions']['normality'].items():
            st.write(f"{group}: {'Normal' if result['is_normal'] else 'Non-normal'} "
                    f"(p-value: {result['p_value']:.4f})")
        
        st.write("#### Variance Homogeneity Test Results")
        variance_result = results['assumptions']['variance_homogeneity']
        st.write(f"Equal variances: {'Yes' if variance_result['has_equal_variance'] else 'No'} "
                f"(p-value: {variance_result['p_value']:.4f})")


if __name__ == "__main__":
    main()