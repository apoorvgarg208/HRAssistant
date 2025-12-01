"""
Enterprise-Level Attrition Analytics Module
Comprehensive KPIs, Insights, and Predictive Analytics
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from attrition_config import (
    DATA_CONFIG, FEATURE_CONFIG, COST_CONFIG, RISK_CONFIG,
    ML_CONFIG, CHART_CONFIG, DEPT_CONFIG, ROLE_CONFIG,
    SATISFACTION_CONFIG, UI_CONFIG, RECOMMENDATIONS
)


class AttritionAnalytics:
    """Comprehensive attrition analytics with enterprise-level KPIs"""
    
    def __init__(self, data_path, optimize_memory=True):
        # Optimize data loading for large datasets
        dtype_mapping = {
            'EmployeeID': 'int32',
            'Age': 'int8',
            'JobSatisfaction': 'int8',
            'EnvironmentSatisfaction': 'int8',
            'WorkLifeBalance': 'int8',
            'PerformanceRating': 'int8',
            'TrainingTimesLastYear': 'int8',
            'EducationLevel': 'int8',
            'YearsAtCompany': 'int16',
            'YearsInCurrentRole': 'int16',
            'MonthlyIncome': 'int32',
            'DistanceFromHome': 'int16'
        }
        
        if optimize_memory:
            self.df = pd.read_csv(data_path, dtype=dtype_mapping)
        else:
            self.df = pd.read_csv(data_path)
        
        # Cache for expensive computations
        self._cache = {}
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and enrich data with derived features using vectorized operations"""
        # Convert categorical to numeric for analysis (vectorized, memory efficient)
        self.df['Attrition_Numeric'] = (self.df[DATA_CONFIG['attrition_column']] == DATA_CONFIG['attrition_positive_value']).astype('int8')
        self.df['OverTime_Numeric'] = (self.df[DATA_CONFIG['overtime_column']] == DATA_CONFIG['overtime_positive_value']).astype('int8')
        
        # Create derived features with efficient dtypes
        self.df['Tenure_Category'] = pd.cut(self.df['YearsAtCompany'], 
                                            bins=FEATURE_CONFIG['tenure_bins'],
                                            labels=FEATURE_CONFIG['tenure_labels'],
                                            ordered=True)
        
        self.df['Age_Group'] = pd.cut(self.df['Age'], 
                                       bins=FEATURE_CONFIG['age_bins'],
                                       labels=FEATURE_CONFIG['age_labels'],
                                       ordered=True)
        
        self.df['Income_Quartile'] = pd.qcut(self.df['MonthlyIncome'], 
                                              q=FEATURE_CONFIG['income_quartiles'], 
                                              labels=FEATURE_CONFIG['income_labels'],
                                              duplicates='drop')
        
        self.df['Performance_Level'] = pd.cut(self.df['PerformanceRating'],
                                              bins=FEATURE_CONFIG['performance_bins'],
                                              labels=FEATURE_CONFIG['performance_labels'],
                                              ordered=True)
        
        # Vectorized computation for satisfaction score
        self.df['Satisfaction_Score'] = (self.df[['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']].mean(axis=1).astype('float32'))
        
        self.df['Distance_Category'] = pd.cut(self.df['DistanceFromHome'],
                                              bins=FEATURE_CONFIG['distance_bins'],
                                              labels=FEATURE_CONFIG['distance_labels'],
                                              ordered=True)
        
        self.df['Training_Level'] = pd.cut(self.df['TrainingTimesLastYear'],
                                           bins=FEATURE_CONFIG['training_bins'],
                                           labels=FEATURE_CONFIG['training_labels'],
                                           ordered=True)
        
        # Vectorized subtraction
        self.df['Role_Stagnation'] = (self.df['YearsAtCompany'] - self.df['YearsInCurrentRole']).astype('int16')
        
    def get_executive_summary(self):
        """Generate executive summary with key metrics (cached for performance)"""
        if 'executive_summary' in self._cache:
            return self._cache['executive_summary']
        
        total_employees = len(self.df)
        attrition_count = int(self.df['Attrition_Numeric'].sum())
        attrition_rate = (attrition_count / total_employees) * 100
        
        # Calculate costs
        avg_salary = self.df['MonthlyIncome'].mean()
        replacement_cost = avg_salary * COST_CONFIG['replacement_cost_multiplier']
        total_attrition_cost = replacement_cost * attrition_count
        
        # Retention metrics
        retention_rate = 100 - attrition_rate
        avg_tenure = self.df['YearsAtCompany'].mean()
        
        # High-risk segments
        high_risk_overtime = self.df[self.df[DATA_CONFIG['overtime_column']] == DATA_CONFIG['overtime_positive_value']]['Attrition_Numeric'].mean() * 100
        high_risk_low_satisfaction = self.df[self.df['JobSatisfaction'] <= RISK_CONFIG['low_satisfaction_threshold']]['Attrition_Numeric'].mean() * 100
        
        summary = {
            'total_employees': total_employees,
            'attrition_count': attrition_count,
            'attrition_rate': attrition_rate,
            'retention_rate': retention_rate,
            'avg_tenure': avg_tenure,
            'total_attrition_cost': total_attrition_cost,
            'avg_replacement_cost': replacement_cost,
            'high_risk_overtime': high_risk_overtime,
            'high_risk_low_satisfaction': high_risk_low_satisfaction
        }
        
        self._cache['executive_summary'] = summary
        return summary
    
    def calculate_department_kpis(self):
        """Calculate comprehensive department-level KPIs (optimized for large datasets)"""
        if 'department_kpis' in self._cache:
            return self._cache['department_kpis']
        
        dept_stats = self.df.groupby('Department', observed=True).agg(DEPT_CONFIG['aggregation_metrics']).round(2)
        
        dept_stats.columns = DEPT_CONFIG['column_names']
        
        dept_stats['Attrition_Rate'] = (dept_stats['Attrition_Rate'] * 100).round(2)
        dept_stats['Overtime_Rate'] = (dept_stats['Overtime_Rate'] * 100).round(2)
        
        result = dept_stats.reset_index()
        self._cache['department_kpis'] = result
        return result
    
    def calculate_role_kpis(self):
        """Calculate role-level KPIs (optimized for large datasets)"""
        if 'role_kpis' in self._cache:
            return self._cache['role_kpis']
        
        role_stats = self.df.groupby('JobRole', observed=True).agg(ROLE_CONFIG['aggregation_metrics']).round(2)
        
        role_stats.columns = ROLE_CONFIG['column_names']
        
        role_stats['Attrition_Rate'] = (role_stats['Attrition_Rate'] * 100).round(2)
        role_stats = role_stats.sort_values('Attrition_Rate', ascending=False)
        
        result = role_stats.reset_index()
        self._cache['role_kpis'] = result
        return result
    
    def analyze_tenure_impact(self):
        """Analyze impact of tenure on attrition (optimized)"""
        tenure_analysis = self.df.groupby('Tenure_Category', observed=True).agg({
            'EmployeeID': 'count',
            'Attrition_Numeric': ['sum', 'mean'],
            'MonthlyIncome': 'mean',
            'JobSatisfaction': 'mean'
        }).round(2)
        
        tenure_analysis.columns = ['Count', 'Attrition_Count', 'Attrition_Rate',
                                   'Avg_Income', 'Avg_Satisfaction']
        tenure_analysis['Attrition_Rate'] = (tenure_analysis['Attrition_Rate'] * 100).round(2)
        
        return tenure_analysis.reset_index()
    
    def analyze_compensation_impact(self):
        """Analyze compensation and attrition relationship (optimized)"""
        income_analysis = self.df.groupby('Income_Quartile', observed=True).agg({
            'EmployeeID': 'count',
            'Attrition_Numeric': ['sum', 'mean'],
            'MonthlyIncome': ['min', 'max', 'mean'],
            'JobSatisfaction': 'mean'
        }).round(2)
        
        income_analysis.columns = ['Count', 'Attrition_Count', 'Attrition_Rate',
                                   'Min_Income', 'Max_Income', 'Avg_Income', 'Avg_Satisfaction']
        income_analysis['Attrition_Rate'] = (income_analysis['Attrition_Rate'] * 100).round(2)
        
        return income_analysis.reset_index()
    
    def analyze_satisfaction_drivers(self):
        """Analyze key satisfaction drivers and their impact"""
        satisfaction_metrics = {metric: [] for metric in SATISFACTION_CONFIG['satisfaction_metrics']}
        
        for level in SATISFACTION_CONFIG['satisfaction_levels']:
            for metric_name in SATISFACTION_CONFIG['satisfaction_metrics']:
                col_name = SATISFACTION_CONFIG['satisfaction_columns'][metric_name]
                attrition_rate = self.df[self.df[col_name] == level]['Attrition_Numeric'].mean() * 100
                satisfaction_metrics[metric_name].append(attrition_rate)
        
        return satisfaction_metrics
    
    def calculate_overtime_impact(self):
        """Calculate overtime impact across dimensions (optimized)"""
        overtime_stats = self.df.groupby(['OverTime', 'Department'], observed=True).agg({
            'EmployeeID': 'count',
            'Attrition_Numeric': 'mean',
            'JobSatisfaction': 'mean',
            'WorkLifeBalance': 'mean'
        }).round(2)
        
        overtime_stats.columns = ['Count', 'Attrition_Rate', 'Avg_JobSat', 'Avg_WLB']
        overtime_stats['Attrition_Rate'] = (overtime_stats['Attrition_Rate'] * 100).round(2)
        
        return overtime_stats.reset_index()
    
    def analyze_education_impact(self):
        """Analyze education level impact on attrition (optimized)"""
        education_analysis = self.df.groupby('EducationLevel', observed=True).agg({
            'EmployeeID': 'count',
            'Attrition_Numeric': ['sum', 'mean'],
            'MonthlyIncome': 'mean',
            'JobSatisfaction': 'mean'
        }).round(2)
        
        education_analysis.columns = ['Count', 'Attrition_Count', 'Attrition_Rate',
                                      'Avg_Income', 'Avg_Satisfaction']
        education_analysis['Attrition_Rate'] = (education_analysis['Attrition_Rate'] * 100).round(2)
        
        return education_analysis.reset_index()
    
    def analyze_distance_impact(self):
        """Analyze commute distance impact (optimized)"""
        distance_analysis = self.df.groupby('Distance_Category', observed=True).agg({
            'EmployeeID': 'count',
            'Attrition_Numeric': 'mean',
            'JobSatisfaction': 'mean',
            'OverTime_Numeric': 'mean'
        }).round(2)
        
        distance_analysis.columns = ['Count', 'Attrition_Rate', 'Avg_Satisfaction', 'Overtime_Rate']
        distance_analysis['Attrition_Rate'] = (distance_analysis['Attrition_Rate'] * 100).round(2)
        distance_analysis['Overtime_Rate'] = (distance_analysis['Overtime_Rate'] * 100).round(2)
        
        return distance_analysis.reset_index()
    
    def analyze_training_roi(self):
        """Analyze training investment and attrition (optimized)"""
        training_analysis = self.df.groupby('Training_Level', observed=True).agg({
            'EmployeeID': 'count',
            'Attrition_Numeric': 'mean',
            'PerformanceRating': 'mean',
            'JobSatisfaction': 'mean'
        }).round(2)
        
        training_analysis.columns = ['Count', 'Attrition_Rate', 'Avg_Performance', 'Avg_Satisfaction']
        training_analysis['Attrition_Rate'] = (training_analysis['Attrition_Rate'] * 100).round(2)
        
        return training_analysis.reset_index()
    
    def identify_high_risk_segments(self):
        """Identify high-risk employee segments (vectorized for performance)"""
        segments = []
        
        # Segment 1: Low satisfaction + Overtime (vectorized boolean indexing)
        mask1 = (self.df['JobSatisfaction'] <= RISK_CONFIG['low_satisfaction_threshold']) & \
                (self.df[DATA_CONFIG['overtime_column']] == DATA_CONFIG['overtime_positive_value'])
        seg1 = self.df[mask1]
        segments.append({
            'Segment': 'Low Satisfaction + Overtime',
            'Count': len(seg1),
            'Attrition_Rate': (seg1['Attrition_Numeric'].mean() * 100).round(2) if len(seg1) > 0 else 0,
            'Avg_Tenure': seg1['YearsAtCompany'].mean().round(2) if len(seg1) > 0 else 0
        })
        
        # Segment 2: Long commute + Poor work-life balance
        mask2 = (self.df['DistanceFromHome'] > RISK_CONFIG['long_commute_threshold']) & \
                (self.df['WorkLifeBalance'] <= RISK_CONFIG['poor_work_life_balance_threshold'])
        seg2 = self.df[mask2]
        segments.append({
            'Segment': 'Long Commute + Poor WLB',
            'Count': len(seg2),
            'Attrition_Rate': (seg2['Attrition_Numeric'].mean() * 100).round(2) if len(seg2) > 0 else 0,
            'Avg_Tenure': seg2['YearsAtCompany'].mean().round(2) if len(seg2) > 0 else 0
        })
        
        # Segment 3: Low training + Low performance
        mask3 = (self.df['TrainingTimesLastYear'] <= RISK_CONFIG['low_training_threshold']) & \
                (self.df['PerformanceRating'] <= RISK_CONFIG['low_performance_threshold'])
        seg3 = self.df[mask3]
        segments.append({
            'Segment': 'Low Training + Low Performance',
            'Count': len(seg3),
            'Attrition_Rate': (seg3['Attrition_Numeric'].mean() * 100).round(2) if len(seg3) > 0 else 0,
            'Avg_Tenure': seg3['YearsAtCompany'].mean().round(2) if len(seg3) > 0 else 0
        })
        
        # Segment 4: Early career + Low income (cache quantile calculation)
        if 'income_q25' not in self._cache:
            self._cache['income_q25'] = self.df['MonthlyIncome'].quantile(RISK_CONFIG['low_income_quantile'])
        mask4 = (self.df['YearsAtCompany'] < RISK_CONFIG['early_career_years_threshold']) & \
                (self.df['MonthlyIncome'] < self._cache['income_q25'])
        seg4 = self.df[mask4]
        segments.append({
            'Segment': 'New Joiners + Low Income',
            'Count': len(seg4),
            'Attrition_Rate': (seg4['Attrition_Numeric'].mean() * 100).round(2) if len(seg4) > 0 else 0,
            'Avg_Tenure': seg4['YearsAtCompany'].mean().round(2) if len(seg4) > 0 else 0
        })
        
        # Segment 5: Role stagnation
        mask5 = self.df['Role_Stagnation'] > RISK_CONFIG['role_stagnation_years_threshold']
        seg5 = self.df[mask5]
        segments.append({
            'Segment': f"Role Stagnation ({RISK_CONFIG['role_stagnation_years_threshold']}+ years)",
            'Count': len(seg5),
            'Attrition_Rate': (seg5['Attrition_Numeric'].mean() * 100).round(2) if len(seg5) > 0 else 0,
            'Avg_Tenure': seg5['YearsAtCompany'].mean().round(2) if len(seg5) > 0 else 0
        })
        
        return pd.DataFrame(segments).sort_values('Attrition_Rate', ascending=False)
    
    def calculate_feature_importance(self):
        """Calculate feature importance using Random Forest (cached for performance)"""
        if 'feature_importance' in self._cache:
            return self._cache['feature_importance']
        
        # Prepare features
        feature_cols = ML_CONFIG['feature_columns']
        
        X = self.df[feature_cols].copy()
        y = self.df['Attrition_Numeric']
        
        # Encode categorical variables if any
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Train Random Forest with n_jobs for parallel processing
        rf = RandomForestClassifier(
            n_estimators=ML_CONFIG['random_forest_estimators'],
            random_state=ML_CONFIG['random_forest_random_state'],
            max_depth=ML_CONFIG['random_forest_max_depth'],
            n_jobs=-1  # Use all available cores
        )
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        importance_df['Importance_Pct'] = (importance_df['Importance'] * 100).round(2)
        
        self._cache['feature_importance'] = importance_df
        return importance_df
    
    def generate_demographic_insights(self):
        """Generate demographic-based insights (optimized)"""
        insights = {}
        
        # Age analysis
        insights['age'] = self.df.groupby('Age_Group', observed=True).agg({
            'EmployeeID': 'count',
            'Attrition_Numeric': 'mean',
            'MonthlyIncome': 'mean'
        }).round(2)
        insights['age']['Attrition_Rate'] = (insights['age']['Attrition_Numeric'] * 100).round(2)
        
        # Gender analysis
        insights['gender'] = self.df.groupby('Gender', observed=True).agg({
            'EmployeeID': 'count',
            'Attrition_Numeric': 'mean',
            'MonthlyIncome': 'mean',
            'YearsAtCompany': 'mean'
        }).round(2)
        insights['gender']['Attrition_Rate'] = (insights['gender']['Attrition_Numeric'] * 100).round(2)
        
        return insights
    
    def calculate_predictive_scores(self):
        """Calculate attrition risk scores for active employees (cached and parallelized)"""
        if 'predictive_scores' in self._cache:
            return self._cache['predictive_scores']
        
        # Prepare features for prediction
        feature_cols = ML_CONFIG['feature_columns']
        
        X = self.df[feature_cols].copy()
        y = self.df['Attrition_Numeric']
        
        # Train model with parallel processing
        rf = RandomForestClassifier(
            n_estimators=ML_CONFIG['random_forest_estimators'],
            random_state=ML_CONFIG['random_forest_random_state'],
            max_depth=ML_CONFIG['random_forest_max_depth'],
            n_jobs=-1  # Use all available cores
        )
        rf.fit(X, y)
        
        # Predict probabilities
        self.df['Attrition_Risk_Score'] = (rf.predict_proba(X)[:, 1] * 100).astype('float32')
        
        # Categorize risk
        self.df['Risk_Category'] = pd.cut(self.df['Attrition_Risk_Score'],
                                          bins=RISK_CONFIG['risk_score_bins'],
                                          labels=RISK_CONFIG['risk_score_labels'],
                                          ordered=True)
        
        # Get high-risk employees (use nlargest for efficiency on large datasets)
        high_risk_mask = self.df['Risk_Category'] == 'High Risk'
        high_risk = self.df[high_risk_mask][
            ['EmployeeID', 'Department', 'JobRole', 'Attrition_Risk_Score',
             'JobSatisfaction', 'YearsAtCompany', 'MonthlyIncome']
        ].sort_values('Attrition_Risk_Score', ascending=False)
        
        return high_risk
    
    def generate_correlation_matrix(self):
        """Generate correlation matrix for numeric features"""
        numeric_cols = ML_CONFIG['correlation_columns']
        
        corr_matrix = self.df[numeric_cols].corr().round(2)
        
        return corr_matrix
    
    def calculate_retention_cohorts(self):
        """Analyze retention by hiring cohorts (vectorized for scalability)"""
        # Use groupby instead of loop for better performance on large datasets
        cohort_analysis = self.df.groupby('YearsAtCompany', observed=True).agg(
            Employee_Count=('EmployeeID', 'count'),
            Attrition_Count=('Attrition_Numeric', 'sum'),
            Attrition_Rate=('Attrition_Numeric', lambda x: (x.mean() * 100).round(2)),
            Avg_Satisfaction=('Satisfaction_Score', lambda x: x.mean().round(2))
        ).reset_index()
        
        cohort_analysis.rename(columns={'YearsAtCompany': 'Years_At_Company'}, inplace=True)
        
        return cohort_analysis


def render_attrition_dashboard():
    """Render comprehensive attrition analytics dashboard"""
    # Custom CSS for compact layout
    st.markdown("""
        <style>
        .block-container {padding-top: 0.5rem !important; padding-bottom: 0.5rem !important;}
        h1 {font-size: 1.6rem !important; margin-bottom: 0.2rem !important; margin-top: 0 !important;}
        h2 {font-size: 1.1rem !important; margin-bottom: 0.3rem !important; margin-top: 0.3rem !important;}
        h3 {font-size: 0.95rem !important; margin-bottom: 0.2rem !important; margin-top: 0 !important;}
        p {font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        .stButton > button {font-size: 0.7rem !important; padding: 0.2rem 0.4rem !important; height: auto !important; margin-bottom: 0.2rem !important;}
        .stTextArea textarea {font-size: 0.85rem !important;}
        .stMarkdown {font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        hr {margin: 0.3rem 0 !important;}
        div[data-testid="stExpander"] {font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        .stAlert {padding: 0.4rem !important; font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        div[data-testid="stHorizontalBlock"] {gap: 0.3rem !important;}
        div[data-testid="column"] {padding: 0 0.2rem !important;}
        [data-testid="stMetricValue"] {font-size: 1.2rem !important;}
        [data-testid="stMetricLabel"] {font-size: 0.8rem !important;}
        [data-testid="stMetricDelta"] {font-size: 0.75rem !important;}
        .stDataFrame {font-size: 0.8rem !important; max-height: 250px !important;}
        div[data-testid="stHorizontalBlock"] > div {min-width: 0 !important;}
        .js-plotly-plot .plotly .gtitle {margin-top: -15px !important; padding-top: 0 !important;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("HR Dashboard")
    st.markdown("##### Comprehensive Data-Driven Diagnostic & Actionable Insights")
    
    # Initialize analytics
    analytics = AttritionAnalytics(DATA_CONFIG['data_file'])
    
    # Download Section at the top
    dept_kpis_preview = analytics.calculate_department_kpis()
    high_risk_employees_preview = analytics.calculate_predictive_scores()
    recommendations_preview = [rec.copy() for rec in RECOMMENDATIONS]
    summary_preview = analytics.get_executive_summary()
    recommendations_preview[0]['Issue'] = recommendations_preview[0]['Issue'].format(high_risk_overtime=summary_preview['high_risk_overtime'])
    recommendations_preview[1]['Issue'] = recommendations_preview[1]['Issue'].format(high_risk_low_satisfaction=summary_preview['high_risk_low_satisfaction'])
    recommendations_df_preview = pd.DataFrame(recommendations_preview)
    
    col1, col2, col3 = st.columns(UI_CONFIG['download_columns'])
    with col1:
        st.download_button("Download Dept KPIs", dept_kpis_preview.to_csv(index=False), "department_kpis.csv", "text/csv", use_container_width=True)
    with col2:
        st.download_button("Download High-Risk Employees", high_risk_employees_preview.to_csv(index=False), "high_risk_employees.csv", "text/csv", use_container_width=True)
    with col3:
        st.download_button("Download Recommendations", recommendations_df_preview.to_csv(index=False), "recommendations.csv", "text/csv", use_container_width=True)
    
    st.divider()
    
    # Executive Summary Section
    st.header("Executive Summary")
    summary = analytics.get_executive_summary()
    
    col1, col2, col3, col4, col5 = st.columns(UI_CONFIG['executive_summary_columns'])
    
    with col1:
        st.metric("Total Employees", f"{summary['total_employees']:,}")
        st.metric("Attrition Count", f"{summary['attrition_count']:,}")
    with col2:
        st.metric("Attrition Rate", f"{summary['attrition_rate']:.1f}%", delta_color="inverse")
        st.metric("Avg Tenure", f"{summary['avg_tenure']:.1f}y")
    with col3:
        st.metric("Attrition Cost", f"₹{summary['total_attrition_cost']/1000000:.1f}M")
        st.metric("Replacement Cost", f"₹{summary['avg_replacement_cost']/1000:.0f}K")
    with col4:
        st.metric("Overtime Risk", f"{summary['high_risk_overtime']:.1f}%")
    with col5:
        st.metric("Low Sat Risk", f"{summary['high_risk_low_satisfaction']:.1f}%")
    
    st.divider()
    
    # Department Analysis
    st.header("Department KPIs")
    dept_kpis = analytics.calculate_department_kpis()
    
    col1, col2, col3 = st.columns(UI_CONFIG['department_columns'])
    
    with col1:
        fig_dept_attrition = px.bar(dept_kpis, x='Department', y='Attrition_Rate',
                                     title='Attrition by Dept', color='Attrition_Rate',
                                     color_continuous_scale='Reds', height=CHART_CONFIG['main_chart_height'])
        fig_dept_attrition.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=30, 
                                                      l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                        title_font_size=CHART_CONFIG['title_font_size_large'], bargap=CHART_CONFIG['bargap_standard'])
        st.plotly_chart(fig_dept_attrition, use_container_width=True)
    with col2:
        fig_dept_income = px.bar(dept_kpis, x='Department', y='Avg_Income',
                                 title='Avg Income by Dept', color='Avg_Income',
                                 color_continuous_scale='Greens', height=CHART_CONFIG['main_chart_height'])
        fig_dept_income.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=30, 
                                                   l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                     title_font_size=CHART_CONFIG['title_font_size_large'], bargap=CHART_CONFIG['bargap_standard'])
        st.plotly_chart(fig_dept_income, use_container_width=True)
    with col3:
        st.dataframe(dept_kpis[['Department', 'Employee_Count', 'Attrition_Rate', 'Avg_Income', 'Avg_Tenure']], 
                     use_container_width=True, hide_index=True, height=UI_CONFIG['dataframe_height'])
    
    st.divider()
    
    # Role Analysis
    st.header("Role Analysis")
    role_kpis = analytics.calculate_role_kpis()
    
    col1, col2 = st.columns(UI_CONFIG['role_columns'])
    
    with col1:
        top_roles = role_kpis.head(CHART_CONFIG['top_roles_display'])
        fig_role = px.bar(top_roles, x='Attrition_Rate', y='JobRole',
                         title='Top Roles by Attrition', orientation='h',
                         color='Attrition_Rate', color_continuous_scale='RdYlGn_r', height=CHART_CONFIG['main_chart_height'])
        fig_role.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                           l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                              title_font_size=CHART_CONFIG['title_font_size_large'], bargap=CHART_CONFIG['bargap_small'])
        st.plotly_chart(fig_role, use_container_width=True)
    with col2:
        fig_role_count = px.scatter(role_kpis, x='Avg_Income', y='Attrition_Rate',
                                    size='Employee_Count', hover_data=['JobRole'],
                                    title='Income vs Attrition', color='Attrition_Rate',
                                    color_continuous_scale='RdYlGn_r', height=CHART_CONFIG['main_chart_height'])
        fig_role_count.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                                  l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                     title_font_size=CHART_CONFIG['title_font_size_large'])
        st.plotly_chart(fig_role_count, use_container_width=True)
    
    with st.expander("View Full Role Data", expanded=False):
        st.dataframe(role_kpis, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Tenure & Compensation Analysis (Combined)
    st.header("Tenure & Compensation Analysis")
    tenure_kpis = analytics.analyze_tenure_impact()
    income_kpis = analytics.analyze_compensation_impact()
    
    col1, col2, col3, col4 = st.columns(UI_CONFIG['tenure_compensation_columns'])
    
    with col1:
        fig_tenure = px.line(tenure_kpis, x='Tenure_Category', y='Attrition_Rate',
                            title='Attrition by Tenure', markers=True, height=CHART_CONFIG['medium_chart_height'])
        fig_tenure.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                            l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                title_font_size=CHART_CONFIG['title_font_size_small'], showlegend=False)
        st.plotly_chart(fig_tenure, use_container_width=True)
    with col2:
        fig_tenure_sat = px.bar(tenure_kpis, x='Tenure_Category', y='Avg_Satisfaction',
                               title='Satisfaction by Tenure', color='Avg_Satisfaction',
                               color_continuous_scale='Blues', height=CHART_CONFIG['medium_chart_height'])
        fig_tenure_sat.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                                 l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                    title_font_size=CHART_CONFIG['title_font_size_small'], showlegend=False, bargap=CHART_CONFIG['bargap_medium'])
        st.plotly_chart(fig_tenure_sat, use_container_width=True)
    with col3:
        fig_income = px.bar(income_kpis, x='Income_Quartile', y='Attrition_Rate',
                           title='Attrition by Income', color='Attrition_Rate',
                           color_continuous_scale='Reds', height=CHART_CONFIG['medium_chart_height'])
        fig_income.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                            l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                title_font_size=CHART_CONFIG['title_font_size_small'], showlegend=False, bargap=CHART_CONFIG['bargap_medium'])
        st.plotly_chart(fig_income, use_container_width=True)
    with col4:
        fig_income_range = px.bar(income_kpis, x='Income_Quartile', 
                                 y=['Avg_Income'], title='Avg Income by Quartile', height=CHART_CONFIG['medium_chart_height'])
        fig_income_range.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                                   l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                      title_font_size=CHART_CONFIG['title_font_size_small'], showlegend=False, bargap=CHART_CONFIG['bargap_medium'])
        st.plotly_chart(fig_income_range, use_container_width=True)
    
    st.divider()
    
    # Satisfaction & Overtime (Combined)
    st.header("Satisfaction & Overtime Impact")
    satisfaction_data = analytics.analyze_satisfaction_drivers()
    overtime_kpis = analytics.calculate_overtime_impact()
    
    col1, col2 = st.columns(UI_CONFIG['satisfaction_overtime_columns'])
    
    with col1:
        satisfaction_df = pd.DataFrame(satisfaction_data, 
                                       index=[f'L{i}' for i in SATISFACTION_CONFIG['satisfaction_levels']])
        fig_satisfaction = go.Figure()
        for col in satisfaction_df.columns:
            fig_satisfaction.add_trace(go.Scatter(x=satisfaction_df.index, 
                                                 y=satisfaction_df[col],
                                                 mode='lines+markers', name=col))
        fig_satisfaction.update_layout(title='Attrition by Satisfaction', height=CHART_CONFIG['main_chart_height'],
                                      margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                                  l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                      title_font_size=CHART_CONFIG['title_font_size_large'])
        st.plotly_chart(fig_satisfaction, use_container_width=True)
    with col2:
        fig_overtime = px.bar(overtime_kpis, x='Department', y='Attrition_Rate',
                             color=DATA_CONFIG['overtime_column'], barmode='group',
                             title='Overtime vs No Overtime by Dept', height=CHART_CONFIG['main_chart_height'])
        fig_overtime.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                              l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                  title_font_size=CHART_CONFIG['title_font_size_large'], bargap=CHART_CONFIG['bargap_standard'])
        st.plotly_chart(fig_overtime, use_container_width=True)
    
    st.divider()
    
    # Education, Distance & Training (Combined)
    st.header("Education, Distance & Training")
    education_kpis = analytics.analyze_education_impact()
    distance_kpis = analytics.analyze_distance_impact()
    training_kpis = analytics.analyze_training_roi()
    
    col1, col2, col3 = st.columns(UI_CONFIG['education_distance_training_columns'])
    
    with col1:
        fig_edu = px.line(education_kpis, x='EducationLevel', y='Attrition_Rate',
                         title='Attrition by Education', markers=True, height=CHART_CONFIG['small_chart_height'])
        fig_edu.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                         l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                             title_font_size=CHART_CONFIG['title_font_size_small'], showlegend=False)
        st.plotly_chart(fig_edu, use_container_width=True)
        
        fig_dist = px.bar(distance_kpis, x='Distance_Category', y='Attrition_Rate',
                         title='Attrition by Distance', color='Attrition_Rate',
                         color_continuous_scale='Oranges', height=CHART_CONFIG['small_chart_height'])
        fig_dist.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                          l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                              title_font_size=CHART_CONFIG['title_font_size_small'], showlegend=False, bargap=CHART_CONFIG['bargap_large'])
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        fig_edu_income = px.bar(education_kpis, x='EducationLevel', y='Avg_Income',
                               title='Income by Education', height=CHART_CONFIG['small_chart_height'])
        fig_edu_income.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                                 l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                    title_font_size=CHART_CONFIG['title_font_size_small'], showlegend=False, bargap=CHART_CONFIG['bargap_medium'])
        st.plotly_chart(fig_edu_income, use_container_width=True)
        
        fig_dist_ot = px.bar(distance_kpis, x='Distance_Category', 
                            y='Overtime_Rate', title='Overtime by Distance', height=CHART_CONFIG['small_chart_height'])
        fig_dist_ot.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                             l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                 title_font_size=CHART_CONFIG['title_font_size_small'], showlegend=False, bargap=CHART_CONFIG['bargap_large'])
        st.plotly_chart(fig_dist_ot, use_container_width=True)
    
    with col3:
        fig_training = px.bar(training_kpis, x='Training_Level', y='Attrition_Rate',
                             title='Attrition by Training', color='Attrition_Rate',
                             color_continuous_scale='RdYlGn_r', height=CHART_CONFIG['small_chart_height'])
        fig_training.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                              l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                  title_font_size=CHART_CONFIG['title_font_size_small'], showlegend=False, bargap=CHART_CONFIG['bargap_large'])
        st.plotly_chart(fig_training, use_container_width=True)
        
        fig_training_perf = px.bar(training_kpis, x='Training_Level', y='Avg_Performance',
                                  title='Performance by Training', height=CHART_CONFIG['small_chart_height'])
        fig_training_perf.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                                    l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                       title_font_size=CHART_CONFIG['title_font_size_small'], showlegend=False, bargap=CHART_CONFIG['bargap_large'])
        st.plotly_chart(fig_training_perf, use_container_width=True)
    
    st.divider()
    
    # High-Risk Segments & Feature Importance (Combined)
    st.header("High-Risk Segments & Key Drivers")
    high_risk_segments = analytics.identify_high_risk_segments()
    importance_df = analytics.calculate_feature_importance()
    
    col1, col2 = st.columns(UI_CONFIG['role_columns'])
    
    with col1:
        fig_segments = px.bar(high_risk_segments, x='Segment', y='Attrition_Rate',
                             title='High-Risk Segments', color='Attrition_Rate',
                             color_continuous_scale='Reds', text='Count', height=CHART_CONFIG['main_chart_height'])
        fig_segments.update_traces(texttemplate='%{text}', textposition='outside')
        fig_segments.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['segments_chart_margin_bottom'], 
                                              l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                  title_font_size=CHART_CONFIG['title_font_size_large'], bargap=CHART_CONFIG['bargap_standard'])
        st.plotly_chart(fig_segments, use_container_width=True)
    with col2:
        fig_importance = px.bar(importance_df.head(CHART_CONFIG['top_features_display']), x='Importance_Pct', y='Feature',
                               title='Top Attrition Drivers', orientation='h',
                               color='Importance_Pct', color_continuous_scale='Viridis', height=CHART_CONFIG['main_chart_height'])
        fig_importance.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                                l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                    title_font_size=CHART_CONFIG['title_font_size_large'], bargap=CHART_CONFIG['bargap_very_small'])
        st.plotly_chart(fig_importance, use_container_width=True)
    
    st.divider()
    
    # Demographics & Correlation (Combined)
    st.header("Demographics & Correlation")
    demo_insights = analytics.generate_demographic_insights()
    corr_matrix = analytics.generate_correlation_matrix()
    
    col1, col2, col3 = st.columns(UI_CONFIG['demographics_correlation_columns'])
    
    with col1:
        age_df = demo_insights['age'].reset_index()
        fig_age = px.bar(age_df, x='Age_Group', y='Attrition_Rate',
                        title='Attrition by Age', color='Attrition_Rate',
                        color_continuous_scale='Purples', height=CHART_CONFIG['main_chart_height'])
        fig_age.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                         l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                             title_font_size=CHART_CONFIG['title_font_size_large'], showlegend=False, bargap=CHART_CONFIG['bargap_medium'])
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        gender_df = demo_insights['gender'].reset_index()
        fig_gender = px.bar(gender_df, x='Gender', y='Attrition_Rate',
                           title='Attrition by Gender', color='Gender', height=CHART_CONFIG['main_chart_height'])
        fig_gender.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                            l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                  title_font_size=CHART_CONFIG['title_font_size_large'], showlegend=False, bargap=CHART_CONFIG['bargap_xlarge'])
        st.plotly_chart(fig_gender, use_container_width=True)
    with col3:
        fig_corr = px.imshow(corr_matrix, title='Correlation Matrix',
                            color_continuous_scale='RdBu_r', aspect='auto',
                            text_auto='.2f', height=CHART_CONFIG['main_chart_height'])
        fig_corr.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=10, 
                                          l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                              title_font_size=CHART_CONFIG['title_font_size_large'])
        fig_corr.update_xaxes(tickfont=dict(size=UI_CONFIG['correlation_tick_font_size']))
        fig_corr.update_yaxes(tickfont=dict(size=UI_CONFIG['correlation_tick_font_size']))
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.divider()
    
    # Retention Cohorts & Predictive Risk (Combined)
    st.header("Retention Cohorts & Risk Scoring")
    cohort_data = analytics.calculate_retention_cohorts()
    high_risk_employees = analytics.calculate_predictive_scores()
    risk_dist = analytics.df['Risk_Category'].value_counts().reset_index()
    risk_dist.columns = ['Risk_Category', 'Count']
    
    col1, col2, col3 = st.columns(UI_CONFIG['cohort_risk_columns'])
    
    with col1:
        fig_cohort = px.line(cohort_data, x='Years_At_Company', y='Attrition_Rate',
                            title='Attrition by Tenure Years', markers=True, height=CHART_CONFIG['cohort_chart_height'])
        fig_cohort.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                            l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                title_font_size=CHART_CONFIG['title_font_size_large'])
        st.plotly_chart(fig_cohort, use_container_width=True)
    with col2:
        fig_cohort_count = px.bar(cohort_data, x='Years_At_Company', y='Employee_Count',
                                 title='Employee Count by Tenure', color='Attrition_Rate',
                                 color_continuous_scale='RdYlGn_r', height=CHART_CONFIG['cohort_chart_height'])
        fig_cohort_count.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=CHART_CONFIG['chart_margin_bottom'], 
                                                   l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                                      title_font_size=CHART_CONFIG['title_font_size_large'], showlegend=False, bargap=CHART_CONFIG['bargap_small'])
        st.plotly_chart(fig_cohort_count, use_container_width=True)
    with col3:
        fig_risk = px.pie(risk_dist, values='Count', names='Risk_Category',
                         title='Risk Distribution', height=CHART_CONFIG['cohort_chart_height'],
                         color='Risk_Category',
                         color_discrete_map={'Low Risk': 'green', 
                                            'Medium Risk': 'orange', 'High Risk': 'red'})
        fig_risk.update_layout(margin=dict(t=CHART_CONFIG['chart_margin_top'], b=10, 
                                          l=CHART_CONFIG['chart_margin_left'], r=CHART_CONFIG['chart_margin_right']), 
                              title_font_size=CHART_CONFIG['title_font_size_large'])
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with st.expander(f"View Top {CHART_CONFIG['top_risk_employees_display']} High-Risk Employees", expanded=False):
        st.dataframe(high_risk_employees.head(CHART_CONFIG['top_risk_employees_display']), use_container_width=True, hide_index=True, height=UI_CONFIG['dataframe_height'])
    
    st.divider()
    
    # Actionable Recommendations
    st.header("Recommendations")
    
    # Format recommendations with dynamic values
    recommendations = [rec.copy() for rec in RECOMMENDATIONS]
    recommendations[0]['Issue'] = recommendations[0]['Issue'].format(high_risk_overtime=summary['high_risk_overtime'])
    recommendations[1]['Issue'] = recommendations[1]['Issue'].format(high_risk_low_satisfaction=summary['high_risk_low_satisfaction'])
    
    recommendations_df = pd.DataFrame(recommendations)
    st.dataframe(recommendations_df, use_container_width=True, hide_index=True, height=UI_CONFIG['recommendations_height'])


if __name__ == "__main__":
    render_attrition_dashboard()
