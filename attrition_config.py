"""
Configuration file for Attrition Analytics Dashboard
Contains all configurable parameters and thresholds
"""

# Data Configuration
DATA_CONFIG = {
    'data_file': 'employee_attrition.csv',
    'attrition_column': 'Attrition',
    'attrition_positive_value': 'Yes',
    'overtime_column': 'OverTime',
    'overtime_positive_value': 'Yes'
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'tenure_bins': [-1, 2, 5, 10, 50],
    'tenure_labels': ['0-2 years', '3-5 years', '6-10 years', '10+ years'],
    
    'age_bins': [0, 25, 35, 45, 100],
    'age_labels': ['<25', '25-35', '36-45', '45+'],
    
    'income_quartiles': 4,
    'income_labels': ['Q1-Low', 'Q2-Medium-Low', 'Q3-Medium-High', 'Q4-High'],
    
    'performance_bins': [0, 2, 3, 4],
    'performance_labels': ['Below Average', 'Average', 'Above Average'],
    
    'distance_bins': [0, 10, 20, 100],
    'distance_labels': ['Near (<10km)', 'Medium (10-20km)', 'Far (>20km)'],
    
    'training_bins': [-1, 1, 3, 10],
    'training_labels': ['Low (0-1)', 'Medium (2-3)', 'High (4+)']
}

# Cost Calculation Configuration
COST_CONFIG = {
    'replacement_cost_multiplier': 6,  # Industry standard: 6 months salary
}

# Risk Segmentation Configuration
RISK_CONFIG = {
    'low_satisfaction_threshold': 2,
    'long_commute_threshold': 20,
    'poor_work_life_balance_threshold': 2,
    'low_training_threshold': 1,
    'low_performance_threshold': 2,
    'early_career_years_threshold': 2,
    'low_income_quantile': 0.25,
    'role_stagnation_years_threshold': 7,
    
    'risk_score_bins': [0, 30, 60, 100],
    'risk_score_labels': ['Low Risk', 'Medium Risk', 'High Risk']
}

# Machine Learning Configuration
ML_CONFIG = {
    'random_forest_estimators': 100,
    'random_forest_max_depth': 10,
    'random_forest_random_state': 42,
    
    'feature_columns': [
        'Age', 'MonthlyIncome', 'YearsAtCompany', 'YearsInCurrentRole',
        'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
        'PerformanceRating', 'TrainingTimesLastYear', 'DistanceFromHome',
        'OverTime_Numeric', 'EducationLevel'
    ],
    
    'correlation_columns': [
        'Age', 'MonthlyIncome', 'YearsAtCompany', 'YearsInCurrentRole',
        'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
        'PerformanceRating', 'TrainingTimesLastYear', 'DistanceFromHome',
        'OverTime_Numeric', 'Attrition_Numeric'
    ]
}

# Chart Configuration
CHART_CONFIG = {
    'main_chart_height': 280,
    'medium_chart_height': 250,
    'small_chart_height': 220,
    'cohort_chart_height': 260,
    
    'chart_margin_top': 40,
    'chart_margin_bottom': 20,
    'chart_margin_left': 10,
    'chart_margin_right': 10,
    'segments_chart_margin_bottom': 60,
    
    'title_font_size_large': 12,
    'title_font_size_small': 11,
    
    'bargap_standard': 0.2,
    'bargap_small': 0.15,
    'bargap_very_small': 0.12,
    'bargap_medium': 0.3,
    'bargap_large': 0.4,
    'bargap_xlarge': 0.5,
    
    'top_roles_display': 8,
    'top_features_display': 10,
    'top_risk_employees_display': 20
}

# Department Analysis Configuration
DEPT_CONFIG = {
    'aggregation_metrics': {
        'EmployeeID': 'count',
        'Attrition_Numeric': ['sum', 'mean'],
        'MonthlyIncome': ['mean', 'median'],
        'YearsAtCompany': 'mean',
        'JobSatisfaction': 'mean',
        'PerformanceRating': 'mean',
        'OverTime_Numeric': 'mean',
        'TrainingTimesLastYear': 'mean'
    },
    'column_names': [
        'Employee_Count', 'Attrition_Count', 'Attrition_Rate',
        'Avg_Income', 'Median_Income', 'Avg_Tenure',
        'Avg_JobSatisfaction', 'Avg_Performance', 
        'Overtime_Rate', 'Avg_Training'
    ]
}

# Role Analysis Configuration
ROLE_CONFIG = {
    'aggregation_metrics': {
        'EmployeeID': 'count',
        'Attrition_Numeric': ['sum', 'mean'],
        'MonthlyIncome': 'mean',
        'YearsAtCompany': 'mean',
        'JobSatisfaction': 'mean'
    },
    'column_names': [
        'Employee_Count', 'Attrition_Count', 'Attrition_Rate',
        'Avg_Income', 'Avg_Tenure', 'Avg_Satisfaction'
    ]
}

# Satisfaction Analysis Configuration
SATISFACTION_CONFIG = {
    'satisfaction_levels': range(1, 5),  # 1 to 4
    'satisfaction_metrics': [
        'Job Satisfaction',
        'Environment Satisfaction',
        'Work-Life Balance'
    ],
    'satisfaction_columns': {
        'Job Satisfaction': 'JobSatisfaction',
        'Environment Satisfaction': 'EnvironmentSatisfaction',
        'Work-Life Balance': 'WorkLifeBalance'
    }
}

# UI Configuration
UI_CONFIG = {
    'executive_summary_columns': 5,
    'department_columns': [1, 1, 1.5],
    'role_columns': 2,
    'tenure_compensation_columns': 4,
    'satisfaction_overtime_columns': 2,
    'education_distance_training_columns': 3,
    'demographics_correlation_columns': [1, 1, 1.5],
    'cohort_risk_columns': [1.2, 1.2, 1],
    'download_columns': 3,
    
    'dataframe_height': 250,
    'recommendations_height': 220,
    'correlation_tick_font_size': 7
}

# Recommendations Configuration
RECOMMENDATIONS = [
    {
        "Priority": "Critical",
        "Area": "Overtime Mgmt",
        "Issue": "OT: {high_risk_overtime:.1f}% attrition",
        "Action": "Monitor overtime, hire staff, optimize workload"
    },
    {
        "Priority": "Critical",
        "Area": "Job Satisfaction",
        "Issue": "Low sat: {high_risk_low_satisfaction:.1f}% attrition",
        "Action": "Pulse surveys, improvement programs, manager training"
    },
    {
        "Priority": "High",
        "Area": "New Hire Retention",
        "Issue": "0-2yr high attrition",
        "Action": "Enhance onboarding, mentors, 30-60-90 check-ins"
    },
    {
        "Priority": "High",
        "Area": "Training",
        "Issue": "Low training = high attrition",
        "Action": "Increase L&D budget, 2 trainings/yr min"
    },
    {
        "Priority": "Medium",
        "Area": "Commute",
        "Issue": "Long commute risk",
        "Action": "Remote work, transport allowance, flexible hours"
    },
    {
        "Priority": "Medium",
        "Area": "Compensation",
        "Issue": "Low income quartile risk",
        "Action": "Market benchmarking, salary adjustments"
    },
    {
        "Priority": "Medium",
        "Area": "Role Progression",
        "Issue": "Role stagnation",
        "Action": "Promotion reviews, lateral moves, skill dev"
    }
]
