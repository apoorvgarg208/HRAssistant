# ACME Corp HR Analytics Platform
Enterprise-level HR analytics solution featuring comprehensive attrition diagnostics, predictive analytics, and AI-powered HR assistance. Optimized for large-scale datasets (50,000+ employees) with advanced RAG-based policy Q&A.
## Features
### HR Dashboard
    -**17 Analytics Features** - Comprehensive workforce analytics
    - **24+ Visualization Charts** - Interactive Plotly charts
    - **Executive Summary** - High-level metrics with financial impact
    - **Department Analysis** - Detailed departmental performance
    - **Role-Based Insights** - Job role attrition patterns
    - **Predictive Risk Scoring** - ML-powered employee risk assessment (Random Forest)
    - **Feature Importance** - Data-driven attrition drivers
    - **Demographic Analysis** - Age, gender, education insights
    - **Satisfaction Drivers** - Key retention factors
    - **Compensation Analysis** - Income vs. attrition correlation
    - **Training ROI** - Learning investment effectiveness
    - **Actionable Recommendations** - Data-driven retention strategies
    - **High-Risk Segments** - Proactive identification of at-risk employees
    - **Retention Cohorts** - Tenure-based survival analysis

### Ask HR Assistant
    - **8 AI Features** - Advanced conversational capabilities
    - **11 Policy Topics** - Comprehensive HR policy coverage
    - **Natural Language Q&A** - Ask questions in plain English
    - **RAG-Based Retrieval** - Semantic search with FAISS vectorstore
    - **MMR Diversity** - Retrieves diverse, relevant policy chunks
    - **Vectorstore Caching** - 10-100x faster initialization
    - **Answer Caching** - Instant responses for repeated questions
    - **Context-Aware Responses** - Accurate, sourced answers with relevance scoring
    - **Chat History** - Full conversation tracking
    - **Common Questions** - Quick access to frequent queries
    - **Ctrl+Enter Support** - Keyboard shortcuts for efficiency
    - **Policy Reference** - Expandable source context
## Project Structure
```.├── app.py                      # Main Streamlit application with navigation
    ├── attrition_analytics.py      # Scalable analytics module (50K+ rows)
    ├── hr_assistant.py             # AI-powered HR assistant with RAG pipeline
    ├── attrition_config.py         # Centralized configuration management
    ├── employee_attrition.csv      # Employee dataset
    ├── acme_hr_policy.txt          # HR policy document
    ├── requirements.txt            # Python dependencies
    ├── HR_Assistant_UserManual.docx# User Manual and documentation
    └── README.md                   # This file```

## Installation & Setup
### Prerequisites
    - Python 3.12 or higher
    - pip package manager

### Step 1: Clone or Download
    ```powershell
    # Navigate to the project directory cd <project_directory>

### Step 2: Create Virtual Environment (Recommended)
    ```powershell
    # Create virtual environmentpython -m venv venv
    # Activate virtual environment.\venv\Scripts\Activate.ps1

### Step 3: Install Dependencies
    ```powershell
    pip install -r requirements.txt```

### Step 4: Run the Application
    ```powershellstreamlit run app.py```

The application will open automatically in your default browser at `http://localhost:8501`

## Usage Guide
### HR Dashboard
1. **Navigate to Analytics**   - Click " HR Dashboard" in the sidebar navigation   - Or use "Launch Analytics Dashboard" button on home page
2. **Explore Insights**   - Review executive summary metrics   - Scroll through various analytical sections   - Examine charts and tables   - Identify high-risk segments
3. **Export Data**   - Use download buttons at the bottom   - Export department KPIs, high-risk employees, or recommendations
### Ask HR Assistant
1. **Access Assistant**   - Click " Ask HR Assistant" in the sidebar navigation   - Or use "Start Conversation with HR Bot" button on home page
2. **Ask Questions**   - Type your question in the text area (supports Ctrl+Enter)   - Or click on common questions for quick answers   - View source policy context with relevance scores in expandable sections
3. **Policy Reference**   - Expand "Quick Policy Reference" for summaries   - Review chat history for previous conversations
## Key Metrics Provided
### Executive Metrics- Total employee count- Attrition rate and count- Retention rate- Average tenure- Total attrition cost (₹)- Average replacement cost- High-risk segment identification
### Department KPIs- Employee count per department- Attrition rate by department- Average and median income- Average tenure- Job satisfaction scores- Performance ratings- Overtime rates- Training hours
### Role Analysis- Attrition rates by job role- Income distribution- Tenure patterns- Satisfaction levels
### Predictive Analytics- Employee risk scores (0-100)- Risk categorization (Low/Medium/High)- Feature importance rankings- Correlation analysis
### Segmentation Analysis- Tenure-based cohorts- Income quartiles- Education levels- Age groups- Distance categories- Training levels
## HR Policy Coverage
The AI assistant can answer questions about:- Leave policies (Annual, Sick, Maternity, Paternity)- Working hours and flexibility- Remote work policies- Travel and expense reimbursement- Performance management- Learning & development- Benefits and health insurance- Exit procedures- Dress code- Internal transfers- Grievance handling- Code of conduct- IT security policies- Employee wellness programs
## Best Practices
### For HR Leaders1. Review analytics dashboard weekly2. Act on high-risk employee alerts immediately3. Share departmental insights with managers4. Use recommendations for strategic planning5. Export reports for executive presentations
### For Employees1. Use HR assistant for instant policy answers2. Save common questions for quick reference3. Review chat history for previous answers4. Contact HR directly for personal matters
### For Managers1. Monitor your department's KPIs2. Focus on high-risk segments in your team3. Address satisfaction drivers proactively4. Track training ROI and effectiveness
## Technical Details
### Technology Stack- **Frontend**: Streamlit with custom CSS- **Data Processing**: Pandas (optimized dtypes), NumPy- **Visualization**: Plotly (24+ interactive charts)- **Machine Learning**: Scikit-learn (Random Forest with parallel processing)- **AI/NLP**: LangChain, FAISS vectorstore, HuggingFace Embeddings- **Statistics**: SciPy- **Caching**: Dictionary-based computation cache, pickle serialization
### Data Requirements- **employee_attrition.csv**: Employee data with 31+ features  - Required columns: EmployeeID, Age, Gender, Department, JobRole, Attrition, etc.- **acme_hr_policy.txt**: HR policy document (supports 50+ pages)
### Performance & Scalability- **Dataset Capacity**: Optimized for 50,000+ employee records- **Memory Efficiency**: 50-70% reduction via optimized dtypes (int8, int16, int32, float32)- **Speed**: Vectorized operations, observed=True groupby (20-50% faster)- **Parallel Processing**: n_jobs=-1 for Random Forest classifier- **Caching**: Executive summary, KPIs, predictions, and feature importance cached- **RAG Pipeline**: Handles 50+ pages, 100+ chunks efficiently- **Vectorstore Cache**: 10-100x faster initialization (pickle + MD5 hashing)- **Answer Cache**: Instant responses for repeated questions (LRU, max 100 entries)- **Retrieval Speed**: < 1 second with MMR diversity (k=4, fetch_k=20)- **Embedding**: Batch processing (32 chunks) with sentence-transformers
## Customization
### Adding New MetricsEdit `attrition_analytics.py` and add new methods to `AttritionAnalytics` class:```pythondef calculate_custom_kpi(self):    # Your analysis logic with caching    if 'custom_kpi' in self._cache:        return self._cache['custom_kpi']    result = # calculation logic    self._cache['custom_kpi'] = result    return result```
### Modifying ConfigurationEdit `attrition_config.py` to adjust:- Risk thresholds and scoring parameters- Chart colors and styling- Cost assumptions and multipliers- Feature engineering parameters
### Updating HR PoliciesUpdate `acme_hr_policy.txt` with your organization's policies. The vectorstore will automatically rebuild with MD5 hash validation.
### StylingCustomize CSS in `app.py`, `attrition_analytics.py`, or `hr_assistant.py` to match your brand colors and styling preferences.
## Sample Insights You'll Get
1. **Cost Impact**: "Employee attrition costs ACME Corp ₹X million annually"2. **Risk Factors**: "Employees with overtime have 45% higher attrition rate"3. **Critical Segments**: "0-2 year employees show 35% attrition rate"4. **ROI Analysis**: "Higher training investment reduces attrition by 20%"5. **Department Trends**: "IT department has highest attrition at 28%"
## Troubleshooting
### Issue: ModuleNotFoundError**Solution**: Ensure all dependencies are installed```powershellpip install -r requirements.txt```
### Issue: File not found error**Solution**: Ensure data files are in the same directory as app.py```employee_attrition.csvacme_hr_policy.txt```
### Issue: Streamlit won't start**Solution**: Check if port 8501 is available, or specify a different port```powershellstreamlit run app.py --server.port 8502```
### Issue: Slow performance**Solution**: Reduce dataset size for testing or increase system resources
### Issue: Virtual environment activation fails**Solution**: Set execution policy```powershellSet-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser```
## Security & Privacy
- All data processing happens locally- No external API calls for analytics- HR assistant uses local embeddings- Chat history stored in session (not persisted)- No data is transmitted outside your environment
## License
This is a demonstration project. Customize as needed for your organization.
## Support
For questions or issues:- Email: hr@acmecorp.com- Internal Portal: portal.acmecorp.com
## Learning Resources
### Understanding the Analytics- **Attrition Rate**: (Employees Left / Total Employees) × 100- **Retention Rate**: 100 - Attrition Rate- **Risk Score**: ML probability (0-100) of employee leaving- **Feature Importance**: Relative impact of factors on attrition
### ML Model Details- **Algorithm**: Random Forest Classifier (n_jobs=-1 for parallel processing)- **Features**: 12 key employee attributes (satisfaction, tenure, income, etc.)- **Performance**: Used for risk scoring (0-100 scale)- **Optimization**: Cached predictions and feature importance- **Interpretation**: Higher score = higher risk of leaving
### RAG Pipeline Details- **Chunking**: RecursiveCharacterTextSplitter (600 tokens, 100 overlap)- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (batch_size=32)- **Vectorstore**: FAISS with L2 distance- **Retrieval**: MMR (Maximal Marginal Relevance) for diversity- **Relevance**: L2 distance threshold of 2.0 (lower = better match)- **Caching**: Pickle-based vectorstore cache with MD5 validation
## Future Enhancements
Potential additions:- Real-time data integration- Advanced ML models (XGBoost, Neural Networks)- Sentiment analysis from employee surveys- Integration with HRMS systems- Email alerts for high-risk employees- Departmental drill-down reports- Time-series forecasting- Comparative benchmarking
## Acknowledgments
Built with:- Streamlit for the web framework- Plotly for interactive visualizations- LangChain for AI capabilities- HuggingFace for embeddings- scikit-learn for ML models
---
**Version**: 2.0.0  **Last Updated**: November 30, 2025  **Developed for**: ACME Corporation  **Optimizations**: Scalable for 50K+ employees, RAG pipeline with caching
Happy Analyzing! 