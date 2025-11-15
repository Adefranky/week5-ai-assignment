1. Problem Definition (6 points)
Hypothetical AI Problem:
Predicting employee turnover in a tech company using machine learning.
Objectives:
1.	Identify employees at high risk of leaving within the next 6 months.
2.	Reduce overall turnover rate by 15% in one year.
3.	Provide actionable insights to HR for targeted retention initiatives.
Stakeholders:
1.	Human Resources Manager – responsible for employee retention and development strategies.
2.	Company Leadership (e.g., CEO) – interested in minimizing hiring costs and maintaining team stability.
Key Performance Indicator (KPI):
Turnover Prediction Accuracy – Percentage of correctly predicted attrition cases compared to actual turnover events.
2. Data Collection & Preprocessing (8 points)
Data Sources:
1.	Internal HR Records: Includes employee demographics, performance ratings, job satisfaction surveys, tenure, and compensation data.
2.	Exit Interviews & Survey Responses: Qualitative data from employees leaving the company, detailing reasons for resignation.
Potential Bias:
•	Survivorship Bias: The dataset may overrepresent employees who remain with the company and underrepresent those who left quickly without completing exit surveys, leading to skewed insights about what factors predict turnover.
Preprocessing Steps:
1.	Handling Missing Data: Impute missing values using techniques such as mean/mode imputation for numerical and categorical fields, or remove records with excessive missing data.
2.	Encoding Categorical Variables: Convert non-numeric fields (e.g., job title, department) into numerical format using one-hot encoding or label encoding.
3.	Feature Scaling: Normalize numerical attributes like salary and years of experience to ensure fair weight during model training (e.g., using Min-Max Scaling or Standardization).
3. Model Development (8 points)
Chosen Model:
Random Forest Classifier
Justification:
•	Handles Mixed Data Types: Can work effectively with both numerical and categorical data without extensive preprocessing.
•	Robust to Overfitting: By aggregating results from multiple decision trees, it reduces variance and improves generalization.
•	Feature Importance: Provides insights into which features (e.g., satisfaction score, tenure) have the greatest impact on turnover.
Data Splitting:
•	Training Set: 70% of the data – used to train the model.
•	Validation Set: 15% – used to tune hyperparameters and avoid overfitting.
•	Test Set: 15% – used to evaluate final model performance on unseen data.
Hyperparameters to Tune:
1.	Number of Trees (n_estimators): Controls the number of decision trees in the forest; more trees generally improve stability but increase computation time.
2.	Maximum Depth (max_depth): Limits the depth of each tree; helps prevent overfitting by restricting overly complex models.
4. Evaluation & Deployment (8 points)
Evaluation Metrics:
1.	Precision:
Measures the proportion of correctly predicted turnover cases out of all predicted positive cases. Relevant because HR might want to avoid falsely labeling an employee as a turnover risk, which could lead to unnecessary intervention.
2.	Recall (Sensitivity):
Measures the proportion of actual turnover cases that the model correctly identifies. Important to minimize missed turnover risks so HR can proactively intervene.
Concept Drift:
Concept drift refers to the change in the underlying patterns of data over time, causing model performance to degrade. For example, factors contributing to turnover may shift due to changes in company policy, remote work trends, or economic conditions.
Monitoring Concept Drift Post-Deployment:
•	Regularly evaluate the model on recent data and compare accuracy to baseline performance.
•	Use drift detection tools (e.g., ADWIN, DDM) to monitor statistical changes in input data distributions.
•	Retrain the model periodically if performance drops below acceptable thresholds.
Technical Challenge During Deployment:
•	Scalability:
If the company has thousands of employees with real-time turnover risk score updates, the system must handle high data volumes and frequent predictions. This requires robust cloud infrastructure and efficient inference pipelines to ensure low latency and consistent performance.
Part 2: Case Study — Predicting 30-day Patient Readmission
Problem Scope (5 points)
Problem: Predict whether a patient will be readmitted to the hospital within 30 days of discharge so care teams can intervene (follow-up calls, home visits, medication review).
Objectives
1.	Identify high-risk patients at discharge so care coordinators can target interventions.
2.	Reduce 30-day readmission rate by X% (target set by hospital leadership).
3.	Provide interpretable risk drivers per patient (why the model predicted high risk).
Stakeholders
•	Clinical Care Team / Discharge Coordinators — need actionable risk scores and explanations to plan follow-up.
•	Hospital Administration / Quality & Safety — care about readmission rates, penalties, and costs.
Data Strategy (10 points)
Proposed Data Sources
1.	Electronic Health Records (EHR): diagnoses, procedure codes, vitals, lab results, nursing notes, length of stay, discharge disposition.
2.	Patient Demographics & Social Determinants: age, sex, address (linked to socioeconomic indicators), insurance/claims data, living situation.
3.	(optional / useful) Pharmacy/medication reconciliation, prior admissions history, outpatient follow-up appointments, home health referrals.
Two Ethical Concerns
1.	Patient Privacy & Confidentiality: risk of exposing Protected Health Information (PHI) during collection, model training, or inference. Must minimize PHI exposure and secure datasets.
2.	Algorithmic Bias / Equity: model may perform worse for underrepresented groups (e.g., certain ethnicities, uninsured patients), causing unequal care. Must evaluate group-wise performance and mitigate disparities.
Preprocessing Pipeline (including feature engineering)
1.	Data Extraction & Temporal Windowing
o	Define index = discharge date. Aggregate features from a fixed lookback window (e.g., 12 months prior to discharge) so label leakage is avoided.
2.	Missing Data Handling
o	Identify variables with missingness patterns. For time-series labs/vitals use forward/backward carry where clinically appropriate; for static fields use domain-aware imputation (median for continuous, mode or “unknown” for categorical). Flag missingness as a feature when informative.
3.	De-duplication & Identity Resolution
o	Merge multiple records per patient, ensure consistent patient IDs; remove test/demo records.
4.	Feature Engineering
o	Count features: number of admissions in past 6/12 months, number of ED visits.
o	Clinical summaries: Charlson comorbidity index or similar comorbidity score computed from diagnosis codes.
o	Recent trends: slope/last value/variance of key labs (e.g., creatinine) and vitals over the last stay.
o	Medication features: number of discharge medications, high-risk meds indicator.
o	Social risk indicators: distance from hospital, prior missed appointments, insurance type.
o	Discharge context: length of stay, discharge disposition (home, skilled nursing), follow-up appointment scheduled.
5.	Encoding & Scaling
o	Categorical encoding (target/ordinal encoding for high-cardinality fields; one-hot for small cardinality).
o	Scale numeric features if using models sensitive to scale (e.g., neural nets); tree models generally don’t require scaling.
6.	Label Construction & Leakage Check
o	Label = 1 if any inpatient admission occurs within 30 days after discharge date; ensure outpatient encounters not counted. Remove features that would only be available after discharge or that directly encode the outcome.
7.	Train/Validation/Test Split
o	Prefer time-based split: train on earlier discharges, validate on a later period, test on the most recent period to mimic deployment and avoid temporal leakage.
8.	Class Imbalance
o	If readmission is rare, consider resampling (SMOTE variants), class weights, or threshold calibration during model training.
Model Development (10 points)
Model Choice & Justification
•	Gradient Boosted Trees (e.g., XGBoost / LightGBM / CatBoost)
Why: strong tabular performance, handles mixed feature types, robust to outliers, supports built-in handling of missing values, provides feature importance and pairs well with SHAP for per-patient explanations — critical for clinical acceptance.
Hypothetical Confusion Matrix (test set of 1000 discharges)
	Predicted Readmit (Positive)	Predicted No Readmit (Negative)	Row total
Actual Readmit	TP = 120	FN = 30	150
Actual No Readmit	FP = 80	TN = 770	850
Column total	200	800	1000
(Checks: 120 + 30 = 150 actual readmits; 80 + 770 = 850 actual non-readmits; total = 1000)
Calculate Precision and Recall (digit-by-digit arithmetic)
•	Precision = TP / (TP + FP)
o	TP + FP = 120 + 80 = 200.
o	Precision = 120 ÷ 200.
	Cancel trailing zeros: 12 ÷ 20 = reduce by 4 → 3 ÷ 5 = 0.6.
o	Precision = 0.60 (60%)
•	Recall = TP / (TP + FN)
o	TP + FN = 120 + 30 = 150.
o	Recall = 120 ÷ 150.
	Divide numerator and denominator by 30: (120 ÷ 30) / (150 ÷ 30) = 4 / 5 = 0.8.
o	Recall = 0.80 (80%)
 
 
Notes on model training
•	Use stratified/time aware cross-validation.
•	Optimize calibration (e.g., isotonic or Platt scaling) so predicted probabilities reflect true risk.
•	Use SHAP or LIME for local explanations and global feature importance; present explanations in clinician-facing UI.
Deployment (10 points)
Steps to integrate model into hospital system
1.	Design inference mode: Decide on batch scoring at discharge vs real-time scoring. Typical choice: score at time of discharge and store risk score in EHR.
2.	Wrap model as a service: Deploy model as a secure REST API or microservice (containerized, e.g., Docker/K8s) that accepts a patient payload and returns risk + explanation.
3.	EHR Integration: Use standard interoperability protocols (FHIR resources) or hospital integration engine to call the model at discharge event; write risk score and key explanatory fields back into EHR flowsheet/summary.
4.	Clinician UI & Alerts: Add a clear risk flag on discharge summary with top 3 contributing factors and recommended actions; avoid alert fatigue by thresholding and batching alerts.
5.	Logging & Audit Trails: Log inputs, outputs (risk and explanation), user access, and clinician actions for quality, debugging, and compliance.
6.	Monitoring & Feedback Loop: Monitor performance metrics (AUC, precision/recall), calibration, and usage. Capture outcomes and clinician feedback to retrain model periodically.
7.	Testing & Validation: Conduct retrospective validation, prospective silent deployment (score but do not surface to clinicians), then pilot limited rollout before full deployment.
8.	Governance: Establish clinical review board/steering committee to approve thresholds and workflows.
Ensuring compliance with healthcare regulations (e.g., HIPAA)
•	Data Minimization & De-identification: Use de-identified datasets for model development where possible; for PHI required during training, ensure strict access controls.
•	Business Associate Agreements (BAAs): Ensure any cloud vendor or third party that handles PHI signs a BAA.
•	Encryption: Encrypt PHI at rest and in transit (TLS for API calls; AES-256 for stored data).
•	Access Controls & RBAC: Enforce role-based access: only authorized personnel and services can query the model or view PHI.
•	Audit Logging & Monitoring: Keep immutable logs of data access, model inferences, and administrative actions; retain logs per policy.
•	Secure DevOps Practices: Use secure CI/CD, vulnerability scans, secret management, and regular penetration tests.
•	Clinical Oversight & Consent: Where applicable, obtain patient consent and have clinical governance for model use; document intended use and limitations in clinical policy.
•	Data Retention & Disposal Policies: Define retention windows and secure deletion for training datasets and intermediate artifacts.
•	Regulatory Documentation: Maintain documentation for model design, validation, risk assessment, and change logs for audits.
Optimization (5 points)
Method to address overfitting:
L2 Regularization (weight decay) — apply L2 penalty to model objective (for models that support it, e.g., logistic regression, neural nets, some gradient boosting frameworks allow regularization parameters). L2 pushes weights toward smaller values, smoothing the model and reducing variance. Combine with early stopping on validation set and proper cross-validation for best effect.
Part 3: Critical Thinking (20 points)
Ethics & Bias (10 points)
Impact of Biased Training Data on Patient Outcomes:
Biased training data — for example, underrepresentation of certain racial groups, rural patients, or women — can lead to a model that systematically underpredicts risk for these populations. This can result in:
•	Unequal care: High-risk but underrepresented patients may not receive necessary interventions.
•	Worsened health outcomes: Increased readmissions or emergency visits for neglected groups.
•	Perpetuation of disparities: Bias amplifies existing gaps in healthcare access and quality.
Strategy to Mitigate Bias:
•	Stratified Performance Evaluation and Fairness Metrics:
Regularly evaluate model performance across sensitive subgroups (e.g., race, gender, insurance type). Use fairness metrics like Equal Opportunity Difference or Demographic Parity. If disparities are detected, apply debiasing methods such as:
o	Reweighing data samples.
o	Oversampling minority groups.
o	Training separate models for different subpopulations.
o	Enforcing fairness constraints during optimization.
Trade-offs (10 points)
Interpretability vs Accuracy in Healthcare:
•	High-performing models like deep neural networks or ensemble models (e.g., XGBoost) may offer better predictive accuracy but lack transparency.
•	Healthcare requires trust and explainability, as clinicians must justify interventions and understand risk drivers.
•	Trade-off: A simpler model (like logistic regression) is highly interpretable but may sacrifice accuracy. More complex models improve predictions but risk clinician distrust and regulatory hurdles.
Impact of Limited Computational Resources on Model Choice:
•	With limited compute, the hospital may need to choose a lightweight model:
o	Prefer simpler models (Logistic Regression, Decision Trees).
o	Avoid complex models (deep learning) that require GPUs or high-memory environments.
•	Resource constraints may also push for batch scoring (vs. real-time) and limit retraining frequency, impacting how often the model can adapt to new data patterns.
Part 4: Reflection & Workflow Diagram (10 points)
Reflection (5 points)
Most Challenging Part of the Workflow:
Designing the data preprocessing and feature engineering pipeline was the most challenging because healthcare data is messy, fragmented across multiple systems, and requires domain expertise to engineer meaningful, clinically-relevant features without introducing label leakage.
Improvements with More Time/Resources:
•	Collaborate with clinicians early on to get deeper insights into patterns and features that matter.
•	Invest in automated pipeline tools (e.g., Apache Airflow, Data Version Control) to streamline data cleaning and model retraining.
•	Conduct a prospective pilot study to gather real-world feedback and continuously improve model relevance and usability.
Diagram (5 points)
Here is a flowchart of the AI Development Workflow:
      
  START
           |
           v
  Define Problem & Objectives
           |
           v
    Data Collection & Access
           |
           v
 Data Cleaning & Preprocessing
           |
           v
    Feature Engineering
           |
           v
    Model Selection & Training
           |
           v
  Validation & Performance Tuning
           |
           v
Interpretability & Fairness Checks
           |
           v
       Deployment Setup
           |
           v
Monitoring & Maintenance (Feedback Loop)
           |
           v
         END

