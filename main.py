import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import boxcox

originalData=pd.read_csv('D600 Task 2 Dataset 1 Housing Information.csv')
df=pd.DataFrame(originalData)
print("PRINT INFO:")
print(df.info())
print("PRINT DESCRIPTIVE STATISTICS FOR EACH COLUMN:")
print(df.describe().transpose().to_string())


# ------------------------------------- ASPECTS C1 and C2: DATA PREPARATION ------------------------------------
# Selected columns
columns = ["Price","NumBedrooms", "IsLuxury"]
# drop missing values
df=df[columns].dropna()

# Identify Dependent and Independent Variables
X = df[['Price', 'NumBedrooms']]
y = df['IsLuxury']

# Describe the dependent variable and all independent variables
print("\nDESCRIPTIVE STATISTICS AFTER REMOVING MISSING VALUES:")
print(df[columns].describe().transpose().to_string())
dfColumns=df[columns]

# ------------------------------------- ASPECT C3: DATA VISUALIZATION ------------------------------------

# Univariate - distribution of independent variables
sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Distribution of Price')
plt.show()

sns.histplot(df['NumBedrooms'], bins=10, kde=True)
plt.title('Distribution of Number of Bedrooms')
plt.show()

# Bivariate - Boxplot for Price vs. IsLuxury
sns.boxplot(x=df['IsLuxury'], y=df['Price'])
plt.title('Price Distribution by Luxury Status')
plt.show()

# Bivariate - Boxplot for NumBedrooms vs. IsLuxury
sns.boxplot(x=df['IsLuxury'], y=df['NumBedrooms'])
plt.title('Number of Bedrooms Distribution by Luxury Status')
plt.show()

#--------------------------- ASSUMPTIONS ----------------------------------

# Check if 'IsLuxury' is binary
print("Is IsLuxury binary:",df['IsLuxury'].nunique() == 2)  # Should return True if binary


# Checking for multicollinearity
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# Scatter plots to check linearity assumption
df['Price_log'] = boxcox(df['Price'] + 1)[0]  # Adding 1 to avoid log(0)
df['NumBedrooms_log'] = boxcox(df['NumBedrooms'] + 1)[0]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(df['Price_log'], df['IsLuxury'])
ax[0].set_title('Log-Transformed Price vs IsLuxury')

ax[1].scatter(df['NumBedrooms_log'], df['IsLuxury'])
ax[1].set_title('Log-Transformed NumBedrooms vs IsLuxury')
plt.show()

#Independence of Observations
print("Independence of Observations",df.duplicated().sum())  # Should be 0 if all rows are independent



# ---------------------------------- ASPECT D1: SPLIT THE DATA AND MODEL BUILDING -----------------------------

# Splitting data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the training and test datasets as CSV files
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save datasets
train_data.to_csv("training.csv", index=False)
test_data.to_csv("test.csv", index=False)

# Adding a constant for the intercept in logistic regression
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# logistic regression using all variables
initialModel = sm.Logit(y_train, X_train_const).fit()
print(initialModel.summary())

# model parameters
aic = initialModel.aic
bic = initialModel.bic
pseudo_r2 = 1 - (initialModel.llf / initialModel.llnull)
coefficients = initialModel.params
p_values = initialModel.pvalues

print("AIC:", aic)
print("BIC:", bic)
print("Pseudo R²:", pseudo_r2)
print("Coefficient Estimates:\n", coefficients)
print("P-values:\n", p_values)


# -------------------------------- ASPECT D2: OPTIMIZATION - BACKWARD STEPWISE ELIMINATION ---------------------
X_train_optimized = X_train_const.copy() #copy of data with intercept

while True:
    optimizedModel=sm.Logit(y_train, X_train_optimized).fit()
    pValues=optimizedModel.pvalues
    pMax=pValues.max() #find the highest p-value
    if pMax > 0.05:  # Check if pMax exceeds threshold
        excludedVariable = optimizedModel.pvalues.idxmax()  # Find variable with max p-value
        print(f"REMOVING {excludedVariable} with p-value: {pMax:.4f}")
        X_train_optimized = X_train_optimized.drop(columns=[excludedVariable])  # Remove it
    else:
        break  # Stop when all p-values are below threshold
print("\n OPTIMIZED MODEL SUMMARY:")
print(optimizedModel.summary())


# Re-extract optimized model parameters
optimized_aic = optimizedModel.aic
optimized_bic = optimizedModel.bic
optimized_pseudo_r2=1-(optimizedModel.llf/optimizedModel.llnull)
optimized_coefficients = optimizedModel.params
optimized_p_values = optimizedModel.pvalues

print("Optimized AIC:", optimized_aic)
print("Optimized BIC:", optimized_bic)
print("Optimized Pseudo R2:", optimized_pseudo_r2)
print("Optimized Coefficient Estimates:\n", optimized_coefficients)
print("Optimized P-values:\n", optimized_p_values)


# -------------------------------- ASPECT D3: CONFUSION MATRIX AND ACCURACY OF THE OPTIMIZED MODEL---------------------

# Predictions on training data
y_train_pred = optimizedModel.predict(X_train_optimized)
y_train_pred_class = (y_train_pred >= 0.5).astype(int)

# Confusion matrix & accuracy
train_conf_matrix = confusion_matrix(y_train, y_train_pred_class)
train_accuracy = accuracy_score(y_train, y_train_pred_class)

print("Training Confusion Matrix:\n", train_conf_matrix)
print("Training Accuracy:", train_accuracy)


# -------------------------------- ASPECT D4: PREDICTION ON TEST DATASET---------------------

X_test_optimized = X_test_const[X_train_optimized.columns]

 #Predictions on test data
y_test_pred = optimizedModel.predict(X_test_optimized)
y_test_pred_class = (y_test_pred >= 0.5).astype(int)

# Confusion matrix & accuracy
test_conf_matrix = confusion_matrix(y_test, y_test_pred_class)
test_accuracy = accuracy_score(y_test, y_test_pred_class)

print("Test Confusion Matrix:\n", test_conf_matrix)
print("Test Accuracy:", test_accuracy)

# Extract the coefficients from the trained logistic regression model
coefficients = initialModel.params



# Dynamical regression equation
regression_equation = "Logit(P(IsLuxury)) = {:.4f}".format(coefficients.iloc[0])

for i in range(1, len(coefficients)):
    regression_equation += " + {:.4f} * {}".format(coefficients.iloc[i], coefficients.index[i])

# Print the equation
print("Logistic Regression Equation:")
print(regression_equation)
