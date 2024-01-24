import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = "C:\\Users\\Hrishikesh\\Downloads\\archive (1)\\data.csv"  # Replace with your dataset path
data = pd.read_csv(file_path)

# Convert 'date' to datetime and extract year, month, and day
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data.drop(['date', 'country'], axis=1, inplace=True)  # Drop 'country' if it's a single value

# Separate features and target variable
X = data.drop(['price'], axis=1)
y = data['price']

# Categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Define the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', rf_model)])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model
rf_pipeline.fit(X_train, y_train)

# Get predictions
rf_preds = rf_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, rf_preds)
r2 = r2_score(y_test, rf_preds)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Function to filter houses based on price range
def filter_houses_by_price(data, min_price, max_price):
    filtered_houses = data[(data['price'] >= min_price) & (data['price'] <= max_price)]
    return filtered_houses.drop(['price'], axis=1)

# Example usage
min_price = 300000
max_price = 500000
suitable_houses = filter_houses_by_price(data, min_price, max_price)
print(suitable_houses.head())  # Display the first few suitable houses
