# Ex.No-3b – Data Preprocessing

## Aim

To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset, and applying feature scaling.

---

## Dataset

The program uses `Data.csv`, containing the following features:

- **Country** – Categorical feature
- **Age** – Numerical feature
- **Salary** – Numerical feature
- **Purchased** – Target/dependent variable

The dataset contains **10 records and 4 columns**.

---

## Procedure

1. Import the required Python libraries.
2. Mount Google Drive and load the dataset using Pandas.
3. Display the dataset and inspect its information and shape.
4. Separate the independent variables (`X`) and dependent variable (`Y`).
5. Convert the independent variables into an array.
6. Handle missing values using `SimpleImputer` with the mean strategy.
7. Encode the categorical `Country` column using LabelEncoder.
8. Apply One-Hot Encoding to the `Country` column.
9. Encode the dependent variable `Purchased` using LabelEncoder.
10. Split the dataset into training and testing sets using `train_test_split`.
11. Apply `StandardScaler` for feature scaling.
12. Display the preprocessed training and testing datasets.

---

## Program

```python
# Ex.No-3b-DATA PREPROCESSING

# Step 1: Import libraries and load dataset
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/Datasets/Data.csv')

# Display dataset
df

# Step 2: Check dataset information
df.info()
print(df.shape)

# Step 3: Separate independent and dependent variables
x = df[['Country', 'Age', 'Salary']]

x = df[['Country', 'Age', 'Salary']].values

y = df['Purchased'].values

# Step 4: Handle missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean'
)

impute = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

# Step 5: Encode categorical Country column
from sklearn.preprocessing import LabelEncoder

lbx = LabelEncoder()
x[:, 0] = lbx.fit_transform(x[:, 0])

print(x)

# Step 6: One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

x = onehotencoder.fit_transform(
    df.Country.values.reshape(-1, 1)
).toarray()

print(x)

# Encode dependent variable
y = lbx.fit_transform(y)

print(y)

# Step 7: Split dataset into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)

print(x_train)
print(x_test)
print(y_train)

# Step 8: Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

print(x_train)
print(x_test)
```

---

## Output

The program produced the following output in Google Colab:

```text
Mounted at /content/drive

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 4 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   Country    10 non-null     object
 1   Age        9 non-null      float64
 2   Salary     9 non-null      float64
 3   Purchased  10 non-null     object

dtypes: float64(2), object(2)
memory usage: 452.0+ bytes

(10, 4)

array([[-1.        ,  2.64575131, -0.77459667],
       [-1.        ,  2.64575131, -0.77459667]])
```

### Output Details

- The dataset contains **10 entries**.
- The dataset contains **4 columns**.
- `Country` has **10 non-null values**.
- `Age` has **9 non-null values**, indicating one missing value.
- `Salary` has **9 non-null values**, indicating one missing value.
- `Purchased` has **10 non-null values**.
- The dataset shape is **(10, 4)**.
- Missing numerical values are handled using **mean imputation**.
- Categorical values are encoded using **LabelEncoder** and **OneHotEncoder**.
- The data is divided into training and testing sets.
- Feature scaling is performed using **StandardScaler**.
- The displayed decimal values are the scaled feature values.

---

## Conclusion

Thus, the given dataset was successfully preprocessed using Python and Scikit-learn by handling missing values using `SimpleImputer`, encoding categorical variables using `LabelEncoder` and `OneHotEncoder`, splitting the dataset into training and testing sets using `train_test_split`, and performing feature scaling using `StandardScaler`.
