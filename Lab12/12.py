from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# Create categorical dataset
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male'],
    'Purchased': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
}
df2 = pd.DataFrame(data)

# Encode to numbers
le = LabelEncoder()
df2['Gender'] = le.fit_transform(df2['Gender'])
df2['Purchased'] = le.fit_transform(df2['Purchased'])

X = df2[['Gender']]
y = df2['Purchased']

chi_vals, p_vals = chi2(X, y)
print("Chi-Square Value:", chi_vals[0])
print("p-Value:", p_vals[0])
