import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('House Price Prediction Dataset.csv')


X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)







def predict_price(Area, Bedrooms, Bathrooms):
    return model.predict([[Area, Bedrooms, Bathrooms]])[0]


print("Predicted price:", predict_price(2000, 3, 2))



