import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Admission_Predict.csv')
df.columns = df.columns.str.strip()

#1 Первинний аналіз даних
# missing_values = df.isnull().sum()
#
# print(missing_values)
# missing_percent = (missing_values / len(df)) * 100
# print(missing_percent)
#
# duplicates_count = df.duplicated().sum()
# print(duplicates_count)
# print(df.dtypes)
#
# categorical_cols = df.select_dtypes(include=['object', 'category']).columns
#
# print('Категоріальні колонки:', list(categorical_cols))
# print('Кількість категоріальних колонок:', len(categorical_cols))


#2 Додаємо 2 фічі що визначають показники академічної та лінгвістичної підготовки (1 фіча) та вагу отриманих балів з урахуванням престижності навчальноо закладу (2 фіча)

df['Total_Score'] = df['GRE Score'] + df['TOEFL Score']
df['Academic_Power'] = df['CGPA'] * df['University Rating']

target_col = [col for col in df.columns if 'Chance of Admit' in col][0]

print(target_col)

target_col = 'Chance of Admit'

X = df.drop(columns=[target_col])
y = df[target_col]

# 3. Поділ на train / val / test

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# 4. Масштабування фіч
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#5 Навчання базової моделі з дефолтними гіперпараметрами

base_model = LinearRegression()
base_model.fit(X_train_scaled, y_train)

y_val_pred = base_model.predict(X_val_scaled)
print(f'Базова модель R2: {r2_score(y_val, y_val_pred):.4f}')

# 6. Підбір гіперпараметрів за допомогою Ridge-регресії та GridSearchCV

ridge = Ridge()
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid_search = GridSearchCV(
    estimator=ridge, param_grid=param_grid, cv=5, scoring='r2'
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f'Найкращий параметр alpha: {grid_search.best_params_}')

y_test_pred = best_model.predict(X_test_scaled)
print(f'Фінальний R2 score (test): {r2_score(y_test, y_test_pred):.4f}')
print(f'Фінальний MAE (test):      {mean_absolute_error(y_test, y_test_pred):.4f}')

#7 Оцінка результатів

print('Використання створених ознак (Total_Score, Academic_Power) та стандартизації даних дозволило досягти високої якості передбачень. На фінальному тестовому наборі даних модель показала коефіцієнт детермінації R^2 ~ 0.80, що свідчить про високу прогнозову здатність. Середня похибка (MAE) склала всього 0.048, підтверджуючи високу точність роботи моделі')