# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import pickle
import os

# %% 
# 1. Cấu hình & Load dữ liệu
# ---------------------------------------------------------
print("Loading data...")

# Xác định đường dẫn tới file CSV dựa trên vị trí của file script hiện tại
# Điều này giúp code chạy được bất kể bạn gọi lệnh python từ thư mục nào
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'AB_NYC_2019.csv')

# Kiểm tra xem file có tồn tại không trước khi đọc
if not os.path.exists(csv_path):
    # Nếu không thấy trong cùng thư mục với script, thử tìm ở thư mục cha (trường hợp data ở root)
    parent_csv_path = os.path.join(os.path.dirname(current_dir), 'AB_NYC_2019.csv')
    if os.path.exists(parent_csv_path):
        csv_path = parent_csv_path
    else:
        raise FileNotFoundError(f"Không tìm thấy file 'AB_NYC_2019.csv' tại {csv_path} hoặc {parent_csv_path}")

df = pd.read_csv(csv_path)

# %%
# 2. Xử lý dữ liệu & Feature Engineering (Theo logic Slide)
# ---------------------------------------------------------

# Điền giá trị thiếu (Data Cleaning)
# reviews_per_month thiếu -> coi như bằng 0
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Danh sách các cột sẽ sử dụng
categorical_cols = ['neighbourhood_group', 'room_type']
numerical_cols = [
    'minimum_nights', 
    'number_of_reviews', 
    'reviews_per_month', 
    'calculated_host_listings_count', 
    'availability_365'
]

# Loại bỏ các cột không cần thiết (Drop features)
# id, name, host_id, host_name, last_review, latitude, longitude, neighbourhood
# (Chúng ta chỉ giữ lại các cột trong list categorical & numerical ở trên)
df = df[categorical_cols + numerical_cols + ['price']]

# Xử lý biến mục tiêu (Target Transformation)
# Lọc bỏ giá = 0 để tránh lỗi log
df = df[df['price'] > 0]

# %%
# 3. Chia tập dữ liệu (Data Splitting)
# ---------------------------------------------------------
# Chia Train/Test theo tỷ lệ 80/20
# (Lưu ý: Trong thực tế, ta train trên Full Train (Train+Val) để tối ưu dữ liệu cho model cuối cùng)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Reset index để tránh lỗi
df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Tách biến mục tiêu và Log-transform (np.log1p)
y_full_train = np.log1p(df_full_train.price.values)
y_test = np.log1p(df_test.price.values)

# Xóa cột price khỏi tập feature
del df_full_train['price']
del df_test['price']

# %%
# 4. Feature Transformation & Vectorization
# ---------------------------------------------------------

def prepare_features(df_input):
    df_out = df_input.copy()
    
    # Áp dụng Log1p cho các biến số lệch phải (Numerical Transformation)
    # Logic theo Slide 13-16: Minimum nights, Reviews, etc.
    for col in numerical_cols:
        df_out[col] = np.log1p(df_out[col])
    
    # Chuyển về dạng dictionary để dùng DictVectorizer
    features_dict = df_out[categorical_cols + numerical_cols].to_dict(orient='records')
    return features_dict

# Chuẩn bị dữ liệu dạng từ điển
train_dicts = prepare_features(df_full_train)
test_dicts = prepare_features(df_test)

# Khởi tạo DictVectorizer (Tự động One-Hot Encoding cho biến Category)
dv = DictVectorizer(sparse=False)

# Fit & Transform
X_full_train = dv.fit_transform(train_dicts)
X_test = dv.transform(test_dicts)

# %%
# 5. Huấn luyện XGBoost (Model Training)
# ---------------------------------------------------------
features_names = list(dv.get_feature_names_out())

# Tạo DMatrix cho XGBoost
dtrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features_names)

# Tham số tối ưu (Theo Slide 6 phần XGBoost)
xgb_params = {
    'eta': 0.1, 
    'max_depth': 6,
    'min_child_weight': 10,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

print("Training XGBoost model...")
model = xgb.train(xgb_params, dtrain, num_boost_round=100)

# %%
# 6. Đánh giá Mô hình (Evaluation)
# ---------------------------------------------------------
y_pred_log = model.predict(dtest)

# Chuyển ngược từ Log scale về USD thực tế
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# Tính RMSE trên giá thực
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"✅ Final RMSE on Test Set: {rmse:.4f} USD")

# %%
# 7. Đóng gói & Lưu Model (Model Export)
# ---------------------------------------------------------
# Xác định vị trí lưu file (lưu vào folder backend)
# current_dir đã được xác định ở đầu file
output_path = os.path.join(current_dir, '..', 'backend', 'model.bin')
output_path = os.path.normpath(output_path)

# Tạo folder nếu chưa có
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Lưu DictVectorizer và Model
with open(output_path, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"✅ Model & DV saved successfully to: {output_path}")
# %%