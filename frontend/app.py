import streamlit as st
import requests
import os
import json
import time

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lấy địa chỉ Backend từ biến môi trường (được set trong Dockerfile hoặc Render)
# Mặc định là localhost:9696 nếu chạy local
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:9696")
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def check_backend_status():
    """Kiểm tra xem Backend có đang online không"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        if response.status_code == 200:
            return True, response.json()
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        pass
    return False, None

# ==========================================
# 3. SIDEBAR - INPUT SECTION
# ==========================================
with st.sidebar:
    st.title("🔧 Cấu hình Căn hộ")
    st.markdown("Nhập thông tin chi tiết để dự đoán giá.")
    
    st.divider()

    # Nhóm 1: Vị trí & Loại phòng
    neighbourhood_group = st.selectbox(
        "Khu vực (Neighbourhood Group)",
        options=["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
        index=0,
        help="Quận mà căn hộ tọa lạc tại New York."
    )

    room_type = st.selectbox(
        "Loại phòng (Room Type)",
        options=["Entire home/apt", "Private room", "Shared room"],
        index=0,
        help="Loại hình lưu trú."
    )

    st.divider()

    # Nhóm 2: Thông số chi tiết
    minimum_nights = st.number_input(
        "Số đêm tối thiểu (Minimum Nights)",
        min_value=1,
        max_value=365,
        value=3,
        step=1
    )

    availability_365 = st.slider(
        "Số ngày trống trong năm (Availability 365)",
        min_value=0,
        max_value=365,
        value=200,
        help="Số ngày căn hộ có sẵn để cho thuê trong năm tới."
    )

    calculated_host_listings_count = st.number_input(
        "Số lượng nhà của Host (Host Listings)",
        min_value=0,
        max_value=500,
        value=1,
        help="Tổng số lượng bất động sản mà chủ nhà này đang cho thuê."
    )

    st.markdown("---")
    
    # Nút bấm dự đoán
    predict_btn = st.button("🚀 Dự đoán Giá ngay", type="primary", use_container_width=True)

    # Hiển thị trạng thái hệ thống ở cuối sidebar
    st.markdown("### 📡 System Status")
    is_online, health_data = check_backend_status()
    if is_online:
        st.success(f"Backend Online (v{health_data.get('version', '1.0.0')})")
    else:
        st.error("Backend Offline / Không kết nối được")

# ==========================================
# 4. MAIN INTERFACE - OUTPUT SECTION
# ==========================================
st.title("🗽 NYC Airbnb Price Prediction")
st.markdown("""
Hệ thống dự đoán giá thuê căn hộ Airbnb tại New York City sử dụng mô hình **XGBoost**.
Nhập thông tin bên thanh menu trái và nhấn nút **Dự đoán**.
""")

# Hiển thị thông số đầu vào dưới dạng JSON (để debug hoặc minh bạch thông tin)
with st.expander("👀 Xem dữ liệu đầu vào (Debug Payload)"):
    input_data = {
        "neighbourhood_group": neighbourhood_group,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365
    }
    st.json(input_data)

# Logic xử lý khi bấm nút
if predict_btn:
    if not is_online:
        st.error(f"❌ Không thể kết nối tới Backend tại: `{BACKEND_URL}`. Vui lòng kiểm tra lại server.")
    else:
        with st.spinner("🤖 Đang gửi dữ liệu tới AI Model..."):
            # 1. Chuẩn bị Payload (Map dữ liệu cho khớp với Pydantic Model bên Backend)
            # Backend mong đợi chuỗi thường (lowercase) cho neighbourhood và room_type
            payload = {
                "neighbourhood_group": neighbourhood_group.lower(),
                "room_type": room_type.lower(),
                "minimum_nights": int(minimum_nights),
                "calculated_host_listings_count": int(calculated_host_listings_count),
                "availability_365": int(availability_365)
            }

            try:
                # 2. Gửi Request POST
                start_time = time.time()
                response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=10)
                process_time = (time.time() - start_time) * 1000

                # 3. Xử lý kết quả
                if response.status_code == 200:
                    result = response.json()
                    price = result.get("price_prediction", 0)
                    confidence = result.get("confidence", "unknown")
                    
                    # Hiển thị kết quả đẹp mắt
                    st.success("✅ Dự đoán thành công!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="💰 Giá dự đoán (mỗi đêm)", value=f"${price}")
                    with col2:
                        # Map màu sắc cho mức độ tự tin
                        conf_color = "off"
                        if confidence == "high": conf_color = "normal" 
                        st.metric(label="🎯 Độ tin cậy", value=confidence.upper())
                    with col3:
                        st.metric(label="⚡ Thời gian xử lý", value=f"{process_time:.0f}ms")

                    # Hiển thị JSON trả về đầy đủ (nếu cần)
                    with st.expander("Xem chi tiết phản hồi từ API"):
                        st.json(result)
                        
                else:
                    st.error(f"⚠️ Server trả về lỗi: {response.status_code}")
                    st.code(response.text)

            except requests.exceptions.Timeout:
                st.error("⏰ Request hết thời gian (Timeout). Backend xử lý quá lâu.")
            except Exception as e:
                st.error(f"❌ Lỗi không xác định: {str(e)}")

# Footer
st.markdown("---")
st.markdown(f"*Connected to API: `{BACKEND_URL}`*")