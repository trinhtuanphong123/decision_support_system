import streamlit as st
import requests
import os
import json
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để làm đẹp UI
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF5A5F 0%, #FF8B94 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 2rem 0;
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF5A5F;
        margin: 1rem 0;
    }
    
    /* Confidence badges */
    .confidence-high {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .confidence-medium {
        background-color: #FF9800;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .confidence-low {
        background-color: #f44336;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Stats card */
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #f0f0f0;
        text-align: center;
    }
    
    /* Button styling improvement */
    .stButton>button {
        background: linear-gradient(90deg, #FF5A5F 0%, #FF8B94 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,90,95,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Backend URL configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:9696")
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
ENCODINGS_ENDPOINT = f"{BACKEND_URL}/encodings"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def check_backend_status():
    """Kiểm tra xem Backend có đang online không"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=3)
        if response.status_code == 200:
            return True, response.json()
    except:
        pass
    return False, None

def get_encodings():
    """Lấy danh sách options hợp lệ từ backend"""
    try:
        response = requests.get(ENCODINGS_ENDPOINT, timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def create_price_comparison_chart(predicted_price, neighbourhood, room_type):
    """Tạo biểu đồ so sánh giá"""
    
    # Dữ liệu trung bình cho các khu vực (có thể cập nhật từ backend sau)
    avg_prices = {
        "manhattan": {"entire home/apt": 180, "private room": 90, "shared room": 50},
        "brooklyn": {"entire home/apt": 120, "private room": 70, "shared room": 40},
        "queens": {"entire home/apt": 100, "private room": 60, "shared room": 35},
        "bronx": {"entire home/apt": 85, "private room": 50, "shared room": 30},
        "staten island": {"entire home/apt": 90, "private room": 55, "shared room": 32}
    }
    
    # Lấy giá trung bình cho khu vực và loại phòng được chọn
    avg_price = avg_prices.get(neighbourhood.lower(), {}).get(room_type.lower(), 100)
    
    # Tạo dataframe
    df = pd.DataFrame({
        'Category': ['Giá trung bình', 'Giá dự đoán'],
        'Price': [avg_price, predicted_price],
        'Color': ['#95a5a6', '#FF5A5F']
    })
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Category'],
            y=df['Price'],
            marker_color=df['Color'],
            text=[f'${p:.0f}' for p in df['Price']],
            textposition='auto',
            textfont=dict(size=16, color='white')
        )
    ])
    
    fig.update_layout(
        title=f"So sánh giá - {neighbourhood.title()} ({room_type.title()})",
        yaxis_title="Giá (USD/đêm)",
        showlegend=False,
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def create_borough_comparison_chart(predicted_price, current_borough):
    """Biểu đồ so sánh giá giữa các quận"""
    
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    avg_prices = [180, 120, 100, 85, 90]
    
    colors = ['#FF5A5F' if b.lower() == current_borough.lower() else '#d3d3d3' for b in boroughs]
    
    fig = go.Figure(data=[
        go.Bar(
            x=boroughs,
            y=avg_prices,
            marker_color=colors,
            text=[f'${p}' for p in avg_prices],
            textposition='auto',
        )
    ])
    
    # Thêm đường giá dự đoán
    fig.add_hline(
        y=predicted_price,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Giá của bạn: ${predicted_price:.0f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Giá trung bình theo quận",
        xaxis_title="Quận",
        yaxis_title="Giá (USD/đêm)",
        showlegend=False,
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_availability_impact_chart():
    """Biểu đồ ảnh hưởng của availability đến confidence"""
    
    availability_ranges = ['0-30', '30-90', '90-180', '180-365']
    confidence_scores = [1, 2, 3, 4]  # Low to High
    
    fig = go.Figure(data=[
        go.Scatter(
            x=availability_ranges,
            y=confidence_scores,
            mode='lines+markers',
            line=dict(color='#FF5A5F', width=3),
            marker=dict(size=12)
        )
    ])
    
    fig.update_layout(
        title="Độ tin cậy theo Availability",
        xaxis_title="Số ngày trống (days)",
        yaxis_title="Độ tin cậy",
        yaxis=dict(tickvals=[1,2,3,4], ticktext=['Low','Medium','High','Very High']),
        showlegend=False,
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ==========================================
# 3. SIDEBAR - INPUT SECTION
# ==========================================

with st.sidebar:
    st.markdown("### 🔧 Cấu hình Căn hộ")
    st.markdown("*Nhập thông tin chi tiết để dự đoán giá*")
    st.divider()

    # Kiểm tra backend và lấy encodings
    is_online, health_data = check_backend_status()
    
    if is_online:
        encodings = get_encodings()
        if encodings:
            neighbourhood_options = [opt.title() for opt in encodings["neighbourhood_group"]["options"]]
            room_type_options = [opt.title() for opt in encodings["room_type"]["options"]]
        else:
            # Fallback nếu không lấy được encodings
            neighbourhood_options = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
            room_type_options = ["Entire Home/Apt", "Private Room", "Shared Room"]
    else:
        neighbourhood_options = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        room_type_options = ["Entire Home/Apt", "Private Room", "Shared Room"]

    # Nhóm 1: Vị trí & Loại phòng
    neighbourhood_group = st.selectbox(
        "🗺️ Khu vực (Borough)",
        options=neighbourhood_options,
        index=0,
        help="Quận mà căn hộ tọa lạc tại New York"
    )

    room_type = st.selectbox(
        "🏡 Loại phòng",
        options=room_type_options,
        index=0,
        help="Loại hình lưu trú"
    )

    st.divider()

    # Nhóm 2: Thông số chi tiết
    minimum_nights = st.number_input(
        "🌙 Số đêm tối thiểu",
        min_value=1,
        max_value=365,
        value=3,
        step=1,
        help="Khách phải đặt tối thiểu bao nhiêu đêm"
    )

    availability_365 = st.slider(
        "📅 Số ngày trống/năm",
        min_value=0,
        max_value=365,
        value=200,
        help="Số ngày căn hộ sẵn sàng cho thuê"
    )

    calculated_host_listings_count = st.number_input(
        "📊 Số listing của Host",
        min_value=1,
        max_value=500,
        value=1,
        help="Tổng số nhà chủ này đang cho thuê"
    )

    st.divider()
    
    # Nút dự đoán
    predict_btn = st.button("🚀 Dự đoán Giá", type="primary", use_container_width=True)

    # System Status
    st.markdown("---")
    st.markdown("#### 📡 System Status")
    if is_online:
        st.success(f"✅ Online (v{health_data.get('version', '1.0.0')})")
        st.caption(f"🔄 Uptime: {health_data.get('uptime_seconds', 0):.0f}s")
        st.caption(f"📊 Predictions: {health_data.get('total_predictions', 0)}")
    else:
        st.error("❌ Backend Offline")
        st.caption(f"URL: `{BACKEND_URL}`")

# ==========================================
# 4. MAIN INTERFACE - HEADER
# ==========================================

st.markdown('<h1 class="main-title">🗽 NYC Airbnb Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Dự đoán giá thuê căn hộ Airbnb tại New York City sử dụng Machine Learning</p>', unsafe_allow_html=True)

# ==========================================
# 5. WELCOME SECTION (khi chưa predict)
# ==========================================

if not predict_btn:
    # Info cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>⚡ Nhanh chóng</h3>
            <p>Dự đoán trong vòng <strong>< 100ms</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Chính xác</h3>
            <p>Model XGBoost được train trên <strong>49,000+ listings</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h3>💡 Thông minh</h3>
            <p>Confidence scoring và market insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How it works
    st.markdown("### 🎓 Cách sử dụng")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        1. **📝 Điền thông tin** căn hộ của bạn ở sidebar bên trái
        2. **🚀 Nhấn nút "Dự đoán Giá"** để AI phân tích
        3. **📊 Xem kết quả** với insights và recommendations
        4. **💰 Quyết định giá** dựa trên data-driven insights
        
        ✨ **Mẹo:** Listings với availability cao và minimum nights thấp thường có booking rate tốt hơn!
        """)
    
    with col2:
        st.info("""
        **📚 Features:**
        - Real-time predictions
        - Market insights
        - Price comparison
        - Confidence scoring
        - Borough analysis
        """)
    
    # Sample data preview
    with st.expander("👀 Xem ví dụ Input Data"):
        sample_data = {
            "neighbourhood_group": "Manhattan",
            "room_type": "Entire home/apt",
            "minimum_nights": 3,
            "calculated_host_listings_count": 5,
            "availability_365": 200
        }
        st.json(sample_data)
        st.caption("Đây là dữ liệu mẫu. Điều chỉnh values ở sidebar và nhấn Dự đoán!")

# ==========================================
# 6. PREDICTION LOGIC
# ==========================================

if predict_btn:
    if not is_online:
        st.error(f"""
        ❌ **Không thể kết nối Backend**
        
        Backend URL: `{BACKEND_URL}`
        
        Vui lòng kiểm tra:
        - Backend service có đang chạy không?
        - BACKEND_URL environment variable đúng chưa?
        - Network connection có ổn định không?
        """)
    else:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Đang chuẩn bị dữ liệu...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        # Chuẩn bị payload
        payload = {
            "neighbourhood_group": neighbourhood_group.lower(),
            "room_type": room_type.lower().replace(" ", "_"),
            "minimum_nights": int(minimum_nights),
            "calculated_host_listings_count": int(calculated_host_listings_count),
            "availability_365": int(availability_365)
        }
        
        status_text.text("🤖 Đang gửi tới AI model...")
        progress_bar.progress(50)
        
        try:
            # Gọi API
            start_time = time.time()
            response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=10)
            process_time = (time.time() - start_time) * 1000
            
            progress_bar.progress(100)
            status_text.text("✅ Hoàn tất!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Xử lý kết quả thành công
            if response.status_code == 200:
                result = response.json()
                price = result.get("price_prediction", 0)
                confidence = result.get("confidence", "unknown")
                
                # ==========================================
                # PREDICTION RESULT DISPLAY
                # ==========================================
                
                st.success("🎉 Dự đoán thành công!")
                
                # Main prediction card
                confidence_class = f"confidence-{confidence}"
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>💰 Giá dự đoán mỗi đêm</h2>
                    <div class="prediction-value">${price:.2f}</div>
                    <p style="font-size: 1.1rem;">Recommended nightly price for your listing</p>
                    <span class="{confidence_class}">
                        {confidence.upper()} CONFIDENCE
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Stats row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("💵 Giá dự đoán", f"${price:.2f}")
                
                with col2:
                    competitive_price = price * 0.95
                    st.metric("🎯 Giá cạnh tranh", f"${competitive_price:.2f}", 
                             delta="-5%", delta_color="normal")
                
                with col3:
                    premium_price = price * 1.10
                    st.metric("⭐ Giá premium", f"${premium_price:.2f}",
                             delta="+10%", delta_color="normal")
                
                with col4:
                    st.metric("⚡ Response Time", f"{process_time:.0f}ms")
                
                st.markdown("---")
                
                # ==========================================
                # INSIGHTS & VISUALIZATIONS
                # ==========================================
                
                st.markdown("## 📊 Market Insights")
                
                tab1, tab2, tab3 = st.tabs(["📈 Phân tích giá", "🗺️ So sánh khu vực", "💡 Recommendations"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = create_price_comparison_chart(price, neighbourhood_group, room_type)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Monthly revenue estimate
                        occupancy_rate = 0.7  # Assume 70% occupancy
                        days_per_month = availability_365 / 12
                        monthly_revenue = price * days_per_month * occupancy_rate
                        
                        st.markdown("""
                        ### 💵 Ước tính doanh thu
                        """)
                        
                        st.metric("Doanh thu/tháng", f"${monthly_revenue:.0f}",
                                 help="Giả định 70% occupancy rate")
                        
                        annual_revenue = monthly_revenue * 12
                        st.metric("Doanh thu/năm", f"${annual_revenue:.0f}")
                        
                        st.info(f"""
                        **💡 Tính toán:**
                        - Giá mỗi đêm: ${price:.2f}
                        - Ngày trống/tháng: {days_per_month:.0f}
                        - Occupancy: 70%
                        - Thu nhập/tháng: ${monthly_revenue:.0f}
                        """)
                
                with tab2:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig2 = create_borough_comparison_chart(price, neighbourhood_group)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 🎯 Vị trí của bạn")
                        
                        avg_prices = {
                            "manhattan": 180,
                            "brooklyn": 120,
                            "queens": 100,
                            "bronx": 85,
                            "staten island": 90
                        }
                        
                        borough_avg = avg_prices.get(neighbourhood_group.lower(), 100)
                        price_diff = ((price - borough_avg) / borough_avg) * 100
                        
                        if price_diff > 0:
                            st.success(f"**{price_diff:.1f}%** cao hơn TB khu vực")
                        else:
                            st.info(f"**{abs(price_diff):.1f}%** thấp hơn TB khu vực")
                        
                        st.markdown(f"""
                        **Thông tin thêm:**
                        - Khu vực: {neighbourhood_group.title()}
                        - Giá TB: ${borough_avg}
                        - Giá của bạn: ${price:.2f}
                        """)
                
                with tab3:
                    st.markdown("### 💡 Pricing Recommendations")
                    
                    rec_col1, rec_col2 = st.columns(2)
                    
                    with rec_col1:
                        st.markdown("""
                        #### 🎯 Để tăng bookings:
                        - **Giá cạnh tranh:** Đặt giá ${:.2f} (giảm 5%)
                        - **Minimum nights:** Giảm xuống 1-2 đêm
                        - **Availability:** Tăng ngày trống
                        - **Photos:** Thêm ảnh chất lượng cao
                        - **Reviews:** Khuyến khích khách review
                        """.format(competitive_price))
                    
                    with rec_col2:
                        st.markdown("""
                        #### ⭐ Để tối đa hóa revenue:
                        - **Giá premium:** Đặt giá ${:.2f} (tăng 10%)
                        - **Amenities:** Thêm tiện nghi (WiFi, AC, etc.)
                        - **Location:** Nhấn mạnh gần các địa điểm hot
                        - **Flexibility:** Có chính sách cancel linh hoạt
                        - **Response rate:** Trả lời nhanh trong 1h
                        """.format(premium_price))
                    
                    st.warning(f"""
                    **⚠️ Về Confidence Level: {confidence.upper()}**
                    
                    - **High:** Availability > 180 days → Prediction rất đáng tin
                    - **Medium:** Availability 30-180 days → Prediction khá tốt
                    - **Low:** Availability < 30 days → Nên thận trọng
                    
                    ⭐ Tip: Tăng availability để có predictions chính xác hơn!
                    """)
                
                # ==========================================
                # ADDITIONAL INFO
                # ==========================================
                
                st.markdown("---")
                
                with st.expander("📋 Xem chi tiết Input & Output"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Input Data:**")
                        st.json(payload)
                    
                    with col2:
                        st.markdown("**API Response:**")
                        st.json(result)
                
                # Tips section
                with st.expander("🎓 Tips để tối ưu giá"):
                    st.markdown("""
                    ### 📈 Chiến lược Pricing
                    
                    **Dynamic Pricing:**
                    - Tăng giá vào mùa cao điểm (Summer, Holidays)
                    - Giảm giá vào low season để maintain occupancy
                    - Theo dõi events lớn tại NYC (conferences, concerts)
                    
                    **Competitive Analysis:**
                    - Check giá của listings tương tự trong khu vực
                    - Monitor reviews và ratings của competitors
                    - Adjust dựa trên demand patterns
                    
                    **Optimization:**
                    - Test giá khác nhau trong 2-4 tuần
                    - Track booking rate và revenue
                    - Sử dụng Airbnb's Smart Pricing như reference
                    - Luôn cập nhật calendar availability
                    """)
            
            # Xử lý lỗi từ API
            else:
                st.error(f"""
                ⚠️ **Server trả về lỗi: {response.status_code}**
                
                Chi tiết lỗi:
                """)
                
                try:
                    error_detail = response.json()
                    st.json(error_detail)
                except:
                    st.code(response.text)
                
                st.info("""
                **💡 Có thể do:**
                - Input data không hợp lệ
                - Server đang xử lý quá tải
                - Model chưa được load
                
                Vui lòng thử lại hoặc kiểm tra input data!
                """)
        
        except requests.exceptions.Timeout:
            st.error("""
            ⏰ **Request Timeout**
            
            Backend xử lý quá lâu (> 10 giây). Vui lòng:
            - Kiểm tra network connection
            - Thử lại sau vài phút
            - Liên hệ admin nếu vấn đề tiếp diễn
            """)
        
        except Exception as e:
            st.error(f"""
            ❌ **Lỗi không xác định**
            
            Error: `{str(e)}`
            
            Vui lòng thử lại hoặc liên hệ support.
            """)

# ==========================================
# 7. FOOTER
# ==========================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🗽 NYC Airbnb Price Predictor**")
    st.caption("Powered by XGBoost ML")

with footer_col2:
    st.markdown("**🔗 Links**")
    st.markdown(f"[API Docs]({BACKEND_URL}/docs) | [Health Check]({BACKEND_URL}/health)")

with footer_col3:
    st.markdown("**📊 Stats**")
    if is_online and health_data:
        st.caption(f"Total Predictions: {health_data.get('total_predictions', 0)}")
        st.caption(f"Uptime: {health_data.get('uptime_seconds', 0)/3600:.1f}h")

st.caption(f"*API Backend: `{BACKEND_URL}`*")