💡 AI Multi-Agent Hỗ Trợ Chuẩn Đoán Bệnh Tiểu Đường và Đột Quỵ
📝 Mô tả dự án: Dự án xây dựng một hệ thống AI Multi-Agent sử dụng công nghệ LangGraph của LangChain để hỗ trợ chẩn đoán bệnh tiểu đường và đột quỵ.
    - Hệ thống bao gồm các thành phần chính:
        + Multi-Agent System: Một agent điều phối và hai agent chuyên biệt chẩn đoán từng loại bệnh.
        + Mô hình học sâu (MLP) cho dự đoán bệnh.
        + Web giao diện chatbox để người dùng tương tác.
        + Kết nối API giữa frontend và backend sử dụng FastAPI.
2. Tài liệu: Langgraph Python: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
3. Cài đặt
    - Khởi tạo môi trường ảo: python -m venv env
    - Kích hoạt môi trường(Windows):env\Scripts\activate    
    - Cài đặt thư viện: pip install -r requirements.txt
4. Dữ liệu: dữ liệu được lấy trên Kaggle
    - Bệnh tiểu đường: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv
    - Bệnh đột quỵ: https://www.kaggle.com/code/soheibkholdine/stork-risk-prediction/input
5.  Các bước thực hiện
    - Khám phá và tiền xử lí dữ liệu. Sau đó sử dụng  mạng Multi-Layer Perceptron(MLP) để xây dựng mô hình cho dự đoán kết quả.
    - Xây dựng multi-agent với một agent điều phối giữa hai agent con chuyên biệt là agent chuẩn đoán bệnh tiểu đường và agent chuẩn đoán đột quỵ.
    - Sử dụng HTML,CSS,JS để xây dựng một trang web đơn giản dạng chatbox để người dùng tương tác.
    - Thiết kế và triểu khai API để kết nối frontend với backend thông qua fastAPI