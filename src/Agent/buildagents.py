from langgraph.graph import MessagesState
from typing import Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
import os
from langchain.tools import Tool
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
import time
# Tải biến môi trường từ file .env
load_dotenv()
#kiểm tra API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#Trạng thái agent là đầu vào của mỗi node
class AgentState(MessagesState):
    next:str

agents=["diabetes","stroke"]
options=agents+["FINISH"]

system_prompt = (
    "Bạn là trợ lý điều phối AI. Nhiệm vụ: chọn đúng tác vụ `diabetes`, `stroke`, hoặc `FINISH` dựa trên tin nhắn người dùng.\n\n"
    "Cách chọn:\n"
    "- Nếu người dùng đề cập đến tiểu đường, đường huyết → chọn `diabetes`\n"
    "- Nếu nhắc đến đột quỵ, tai biến, đau đầu, yếu nửa người → chọn `stroke`\n"
    "- Nếu nói kiểm tra xong hoặc không cần nữa → chọn `FINISH`\n"
    "- Nếu không rõ → hỏi lại người dùng: 'Bạn muốn kiểm tra bệnh tiểu đường hay đột quỵ?'\n\n"
    "Chỉ trả về 1 từ: `diabetes`, `stroke`, hoặc `FINISH`. Không viết thêm gì khác."
)

#Xác định xem tác tử tiếp theo là gì.
class Router(TypedDict):
    next: Literal["diabetes", "stroke", "FINISH"]
#Gọi chatpgt
llm=ChatOpenAI(model="gpt-4o-mini")
# llm=ChatOpenAI(model="gpt-4.1")
#Xây dụng nốt supervisor
def supervisor_node(state: AgentState) -> AgentState:
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    try:
        response = llm.with_structured_output(Router).invoke(messages)
        next_ = response["next"]
        if next_ not in options:
            print(f"Cảnh báo: Giá trị next không hợp lệ: {next_}. Chuyển sang FINISH.")
            next_ = "FINISH"
    except Exception as e:
        print(f"Lỗi trong supervisor_node: {e}. Chuyển sang FINISH.")
        # Gửi thông báo lỗi cho người dùng luôn
        return {
            "messages": state["messages"] + [HumanMessage(content="Xin lỗi, hệ thống đang tạm quá tải. Bạn hãy thử lại sau vài phút nhé.", name="system")],
            "next": END
        }

    if next_ == "FINISH":
        next_ = END
    return {"next": next_}

"""Xây dựng agent con"""
# tool gọi model đã train của bệnh tiểu đường
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Đường dẫn tới buildagents.py
MODEL_PATH = os.path.join(BASE_DIR, '..', '..', 'output_model', 'diabetes_model.h5')
diabetes_model = load_model(MODEL_PATH, compile=False)
scaler_path = os.path.join(BASE_DIR, '..', '..', 'output_model', 'diabetesscaler.pkl')
scaler_diabetes= joblib.load(scaler_path)
def predict_diabetes_risk(bmi,highBP,highchol,age,gentHth):
    if age < 18:
        raise ValueError("Tuổi phải từ 18 trở lên")
    age_group = min((age - 18) // 5 + 1, 13)  # nhóm tuổi từ 1–13
    # Giá trị mặc định nếu người dùng không nhập
    input_data = {
        'HighBP': highBP, #Có bị huyết áp cao không
        'HighChol': highchol,#có bị cholesterol cao không
        'CholCheck': 1, #có kiểm tra cholesterol trong năm nay không
        'BMI': bmi,#chỉ số cơ thể
        'Smoker': 0,#có hút thuốc không
        'Stroke': 0,#có bị đột quỵ không
        'HeartDiseaseorAttack': 0,# Có bệnh tim hay từng đau tim (0: không, 1: có)
        'PhysActivity': 1,# có hoạt động thường xuyên không
        'Fruits': 0,# Ăn trái cây thường xuyên (0: không, 1: có)
        'Veggies': 1,# Ăn rau thường xuyên (0: không, 1: có)
        'HvyAlcoholConsump': 0,# Uống nhiều rượu (0: không, 1: có)
        'AnyHealthcare': 1,# Có bảo hiểm/y tế (0: không, 1: có)
        'NoDocbcCost': 0,# Không đi khám vì chi phí (0: không, 1: có)
        'GenHlth': gentHth, # Sức khỏe tổng thể (1: rất tốt, 5: rất tệ)
        'MentHlth': 0, # Số ngày sức khỏe tinh thần kém (0–30)
        'DiffWalk': 0,# Gặp khó khăn khi đi bộ (0: không, 1: có)
        'Sex': 0,#Giới tính(0: nữ, 1:nam)
        'Age': age_group,#Nhóm tuổi.
        'Education': 3,#trình độ học vấn
        'Income': 3#thu nhập trong 1 năm
    }
    df_input = pd.DataFrame([input_data])  # Tạo DataFrame 1 dòng
    num_col = ['BMI', 'GenHlth', 'MentHlth', 'Age', 'Education', 'Income']
    df_input[num_col] = scaler_diabetes.transform(df_input[num_col])
    prediction = diabetes_model.predict(df_input)
    return prediction[0][0]
diabetes_model_tool= Tool(
    name='diabetes_model_predictor',
    func=predict_diabetes_risk,
    description=' Dự đoán nguy cơ bị tiểu đường dựa trên các thông tin đầu vào.'
)
diabetes_agent=create_react_agent(
    llm, tools=[diabetes_model_tool], #Xây dựng model để truy vấn
   state_modifier = """
Bạn là một trợ lý AI chuyên đánh giá nguy cơ mắc bệnh tiểu đường, dựa trên dữ liệu từ bộ chỉ số sức khỏe Diabetes Health Indicators (BRFSS 2015). Nhiệm vụ của bạn gồm các bước sau:

1. Thu thập **5 thông tin cơ bản** từ người dùng để đánh giá nguy cơ tiểu đường:
   - Cân nặng (kg) và chiều cao (m) để tính chỉ số BMI.
   - Người dùng có bị huyết áp cao không? (Có/Không).
   - Người dùng có bị cholesterol cao không? (Có/Không).
   - Tuổi.
   - Đánh giá sức khỏe tổng quát của người dùng (Rất tốt/Tốt / Bình thường / Kém/rất kém).

2. Phân tích dữ liệu theo 3 trường hợp sức khỏe:
     - Tính chỉ số BMI theo công thức: `BMI = cân nặng / (chiều cao * chiều cao)`.
   - Chuyển đổi mức đánh giá sức khỏe (`gen_health`) thành số nguyên:
     - Rất tốt → 1
     - Tốt → 2
     - Bình thường → 3
     - Kém → 4
     - Rất kém → 5
   - Gửi các thông tin sau vào công cụ `diabetes_model_tool` để dự đoán nguy cơ tiểu đường:
     - `bmi`: chỉ số BMI đã tính.
     - `highBP`: 1 nếu người dùng bị huyết áp cao, 0 nếu không.
     - `highChol`: 1 nếu người dùng bị cholesterol cao, 0 nếu không.
     - `age`: Tuổi người dùng(số nguyên).
     - `gen_health`: giá trị số từ 1 đến 5, như quy ước trên.
3. Diễn giải kết quả:
   - Nếu mô hình trả về 1 → Đã mắc bệnh. “Dựa trên phân tích và kết quả mô hình, bạn có khả năng **đã mắc bệnh tiểu đường**. Các yếu tố như [liệt kê các yếu tố nguy cơ] làm tăng nguy cơ bệnh. Bạn nên đi khám bác sĩ và xét nghiệm đường huyết càng sớm càng tốt nhé.”
   - Nếu trả về 0:
     - Nếu có ≥ 2 yếu tố nguy cơ (BMI >25, huyết áp cao, tuổi >45...) → Nguy cơ tiền tiểu đường.“Hiện tại bạn chưa mắc bệnh nhưng có **nguy cơ cao bị tiểu đường** do các yếu tố như [liệt kê yếu tố]. Hãy thay đổi lối sống (ăn uống lành mạnh, tập thể dục đều đặn) và theo dõi sức khỏe thường xuyên.”
     - Ngược lại → Khỏe mạnh.“Tuyệt vời! Bạn có lối sống khá lành mạnh và các chỉ số sức khỏe tốt. Nguy cơ tiểu đường hiện tại của bạn **rất thấp**. Hãy tiếp tục duy trì lối sống tốt này nhé!”

4. Ngôn ngữ phản hồi cần thân thiện, dễ hiểu, khích lệ tinh thần. Nếu người dùng mô tả có triệu chứng bất thường (ví dụ: khát nước nhiều, tiểu nhiều, mệt mỏi kéo dài…), hãy **khuyến nghị đi khám bác sĩ ngay**, kể cả khi mô hình dự đoán chưa mắc bệnh.

Bắt đầu bằng lời chào và câu hỏi như sau:
“Chào bạn! Mình là trợ lý đánh giá nguy cơ tiểu đường. Mình cần bạn cung cấp vài thông tin đơn giản nhé:
1. Cân nặng của bạn là bao nhiêu kg?
2. Chiều cao của bạn là bao nhiêu mét?
3. Bạn có bị huyết áp cao không? (Có/Không)
4. Ban có bị tolesterol cao không?(Có/Không)
5. Tuổi của bạn (hoặc chọn nhóm tuổi: 18–24, 25–34, 35–44, 45–54, 55–64, 65+)?
6. Bạn đánh giá sức khỏe tổng quát của mình như thế nào? (Rất tốt/Tốt / Bình thường / Kém/rất kém)

Bạn chỉ cần trả lời những câu hỏi trên, mình sẽ phân tích nguy cơ tiểu đường giúp bạn nhé!”
"""
)
def diabetes_node(state: AgentState)-> AgentState:
    #Tìm kiếm thông tin và trả về kết quả
    result=diabetes_agent.invoke(state)
    return{
      "messages":[HumanMessage(content=result["messages"][-1].content)]
    }

percent_risk_MODEL_PATH = os.path.join(BASE_DIR, '..', '..', 'output_model', 'PercentStroke_model.h5')
scaler_percent_stroke_path = os.path.join(BASE_DIR, '..', '..', 'output_model', 'scaler_%risk.pkl')
# tool của agent stroke
percent_stroke_model = load_model(percent_risk_MODEL_PATH, compile=False)
scaler_percent_stroke= joblib.load(scaler_percent_stroke_path)
def predict_percent_risk(highBP,age,fatigue_weakness,dizziness,nausea,chestpain):
    input_data = {
        'Chest Pain': chestpain,#Có đau ngực không.
        'Shortness of Breath': 0,#Có khó thở không.
        'Irregular Heartbeat': 0,#Nhịp tim có không đều không
        'Fatigue & Weakness': fatigue_weakness,#Có mệt mỏi và yếu dai dẳng không
        'Dizziness': dizziness,#Có chóng mặt không
        'Swelling (Edema)': 0,#Có bị sưng/phù nề không
        'Pain in Neck/Jaw/Shoulder/Back': 0,#Có đau cổ hàm vai lưng không
        'Excessive Sweating': 0,#Có đổ mồ hôi nhiều không
        'Persistent Cough': 0,#Có ho kéo dài không
        'Nausea/Vomiting': nausea,#Có buồn nôn/nôn không
        'High Blood Pressure': highBP,#Có huyết áp cao không
        'Chest Discomfort (Activity)': 0,#Có  Khó chịu vùng ngực khi vận động không.
        'Cold Hands/Feet': 0,# Có Lạnh tay/chân không
        'Snoring/Sleep Apnea': 0,#Có Ngáy / ngưng thở khi ngủ
        'Anxiety/Feeling of Doom': 0,#CóLo âu / linh cảm xấu không
        'Age': age #tuổi
    }
    df_input = pd.DataFrame([input_data])  # Tạo DataFrame 1 dòng
    num_col = ['Age', ]
    df_input[num_col] = scaler_percent_stroke.transform(df_input[num_col])
    prediction = percent_stroke_model.predict(df_input)
    return prediction[0][0]
percent_risk_tool= Tool(
    name='%risk_model_predictor',
    func=predict_percent_risk,
    description=' Dự đoán phần trăm nguy cơ bị đột quỵ.'
)
#tool model stroke
stroke_model_PATH = os.path.join(BASE_DIR, '..', '..', 'output_model', 'stroke_model.h5')
scaler_stroke_path = os.path.join(BASE_DIR, '..', '..', 'output_model', 'strokescaler.pkl')
stroke_model = load_model(stroke_model_PATH, compile=False)
scaler_stroke= joblib.load(scaler_stroke_path)
def predict_stroke(age,percentrisk):
    input_data = {
        'Chest Pain': 0,
        'Shortness of Breath': 0,
        'Irregular Heartbeat': 0,
        'Fatigue & Weakness': 0,
        'Dizziness': 0,
        'Swelling (Edema)': 0,
        'Pain in Neck/Jaw/Shoulder/Back': 0,
        'Excessive Sweating': 0,
        'Persistent Cough': 0,
        'Nausea/Vomiting': 0,
        'High Blood Pressure': 0,
        'Chest Discomfort (Activity)': 0,
        'Cold Hands/Feet': 0,
        'Snoring/Sleep Apnea': 0,
        'Anxiety/Feeling of Doom': 0,
        'Age': age,
        'Stroke Risk (%)': percentrisk
    }
    df_input = pd.DataFrame([input_data])  # Tạo DataFrame 1 dòng
    num_col = ['Age','Stroke Risk (%)']
    df_input[num_col] = scaler_stroke.transform(df_input[num_col])
    prediction = stroke_model.predict(df_input)
    return prediction[0][0]
stroke_model_tool= Tool(
    name='stroke_model_predictor',
    func=predict_stroke,
    description=' Dự đoán nguy cơ bị đột quỵ.'
)
stroke_agent=create_react_agent(
    llm, tools=[percent_risk_tool,stroke_model_tool],
   state_modifier = """
Bạn là một bác sĩ AI chuyên dự đoán nguy cơ đột quỵ dựa trên bộ dữ liệu Stroke Prediction Dataset. Nhiệm vụ của bạn là:

1. Thu thập **6 thông tin cơ bản** từ người dùng để đánh giá nguy cơ đột quỵ:
   - Tuổi (ví dụ: 45).
   - Có bị tăng huyết áp không? (Có/Không).
   - Có bị đau ngực không? (Có/Không).
   - Có bị mệt mỏi và yếu dai dẳng không?(Có/không)
   - Có thường xuyên bị chóng mặt không?(Có/không)
   - Có thường xuyên bị buồn nôn hay nôn mửa không?(Có/không)?
2. Phân tích và xử lý dữ liệu đầu vào:
   - Sử dụng công cụ `percent_risk_tool` để dự đoán **% nguy cơ đột quỵ**. 
     → Gọi hàm `percent_risk_tool` với 6 tham số như sau:
       (highBP, age, fatigue_weakness, dizziness, nausea, chestpain)
       trong đó:
         - highBP: Có tăng huyết áp không? (1 nếu Có, 0 nếu Không)
         - age: Tuổi người dùng (số nguyên)
         - fatigue_weakness: Có bị mệt mỏi và yếu dai dẳng không? (1/0)
         - dizziness: Có thường xuyên chóng mặt không? (1/0)
         - nausea: Có thường xuyên buồn nôn hoặc nôn mửa không? (1/0)
         - chestpain: Có bị đau ngực không? (1/0)
   -Tiếp theo sử dụng công cụ `stroke_model_tool` với giá trị đầu vào là tuổi và kết quả dự đoán của 'percent_risk_tool' để dự đoán biến `stroke`:
     - Kết quả `stroke = 1`: Dự đoán có khả năng đã bị đột quỵ.
     - Kết quả `stroke = 0`: Dự đoán chưa bị đột quỵ, nhưng cần xem **xác suất nguy cơ (%)**.

3. Phản hồi theo các trường hợp:
   - **Trường hợp 1 – Đã bị đột quỵ (`stroke = 1`)**:
     → Thông báo người dùng có khả năng đã bị đột quỵ.
     → Khuyến nghị đi khám bác sĩ hoặc đến bệnh viện ngay để kiểm tra và điều trị kịp thời.

   - **Trường hợp 2 – Nguy cơ cao (`stroke = 0` và **% nguy cơ đột quỵ** > 40%)**:
     → Thông báo người dùng hiện chưa bị đột quỵ, nhưng có nguy cơ cao (nêu cụ thể phần trăm).
     → Chỉ ra các yếu tố nguy cơ như dựa trên thông tin người dùng nhập
     → Khuyến nghị thay đổi lối sống: ăn nhạt, tập thể dục đều đặn, kiểm tra sức khỏe định kỳ.

   - **Trường hợp 3 – Nguy cơ thấp (`stroke = 0` và xác suất ≤ 40%)**:
     → Thông báo người dùng có sức khỏe tim mạch tốt, nguy cơ đột quỵ thấp (nêu cụ thể phần trăm).
     → Khen ngợi lối sống lành mạnh và khuyến khích tiếp tục duy trì.

   - **Trường hợp 4 – Thiếu dữ liệu đầu vào quan trọng (tuổi hoặc tăng huyết áp)**:
     → Nhắc nhở nhẹ nhàng và yêu cầu người dùng cung cấp thông tin còn thiếu.

4. Nếu người dùng mô tả có triệu chứng bất thường như:
   - Yếu một bên người
   - Méo miệng
   - Khó nói, nói ngọng bất thường
   - Mất thăng bằng, chóng mặt, đau đầu dữ dội

→ Dù mô hình dự đoán chưa có nguy cơ, bạn vẫn **phải khuyến nghị người dùng đến bệnh viện hoặc gọi cấp cứu ngay lập tức**.

5. Ngôn ngữ trả lời:
   - Luôn thân thiện, dễ hiểu, không gây hoang mang.
   - Gợi ý thay đổi tích cực, động viên chăm sóc sức khỏe.

Bắt đầu bằng lời chào và đặt câu hỏi như sau:

“Chào bạn! Mình là trợ lý đánh giá nguy cơ đột quỵ. Để giúp bạn, mình chỉ cần vài thông tin đơn giản:
1. Tuổi của bạn?
2. Bạn có bị tăng huyết áp không? (Có/Không)
3. Bạn Có bị mệt mỏi và yếu dai dẳng không?(Có/không)
4. Bạn có  bị đau ngực không? (Có/Không).
5. Bạn có thường xuyên bị chóng mặt không?(Có/không).
6. Bạn có thường xuyên buồn nôn hay nôn mửa không?(Có/không).

Hãy trả lời và mình sẽ đánh giá nguy cơ đột quỵ cho bạn nhé!”
"""
)
def stroke_node(state: AgentState)-> AgentState:
  #Tìm kiếm thông tin và trả về kết quả
  result=stroke_agent.invoke(state)
  return{
      "messages":[HumanMessage(content=result["messages"][-1].content)]
  }

#Xây dựng flow
builder = StateGraph(AgentState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("diabetes", diabetes_node)
builder.add_node("stroke", stroke_node)

for agent in agents:
    builder.add_edge(agent, "supervisor")

# Để supervisor điều phối luồng công việc giữa các agent và quyết định khi nào hoàn thành
builder.add_conditional_edges("supervisor", lambda state: state["next"])

# === Bộ nhớ short-term và long-term
checkpointer = InMemorySaver()
store = InMemoryStore()

# Biên dịch graph kèm theo bộ nhớ
graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)

#Xây dựng hàm để trả lời lại người dùng
def run_graph(messeages: list)->dict:
    state={"messages": messeages}
    result=graph.invoke(state)
    return result

# state = {"messages": []}
# #Code chạy thử trên console
# print("Chào bạn! Tôi là trợ lý AI chuyên hỗ trợ kiểm tra bệnh tiểu đường và đột quỵ. Bạn muốn kiểm tra bệnh nào trước?")
# session_id = input("🔐 Nhập mã người dùng (ví dụ: user1, admin, 123): ").strip()

# state = {"messages": []}

# while True:
#     user_input = input(session_id+": ").strip()
#     if not user_input:
#         continue

#     # Lưu câu hỏi người dùng vào state
#     state["messages"].append(HumanMessage(content=user_input))

#     # Gọi graph, truyền session_id để kích hoạt long-term memory
#     result = graph.invoke(
#         state,
#         config={"configurable": {"thread_id": session_id}}
#     )

#     # Phản hồi từ agent
#     ai_reply = result["messages"][-1].content
#     print(f"🤖 AI: {ai_reply}")

#     # Nếu agent trả về kết thúc
#     if result.get("next") == END:
#         print("🛑 Kết thúc phiên.")
#         break
#     time.sleep(3) 

#     # Cập nhật state cho vòng tiếp theo
#     state = result