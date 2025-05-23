#Code chạy thử trên console
state = {"messages": []}
print("Chào bạn! Tôi là trợ lý AI chuyên hỗ trợ kiểm tra bệnh tiểu đường và đột quỵ. Bạn muốn kiểm tra bệnh nào trước?")
session_id = input("🔐 Nhập mã người dùng (ví dụ: user1, admin, 123): ").strip()

state = {"messages": []}

while True:
    user_input = input(session_id+": ").strip()
    if not user_input:
        continue

    # Lưu câu hỏi người dùng vào state
    state["messages"].append(HumanMessage(content=user_input))

    # Gọi graph, truyền session_id để kích hoạt long-term memory
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": session_id}}
    )

    # Phản hồi từ agent
    ai_reply = result["messages"][-1].content
    print(f"🤖 AI: {ai_reply}")

    # Nếu agent trả về kết thúc
    if result.get("next") == END:
        print("🛑 Kết thúc phiên.")
        break

    # Cập nhật state cho vòng tiếp theo
    state = result