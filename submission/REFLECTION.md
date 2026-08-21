# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Điều làm tôi ngạc nhiên nhất là fine-tune đạt target 0.965 và JSON hợp lệ 100% nhưng
regression lại giảm từ 0.7578 xuống 0.4556. Nếu chỉ nhìn độ chính xác tác vụ, tôi đã có
thể kết luận nhầm rằng model sẵn sàng triển khai.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần tốn thời gian nhất là bốn lượt train và nhiều lượt generation trên T4, đặc biệt là
NB3–NB5. Ban đầu tôi nghĩ huấn luyện sẽ chiếm gần như toàn bộ thời gian, nhưng đánh giá
nhiều baseline và adapter cũng tốn đáng kể vì phải sinh output thật cho từng mẫu.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước lab tôi cho rằng target accuracy tăng đủ mạnh là bằng chứng chính để chọn bản
fine-tune. Sau lab, tôi không còn tin một chỉ số đơn lẻ như accuracy, perplexity hay
training loss có thể đại diện cho chất lượng triển khai nếu chưa có regression gate.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để đọc yêu cầu, đối chiếu rubric, hướng dẫn chạy local/Colab,
chẩn đoán lỗi quyền truy cập thư mục temp của Pytest và tổng hợp artefact thành report.
Điểm hạn chế là assistant không thể khôi phục prediction theo từng mẫu của baseline (b)
vì NB2 chỉ lưu metric tổng hợp; nếu tự điền chúng thì sẽ là suy đoán không có bằng chứng.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ viết tiêu chí chấp nhận và đóng băng bộ eval trước khi train: ít nhất phải có
target, format, latency và một tập regression đại diện cho hành vi không được phép suy
giảm. Sau đó tôi mới đo base model với một prompt được tối ưu tử tế để xác định fine-tune
có thực sự cần thiết hay không.
