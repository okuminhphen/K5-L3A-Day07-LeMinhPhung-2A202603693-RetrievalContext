# Tiêu chí chọn cấu hình trên dev

Ưu tiên số câu có đáp án được chấm 2 trên tổng 15 câu dev. Trong trường hợp bằng điểm, lần lượt ưu tiên số câu thiếu bằng chứng được từ chối đúng, số citation thực sự hỗ trợ câu trả lời, số câu giữ được bằng chứng trong context, kết quả retrieval trên truy vấn gốc, chi phí API và thời gian trung vị. Lỗi model được ghi riêng và không tính là câu trả lời đúng. So sánh hiệu quả citation số với cùng Dense, truy vấn tiếng Việt, top 3 và ngân sách ghép context 1.600 ký tự.

Baseline để đánh giá test là Dense, truy vấn tiếng Việt, context 800 ký tự và citation bằng ID. Model `llama3.2:latest` chạy local, giữ tối đa hai lượt tìm cho mỗi câu. Chốt cấu hình được chọn trước khi chạy test.

## Cấu hình đã chốt trước lượt test

Chọn Dense, truy vấn tiếng Việt, context 1.600 ký tự và citation bằng ID. Trên dev cấu hình này trả lời đúng, đủ và có nguồn 3/15 câu. Baseline đạt 0/15, dịch truy vấn đầu đạt 2/15 và citation một số đoạn đạt 2/15. Cả bốn cấu hình từ chối đúng 2/2 câu thiếu bằng chứng. Lượt citation danh sách số đoạn có 5 lỗi JSON và 0 câu trả lời được xuất. Cả hai lượt test giữ cùng model, prompt, top-k và giới hạn lượt tìm. Thay đổi duy nhất giữa baseline và cấu hình đã chốt là ngân sách context.
