# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

## Benchmark Day 7

| Tập | Có đáp án | Thiếu bằng chứng |
|---|---:|---:|
| [Dev](benchmark_dev.json) | 15 | 2 |
| [Test](benchmark_test.json) | 5 | 2 |

Bộ câu hỏi dùng 211 chunk cấu hình B của Day 6, trích từ 38 trang thuộc chương 1, 2, 3 và 5 của *Think Python, 2nd Edition* (Allen B. Downey, Green Tea Press, 2015, [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/)). `page` là số trang PDF tính từ 1. [Nguồn sách](https://greenteapress.com/wp/think-python-2e/) · [Cách kết nối kho Day 6](../src/README.md)

Dev giữ 15 câu có đáp án của Day 6, gồm các filter chương đã có trong câu hỏi. Cả 7 câu test đều mới. Các trích đoạn `gold` giữ nguyên văn bản trong kho, kể cả xuống dòng và khoảng trắng do trích PDF. Mỗi trích đoạn khớp trọn trong ít nhất một chunk đúng nguồn, đúng trang. Các câu thiếu bằng chứng ghi phạm vi và cách kiểm tra trên toàn bộ corpus.

Dùng dev để thử và chọn cấu hình. Chỉ đánh giá test sau khi chốt cấu hình. Chỉ đưa câu hỏi và filter do người hỏi cung cấp vào retriever hoặc agent. Khi dùng TF-IDF starter để tính chỉ số retrieval, truyền các câu `answerable: true` vì starter yêu cầu mỗi câu có `gold`.
