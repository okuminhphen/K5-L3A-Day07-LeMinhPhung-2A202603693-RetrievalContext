# Báo cáo Day 7

**Lê Minh Phụng · 2A202603693 · L3A**

## Dữ liệu và benchmark

- Dùng 211 chunk của cấu hình B từ Day 6, gồm 38 trang PDF thuộc chương 1, 2, 3 và 5 của *Think Python, 2nd Edition*, Allen B. Downey, Green Tea Press, 2015. Tài liệu dùng cho mục đích phi thương mại có ghi công theo CC BY-NC 3.0. Trang trong trace là trang vật lý tính từ 1. [Nguồn sách](https://greenteapress.com/wp/think-python-2e/) · [Giấy phép](https://creativecommons.org/licenses/by-nc/3.0/)
- Day 7 truy vấn PostgreSQL 16 và pgvector 0.8.7 từ Day 6. Embedding dùng `intfloat/multilingual-e5-small`, 384 chiều, với tiền tố `query:` và cosine. Model trả lời `llama3.2:latest` chạy qua Ollama local, nhiệt độ 0. [Lệnh chạy và kết nối](../../src/README.md)
- Môi trường kiểm tra trên máy dùng Python 3.12.2, Sentence Transformers 5.7.0 và psycopg 3.3.8. Snapshot E5 có trong cache là `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Manifest Ollama của `llama3.2:latest` có ID `a80c4f17acd5`. Các ID này được xác minh từ cache và manifest hiện có. Trace cũ ghi tên model và tag.
- Dev có 15 câu có đáp án và 2 câu thiếu bằng chứng. Test mới có 5 câu có đáp án và 2 câu thiếu bằng chứng. Điểm retrieval dùng đoạn chuẩn khớp nguyên văn, đúng nguồn và trang. Câu thiếu evidence có ghi cách rà corpus, được chấm riêng về từ chối. Agent chỉ nhận câu hỏi và bộ lọc chương đã được chỉ rõ trong câu hỏi. [Dữ liệu và phương pháp chấm](../../data/README.md)

## Retrieval và thử nghiệm

So sánh trên cùng kho và câu hỏi. Cột context của các hàng retrieval dev đo context ghép từ lượt tìm đầu. Hai hàng test đo context cuối của agent. Chỉ số ranking của test lấy từ truy vấn gốc ở lượt đầu. Thời gian là trung vị mỗi câu. Chi phí API bằng 0 USD ở mọi lượt chạy.

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Context có đủ evidence | Thời gian |
|---|---|---:|---:|---:|---:|
| Dense, 800 ký tự | dev | 4/15 (26,7%) | 6/15 (40,0%) | 4/15 (26,7%) | 32,8 ms truy xuất |
| TF-IDF, 800 ký tự | dev | 3/15 (20,0%) | 4/15 (26,7%) | 3/15 (20,0%) | 0,18 ms truy xuất |
| Hybrid RRF, 800 ký tự | dev | 3/15 (20,0%) | 4/15 (26,7%) | 3/15 (20,0%) | 34,5 ms truy xuất |
| Dense, 1.600 ký tự | dev | 4/15 (26,7%) | 6/15 (40,0%) | 6/15 (40,0%) | 37,8 ms truy xuất |
| Dense + rerank, 800 ký tự | dev | 3/15 (20,0%) | 4/15 (26,7%) | 3/15 (20,0%) | 36,1 ms truy xuất |
| Dịch sang Anh + Dense, 1.600 ký tự | dev | 8/15 (53,3%) | 9/15 (60,0%) | 9/15 (60,0%) | 573 ms gồm dịch và truy xuất |
| Baseline Dense, 800 ký tự | test | 1/5 (20%) | 3/5 (60%) | 2/5 (40%) | 3,44 giây toàn luồng |
| Cấu hình chọn, Dense 1.600 ký tự | test | 1/5 (20%) | 3/5 (60%) | 3/5 (60%) | 2,66 giây toàn luồng |

TF-IDF dùng tokenizer của starter. Hybrid lấy tối đa 10 ứng viên từ mỗi phương pháp và gộp theo RRF với hằng số 60. Thử nghiệm rerank lấy 10 ứng viên Dense và xếp theo `0.8 × cosine_dense + 0.2 × cosine_tfidf`. Thử nghiệm dịch giữ nguyên kho, filter và ngân sách cùng mốc Dense 1.600 ký tự. So sánh tiếng Việt và tiếng Anh trên cùng lượt chạy cho Hit@3 lần lượt là 6/15 và 9/15. [Ba phương pháp](../results/retrieval_dev.json) · [Context 1.600](../results/dense_context1600_dev.json) · [Rerank](../results/dense_rerank_dev.json) · [Dịch](../results/retrieval_translation_dev.json)

- **Q04 và Q14.** Dense đã tìm đúng ở hạng 3. Context 800 ký tự bỏ các đoạn đó. Tăng lên 1.600 ký tự giữ được cả hai, nhưng agent vẫn không trả lời đúng. Q04 lặp lại từ `string` và chọn ví dụ sai. Q14 chọn sai đoạn chứa định nghĩa đầy đủ.
- **Q03, Q04 và Q14.** Rerank đẩy cả ba đoạn đúng ra khỏi top 3, trong khi đưa Q07 lên hạng 3. Trên tập này, tín hiệu trùng từ trong ứng viên chưa đủ để xếp ưu tiên đáng tin.
- **Dịch truy vấn và đánh số citation.** Dịch thêm bằng chứng cho bốn câu ở lượt tìm đầu và làm mất Q04 do bỏ chi tiết bảng thuật ngữ. Agent sau khi dịch đạt 2/15 câu đầy đủ, so với 3/15 khi dùng câu gốc. Citation bằng danh sách số có 5 lỗi JSON. Đổi thành một số đoạn và một trích dẫn loại bỏ lỗi JSON ở lượt dev đó và giảm số câu từ chối nhầm từ 7 xuống 5, nhưng chỉ có 2/15 câu đúng đầy đủ. [Trace danh sách số](../results/agent_dev_numbered.json) · [Trace một số đoạn](../results/agent_dev_numbered_single.json) · [Nhãn dev](../results/agent_dev_numbered_single_reviews.json)

Theo tiêu chí ghi trước lần thử citation số, ưu tiên số câu đúng và đủ có nguồn. Chốt Dense, câu gốc tiếng Việt, 1.600 ký tự và citation bằng ID trước khi chạy test. Giữ baseline với 800 ký tự. [Quyết định chọn cấu hình](../results/selection_criteria.md)

## Agent và đánh giá

Agent có tool tìm kiếm Dense, tối đa 2 lượt và 3 đoạn mỗi lượt. Lượt đầu dùng câu hỏi gốc. Model có thể tự tạo truy vấn thứ hai. Context bỏ đoạn trùng ID, ưu tiên kết quả mới nhất và lấy nguyên đoạn vừa ngân sách ký tự, kể cả nhãn nguồn. Model chọn trả lời hoặc từ chối. Code kiểm tra ID và đoạn trích có trong context. Codex đọc trace để chấm câu trả lời và mức hỗ trợ của nguồn. Người nộp có thể đối chiếu nhãn với văn bản gốc. Thời gian toàn luồng bao gồm tạo vector, truy vấn và lượt model. Lượt đầu mỗi tiến trình có thể chịu chi phí khởi động model.

Top 3 giữ cùng ngưỡng với Evidence Hit@3 và giới hạn lượng văn bản phải ghép sau mỗi lượt tìm. Giới hạn hai lượt cho model một cơ hội đổi truy vấn khi lượt đầu thiếu bằng chứng và giới hạn độ trễ do tìm lặp. Hai tham số này được giữ cố định trong phép so sánh ngân sách context. Trên dev, mức 1.600 ký tự giữ lại được đoạn hạng 3 của Q04 và Q14, tăng số câu có bằng chứng trong context từ 4/15 lên 6/15. Thời gian trung vị toàn luồng tăng từ 2,44 lên 2,93 giây/câu trong hai lượt dev đã chấm.

| Cấu hình | Tool use | Phân bố 1/2 lượt tìm | Điểm trả lời 0/1/2, TB trên 2 | Citation hỗ trợ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm | Lỗi |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Dev baseline | 17/17 | 17/0 | 12/3/0, 0,20 | 1/8 | 8/8 | 2/2 | 7/15 | 0 |
| Dev Dense 1.600 | 17/17 | 15/2 | 12/0/3, 0,40 | 3/8 | 8/8 | 2/2 | 7/15 | 0 |
| Dev dịch truy vấn đầu | 16/16 | 15/2 | 12/0/2, 0,29 | 2/5 | 5/5 | 2/2 | 9/14 | 1 |
| Dev một số đoạn | 17/17 | 13/4 | 10/3/2, 0,47 | 3/10 | 10/10 | 2/2 | 5/15 | 0 |
| Test baseline | 7/7 | 5/2 | 4/0/1, 0,40 | 1/1 | 1/1 | 2/2 | 4/5 | 0 |
| Test cấu hình chọn | 6/6 | 6/1 | 3/0/1, 0,50 | 1/3 | 2/2 | 2/2 | 2/4 | 1 |

Theo benchmark, mẫu số của điểm trả lời và tool use loại lỗi model. Phân bố lượt tìm tính trên mọi trace, gồm cả câu tìm xong rồi gặp lỗi JSON. Khi tính trên toàn bộ câu có đáp án của test, cả hai cấu hình trả lời đúng đầy đủ **1/5 (20%)**. T01 lỗi JSON ở cấu hình chọn được giữ nguyên trong trace và vẫn chấm được context. Phân bố context 0/1/2 ở baseline là **2/1/2** và cấu hình chọn là **1/1/3**.

| Câu test | Baseline | Cấu hình đã chốt |
|---|---|---|
| T01 · dấu nhắc `>>>` | Context 0, từ chối vì citation sai định dạng | Context 0, lỗi JSON sau truy xuất |
| T02 · `6 + 4 / 2` | Context 2, trả lời `8`, dẫn trang PDF 34 | Context 2, trả lời `8`, dẫn trang PDF 34 |
| T03 · phạm vi biến | Context 0, từ chối | Context 2, có trang 44 nhưng citation kèm tên file và trang, nên từ chối |
| T04 · điểm dừng đệ quy | Context 2, model chỉ nêu `n = 0` và citation sai định dạng | Context 2, model từ chối dù trang 66 có cả hai ý |
| T05 · phân loại lỗi | Context 1, từ chối | Context 1, trả lời sai là lỗi runtime |
| T06 · `lru_cache` | Ứng dụng chặn câu trả lời vì trích dẫn không khớp | Model chủ động từ chối |
| T07 · `requirements.txt` | Ứng dụng chặn câu trả lời vì trích dẫn không khớp | Model chủ động từ chối |

Context lớn giữ thêm được đoạn đúng ở T03. Số câu trả lời đúng đầy đủ giữ ở 1/5 vì model vẫn từ chối T03 và T04. Một lần đo trên 7 câu chưa đủ để kết luận về tốc độ hoặc hiệu quả trên corpus khác. Các cấu hình TF-IDF, Hybrid và rerank được so sánh ở tầng retrieval. Các lượt chấm agent dùng Dense và các biến thể context, dịch, citation.

- **Dev.** [Baseline](../results/agent_dev_baseline_summary.json) · [Bản 1.600 ký tự](../results/agent_dev_summary.json) · [Bản dịch](../results/agent_dev_translation_summary.json) · [Bản một số đoạn](../results/agent_dev_numbered_single_summary.json)
- **Test baseline.** [Trace](../results/agent_test_baseline.json) · [Nhãn và giải thích](../results/agent_test_baseline_reviews.json) · [Chỉ số](../results/agent_test_baseline_summary.json)
- **Test cấu hình chọn.** [Trace](../results/agent_test_selected.json) · [Nhãn và giải thích](../results/agent_test_selected_reviews.json) · [Chỉ số](../results/agent_test_selected_summary.json)

### Đo token trên test

Hai lượt test gốc chưa lưu số token. Mỗi cấu hình được chạy lại trên cùng 7 câu test để lấy trực tiếp `prompt_eval_count` và `eval_count` từ Ollama ở mỗi lượt gọi model chat. Tổng prompt tính cả hướng dẫn hệ thống, câu hỏi và context gửi lặp lại qua các lượt. Chi phí API vẫn là 0 USD. [Định nghĩa các trường Ollama](https://docs.ollama.com/api/chat)

| Cấu hình | Lượt gọi có số liệu | Token vào | Token ra | Tổng token | Trung vị token/câu |
|---|---:|---:|---:|---:|---:|
| Baseline 800 ký tự | 17 | 7.957 | 825 | 8.782 | 1.111 |
| Cấu hình chọn 1.600 ký tự | 15 | 7.812 | 901 | 8.713 | 1.092 |

Lượt đo token có đầu ra khác lượt gốc. Baseline từ chối cả 7 câu, cấu hình chọn từ chối 6 câu và lỗi JSON ở T01. Tổng token chịu ảnh hưởng của số lượt gọi model. Các điểm chất lượng ở trên tính từ lượt gốc đã chấm. [Trace 800 ký tự](../results/agent_test_tokens_800.json) · [Trace 1.600 ký tự](../results/agent_test_tokens_1600.json) · [Tổng hợp token từng câu](../results/agent_test_token_summary.json)

## Tái lập kết quả

Chuẩn bị môi trường Python và PostgreSQL ở repo Day 6. Nạp lại cấu hình B nếu bảng `ingestion_chunks` chưa có dữ liệu. Từ thư mục gốc Day 7, dùng lệnh trong hướng dẫn để chạy ba phương pháp retrieval trên dev, chạy agent với hai ngân sách context và tổng hợp từng trace với bộ nhãn tương ứng. Khi chạy lại agent, ghi output vào file riêng và chấm lại câu trả lời nếu model sinh kết quả khác. [Hướng dẫn chạy](../../src/README.md)

## Thử thách nâng cao

Chưa thực hiện. Báo cáo này trình bày phần cơ bản trên 38 trang đã index từ Day 6. [Yêu cầu thử thách nâng cao](../../ADVANCED.md)
