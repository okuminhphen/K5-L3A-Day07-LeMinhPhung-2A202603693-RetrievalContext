# Code

Code bài làm nằm trong thư mục này. Xem [starter](../starter/README.md) để tra các ví dụ của đề.

## Thử dense retrieval trên PostgreSQL Day 6

Với repo Day 6 cùng cấp, bật PostgreSQL theo README Day 6 rồi chạy từ thư mục gốc Day 7 trong PowerShell:

```powershell
$day6 = '..\K5-L3A-Day06-LeMinhPhung-2A202603693-DataIngestionEngineering\starter'
$strategy = 'f1981f439e266dff42b2827e190fc629a62e280420cce4f8b8fe89b533846305:B'
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.dense "Khi nhiều điều kiện trong if/elif cùng đúng thì nhánh nào chạy?" --env-file "$day6\.env" --strategy $strategy --top-k 3
```

Lệnh dùng lại môi trường Python Day 6, model `intfloat/multilingual-e5-small` (384 chiều) và bộ chunk B. Nếu nạp một phiên bản index khác, lấy khóa `strategy` đầy đủ từ bảng `ingestion_chunks`. `DATABASE_URL` trong môi trường được ưu tiên hơn giá trị trong file.

Kết quả JSON gồm ID, nội dung đoạn, nguồn, trang PDF, metadata và điểm tương đồng cosine. Trong Python, import `DenseRetriever` từ `src.dense` để gọi `DenseRetriever(database_url, strategy).search(query, filters=None, top_k=3)`.

## Đo dense retrieval trên dev

Giữ hai biến PowerShell ở trên và chạy:

```powershell
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.retrieval_benchmark --env-file "$day6\.env" --strategy $strategy --output report/results/dense_dev.json
```

Script chạy 15 câu có đáp án từ `data/benchmark_dev.json`. Nó tìm ba đoạn cho mỗi câu, áp dụng filter chương nếu câu hỏi có chỉ rõ và chấm khớp nguyên văn bằng chứng theo nguồn và trang. JSON lưu điểm từng đoạn, hạng bằng chứng đầu tiên, context và thời gian truy xuất. Hai câu thiếu bằng chứng được dành để chấm hành vi từ chối của agent.

Chỉ số `packed_context_evidence` đo bằng chứng trong context ghép từ top 3 với ngân sách mặc định 800 ký tự, gồm cả nhãn nguồn. Context này dùng để mô phỏng đầu vào của model. Thời gian truy xuất mỗi câu gồm tạo vector và truy vấn database. Thời gian nạp model được lưu riêng.

## So sánh Dense, TF-IDF và Hybrid trên dev

```powershell
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.retrieval_benchmark --env-file "$day6\.env" --strategy $strategy --method all --output report/results/retrieval_dev.json
```

Cả ba phương pháp dùng cùng bộ chunk, câu hỏi gốc, bộ lọc chương khi có, top 3 và ngân sách context 800 ký tự. TF-IDF dùng cách tách từ của starter và tính trọng số trên toàn bộ corpus. Hybrid lấy tối đa 10 ứng viên từ mỗi phương pháp, rồi gộp bằng Reciprocal Rank Fusion với hằng số 60. Mỗi hạng `r` đóng góp `1 / (60 + r)` vào điểm của chunk. JSON ghi các hạng thành phần cho từng đoạn Hybrid trả về.

Thời gian truy xuất tính riêng cho từng phương pháp. Dense gồm embedding và truy vấn PostgreSQL, TF-IDF chạy trên chỉ mục trong bộ nhớ, Hybrid gồm cả hai lượt tìm và bước gộp hạng. Báo cáo ghi riêng thời gian tải model và tạo chỉ mục TF-IDF.

## Thử hai cấu hình cải thiện trên dev

```powershell
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.retrieval_benchmark --env-file "$day6\.env" --strategy $strategy --method dense --context-budget 1600 --output report/results/dense_context1600_dev.json
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.retrieval_benchmark --env-file "$day6\.env" --strategy $strategy --method dense-rerank --context-budget 800 --output report/results/dense_rerank_dev.json
```

- **Dense với 1.600 ký tự** giữ thứ hạng top 3 và thử giữ thêm các đoạn đã tìm được, đặc biệt ở Q04 và Q14.
- **Dense + rerank** giữ ngân sách 800 ký tự. Lấy tối đa 10 đoạn từ Dense và xếp lại bằng `0.8 × cosine_dense + 0.2 × cosine_tfidf`. Cả hai điểm là cosine. Đoạn không khớp từ khóa nhận điểm TF-IDF bằng 0. Giả thuyết là điểm TF-IDF giúp đẩy ứng viên có tên hàm hoặc thuật ngữ trùng với câu hỏi lên cao hơn. Trace ghi hạng Dense ban đầu và hai điểm thành phần cho top 3 sau xếp lại.

## Chạy agent với Ollama local

### Thử dịch câu hỏi sang tiếng Anh trước khi tìm

```powershell
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.retrieval_benchmark --env-file "$day6\.env" --strategy $strategy --method translation-ab --context-budget 1600 --output report/results/retrieval_translation_dev.json
```

So sánh trực tiếp câu tiếng Việt với bản dịch từ `llama3.2:latest` qua Ollama, trên 15 câu dev có đáp án. Ollama chỉ nhận câu hỏi. Dense nhận truy vấn và bộ lọc chương đã có. Cả hai lượt dùng cùng kho Day 6, top 3 và giới hạn 1.600 ký tự. JSON ghi câu gốc, câu đã dịch, bằng chứng và thời gian dịch riêng với thời gian tạo vector và tìm kiếm.

### Thử hỏi đáp có nguồn

Mở Ollama với model `llama3.2:latest` đã cài. Thử hai câu có đáp án và hai câu thiếu bằng chứng trên dev:

```powershell
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.agent --env-file "$day6\.env" --strategy $strategy --case Q01 --case Q04 --case D16 --case D17 --output report/results/agent_dev_smoke.json
```

Bỏ các tham số `--case` để chạy toàn bộ dev. Agent dùng API chat local của Ollama với hành động JSON. Lượt đầu tìm bằng câu hỏi gốc. Model có thể viết lại truy vấn cho lượt thứ hai. Mỗi lượt lấy 3 đoạn. Context ưu tiên kết quả lượt gần nhất và tối đa 1.600 ký tự theo mặc định. `--context-budget 800` chạy cấu hình baseline.

Trace JSON lưu hành động model, query, bộ lọc, kết quả, context, câu trả lời, citation, thời gian và lỗi. Chương trình kiểm tra ID citation có trong context và đoạn trích có trong nội dung của chunk được dẫn. Khi chấm, đối chiếu để xác nhận đoạn trích thực sự hỗ trợ câu trả lời.

Các lượt chạy mới ghi `ollama_usage` theo từng câu và theo thứ tự gọi model, gồm cả lượt dịch nếu bật. `prompt_tokens` là `prompt_eval_count` do Ollama trả về, bao gồm prompt hệ thống và phần câu hỏi, lịch sử truy vấn, context được gửi. `output_tokens` là `eval_count` của JSON do model sinh. Số này dùng cho lượt gọi model chat. Ngân sách context vẫn đo bằng ký tự, còn E5 và PostgreSQL được đo thời gian riêng.

Để đo ảnh hưởng của bản dịch lên câu trả lời cuối, chạy toàn bộ dev với cùng model và ngân sách context 1.600 ký tự:

```powershell
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.agent --env-file "$day6\.env" --strategy $strategy --translate-first-query --output report/results/agent_dev_translation.json
```

Lượt tìm đầu dùng bản dịch tiếng Anh. Model vẫn nhận câu hỏi tiếng Việt và có thể tự viết truy vấn cho lượt tìm thứ hai. Trace của lượt đầu ghi `original_query`, `query` thực tế và `translation_ms`.

Thêm `--numbered-citations` để model chọn `source_number` và một đoạn trích. Chương trình đổi số thành ID, tên tài liệu và trang PDF. Chế độ này phù hợp với các câu mà một chunk có thể cung cấp đủ bằng chứng.

Hai lệnh sau tái lập baseline và cấu hình đã chốt cho test. Dùng đường dẫn output mới để giữ lần đo đã chấm.

```powershell
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.agent --env-file "$day6\.env" --strategy $strategy --split test --context-budget 800 --output report/results/replay_test_baseline.json
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.agent --env-file "$day6\.env" --strategy $strategy --split test --context-budget 1600 --output report/results/replay_test_selected.json
```

Đọc và chấm lại trace nếu chạy lại model. Lệnh sau tổng hợp lượt baseline gốc với nhãn tương ứng:

```powershell
python -m src.agent_report --cases data/benchmark_test.json --run report/results/agent_test_baseline.json --reviews report/results/agent_test_baseline_reviews.json --output report/results/agent_test_baseline_summary.json
```

## Thử thách nâng cao (100 trang đa phương thức)

Phần thử thách nâng cao index toàn bộ 100 trang của báo cáo *GPT-4 Technical Report* (`arXiv:2303.08774v6`, gồm văn bản, bảng, biểu đồ vector/bitmap và ảnh/sơ đồ):

```powershell
$day6 = '..\K5-L3A-Day06-LeMinhPhung-2A202603693-DataIngestionEngineering\starter'
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.advanced_index
```

Lệnh tự động tải bản gốc PDF về `starter/data/gpt4-technical-report.pdf` (nếu chưa có), trích xuất các ảnh/biểu đồ crop vào `data/advanced/crops/`, xuất 221 chunks đa phương thức vào `data/advanced/multimodal_chunks.json`, lưu thống kê vào `data/advanced/index_manifest.json` và tạo vector 384 chiều từ `intfloat/multilingual-e5-small` tại `starter/data/advanced_embeddings.npz`.

So sánh 3 phương pháp retrieval (`baseline-text-dense`, `multimodal-dense`, `multimodal-hybrid`) trên tập dev nâng cao (18 câu):

```powershell
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.advanced_agent --split dev --mode compare-retrieval --output report/results/advanced_retrieval_dev.json
```

Chạy agent với `llama3.2:latest` qua Ollama cho cấu hình baseline và cấu hình đã chốt trên dev/test (ghi sang đường dẫn mới nếu muốn giữ nguyên các trace đã chấm):

```powershell
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.advanced_agent --split dev --mode baseline-text-dense --top-k 3 --context-budget 1600 --output report/results/replay_advanced_dev_baseline.json
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.advanced_agent --split dev --mode multimodal-hybrid --output report/results/replay_advanced_dev_selected.json
& "$day6\.venv\Scripts\python.exe" -X utf8 -m src.advanced_agent --split test --mode multimodal-hybrid --output report/results/replay_advanced_test_selected.json
```

Tổng hợp kết quả chấm đạt/không đạt theo từng nhóm của bộ test nâng cao (30 câu):

```powershell
python -X utf8 -m src.advanced_evaluate --cases data/advanced_benchmark_test.json --run report/results/advanced_test_selected.json --reviews report/results/advanced_test_selected_reviews.json --output report/results/advanced_test_selected_summary.json
```
