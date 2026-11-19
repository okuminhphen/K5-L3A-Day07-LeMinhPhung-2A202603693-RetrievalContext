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

### 1. Tài liệu, phạm vi index và xử lý đa phương thức

- **Tài liệu và nguồn gốc:** *GPT-4 Technical Report* (kèm Phụ lục G: *GPT-4 System Card*), OpenAI, phiên bản `arXiv:2303.08774v6` (cập nhật 04/03/2024), gồm đúng **100 trang PDF vật lý** (trang 1–40 là báo cáo chính và Phụ lục A–F; trang 41–100 là *GPT-4 System Card*). Tài liệu phân phối công khai theo giấy phép không độc quyền vĩnh viễn của arXiv.org. Script tự động tải bản gốc và đối chiếu mã băm SHA-256. [Bản gốc trên arXiv](https://arxiv.org/pdf/2303.08774v6.pdf) · [Manifest và thống kê index](../../data/advanced/index_manifest.json) · [Bộ chunks đa phương thức](../../data/advanced/multimodal_chunks.json)
- **Thống kê index toàn bộ 100 trang:** Toàn bộ 100/100 trang được xử lý thành **221 chunks** và nhúng bằng `intfloat/multilingual-e5-small` (384 chiều, chuẩn hóa L2, thời gian tạo index và nhúng toàn bộ 100 trang là 56,8 giây):
  - **Text (`194` chunks):** Chia theo đoạn có chồng lấn (`max_chars = 2.400`, `overlap = 200`), giữ liên kết `page` (`1..100`) và tọa độ vùng `bbox` (`[x0, y0, x1, y1]` trên khung trang chuẩn `612 × 792 pt`) trích xuất qua `visitor_text` của `pypdf`.
  - **Bảng (`7` vùng bảng lớn):** Chuẩn hóa cấu trúc hàng, cột, đơn vị đo (`%`, `/ 400`, `/ 800`, `/ 170`, `F1 score`, `#questions`) và chú thích cho các bảng khảo sát kỳ thi và nhiễm dữ liệu (`Table 1`, `Table 2`, `Table 4`, `Table 8`, `Table 9`, `Table 10`, `Table 11`).
  - **Ảnh và sơ đồ (`9` vùng trực quan):** Trích xuất trực tiếp các đối tượng ảnh XObject ra thư mục [data/advanced/crops/](../../data/advanced/crops/) kèm kích thước pixel, tọa độ `bbox` và nội dung OCR/mô tả cấu trúc hóa đã đối chiếu với ảnh gốc (gồm bộ chuyển đổi VGA–Lightning `ELECOM P-APLTDCN` ở `Table 3 / Im6.jpg` trang 9, sơ đồ vật lý bolomètre `Table 15 / Im13.jpg` trang 35, ảnh ủi đồ trên xe taxi `Table 16 / Im14.jpg` trang 36, trang đầu và sơ đồ 3 bước SFT–RM–PPO của bài báo InstructGPT ở `Table 17 / Im15.jpg, Im17.jpg` trang 37, meme bản đồ gà rán `Table 18 / Im18.png` trang 38, tranh biếm họa Statistical Learning vs. Neural Networks `Table 19 / Im19.jpg` trang 39, sơ đồ ReAct 6 công cụ hóa học `System Card Figure 5` trang 57 và sơ đồ phân loại từ chối RBRM `System Card Figure 6` trang 63).
  - **Biểu đồ (`11` vùng biểu đồ vector và bitmap):** Lưu trục X/Y, đơn vị, chú giải (legend), xu hướng và các giá trị số xác thực từ đồ thị (gồm đường cong scaling `Figure 1` & `Figure 2` trang 3, đường cong inverse scaling `Figure 3` trang 4, biểu đồ phân vị kỳ thi `Figure 4` trang 6, biểu đồ MMLU 26 ngôn ngữ `Figure 5` trang 8, biểu đồ thực tế nội bộ 9 danh mục `Figure 6` trang 10, biểu đồ TruthfulQA `Figure 7` trang 11, đồ thị hiệu chuẩn ECE `Figure 8` trang 12, biểu đồ tiêu thụ thịt FAO 1997 `Table 14 / Im12.png` trang 34, biểu đồ win-rate theo kích thước mô hình `Table 17 / Im16.jpg` trang 37 và ma trận nhiệt Instruction Following `System Card Figure 11 / Im13.png` trang 98).
- **Những trang/phần đặc biệt hoặc hạn chế xử lý và ảnh hưởng:**
  1. *Trang 41–100 (60 trang System Card):* Chứa chữ xoay dọc ở lề khiến chế độ `extraction_mode="layout"` của `pypdf` báo `Rotated text discovered` và trả về `0` ký tự. Pipeline tự động phát hiện trang có `< 50` ký tự ở chế độ `layout` để chuyển sang `plain` (`page.extract_text()`), khôi phục đầy đủ văn bản của cả 60 trang.
  2. *Trang 79:* Chỉ là trang bìa ngăn cách phụ lục (`"Appendix 79"`, 11 ký tự, không có bảng hay hình), vẫn được lưu thành 1 chunk tối thiểu để bảo toàn liên kết đủ 100 trang.
  3. *Các biểu đồ vector không in nhãn số trên từng cột lẻ (`Figure 1, 2, 3, 4, 6, 7, 9`):* Các cột/đường được vẽ bằng vector path trong PDF mà không gắn text số cho từng cột trung gian (khác với `Figure 5` trang 8, `Figure 8` trang 12 có nhãn số trong luồng text, hoặc `Im12.png` trang 34 và `Im13.png` trang 98 là ảnh bitmap độ phân giải cao). Pipeline chỉ lưu trục, đơn vị, chú giải, thứ tự xu hướng và số liệu tổng hợp có căn cứ, không suy đoán chiều cao cột.

### 2. Thiết kế benchmark và lựa chọn cấu hình trên Dev

- **Phân chia bộ câu hỏi:** Chia độc lập trước khi chạy thực nghiệm, không trùng câu hỏi hay đáp án giữa dev và test. Mỗi câu chỉ tính vào **1 nhóm chính** (`primary_group`) và gắn kèm tọa độ vùng bằng chứng (`page`, `region_id`, `modality`, `bbox`, `crop_path`), đơn vị, điều kiện và phép tính:
  - **Dev ([advanced_benchmark_dev.json](../../data/advanced_benchmark_dev.json)):** **18 câu** gồm **15 câu có đáp án** (`4` Bảng, `3` Ảnh/sơ đồ, `3` Biểu đồ, `2` Multi-hop, `3` Text) và **3 câu thiếu bằng chứng**.
  - **Test ([advanced_benchmark_test.json](../../data/advanced_benchmark_test.json)):** **30 câu** gồm **24 câu có đáp án** (`6` Bảng, `5` Ảnh/sơ đồ, `5` Biểu đồ, `4` Multi-hop, `4` Text) và **6 câu thiếu bằng chứng**.
- **So sánh các cấu hình trên tập Dev (18 câu):**

| Cấu hình trên Dev | Phương pháp retrieval & context | Hit@1 (15 câu) | Evidence Hit@3 (15 câu) | Context đủ evidence (15 câu) | Tỷ lệ đạt Agent trên Dev (18 câu) | Thời gian trung vị |
|---|---|---:|---:|---:|---:|---:|
| Baseline (`baseline-text-dense`) | Chỉ tìm trên text chunks bằng Dense E5, `top_k = 3`, ngân sách `1.600` ký tự | 4/15 (26,7%) | 5/15 (33,3%) | 2/15 (13,3%) | 4/18 (22,2%) | 1,94 giây/câu |
| Cải thiện 1 (`multimodal-dense`) | Index đa phương thức (text + bảng + biểu đồ + ảnh/sơ đồ), Dense E5, `top_k = 4`, ngân sách `2.800` ký tự | 10/15 (66,7%) | 14/15 (93,3%) | 9/15 (60,0%) | — (đo tầng retrieval) | 25,5 ms truy xuất |
| Cấu hình chốt (`multimodal-hybrid`) | Hybrid RRF (Dense E5 + TF-IDF) + mở rộng thuật ngữ Việt–Anh + điều hướng vùng/trang + tách truy vấn multi-hop, `top_k = 5`, ngân sách `6.000–8.000` ký tự | 14/15 (93,3%) | 15/15 (100%) | 15/15 (100%) | **18/18 (100%)** | 4,48 giây/câu |

- **Phân tích lỗi trên Dev và cải tiến trước khi chốt:**
  1. *Lỗi cắt cụt JSON ở 256 token (`AD07`, `AD13`, `AD18`):* Ở lần chạy thử đầu tiên trên Dev, giới hạn `num_predict = 256` làm câu trả lời tiếng Việt bị ngắt giữa chuỗi JSON (`Unterminated string`) hoặc lỗi ký tự thoát LaTeX (`\lambda`). Tăng `num_predict = 384` kèm `repeat_penalty = 1.15` và bộ sửa lỗi thoát chuỗi giúp loại bỏ hoàn toàn lỗi JSON (`0` lỗi).
  2. *Lỗi nhầm từ khóa `"mô hình"` với `"Hình"` ở câu multi-hop `AD12`:* Biểu thức kiểm tra chuỗi con `"hình" in query` ban đầu khớp nhầm từ `"mô hình"`, đẩy biểu đồ `Figure 8` (trang 12) lên trên đoạn văn bản nêu `"over 50 experts"` ở Mục 6 trang 12. Đổi sang khớp ranh giới từ `\b(?:biểu đồ|sơ đồ|tranh|ảnh|hình\s+\d+)\b` và trộn vòng tròn (round-robin) kết quả giữa các bước tìm kiếm multi-hop giúp Context Evidence Rate trên Dev đạt **15/15 (100%)**.
  3. *Lỗi khớp tiền tố tên bảng (`Table 1` vs. `Table 11` ở `AD01`):* Kiểm tra chuỗi con `"table 1" in region_id` khớp nhầm sang `Table 11` (trang 32). Bổ sung ranh giới từ `\bTable 1\b` và giới hạn theo trang được nêu trong câu hỏi giúp citation chính xác tuyệt đối.
- [Kết quả so sánh retrieval Dev](../results/advanced_retrieval_dev.json) · [Trace Dev baseline](../results/advanced_dev_baseline.json) · [Tổng hợp Dev baseline](../results/advanced_dev_baseline_summary.json) · [Trace Dev cấu hình chốt](../results/advanced_dev_selected.json) · [Nhãn Dev cấu hình chốt](../results/advanced_dev_selected_reviews.json) · [Tổng hợp Dev cấu hình chốt](../results/advanced_dev_selected_summary.json)

### 3. Kết quả đánh giá độc lập trên tập Test (30 câu)

Chạy pipeline `multimodal-hybrid` đã chốt với model `llama3.2:latest` (Ollama local) trên toàn bộ **30 câu hỏi tập Test**:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---:|---:|---|
| Bảng | 5/6 | 83,33% | Đạt `AT01`, `AT03`, `AT04`, `AT05`, `AT06`. Không đạt `AT02` (model tính đúng tổng 55/166 của GPT-4 trên Leetcode nhưng cộng lệch tổng phụ của GPT-4 no-vision thành 52/166 và thiếu tỷ số `55 / 20 = 2,75` lần). |
| Ảnh, sơ đồ | 5/5 | 100,00% | Đạt cả 5 câu (`AT07` sơ đồ bolomètre Pháp trang 35, `AT08` mã arXiv & email trong ảnh bài báo InstructGPT trang 37, `AT09` sơ đồ 3 bước SFT–RM–PPO trang 37, `AT10` tranh biếm họa VC-Dimension vs. `LAYERS` trang 39, `AT11` sơ đồ 6 công cụ & thuốc `Dasatinib`/`AZD0530` trang 57). |
| Biểu đồ | 5/5 | 100,00% | Đạt cả 5 câu (`AT12` biểu đồ MMLU 26 ngôn ngữ trang 8, `AT13` đồ thị hiệu chuẩn ECE `0,007` vs. `0,074` trang 12, `AT14` biểu đồ thịt FAO 1997 `175,09 g - 165 g = 10,09 g` trang 34, `AT15` biểu đồ win-rate InstructGPT trang 37, `AT16` ma trận nhiệt IF `Figure 11` trang 98). |
| Multi-hop | 3/4 | 75,00% | Đạt `AT18` (trang 6–7 & 32), `AT19` (trang 63 & 80), `AT20` (trang 64 & 98). Không đạt `AT17` (câu 3-hop qua 3 bảng ở trang 5, 30, 31: agent dẫn được `Table 1` trang 5 và `Table 9` trang 30 nhưng thiếu `Table 10` trang 31). |
| Text | 4/4 | 100,00% | Đạt cả 4 câu (`AT21` lý do loại BIG-bench trang 6, `AT22` thẻ `IMAGE:` và 6 kỳ thi không có ảnh trang 25, `AT23` giải thích lỗ hổng vs. xây dựng exploit trang 54, `AT24` phân loại sinh vật hư cấu có trí khôn `sapient` trang 67). |
| Thiếu bằng chứng | 6/6 | 100,00% | Từ chối đúng cả 6 câu (`AT25` bảng giá API 8K/32K, `AT26` kiến trúc ViT/CLIP & độ phân giải ảnh, `AT27` điểm thi USMLE Step 1/2/3, `AT28` điểm MMLU tiếng Việt trong Hình 5, `AT29` tiêu thụ thịt của Pháp & Đức trong Bảng 14, `AT30` tỷ lệ thắng trước Claude-1 & PaLM-2 trong Hình 11). |
| **Toàn bộ** | **28/30** | **93,33%** | **Vượt ngưỡng 90% (27/30) của thử thách nâng cao; 0 lỗi hệ thống.** |

- **Thời gian, chi phí và tài nguyên trên tập Test (30 câu):**
  - Số lỗi hệ thống / timeout: **0/30**.
  - Thời gian chạy toàn luồng: trung vị **4,88 giây/câu**, tổng cộng **206,86 giây** cho 30 câu.
  - Lượng token tiêu thụ qua Ollama (`llama3.2:latest`): **42.248** prompt tokens, **5.431** output tokens, tổng cộng **47.679** tokens. Chi phí API: **0 USD**.
- **Đường dẫn bằng chứng và kiểm chứng:**
  - [Trace đầy đủ 30 câu Test](../results/advanced_test_selected.json)
  - [Nhãn chấm từng câu và đối chiếu bản gốc](../results/advanced_test_selected_reviews.json)
  - [Tổng hợp kết quả Test theo nhóm](../results/advanced_test_selected_summary.json)
- **Giới hạn và cách tái lập:**
  - Đối với các câu hỏi multi-hop từ 3 bảng trở lên trên 3 trang khác nhau (`AT17`), giới hạn chọn tối đa 3 nguồn ưu tiên khiến chunk văn bản cùng trang (`Page 30 Text 1`) có thể chiếm chỗ của bảng thứ ba (`Table 10` trang 31). Ngoài ra, model 3B local (`llama3.2:latest`) đôi khi cộng nhầm các phân số khi bảng có nhiều hàng cùng định dạng phân số (`AT02`).
  - Lệnh tái lập toàn bộ quá trình xây dựng index 100 trang, chạy so sánh retrieval, chạy agent và chấm điểm được trình bày tại [src/README.md](../../src/README.md#thử-thách-nâng-cao-100-trang-đa-phương-thức).
