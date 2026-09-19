# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Nguyễn Tiến Đức (2A202602871)
**Nhóm:** Học bổng HUST — Trần Nguyễn Tiến Đức, Lê Nguyễn Quốc Bảo, Hoàng Anh Tài, Nguyễn Anh Dũng
**Ngày:** 2026-09-19

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai đoạn văn được mã hoá thành hai vector gần cùng hướng trên mặt cầu đơn vị. Mô hình xem chúng nói về cùng chủ đề / cùng ý, nên khi search chúng sẽ đứng gần nhau trên bảng xếp hạng.

**Ví dụ có độ tương tự CAO:**
- Câu A: Học bổng KKHT loại A yêu cầu GPA tối thiểu 3,6.
- Câu B: Sinh viên cần điểm rèn luyện học kỳ từ 90 trở lên để đạt học bổng xuất sắc.
- Tại sao tương đồng: cùng điều kiện xét học bổng KKHT loại xuất sắc (GPA + rèn luyện).

**Ví dụ có độ tương tự THẤP:**
- Câu A: GPA loại A phải đạt từ 3,6 trở lên.
- Câu B: Mức lương nhân viên LG Hải Phòng khoảng 450 đến 800 USD.
- Tại sao khác: một bên là tiêu chuẩn học bổng nhà trường, một bên là lương tuyển dụng doanh nghiệp — khác thực thể, khác đơn vị, khác đối tượng.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Embedding thường khác độ lớn (câu dài / mô hình khác scale). Cosine chỉ so góc, không bị độ dài vector chi phối; Euclid sẽ phạt vector dài dù cùng hướng. Công thức lab: `dot(a,b) / (||a|| * ||b||)`, vector 0 trả 0.0.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức lab: `số chunk = làm_tròn_lên((độ_dài - overlap) / (chunk_size - overlap))`
> `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.111…) = 23`
> *Đáp án:* **23 chunks** (chunk cuối chứa phần dư).

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25` — tăng 2 chunk. Overlap lớn hơn giữ câu/điều khoản nằm ở ranh giới, tránh cắt đôi một quy định GPA giữa hai chunk (đúng lỗi câu 1–2 lúc Recursive cắt theo `\n`).

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tách **sau** terminator bằng lookbehind cố định 2 ký tự: `(?<=\. )|(?<=! )|(?<=\? )|(?<=\.\n)` để không nuốt dấu câu. `strip()` từng câu, gom `max_sentences_per_chunk`, nối lại bằng một khoảng trắng. Text rỗng / không còn câu sau strip trả `[]`.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thử separator theo thứ tự `["\n\n", "\n", ". ", " ", ""]`. Base case: text rỗng → `[]`; `len <= chunk_size` → giữ nguyên; hết separator hoặc separator `""` → cắt cứng theo `chunk_size`. Các mảnh nhỏ được ghép lại (kèm separator) cho đến sát ngưỡng; mảnh vẫn dài thì đệ quy với phần separator còn lại.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `_make_record` copy metadata (không dùng chung object người gọi) và luôn gán `metadata["doc_id"]` nếu thiếu. Mỗi record lưu `id`, `content`, `embedding`, `metadata`. `search` và `search_with_filter` cùng gọi `_search_records`: cosine, sort giảm dần, bỏ `embedding` khỏi kết quả in ra.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Lọc metadata **trước**, rồi mới search trên tập còn lại — nếu lọc sau top-k, k slot có thể bị tài liệu sai chiếm hết. `delete_document` xoá mọi record có `id` hoặc `metadata["doc_id"]` khớp, trả `True` nếu có xoá.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> `store.search(question, top_k)` → đánh số `[1] … [k]` thành context → prompt bắt buộc trả lời **chỉ** từ context, thiếu thì nói rõ. `llm_fn` được inject (test dùng lambda; `bench.py` dùng extractive: có `must_contain` thì trả gold).

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts ==============================
platform darwin -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- .venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/duckk/Dev/ai-in-action-k4/K4-DAY07-TranNguyenTienDuc-2A202602871
plugins: anyio-4.15.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.02s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Đo bằng OpenAI `text-embedding-3-small` + `compute_similarity` (cosine). Ngưỡng dùng khi dự đoán: cao ≈ > 0.50, thấp ≈ < 0.35.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Học bổng KKHT loại A yêu cầu GPA tối thiểu 3,6. | Sinh viên cần điểm rèn luyện học kỳ từ 90 trở lên để đạt học bổng xuất sắc. | cao | 0.534 | Đúng hướng (cùng loại A, khác điều kiện GPA vs rèn luyện nên không cực cao) |
| 2 | Hạn nộp hồ sơ Trần Đại Nghĩa 16h30 ngày 09/10/2026 tại phòng 102 nhà C1. | Sinh viên đăng ký học bổng Trần Đại Nghĩa trên eHUST hoặc qldt.hust.edu.vn. | cao | 0.528 | Đúng — cùng quy trình, khác kênh nộp |
| 3 | Học bổng KKHT được xét theo GPA và điểm rèn luyện. | Học bổng MB The Best of MB Chasing trị giá 10 đến 30 triệu đồng. | thấp | 0.412 | **Sai** — tưởng thấp vì khác quỹ; model vẫn thấy chung chủ đề “học bổng” |
| 4 | GPA loại A phải đạt từ 3,6 trở lên. | Mức lương nhân viên LG Hải Phòng khoảng 450 đến 800 USD. | thấp | 0.298 | Đúng |
| 5 | Hội đồng xét cấp học bổng KKHT do Hiệu trưởng thành lập. | Chủ tịch Hội đồng là một thành viên Ban Giám hiệu, ủy viên thường trực là Trưởng phòng CTSV. | cao | 0.465 | Hơi cao thấp hơn dự đoán — cùng Hội đồng nhưng một câu thủ tục, một câu nhân sự |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp 3 bất ngờ nhất: KKHT (tiêu chuẩn GPA) và MB Chasing (tiền doanh nghiệp) vẫn 0.41 vì cùng cụm “học bổng”. Embedding bắt **đồng chủ đề từ vựng**, không bắt “đây là hai quỹ khác nhau”. Đó cũng là lý do câu 2 benchmark cần `audience=student`: không lọc thì `council-staff` (cũng nói KKHT) bám sát query “tiêu chuẩn xét cấp”.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

> Backend: **OpenAI `text-embedding-3-small`**. Chiến lược cá nhân: RecursiveChunker (`chunk_size=900`). Chi tiết: `ket_qua_benchmark.txt`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | GPA + rèn luyện KKHT loại A? | Criteria — tiêu đề + Điều 3 GPA 3,6 / RL 90, score 0.651 | 0.651 | Có, top-1 + span | Đúng (2/2) |
| 2 | Tiêu chuẩn xét KKHT? (filter student) | Cùng chunk Điều 3 (C 2,5 / B 3,2 / A 3,6), score 0.760 | 0.760 | Có, top-1 + span. A/B: không filter thì `council-staff` hạng 2 | Đúng (2/2) |
| 3 | Hạn nộp Trần Đại Nghĩa 2026.1? | Mục 4 — eHUST / phòng 102 C1 / 09/10/2026, score 0.616 | 0.616 | Có, top-1 + span | Đúng (2/2) |
| 4 | Số SV KKHT kỳ II 2025-2026 A/B/C? | 1.888 SV; 1.343 A / 456 B / 89 C, score 0.701 | 0.701 | Có, top-1 + span | Đúng (2/2) |
| 5 | MB Chasing trị giá + hạn? | Intro MB (thiếu ngày), score 0.754 | 0.754 | Có: file gold top-1; span `09/01/2026` **hạng 3** (`## Thời hạn`) | Đúng gold (1/2) |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5. Điểm Recursive: **9/10** (câu 5 đúng file nhưng ngày hạn nằm chunk riêng).

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Tăng Recursive `chunk_size` 500 → 900 để tiêu đề không tách khỏi bảng GPA — câu 1–2 từ 0–1đ lên 2đ. Heading gắn H1 vào từng mục: câu 5 lấy đúng `## Thời hạn` top-1 (2/2). FixedSize vẫn 8/10; Sentence 6/10 vì tách `10–30 triệu` khỏi ngày. A/B câu 2: không filter `council-staff` hạng 2; có `audience=student` thì staff biến mất.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 9 / 10 |
| **Tổng phần cá nhân** | **59 / 60** |
