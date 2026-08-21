# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Văn Linh  **MSSV**: 2A202601971  **Ngày**: 21/08/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4, 14.6 GB`

> Mọi con số dưới đây được lấy từ các artefact trong `results/`. Run dùng fp16 vì T4
> (sm_75) không có phần cứng bfloat16.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 theo recipe tier T4; p95 đo được là 98 và mức gợi ý là 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch / 30 optimizer step cho cả bốn run |

`max_length=1024` lớn hơn mức 256 suy ra từ p95. Tôi giữ cấu hình tier T4 để bốn run
dùng cùng một recipe và còn dư biên cho scaffold của chat template; tập hiện tại có
độ dài tối đa 101 token nên không mẫu nào bị cắt. Nếu tối ưu riêng cho corpus này, 256
là lựa chọn tiết kiệm activation memory hợp lý hơn.

**Template có giữ khối `<think>` không?** Có. `template_check.json` ghi nhận cả thẻ mở,
nội dung reasoning và verdict `reasoning preserved — safe to train on traces`. Tuy nhiên
corpus mặc định chỉ chứa câu trả lời JSON, nên `valid_trace_rate=0.0` ở NB5 không phải
bằng chứng riêng rằng reasoning bị collapse.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Ba dòng đầu của đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Mask không tính loss trên system prompt hay ticket của người dùng. Việc token kết thúc
`<|im_end|>` vẫn được giám sát cũng giúp model học đúng điểm dừng thay vì tiếp tục sinh
văn bản sau object JSON.

---

## 3. Ba baseline (NB2 đo a/b trước train; NB5 đo c sau train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3221.7 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1018.9 |
| (c) LoRA fine-tune | **0.965** | **0.4556** | 1.000 | 1401.9 |

**(b) có thật sự mạnh hơn (a) không?** Có. Prompt tối ưu tăng target từ 0.000 lên
0.765, tăng format từ 0.000 lên 1.000 và giảm latency khoảng 68.4%. Vì vậy đây là một
mốc mạnh và công bằng, không phải baseline yếu được dựng lên để fine-tune dễ thắng.

Tôi không sửa `OPTIMIZED_PROMPT`; SHA `719e74d3b6232053` khớp prompt được đóng băng ở
NB2. Fine-tune thắng (b) 0.200 target nhưng chậm hơn 383.0 ms/mẫu và làm regression
giảm 0.3022.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | **target (NB5 §4)** | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6260 | 0.965 | 929.2 | 12.01 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 1e-4 | **0.5369** | **0.970** | 809.4 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.000 | 947.7 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7074 | 0.940 | 1009.5 | **7.09** |

`attn_only` lệch 8,192 tham số so với `correct`, tương đương khoảng 0.025%, thấp hơn
nhiều so với ngưỡng công bằng 5%. Cả bốn run đều chạy 30 optimizer step.

**4.1 — Vị trí so với rank.** Trên target, `attn_only` thắng `correct` rất sát:
0.970 so với 0.965, tức chỉ một trường nhãn trên 200 trường được chấm. Thứ tự này giống
train loss vì `attn_only` cũng có loss thấp hơn, 0.5369 so với 0.6260; tuy nhiên mức
chênh target quá nhỏ nên hợp lý hơn khi xem hai cấu hình gần như hòa thay vì tuyên bố
q,v luôn tốt hơn. Với ngân sách tham số đã khớp, kết quả không ủng hộ giả thuyết rằng
all-linear chắc chắn là đòn bẩy trên corpus này: rank 283 tập trung vào q,v đã bù được
phạm vi module hẹp. Thí nghiệm chỉ cho kết luận trong recipe và dataset hiện tại, không
đủ để tách hoàn toàn tác động của rank khỏi placement ở mọi model.

**4.2 — Learning rate.** `wrong_lr` chỉ giảm LR từ 1e-4 xuống 1e-5 nhưng train loss
cuối tăng từ 0.6260 lên 1.5704 sau cùng 30 step, cho thấy quá trình học tiến chậm và chưa
đưa model vào vùng lời giải của tác vụ. Hậu quả ngoài tập train còn rõ hơn: target và
format đều bằng 0, latency tăng lên 5150.5 ms vì model không học được cách trả lời JSON
ngắn. Nếu chỉ thấy loss cao mà không biết LR, tôi có thể kết luận sai rằng dữ liệu, mask
hoặc LoRA thiếu năng lực; phép đối chứng cho thấy nguyên nhân trực tiếp là LR ở thang
full-fine-tuning không đủ lớn cho LoRA trong ngân sách 30 step.

**4.3 — QLoRA.** QLoRA giảm peak VRAM từ 12.01 xuống 7.09 GB, tiết kiệm 4.92 GB,
tương đương khoảng 41.0%. Đổi lại target giảm từ 0.965 xuống 0.940, train loss tăng từ
0.6260 lên 0.7074, latency tăng từ 1401.9 lên 1788.5 ms và thời gian train tăng khoảng
80 giây. QLoRA vẫn đạt format 1.000 và chất lượng target khá cao, nên không thể gọi nó
là thất bại; nhưng khi LoRA 16-bit đã vừa T4, số đo này ủng hộ khuyến nghị không lượng
tử hóa Qwen3.5 nếu ưu tiên chất lượng và tốc độ hơn phần VRAM tiết kiệm.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.200` · `regression Δ = -0.302` · `valid_trace_rate = 0.00`

Fine-tune đã học tác vụ rất rõ: target đạt 0.965, cao hơn baseline prompt tối ưu 0.200,
và mọi output đều đúng định dạng JSON. Tuy nhiên cổng hồi quy phải đánh trượt vì điểm
kiến thức/chỉ dẫn phổ thông giảm từ 0.7578 xuống 0.4556, tức giảm 0.3022, vượt xa mức
chịu đựng 0.020. Đây là catastrophic forgetting có ý nghĩa vận hành: một model triage
chính xác hơn nhưng suy giảm mạnh ở yêu cầu ngoài miền sẽ không an toàn nếu endpoint
có thể nhận câu hỏi hỗn hợp. `valid_trace_rate=0.0` không được dùng để khẳng định
reasoning collapse trong run này, vì dữ liệu train không có reasoning trace và generation
tắt thinking. Verdict FAILED không phủ nhận cải thiện target; nó nói rằng lợi ích đó
được mua bằng một chi phí năng lực tổng quát quá lớn. Bước tiếp theo hợp lý là trộn
1–5% replay data phổ thông, train lại cùng recipe rồi đo lại cả bốn nhóm trên đúng tập
eval đã đóng băng, thay vì nới lỏng regression gate.

---

## 6. Định tính — bắt buộc có cả ca thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Shipper không gọi; hỏi cho biết; shop hỗ trợ tốt | van_chuyen · thap · ốp lưng điện thoại · tich_cuc | Output từng mẫu không được NB2 lưu | Đúng 4/4 trường | ✅ FT đúng hoàn toàn |
| 2 | Hỏi giá; mong shop phản hồi | hoi_thong_tin · trung_binh · ốp lưng điện thoại · trung_tinh | Output từng mẫu không được NB2 lưu | Đúng 4/4 trường | ✅ FT đúng hoàn toàn |
| 3 | Chưa thấy tiền; khi nào tiện; cảm ơn shop | hoan_tien · thap · bình giữ nhiệt · tich_cuc | Output từng mẫu không được NB2 lưu | `urgency=trung_binh`; 3 trường còn lại đúng | ❌ **FT sai urgency** |
| 4 | Thiếu phụ kiện; khi nào tiện | san_pham_loi · thap · nồi chiên không dầu · trung_tinh | Output từng mẫu không được NB2 lưu | `urgency=trung_binh`; 3 trường còn lại đúng | ❌ **FT sai urgency** |
| 5 | Áo khoác bị lỗi; khi nào tiện; cảm ơn shop | san_pham_loi · thap · áo khoác gió · tich_cuc | Output từng mẫu không được NB2 lưu | `urgency=trung_binh`; 3 trường còn lại đúng | ❌ **FT sai urgency** |

NB2 chỉ lưu điểm tổng hợp của baseline (b), không lưu prediction theo từng mẫu, nên tôi
ghi rõ giới hạn artefact thay vì dựng lại câu trả lời không có bằng chứng. Năm trường hợp
trên lấy từ `qualitative.json` và nhãn gốc trong `eval_target.jsonl`. Mẫu chung của cả ba
ca fine-tune thua là câu có tín hiệu giảm khẩn cấp “khi nào tiện”; model vẫn dự đoán
`trung_binh` thay vì `thap`, trong khi intent, product và sentiment đều đúng. Như vậy
lỗi còn lại tập trung ở cách diễn giải sắc thái urgency, không phải schema hay intent.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi chưa nên deploy adapter này ở trạng thái hiện tại. Nếu chỉ nhìn target,
kết quả 0.965 so với 0.765 của prompt tối ưu rất thuyết phục, format cũng đạt 1.000 và
latency vẫn thấp hơn naive prompt. Nhưng regression giảm 0.3022 cho thấy model đã đánh
đổi quá nhiều năng lực phổ thông để ghi nhớ phân phối triage nhỏ gồm 225 mẫu train. Đây
không phải thất bại của việc học tác vụ mà là thất bại của recipe dữ liệu đối với yêu
cầu triển khai rộng. Trong các đòn bẩy đã đo, learning rate có ảnh hưởng rõ nhất:
giảm 10 lần làm target và format về 0. Mask là điều kiện nền tảng vì nếu prompt lọt vào
loss thì mọi kết quả sau mất ý nghĩa; run này chứng minh mask đúng. Placement không cho
thấy ưu thế chắc chắn của all-linear vì q,v với rank khớp ngân sách đạt target nhỉnh hơn
đúng 0.005. QLoRA mua được 41% VRAM nhưng giảm 0.025 target và tăng latency. Đòn bẩy
cần thử tiếp theo là chất lượng/phối trộn dữ liệu: thêm một lượng nhỏ replay data phổ
thông để bảo toàn regression, đồng thời bổ sung các ví dụ phân biệt “khi nào tiện” với
mức khẩn `trung_binh`. Chỉ khi adapter vẫn giữ target cao và regression nằm trong gate
trên tập eval đã đóng băng, tôi mới cân nhắc deploy.

**Ba điều tôi học được**:

1. Một prompt tốt là baseline bắt buộc: nó tăng target lên 0.765, đạt JSON tuyệt đối và nhanh hơn naive prompt khoảng 3.2 lần trước khi train.
2. Training loss không đủ để quyết định deploy; target tăng mạnh vẫn có thể đi cùng catastrophic forgetting nghiêm trọng.
3. Một phép đối chứng chỉ có ý nghĩa khi khớp ngân sách tham số và số step; cùng rank 16 cho q,v và all-linear sẽ không trả lời được câu hỏi về placement.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** thêm 1–5% replay data phổ thông và một nhóm mẫu
hard-negative về urgency, train lại đúng 30 step, sau đó đo lại target, regression,
format và latency trên tập eval hiện tại mà không sửa nhãn hay prompt.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
