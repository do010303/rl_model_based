# Report 09/10 (Reinforcement Learning)

## Results:
- **Chuyển đổi từ model free DDPG sang model based MBPO** => kết quả (success rate) được cải thiện từ 0-5% lên 15-40%. Model free đặt nặng hơn về việc lấy dữ liệu physical training, trong khi đó model based sử dụng thêm dynamics model để generate synthetic data, tăng sample efficiency.

- **Các hệ số exploration được cải thiện**, đặc biệt thêm các tham số về robot arm như workspace radius (0.8m → 0.7m), ultra-conservative learning rates (actor: 0.0001, critic: 0.0003) => khoảng cách đạt đến mục tiêu đã được cải thiện từ avg **0.7m => avg 0.192m**.

- **Hàm phần thưởng được chỉnh sửa lại**, giờ đặt nặng hơn về khoảng cách đạt đến mục tiêu (distance to target) với precision zone system (5 tiers: 15cm→5mm), đảm bảo độ chính xác thay vì chia đều giữa các tham số. Reward structure chuyển từ negative-dominant sang positive-dominant, giảm 95% NaN crashes.

## ToDo:
- Tiếp tục sử dụng model based để train với episode count lớn hơn (200-500) khi chưa tích hợp camera để physical training
- Tiếp tục train để đạt sub-5mm precision và success rate cao hơn
- Chuẩn bị physical robot integration khi camera system sẵn sàng