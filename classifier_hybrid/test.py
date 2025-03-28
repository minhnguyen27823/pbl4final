import torch
from net import classifier
def predict(model, x_aff, x_gait, device='cuda'):
    """
    Sử dụng mô hình để dự đoán đầu ra từ dữ liệu đầu vào.

    Args:
        model (Classifier): Mô hình đã được huấn luyện.
        x_aff (torch.Tensor): Đặc trưng affective, dạng (N, in_features).
        x_gait (torch.Tensor): Dữ liệu khung xương, dạng (N, C, T, V, M).
        device (str): Thiết bị sử dụng ('cuda' hoặc 'cpu').

    Returns:
        torch.Tensor: Kết quả dự đoán dạng xác suất cho từng lớp.
    """
    # Đưa model về chế độ đánh giá
    model.eval()

    # Chuyển dữ liệu sang thiết bị
    x_aff = x_aff.to(device)
    x_gait = x_gait.to(device)

    with torch.no_grad():
        output = model(x_aff, x_gait)
        probabilities = torch.softmax(output, dim=1)  # Chuẩn hóa thành xác suất

    return probabilities

# Khởi tạo mô hình
model = classifier.Classifier(in_channels=3, in_features=10, num_classes=5, graph_args={'layout': 'openpose', 'strategy': 'spatial'})

# Chuyển mô hình sang thiết bị (nếu có GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Tạo dữ liệu giả lập (batch size = 1)
x_aff = torch.randn(1, 10)  # Affective features
x_gait = torch.randn(1, 3, 75, 18, 2)  # Skeleton sequence (N, C, T, V, M)

# Dự đoán
predictions = predict(model, x_aff, x_gait, device)
print(predictions)
