from scipy.io import loadmat
import h5py
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
from multiprocessing import freeze_support
from Resnet import *
# 启用 matplotlib 交互模式（适配 PyCharm）
plt.ion()


# 训练、验证循环 & Early Stopping
def train_model(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    num_epochs = 20
    patience = 20

    # 初始化记录列表（新增训练准确率）
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    iterations, val_iterations = [], []
    best_val_loss = float('inf')
    no_improve = 0
    total_iter = 0  # 总迭代次数

    # 初始化图表（适配 PyCharm）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Training Progress')

    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # ---- 训练阶段 ----
        model.train()
        running_loss, running_correct = 0.0, 0
        for batch_idx, (Xb, yb) in enumerate(train_loader):
            Xb, yb = Xb.to(device), yb.to(device)

            # 前向传播
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录每个batch的数据（新增准确率）
            batch_acc = (preds.argmax(1) == yb).float().mean().item()
            train_accs.append(batch_acc)
            train_losses.append(loss.item())
            total_iter += 1
            iterations.append(total_iter)

            # 累加数据
            running_loss += loss.item() * Xb.size(0)
            running_correct += (preds.argmax(1) == yb).sum().item()

            # 每100个batch更新一次图表
            if batch_idx % 100 == 0:
                # 动态更新训练曲线
                ax1.clear()
                ax1.plot(iterations, train_losses, label='Train Loss', alpha=0.6)
                if val_iterations:
                    ax1.plot(val_iterations, val_losses, 'o-', label='Val Loss')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Loss')
                ax1.legend()

                ax2.clear()
                ax2.plot(iterations, train_accs, label='Train Acc', alpha=0.6, color='green')
                if val_iterations:
                    ax2.plot(val_iterations, val_accs, 'o-', label='Val Acc', color='orange')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Accuracy')
                ax2.legend()

                plt.pause(0.01)  # 允许图表更新

        # ---- 验证阶段 ----
        model.eval()
        val_running, val_correct = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                loss = criterion(preds, yb)
                val_running += loss.item() * Xb.size(0)
                val_correct += (preds.argmax(1) == yb).sum().item()

        # 记录验证数据
        val_iter = total_iter  # 当前总迭代次数
        val_loss = val_running / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_iterations.append(val_iter)

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_resnet18.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 打印进度（新增训练准确率）
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_correct / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs}  "
              f"Train Loss: {epoch_train_loss:.4f}  "
              f"Train Acc: {epoch_train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  "
              f"Val Acc: {val_acc:.4f}")

    # 训练结束保持图表显示
    plt.ioff()
    plt.show()
    return model



def main():
    # # 初始化模型，假设输入图像为单通道，分类数为26
    model = resnet18(num_classes=26)
    print(model)  #

    # 1) 读 .mat
    with h5py.File('X_MST_20db.mat', 'r') as f:
        X = f['X'][:]  # 一次性读完

    data = loadmat(r"D:\ProgramFiles\PyCharm\Py_Projects\PythonProject\Y.mat")
    print(data.keys())
    Y = data['Y']
    Y = Y - 1

    # 3) 打乱并按 60/20/20 划分
    N = X.shape[0]
    perm = np.random.permutation(N)
    n_train = int(0.6 * N)
    n_val = int(0.2 * N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    X_train = X[train_idx]
    Y_train = Y[train_idx].squeeze()  # 去掉 shape 中多余的 1，例如 (7800,1) -> (7800,)

    X_val = X[val_idx]
    Y_val = Y[val_idx].squeeze()

    X_test = X[test_idx]
    Y_test = Y[test_idx].squeeze()

    # 4) 标准化（z-score）
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # 5) 构造 DataLoader（同时把 numpy -> Tensor，并指定 dtype）
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).long()  # 一定要 long

    X_val_t = torch.from_numpy(X_val).float()
    Y_val_t = torch.from_numpy(Y_val).long()

    X_test_t = torch.from_numpy(X_test).float()
    Y_test_t = torch.from_numpy(Y_test).long()

    train_ds = TensorDataset(X_train_t, Y_train_t)
    val_ds = TensorDataset(X_val_t, Y_val_t)
    test_ds = TensorDataset(X_test_t, Y_test_t)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    # //////////////////////////////////////////////////////////////////////////////////
    # ---- 训练部分 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18().to(device)

    # 初始化 CUDA 事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 训练模型并记录时间
    start_event.record()
    model = train_model(model, train_loader, val_loader, device)  # 接收返回的模型
    end_event.record()
    torch.cuda.synchronize()
    gpu_time_ms = start_event.elapsed_time(end_event)
    print(f"Training GPU Time: {gpu_time_ms} ms")



    model.load_state_dict(torch.load('best_resnet18.pth'))
    model.eval()

    # 初始化计时器（需要 GPU）
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 记录 GPU 操作时间
    start_event.record()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            preds = model(Xb).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    test_acc = (y_pred == y_true).mean()
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    end_event.record()

    # 等待 GPU 完成所有操作
    torch.cuda.synchronize()
    gpu_time_ms = start_event.elapsed_time(end_event)
    print(f"test Time: {gpu_time_ms} ms")


    cm = confusion_matrix(y_true, y_pred)
    mask_diag = np.eye(len(cm), dtype=bool)  # 对角线掩膜

    plt.figure(figsize=(10, 8))

    # 第一步：用红色高亮非对角线（错误分类）
    sns.heatmap(
        cm,
        mask=mask_diag,  # 仅显示非对角线
        cmap='Reds',  # 错误分类用红色
        annot=True,
        fmt='d',
        annot_kws={'color': 'white', 'weight': 'bold'},  # 白字加粗
        cbar=False
    )

    # 第二步：用蓝色显示对角线（正确分类）
    sns.heatmap(
        cm,
        mask=~mask_diag,  # 仅显示对角线
        cmap='Blues',  # 正确分类用蓝色
        annot=True,
        fmt='d',
        annot_kws={'color': 'black'},  # 黑字
        cbar=False
    )

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    pass


if __name__ == '__main__':
    # 在 Windows 下如果你要打包 exe 可以加这行，
    # 不打包的时候也可以写出来以防万一
    freeze_support()
    main()