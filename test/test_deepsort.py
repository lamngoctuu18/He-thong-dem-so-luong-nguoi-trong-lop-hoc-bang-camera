from deep_sort_realtime.deepsort_tracker import DeepSort
import traceback

print("🔄 Đang kiểm tra DeepSORT...")

try:
    tracker = DeepSort(max_age=50, n_init=2)
    print("✅ DeepSORT hoạt động bình thường!")
except Exception as e:
    print("❌ Lỗi khi chạy DeepSORT:")
    traceback.print_exc()
