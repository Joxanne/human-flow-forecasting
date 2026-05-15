import pandas as pd
import os

def process_city_data(file_path, x0, y0, R=4, output_suffix="_patch"):
    """
    處理單一城市的資料，擷取特定區域的人流資料
    
    Args:
        file_path (str): 輸入CSV檔案路徑
        x0, y0 (int): 目標中心點座標
        R (int): 半徑 (預設4，生成9x9的區域)
        output_suffix (str): 輸出檔案的後綴
    """
    try:
        # 讀取原始資料
        df = pd.read_csv(file_path)
        print(f"\n處理檔案: {os.path.basename(file_path)}")
        print(f"原始資料大小: {df.shape}")

        # 篩選範圍
        df_subset = df[
            (df["x"] >= x0 - R) & (df["x"] <= x0 + R) &
            (df["y"] >= y0 - R) & (df["y"] <= y0 + R)
        ].reset_index(drop=True)

        print(f"篩選範圍: x=[{x0-R}, {x0+R}], y=[{y0-R}, {y0+R}]")
        print(f"篩選後資料大小: {df_subset.shape}")

        # 檢查是否有資料
        if df_subset.empty:
            print("警告: 指定範圍內沒有資料!")
            return

        # 生成輸出檔名
        output_name = os.path.splitext(file_path)[0] + output_suffix + ".csv"
        df_subset.to_csv(output_name, index=False)
        print(f"已儲存至: {os.path.basename(output_name)}")

    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {file_path}")
    except Exception as e:
        print(f"錯誤: 處理檔案時發生問題 - {str(e)}")

# 定義各城市的中心點
city_centers = {
    "A": {"file": "task1_dataset_kotae.csv", "x": 135, "y": 77},
    "B": {"file": "hiroshima_challengedata(B).csv", "x": 79, "y": 93},
    "C": {"file": "sapporo_challengedata(C).csv", "x": 24, "y": 151},
    "D": {"file": "kumamoto_challengedata(D).csv", "x": 101, "y": 103}
}

# 處理每個城市的資料
print("開始處理各城市資料...")
for city, info in city_centers.items():
    print(f"\n=== 處理 City {city} ===")
    process_city_data(
        file_path=info["file"],
        x0=info["x"],
        y0=info["y"]
    )

print("\n所有資料處理完成!")
