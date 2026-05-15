import pandas as pd
import os

def find_top_pois(file_path, city_name):
    """分析單一城市的 POI 資料檔案"""
    print(f"\n=== {city_name} 的 POI 分析結果 ({os.path.basename(file_path)}) ===")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {file_path}")
        print("=" * 50)
        return

    # POI 類別對照表
    category_names = {
        56: "School",
        48: "Transit Station",
        1: "Food"
    }

    # 1. 找出 POI 總數最多的位置
    total_poi_per_location = df.groupby(['x', 'y'])['POI_count'].sum()
    top_total_poi_loc = total_poi_per_location.idxmax()
    top_total_poi_count = total_poi_per_location.max()
    
    print("\n📍 POI 總數最多的位置:")
    print(f"   座標: (x={top_total_poi_loc[0]}, y={top_total_poi_loc[1]})")
    print(f"   總數: {int(top_total_poi_count)} 個 POI")

    # 2. 找出特定類別中 POI 數量最多的位置
    categories_to_find = [56, 48, 1]
    print("\n📊 各類別 POI 分布:")
    
    for category in categories_to_find:
        df_category = df[df['category'] == category]
        category_name = category_names.get(category, f"Category {category}")
        
        if not df_category.empty:
            top_loc = df_category.loc[df_category['POI_count'].idxmax()]
            x = int(top_loc['x'])
            y = int(top_loc['y'])
            count = int(top_loc['POI_count'])
            print(f"\n   🔹 {category_name} (類別 {category})")
            print(f"      座標: (x={x}, y={y})")
            print(f"      數量: {count} 個")
        else:
            print(f"\n   ❌ {category_name} (類別 {category}): 無資料")
            
    print("\n" + "=" * 50)


# --- 主程式 ---
poi_dir = './poi_data'  # 將 POI CSV 檔案放置於此目錄
files = {
    'City A': os.path.join(poi_dir, 'POIdata_cityA.csv'),
    'City B': os.path.join(poi_dir, 'POIdata_cityB.csv'),
    'City C': os.path.join(poi_dir, 'POIdata_cityC.csv'),
    'City D': os.path.join(poi_dir, 'POIdata_cityD.csv')
}

for city, path in files.items():
    find_top_pois(path, city)