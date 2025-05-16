import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse, parse_qs, urlencode
import time

# 配置参数
OUTPUT_FILE = "C:\\Users\\86150\\Desktop\\event_links.txt"
INPUT_FILE = r"C:\Users\86150\Desktop\dataset\unfind.csv"
HEADLESS_MODE = True # 是否使用无头模式
RETRY_LIMIT = 3  # 失败重试次数

# 从这个事件 ID 开始，如果不想从特定事件开始，可以将其设置为 None
start_event_id = "us10002anx"  # 改为你希望开始的事件 ID（如果不想指定则设置为 None）

def process_img_url(original_url):
    """优化URL处理函数，确保时间格式的正确保留"""
    parsed = urlparse(original_url)
    query_params = parse_qs(parsed.query)

    # 设置输出参数和清除无用参数
    query_params["output"] = ["geocsv.tspair"]
    query_params.pop("width", None)
    query_params.pop("height", None)

    # 重新构造URL，但保留时间参数
    new_query = urlencode(query_params, doseq=True)
    # 替换 %3A 为 :
    new_query = new_query.replace("%3A", ":")
    return parsed._replace(query=new_query).geturl()

def cleanup_windows(driver):
    """窗口管理函数：关闭所有多余窗口"""
    while len(driver.window_handles) > 1:
        driver.switch_to.window(driver.window_handles[-1])
        driver.close()
    driver.switch_to.window(driver.window_handles[0])

def get_iris_event_id(url):
    """从IRIS链接中提取事件ID"""
    parts = url.split('/')
    for part in reversed(parts):
        if part.isdigit():
            return part
    raise ValueError("未找到有效的事件ID")

def initialize_driver():
    """初始化浏览器驱动"""
    options = webdriver.ChromeOptions()
    if HEADLESS_MODE:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def main():
    # 初始化浏览器
    driver = initialize_driver()

    # 读取数据
    df = pd.read_csv(INPUT_FILE)

    # 查找开始处理的索引
    if start_event_id:
        try:
            start_index = df[df['id'] == start_event_id].index[0]
        except IndexError:
            print(f"事件 ID {start_event_id} 未在数据中找到，开始从头处理。")
            start_index = 0
    else:
        start_index = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8", buffering=1) as output:
        for idx in range(start_index, len(df)):
            row = df.iloc[idx]
            event_id = row['id']
            print(f"\n{'=' * 40}\n处理事件: {event_id} ({idx + 1}/{len(df)})")

            for retry in range(RETRY_LIMIT):
                try:
                    # Step 1: 访问USGS页面
                    driver.get(f'https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}/waveforms')

                    # Step 2: 定位IRIS链接
                    iris_link = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.LINK_TEXT, "IRIS Seismic Waveform Data (Wilber 3)"))
                    )
                    print("成功定位IRIS链接")

                    # Step 3: 打开新标签页
                    iris_link.click()
                    WebDriverWait(driver, 10).until(lambda d: len(d.window_handles) == 2)
                    driver.switch_to.window(driver.window_handles[1])

                    try:
                        # Step 4: 获取IRIS事件ID
                        current_url = driver.current_url
                        iris_event_id = get_iris_event_id(current_url)
                        print(f"解析到IRIS事件ID: {iris_event_id}")

                        # Step 5: 定位事件链接
                        event_link = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/tools/event/')]"))
                        )
                        print("成功定位事件页面链接")

                        # Step 6: 打开事件页面
                        event_link.click()

                        # Step 7: 定位波形图
                        li_element = WebDriverWait(driver, 55).until(
                            EC.presence_of_element_located((By.ID, 'plots'))
                        )

                        # 查找其中的 img 标签
                        img_element = li_element.find_element(By.TAG_NAME, 'img')

                        # 获取img的src属性
                        img_src = img_element.get_attribute('src')
                        print(f"原始图片URL获取成功")

                        # Step 8: 处理URL
                        modified_url = process_img_url(img_src)
                        output.write(f"{event_id}: {modified_url}\n")
                        print(f"√ 成功保存 {event_id}")
                        break  # 成功时退出重试循环

                    except Exception as e:
                        print(f"! 第{retry + 1}次尝试失败: {str(e)}")
                        # 备用方案：直接构造URL访问
                        direct_url = f"https://ds.iris.edu/ds/nodes/dmc/tools/event/{iris_event_id}"
                        driver.get(direct_url)
                        # 重新尝试定位元素...

                except Exception as e:
                    print(f"!! 整体处理失败: {str(e)}")
                    if retry == RETRY_LIMIT - 1:
                        output.write(f"{event_id}: ERROR\n")
                finally:
                    cleanup_windows(driver)
                    time.sleep(2)  # 礼貌等待

    driver.quit()
    print("\n任务完成！结果保存至:", OUTPUT_FILE)

if __name__ == "__main__":
    main()