import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# 读取CSV文件
input_file = "C:\\Users\\86150\\Desktop\\25 years.csv"  # 替换为你的CSV文件名
output_file = "C:\\Users\\86150\\Desktop\\25 years-mt.csv"  # 输出的Excel文件名
df = pd.read_csv(input_file)

# 初始化Selenium WebDriver
options = webdriver.ChromeOptions()
options.headless = True  # 设置为无头模式
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 爬虫函数
def fetch_earthquake_data(event_id):
    url = f'https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}/moment-tensor'
    driver.get(url)
    # 等待页面加载
    time.sleep(4)
    # 这里可以根据实际情况调整等待时间 # 获取页面源码
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # 提取 Nodal Planes 的数据
    nodal_planes = {}
    axes_data = {}
    rows = soup.find_all('tr')

    # 遍历每一行并提取数据
    for row in rows[1:]:  # 跳过表头行
        cells = row.find_all('td')
        if len(cells) == 4:
            axis = cells[0].get_text(strip=True)
            value = cells[1].get_text(strip=True)
            plunge = cells[2].get_text(strip=True)
            azimuth = cells[3].get_text(strip=True)

            # 存储数据
            axes_data[axis] = {
                'v': value,  # Value
                'p': plunge,  # Plunge
                'a': azimuth  # Azimuth
                }
        print(axes_data)
    # 提取 Nodal Planes 数据
    for row in soup.select('shared-nodal-planes usa-table tbody tr'):
        plane = row.find('td', {'data-label': 'Plane'}).text
        strike = row.find('td', {'data-label': 'Strike'}).text
        dip = row.find('td', {'data-label': 'Dip'}).text
        rake = row.find('td', {'data-label': 'Rake'}).text  # 将数据存储在字典中
        nodal_planes[plane] = {
            's': strike,
            'd': dip,
            'r': rake
        }

    # 提取 NP1 和 NP2 的信息
    np1 = nodal_planes.get('NP1', {})
    np2 = nodal_planes.get('NP2', {})

    # 返回 NP1 和 NP2 的数据以及 T、N、P 的数据
    return {
        "NP1_Strike": np1.get('s'),
        "NP1_Dip": np1.get('d'),
        "NP1_Rake": np1.get('r'),
        "NP2_Strike": np2.get('s'),
        "NP2_Dip": np2.get('d'),
        "NP2_Rake": np2.get('r'),
        "T_Value": axes_data.get('T', {}).get('v'),
        "T_Plunge": axes_data.get('T', {}).get('p'),
        "T_Azimuth": axes_data.get('T', {}).get('a'),
        "N_Value": axes_data.get('N', {}).get('v'),
        "N_Plunge": axes_data.get('N', {}).get('p'),
        "N_Azimuth": axes_data.get('N', {}).get('a'),
        "P_Value": axes_data.get('P', {}).get('v'),
        "P_Plunge": axes_data.get('P', {}).get('p'),
        "P_Azimuth": axes_data.get('P', {}).get('a')
    }

# 创建一个空的DataFrame来存储爬取的数据
results = []

# 遍历每一行，根据ID爬取数据
for idx, row in df.iterrows():
    event_id = row['id']
    data = fetch_earthquake_data(event_id)

    if data:
        results.append({
            "id": event_id,
            **data
        })

# 创建一个新的DataFrame并保存到csv文件
result_df = pd.DataFrame(results)
result_df.to_csv(output_file, index=False,encoding='utf-8-sig')
print("数据已成功爬取并保存到csv文件。")

# 退出WebDriver
driver.quit()