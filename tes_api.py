import requests

url = "https://api.finmindtrade.com/api/v4/data"
parameter = {
    "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
    "data_id": "2330",
    "start_date": "2024-01-01",
    "end_date": "2024-01-10",
}

response = requests.get(url, params=parameter)
data = response.json()

print("API 回傳狀態碼:", response.status_code)
print("API 回傳訊息:", data.get("msg"))
print("資料內容長度:", len(data.get("data", [])))

# 印出前兩筆資料看看長怎樣，或是印出錯誤細節
if data.get("data"):
    print("第一筆資料範例:", data["data"][0])
else:
    print("完整回傳內容:", data)