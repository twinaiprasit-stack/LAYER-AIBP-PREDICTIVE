# AI Super Egg – Streamlit Viewer

UI สำหรับสำรวจผลพยากรณ์ราคาไข่ไก่จากโมเดลที่เทรนใน Google Colab (เช่น Prophet)  
รองรับการเลือกช่วงเวลาแบบ **Dynamic Range** + KPI (MAE / RMSE / MAPE) + กราฟสวยด้วย Plotly

## วิธีใช้งาน
1. เตรียมไฟล์จาก Colab:
   - `forecast.csv` : ต้องมีคอลัมน์อย่างน้อย `ds, yhat` และถ้ามี `yhat_lower, yhat_upper` จะวาดช่วงความเชื่อมั่นให้
   - (แนะนำ) `history.csv` หรือ `actual.csv` : ต้องมีคอลัมน์ `ds, y`
   - แอปรองรับการแม็พชื่อคอลัมน์อัตโนมัติ ถ้าใช้ `Date` จะถูกแปลงเป็น `ds`, ถ้าใช้ `Price/PriceMarket/yMarket` จะถูกแปลงเป็น `y`

2. รันบนเครื่อง:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
