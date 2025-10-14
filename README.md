# 🥚 CPF War Room – AI Super Egg Forecast Dashboard

Dashboard สำหรับพยากรณ์ราคาไข่ไก่โดยใช้ผลลัพธ์จากโมเดล AI (Prophet / AutoML)  
นำข้อมูลจาก Google Colab มาแสดงผลอย่างสวยงามผ่าน Streamlit

## Features
- 🎛️ เลือกช่วงวันที่ (Dynamic Range)
- 📊 แสดงกราฟ Actual vs Forecast
- 🧮 KPI Metrics (MAE / RMSE / MAPE)
- 🧭 Theme สี War Room CPF (#76d9c1 / #f8cf40 / #cfd3da)
- 🏢 โลโก้ CPF มุมขวาบน

## วิธีใช้งาน
1. อัปโหลดไฟล์ผลลัพธ์จาก Colab:
   - `forecast.csv` → ต้องมี `ds`, `yhat`
   - `actual.csv` → ต้องมี `ds`, `y`
2. รันแอป:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
