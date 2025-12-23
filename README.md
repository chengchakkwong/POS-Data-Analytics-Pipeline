📦 零售 POS 數據自動化分析與 BI 管道

Retail POS Data Automation & Business Intelligence Pipeline

📖 專案背景

在本專案中，我針對中型零售店（五金家品類）開發了一套全自動化的數據處理流程。該專案解決了傳統零售業常見的挑戰：數據孤島、手工報表耗時、以及萬用條碼（如雜項、五金件）導致的獲利分析失真。

透過 Python 進行 ETL（擷取、轉換、加載），並連接 Power BI 進行視覺化，實現了從原始 SQL 數據到商業決策洞察的閉環。

🚀 核心功能與商業邏輯

1. 高效數據同步 (ETL)

增量同步：利用 pyarrow 與 parquet 格式建立銷售快取，僅抓取變動數據，大幅減少資料庫壓力與讀取時間。

源頭清洗：自動修正 POS 系統匯出時常見的換行符 (\n, \r) 與空白字符導致的格式錯亂。

2. 精準商業洞察 (Business Insights)

ABC 產品分級：基於利潤貢獻度（Pareto Principle）自動將數千種商品劃分為 A、B、C 三類，輔助採購決策。

雜項成本估算邏輯：

挑戰：店內「五金雜項」使用萬用條碼 202320232023，系統中常無進貨成本（成本為 0），導致利潤虛高。

對策：自定義 AdjustedCost 邏輯，針對雜項自動回推 80% 保守成本（即預設 20% 毛利），確保 ABC 分析的真實性。

庫存健康監控：計算日均銷量與支撐天數（Days of Inventory），自動標記「低庫存預警」與「滯銷死貨」。

📁 檔案結構

pos_system_v2.py: 系統主入口，統籌數據同步與分析流程。

pos_service.py: 負責與 SQL Server 對接及數據增量同步邏輯。

business_insights.py: 核心商業邏輯模組，包含 ABC 分級與成本修正函式。

db_utils.py: 資料庫連線與環境變數管理封裝。

environment.yml: 專案環境設定檔，方便一鍵還原開發環境。

🛠️ 技術棧 (Tech Stack)

語言: Python 3.9+

數據處理: Pandas, NumPy, PyArrow

資料庫管理: SQLAlchemy, SQL Server (pyodbc)

視覺化: Power BI (Pareto Charts, Scatter Matrix, Slicers)

環境管理: Conda, Python-Dotenv

⚙️ 如何執行

複製專案:

git clone [https://github.com/chengchakkwong/POS-Data-Analytics-Pipeline.git](https://github.com/chengchakkwong/POS-Data-Analytics-Pipeline.git)


建立環境:

conda env create -f environment.yml
conda activate [your_env_name]


設定環境變數:
在根目錄建立 .env 檔案並填入你的資料庫連線資訊。

運行分析:

python pos_system_v2.py


🔒 隱私與安全聲明

為了保護商業機密，本專案：

已透過 .gitignore 排除所有真實交易數據 (data/)。

敏感連線資訊透過 .env 進行管理，不進入版本控制。

提供的程式碼範例已進行數據脫敏處理。

專案開發者: ChakKwong (Cheng Chak Kwong)

聯繫方式: chengchakkwong@gmail.com