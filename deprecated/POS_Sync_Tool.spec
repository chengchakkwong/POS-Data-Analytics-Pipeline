# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# --- 關鍵修改 1: 自動收集 Firebase Admin 的所有依賴 ---
# 這會自動抓取 grpc, google-cloud-firestore 等所有相關檔案
# 這是解決 "Certificate error" 或 "Module not found" 最強的方法
datas, binaries, hiddenimports = collect_all('firebase_admin')

# 定義你自己的隱藏模組
my_hidden_imports = [
    'pandas',
    'sqlalchemy',
    'pyodbc',
    'dotenv',
    'pyarrow',
    'firebase_admin',  # 明確加入 firebase_admin 避免漏抓
    'logger_config',
    'db_utils',
    'pos_service',
    'firebase_service',  # 記得加上這個新寫的模組！
]

# 將兩者合併
hiddenimports.extend(my_hidden_imports)

a = Analysis(
    ['POS_Sync_Tool.py'],           # 修改為你現在的主程式 main.py
    pathex=[],             # 如果 spec 檔跟 main.py 在同一層，這裡留空即可
    binaries=binaries,     # 加入 firebase 的二進位檔
    datas=datas,           # 加入 firebase 的數據檔
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='POS_Sync_Tool',  # 改個好聽的名字
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,              # 如果你有安裝 UPX 可以開 True，沒有就 False
    console=True,          # 設為 True，這樣在店鋪電腦跑時可以看到黑視窗報錯
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='POS_Sync_Tool',  # 這會是 dist 資料夾下的子資料夾名稱
)