import firebase_admin
from firebase_admin import credentials, firestore
import logging
import pandas as pd
import hashlib
import json
import os

# 設定 logger
logger = logging.getLogger(__name__)

class FirebaseManager:
    def __init__(self, key_path='serviceAccountKey.json', cache_file='data/sync_cache.json'):
        """初始化 Firebase 連線與快取機制"""
        self.cache_file = cache_file
        self.local_cache = self._load_cache()
        
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(key_path)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            logger.info("✅ Firebase Firestore 連線成功")
        except Exception as e:
            logger.error(f"❌ Firebase 初始化失敗: {e}")
            raise e

    def _load_cache(self):
        """讀取本地快取檔案"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """儲存快取檔案"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.local_cache, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"⚠️ 無法儲存快取: {e}")

    def _generate_hash(self, item):
        """
        為單筆商品資料生成指紋 (Hash)
        如果商品名稱、價格、庫存都沒變，Hash 就不會變
        """
        # 將關鍵欄位串接成字串
        unique_str = f"{item.get('ProductCode')}{item.get('CurrStock')}{item.get('RetailPrice')}{item.get('Name')}"
        return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

    def upload_stock_data(self, df_stock):
        """
        智慧上傳：只更新有變動的資料
        """
        if df_stock is None or df_stock.empty:
            logger.warning("⚠️ 沒有庫存資料需要上傳")
            return

        collection_name = 'products'
        logger.info(f"🚀 開始比對資料 (共 {len(df_stock)} 筆)...")

        # 資料前處理
        df_stock = df_stock.where(pd.notnull(df_stock), None)
        records = df_stock.to_dict(orient='records')

        batch = self.db.batch()
        batch_count = 0
        total_updated = 0
        skipped_count = 0
        
        # 暫存這次的新快取
        new_cache = self.local_cache.copy()

        for item in records:
            # 1. 取得 ID
            product_code = str(item.get('ProductCode', '')).strip()
            if not product_code:
                product_code = str(item.get('Barcode', '')).strip()
            if not product_code:
                continue

            # 2. 計算指紋
            current_hash = self._generate_hash(item)
            
            # 3. 比對快取：如果指紋一樣，代表資料沒變，跳過！
            if product_code in self.local_cache and self.local_cache[product_code] == current_hash:
                skipped_count += 1
                continue

            # 4. 如果不一樣，加入上傳排程
            doc_ref = self.db.collection(collection_name).document(product_code)
            batch.set(doc_ref, item, merge=True)
            
            # 更新快取
            new_cache[product_code] = current_hash
            
            batch_count += 1
            total_updated += 1

            if batch_count >= 400:
                batch.commit()
                logger.info(f"   ...已更新 {total_updated} 筆異動資料")
                batch = self.db.batch()
                batch_count = 0

        # 提交剩餘的
        if batch_count > 0:
            batch.commit()

        # 5. 儲存新的快取到硬碟
        self.local_cache = new_cache
        self._save_cache()
        
        logger.info(f"✨ 同步完成！")
        logger.info(f"   - 實際寫入: {total_updated} 筆 (消耗額度)")
        logger.info(f"   - 略過未變: {skipped_count} 筆 (節省額度)")