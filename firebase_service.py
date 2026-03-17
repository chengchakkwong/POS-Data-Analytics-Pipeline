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
        使用「全部欄位」偵測變動，但忽略 AvgCost：
        - 任何欄位（除了 AvgCost）只要有變化，就會觸發重新上傳
        - 以欄位名稱排序後組合，確保順序穩定
        """
        if not isinstance(item, dict):
            return ""

        parts = []
        for key in sorted(item.keys()):
            if key == "AvgCost":
                continue
            parts.append(f"{key}={item.get(key)}")

        unique_str = "|".join(parts)
        return hashlib.md5(unique_str.encode("utf-8")).hexdigest()

    def _generate_classification_hash(self, item):
        """
        為分類資料生成指紋 (Hash)
        """
        unique_str = f"{item.get('ProductCode')}{item.get('ABC_Class')}{item.get('XYZ_Class')}{item.get('note')}"
        return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

    def _generate_replenishment_hash(self, item):
        """
        為補貨建議資料生成指紋，用於增量上傳快取。
        規則與 _generate_hash 一致：
        - 使用該筆補貨紀錄的「全部欄位」偵測變動
        - 唯一例外：忽略 AvgCost，不讓 AvgCost 影響是否重傳
        """
        if not isinstance(item, dict):
            return ""

        parts = []
        for key in sorted(item.keys()):
            if key == "AvgCost":
                continue
            parts.append(f"{key}={item.get(key)}")

        unique_str = "|".join(parts)
        return hashlib.md5(unique_str.encode("utf-8")).hexdigest()

    def _generate_min_multiple_hash(self, item):
        """為 guessed_min / guessed_multiple 生成指紋，用於增量上傳快取"""
        unique_str = (
            f"{item.get('ProductCode', '')}"
            f"{item.get('guessed_min')}"
            f"{item.get('guessed_multiple')}"
        )
        return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

    def upload_guessed_min_multiple(self, df):
        """
        僅將 guessed_min、guessed_multiple 寫入 Firestore replenishment collection。
        使用 cache key repl_min_mult:{ProductCode}，只更新有變動的 document；merge=True 不覆蓋其他欄位。
        """
        if df is None or df.empty:
            logger.warning("⚠️ 沒有 Min/Multiple 資料需要上傳")
            return

        collection_name = 'replenishment'
        required = ['ProductCode', 'guessed_min', 'guessed_multiple']
        if not all(c in df.columns for c in required):
            logger.warning(f"⚠️ 缺少欄位 {required}，跳過 guessed_min/guessed_multiple 上傳")
            return

        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient='records')

        batch = self.db.batch()
        batch_count = 0
        total_updated = 0
        skipped_count = 0
        new_cache = self.local_cache.copy()

        for item in records:
            product_code = str(item.get('ProductCode', '')).strip()
            if not product_code:
                continue

            cache_key = f"repl_min_mult:{product_code}"
            current_hash = self._generate_min_multiple_hash(item)

            if cache_key in self.local_cache and self.local_cache[cache_key] == current_hash:
                skipped_count += 1
                continue

            payload = {
                'guessed_min': item.get('guessed_min'),
                'guessed_multiple': item.get('guessed_multiple'),
            }
            doc_ref = self.db.collection(collection_name).document(product_code)
            batch.set(doc_ref, payload, merge=True)

            new_cache[cache_key] = current_hash
            batch_count += 1
            total_updated += 1

            if batch_count >= 400:
                batch.commit()
                logger.info(f"   ...已更新 {total_updated} 筆 guessed_min/guessed_multiple")
                batch = self.db.batch()
                batch_count = 0

        if batch_count > 0:
            batch.commit()

        self.local_cache = new_cache
        self._save_cache()

        logger.info("✨ guessed_min / guessed_multiple 同步完成！")
        logger.info(f"   - 實際寫入: {total_updated} 筆 (消耗額度)")
        logger.info(f"   - 略過未變: {skipped_count} 筆 (節省額度)")

    def _safe_doc_id(self, value):
        """Firestore document ID 不可含 / \\ 等字元，轉成底線。"""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        s = str(value).strip()
        for c in r'/\:*?"<>|':
            s = s.replace(c, '_')
        return s or ""

    def upload_supplier_data(self, df):
        """
        將供應商名單上傳至 Firestore collection「supplier」。
        document ID 依序採用：ID 欄位、SID 欄位、否則用 Name（會做安全字元替換）。
        """
        if df is None or df.empty:
            logger.warning("⚠️ 沒有供應商資料需要上傳")
            return

        collection_name = 'supplier'
        logger.info(f"🚀 開始上傳供應商資料至 {collection_name} (共 {len(df)} 筆)...")

        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient='records')
        batch = self.db.batch()
        batch_count = 0

        for i, item in enumerate(records):
            doc_id = (
                self._safe_doc_id(item.get('ID'))
                or self._safe_doc_id(item.get('SID'))
                or self._safe_doc_id(item.get('Name'))
                or f"row_{i}"
            )
            if not doc_id:
                continue
            doc_ref = self.db.collection(collection_name).document(doc_id)
            batch.set(doc_ref, item, merge=True)
            batch_count += 1
            if batch_count >= 400:
                batch.commit()
                logger.info(f"   ...已寫入 {batch_count} 筆")
                batch = self.db.batch()
                batch_count = 0

        if batch_count > 0:
            batch.commit()
        logger.info(f"✨ 供應商資料已同步至 Firestore {collection_name}，共 {len(records)} 筆")

    def upload_replenishment_data(self, df):
        """
        上傳補貨建議資料到 Firestore replenishment collection。
        只更新有變動的 document，使用 cache key repl:{ProductCode}。
        merge=True：僅更新本次提供的欄位，不覆蓋其他欄位（如 guessed_min、guessed_multiple），
        方便 Min/Multiple 估算流程獨立寫入。
        """
        if df is None or df.empty:
            logger.warning("⚠️ 沒有補貨資料需要上傳")
            return

        collection_name = 'replenishment'
        logger.info(f"🚀 開始比對補貨資料 (共 {len(df)} 筆)...")

        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient='records')

        batch = self.db.batch()
        batch_count = 0
        total_updated = 0
        skipped_count = 0
        new_cache = self.local_cache.copy()

        for item in records:
            product_code = str(item.get('ProductCode', '')).strip()
            if not product_code:
                continue

            cache_key = f"repl:{product_code}"
            current_hash = self._generate_replenishment_hash(item)

            if cache_key in self.local_cache and self.local_cache[cache_key] == current_hash:
                skipped_count += 1
                continue

            doc_ref = self.db.collection(collection_name).document(product_code)
            batch.set(doc_ref, item, merge=True)  

            new_cache[cache_key] = current_hash
            batch_count += 1
            total_updated += 1

            if batch_count >= 400:
                batch.commit()
                logger.info(f"   ...已更新 {total_updated} 筆補貨資料")
                batch = self.db.batch()
                batch_count = 0

        if batch_count > 0:
            batch.commit()

        self.local_cache = new_cache
        self._save_cache()

        logger.info("✨ 補貨資料同步完成！")
        logger.info(f"   - 實際寫入: {total_updated} 筆 (消耗額度)")
        logger.info(f"   - 略過未變: {skipped_count} 筆 (節省額度)")

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

    def upload_classification_df(self, df, collection_name='products'):
        """
        上傳分類結果：只寫入 ABC_Class、XYZ_Class 與 note
        """
        if df is None or df.empty:
            logger.warning("⚠️ 分類資料為空，略過上傳")
            return

        required_cols = ['ProductCode', 'ABC_Class', 'XYZ_Class', 'Note']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"❌ 分類 CSV 缺少欄位: {col}")
                return

        df = df[required_cols].copy()
        df = df.where(pd.notnull(df), None)

        records = []
        for _, row in df.iterrows():
            product_code = str(row.get('ProductCode', '')).strip()
            if not product_code:
                continue
            records.append({
                'ProductCode': product_code,
                'ABC_Class': row.get('ABC_Class'),
                'XYZ_Class': row.get('XYZ_Class'),
                'note': row.get('Note')
            })

        if not records:
            logger.warning("⚠️ 分類資料為空，略過上傳")
            return

        batch = self.db.batch()
        batch_count = 0
        total_updated = 0
        skipped_count = 0

        new_cache = self.local_cache.copy()

        for item in records:
            product_code = item['ProductCode']
            cache_key = f"class:{product_code}"
            current_hash = self._generate_classification_hash(item)

            if cache_key in self.local_cache and self.local_cache[cache_key] == current_hash:
                skipped_count += 1
                continue

            # 要寫入的 payload：只含 ABC/XYZ 與 note
            doc_payload = {
                'ABC_Class': item.get('ABC_Class'),
                'XYZ_Class': item.get('XYZ_Class'),
                'note': item.get('note')
            }

            # 1) 寫入主要 collection（預設 products）
            primary_ref = self.db.collection(collection_name).document(product_code)
            batch.set(primary_ref, doc_payload, merge=True)

            # 2) 同步一份到 replenishment collection，方便補貨後台直接查詢 ABC/XYZ
            repl_ref = self.db.collection('replenishment').document(product_code)
            batch.set(repl_ref, doc_payload, merge=True)

            new_cache[cache_key] = current_hash
            batch_count += 1
            total_updated += 1

            if batch_count >= 400:
                batch.commit()
                logger.info(f"   ...已更新 {total_updated} 筆異動分類")
                batch = self.db.batch()
                batch_count = 0

        if batch_count > 0:
            batch.commit()

        self.local_cache = new_cache
        self._save_cache()

        logger.info("✨ 分類同步完成！")
        logger.info(f"   - 實際寫入: {total_updated} 筆 (消耗額度)")
        logger.info(f"   - 略過未變: {skipped_count} 筆 (節省額度)")