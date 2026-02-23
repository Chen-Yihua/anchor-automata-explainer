import tarfile
import json
from io import BytesIO, StringIO
from typing import Union, Tuple, Optional, Dict, List
import numpy as np
import pandas as pd
import requests
from requests import RequestException
from sklearn.preprocessing import LabelEncoder
from alibi.utils.data import Bunch
from PIL import Image
import os

def fetch_custom_dataset(
    source: str,
    mode: str,  # "tabular" | "text" | "image"
    return_X_y: bool = False,
    target_col: Optional[str] = None,
    target_size: Optional[Tuple[int, int]] = None,
    timeout: int = 2,
    y_array: Optional[Union[np.ndarray, List[int]]] = None,
    class_names: Optional[List[str]] = None,
    color_mode: str = "rgb",     # "rgb" 或 "grayscale"
    normalize: bool = False,     # True→回傳 float32/255；False→uint8
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    通用載入器樣板：
    - source: 可以是 URL（csv / tar.gz）或本地資料夾路徑
    - mode: 指定資料型別
    - return_X_y: True→(X,y)，False→Bunch
    - target_col: tabular/text 的標籤欄名（若用資料夾結構可不填）
    - target_size: image resize 用
    - timeout: 下載逾時秒數
    - y_array: 若提供，則直接用這個當 y（不做 LabelEncoder）
    - class_names: 若提供，則直接用這個當 target_names
    - color_mode: image 模式，"rgb" 或 "grayscale"
    - normalize: image 是否回傳 float32/255（否則 uint8）
    """

    def _download(url: str) -> bytes:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except RequestException:
            raise

    if mode == "tabular":
        
        # 支援 URL 或本地檔
        if source.startswith("http"):
            content = _download(source).decode("utf-8")
            df = pd.read_csv(StringIO(content))
        else:
            df = pd.read_csv(source)
        df = df.where(pd.notnull(df), None)

        if target_col is None or target_col not in df.columns:
            raise ValueError("請提供正確的 target_col（標籤欄位）")

        y_raw = df[target_col].astype(str).values
        X_df = df.drop(columns=[target_col])

        # raw_data = X_df.values
        raw_data = []
        for row in X_df.itertuples(index=False, name=None):
            clean_row = [x for x in row if x not in (None, "", np.nan)]
            raw_data.append(clean_row)

        # 對 object 欄位做 LabelEncoder，並保留對應表
        # feature_names = list(X_df.columns)
        # category_map: Dict[int, List[str]] = {}
        # for i, f in enumerate(feature_names):
        #     if X_df[f].dtype == "object":
        #         le = LabelEncoder()
        #         X_df[f] = le.fit_transform(X_df[f].astype(str).values)
        #         category_map[i] = list(le.classes_)
        feature_names = list(X_df.columns)
        category_map: Dict[int, List[str]] = {}
        for i, f in enumerate(feature_names):
            col = X_df[f].astype(str).values
            col = [c for c in col if c not in ("nan", "None", "NaN", "", None)]
            if len(col) == 0:
                continue
            le = LabelEncoder()
            le.fit(col)
            category_map[i] = list(le.classes_)

        # 轉 label 為 int
        y_le = LabelEncoder()
        y = y_le.fit_transform(y_raw)
        target_names = list(y_le.classes_)

        X = []
        for row in X_df.itertuples(index=False, name=None):
            clean_row = [x for x in row if pd.notna(x) and x not in ("", None, "nan", "NaN", "None")]
            X.append(clean_row)
        X = np.array(X, dtype=object)


        if return_X_y:
            return X, y

        return Bunch(
            data=X,
            raw_data=raw_data,
            target=y,
            feature_names=feature_names,
            target_names=target_names,
            category_map=category_map
        )

    elif mode == "text":
        # CSV/TSV with columns: text, label
        if source.endswith(".csv") or source.endswith(".tsv") or not source.startswith(("http", "https")):
            sep = "\t" if source.endswith(".tsv") else ","
            if source.startswith("http"):
                content = _download(source).decode("utf-8")
                df = pd.read_csv(StringIO(content), sep=sep)
            else:
                df = pd.read_csv(source, sep=sep)

            if "text" not in df.columns or "label" not in df.columns:
                raise ValueError("text 模式需要 columns: text, label")
            texts = df["text"].astype(str).tolist()
            y_le = LabelEncoder()
            y = y_le.fit_transform(df["label"].astype(str).values)
            target_names = list(y_le.classes_)

        # tar.gz 內部每個檔案夾/檔案代表一個 label（類似 movie sentiment）
        elif source.endswith(".tar.gz") or source.endswith(".tgz") or source.startswith("http"):  
            if source.startswith("http"):
                blob = requests.get(source, timeout=2)
                blob.raise_for_status()
                tar = tarfile.open(fileobj=BytesIO(blob.content), mode="r:gz")
            else:
                with open(source, "rb") as f:
                    blob = f.read()
                    tar = tarfile.open(fileobj=BytesIO(blob), mode="r:gz")
            texts, labels = [], []
            # 這裡依你的檔案調整 texts, labels 和 target_names，參考以下範例
            for i, member in enumerate(tar.getmembers()[1:]):
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                for line in f.readlines(): # 每行一個樣本
                    try:
                        s = line.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        continue
                    texts.append(s)
                    labels.append(i)  # 每個成員一類，依需要改
            y = np.array(labels)
            target_names = ['negative', 'positive']
        else:
            raise ValueError("不支援的 text source")

        if return_X_y:
            return texts, y

        return Bunch(
            data=texts,
            target=y,
            target_names=target_names if target_names is not None else []
        )

    elif mode == "image":
        # 支援資料夾結構 dataset/{split}/{class}/*.jpg
        def _read_image(fp: BytesIO, target_size: Optional[Tuple[int, int]]):
            im = Image.open(fp).convert("RGB")
            if target_size:
                im = im.resize(target_size)
            return np.array(im)

        def _ensure_4d(X0: np.ndarray) -> np.ndarray:
            """把 (N,H,W) 或 (N,H,W,1/3) 統一成 (N,H,W,C)。"""
            X0 = np.asarray(X0)
            if X0.ndim == 3:  # (N,H,W) → 灰階
                X0 = X0[..., None]
            return X0

        def _to_rgb_if_needed(X0: np.ndarray) -> np.ndarray:
            if color_mode == "rgb" and X0.shape[-1] == 1:
                return np.repeat(X0, 3, axis=-1)
            if color_mode == "grayscale" and X0.shape[-1] != 1:
                # 強制轉灰階（逐張保險做）
                imgs = []
                for img in X0:
                    im = Image.fromarray(img).convert("L")
                    if target_size:
                        im = im.resize(target_size)
                    arr = np.array(im)[..., None]
                    imgs.append(arr)
                return np.stack(imgs, axis=0)
            return X0

        def _resize_batch_if_needed(X0: np.ndarray) -> np.ndarray:
            if target_size is None or (X0.shape[1], X0.shape[2]) == target_size:
                return X0
            imgs = []
            # 用 RGB 路徑 resize，比較穩（即使是 3 通道也 safe）
            for img in X0:
                im = Image.fromarray(img).convert("RGB").resize(target_size)
                imgs.append(np.array(im))
            return np.stack(imgs, axis=0)

        # ndarray / (X, y) / dict
        if isinstance(source, np.ndarray) or \
        (isinstance(source, (list, tuple)) and len(source) == 2 and isinstance(source[0], np.ndarray)) or \
        (isinstance(source, dict) and ("X" in source or "x" in source)):

            # 解析輸入
            if isinstance(source, np.ndarray):
                X0 = source
                y = None if y_array is None else np.asarray(y_array)
            elif isinstance(source, (list, tuple)):
                X0, y = source[0], np.asarray(source[1])
            else:  # dict
                keyX = "X" if "X" in source else "x"
                X0 = source[keyX]
                y  = np.asarray(source.get("y")) if "y" in source else None
                # 允許 dict 直接帶 class_names 覆寫
                if source.get("class_names") is not None:
                    class_names = list(source["class_names"])

            # 標準化 shape / 通道數
            X0 = _ensure_4d(X0)               # → (N,H,W,C)
            X0 = _to_rgb_if_needed(X0)        # 視 color_mode 決定是否擴成 RGB
            X  = _resize_batch_if_needed(X0)  # 視 target_size 決定是否 resize

            # dtype / normalize
            if normalize:
                X = X.astype(np.float32) / 255.0
                x_dtype = np.float32
            else:
                X = X.astype(np.uint8, copy=False)
                x_dtype = np.uint8

            # 處理標籤與類別名稱
            if y is None:
                y = np.zeros((len(X),), dtype=int)
                target_names = ["0"]
                int_to_str_labels = {0: "0"}
                str_to_int_labels = {"0": 0}
            else:
                y = y.astype(int)
                uniq = sorted(np.unique(y).tolist())
                # 將 y 重新壓縮成 0..K-1（避免跳號）
                remap = {old: i for i, old in enumerate(uniq)}
                y = np.array([remap[v] for v in y], dtype=int)
                if class_names is not None:
                    target_names = list(class_names)
                else:
                    target_names = [str(u) for u in uniq]
                int_to_str_labels = {i: target_names[i] for i in range(len(target_names))}
                str_to_int_labels = {target_names[i]: i for i in range(len(target_names))}
            
            if return_X_y:
                return X, y
            return Bunch(
                data=X,
                target=y,
                target_names=target_names,
                int_to_str_labels=int_to_str_labels,
                str_to_int_labels=str_to_int_labels,
                dtype=X.dtype,
                color_mode=color_mode,
            )

        
        X_list, y_list = [], []
        class_to_id: Dict[str, int] = {}

        # tar.gz（本地或 URL）
        if (isinstance(source, str) and
            (source.endswith(".tar.gz") or source.endswith(".tgz") or source.startswith("http"))):
            if source.startswith("http"):
                blob = _download(source)
            else:
                with open(source, "rb") as f:
                    blob = f.read()
            tar = tarfile.open(fileobj=BytesIO(blob), mode="r:gz")
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                parts = member.name.split("/")
                # 假設結構: split/class/filename 或 class/filename
                if len(parts) < 2:
                    continue
                class_name = parts[-2]
                if class_name not in class_to_id:
                    class_to_id[class_name] = len(class_to_id)
                f = tar.extractfile(member)
                if f is None:
                    continue
                arr = _read_image(BytesIO(f.read()), target_size)
                X_list.append(arr)
                y_list.append(class_to_id[class_name])
        
        else:
            # 本地資料夾：支援 dataset/{split}/{class}/*.jpg 或 dataset/{class}/*.jpg
            if not isinstance(source, str) or not os.path.isdir(source):
                raise ValueError("image 模式：source 必須是 ndarray/(X,y)/dict，或有效的資料夾/壓縮檔/URL。")
            for split in ("train", "test"):
                split_dir = os.path.join(source, split)
                if not os.path.isdir(split_dir):
                    continue
                for class_name in sorted(os.listdir(split_dir)):
                    cls_dir = os.path.join(split_dir, class_name)
                    if not os.path.isdir(cls_dir):
                        continue
                    if class_name not in class_to_id:
                        class_to_id[class_name] = len(class_to_id)
                    for fname in os.listdir(cls_dir):
                        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            continue
                        with open(os.path.join(cls_dir, fname), "rb") as f:
                            arr = _read_image(BytesIO(f.read()), target_size)
                        X_list.append(arr)
                        y_list.append(class_to_id[class_name])

            # 若沒有 train/test 結構，試 dataset/{class}/*.jpg
            if not X_list:
                for class_name in sorted(os.listdir(source)):
                    cls_dir = os.path.join(source, class_name)
                    if not os.path.isdir(cls_dir):
                        continue
                    if class_name not in class_to_id:
                        class_to_id[class_name] = len(class_to_id)
                    for fname in os.listdir(cls_dir):
                        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            continue
                        with open(os.path.join(cls_dir, fname), "rb") as f:
                            arr = _read_image(BytesIO(f.read()), target_size)
                        X_list.append(arr)
                        y_list.append(class_to_id[class_name])

        if not X_list:
            raise ValueError("找不到影像，請確認路徑/壓縮結構是否正確。")

        X = np.stack(X_list, axis=0).astype(np.uint8)
        y = np.array(y_list, dtype=int)
        int_to_str_labels = {v: k for k, v in class_to_id.items()}
        str_to_int_labels = class_to_id

        if return_X_y:
            return X, y

        return Bunch(
            data=X,
            target=y,
            target_names=[int_to_str_labels[i] for i in range(len(int_to_str_labels))],
            int_to_str_labels=int_to_str_labels,
            str_to_int_labels=str_to_int_labels,
        )

    else:
        raise ValueError("mode 需為 'tabular' | 'text' | 'image'")
